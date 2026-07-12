"""
Turns a topic into a scene-by-scene documentary narration script.

Primary LLM: SambaNova Cloud (OpenAI-compatible endpoint, running the
open-weight gpt-oss-120b model — fast RDU-hosted inference).
Falls back to Groq if SambaNova fails (rate limit/quota exhausted, API
error, or unparseable output after retries).

Gemini and SambaNova support were both removed: Gemini's free-tier key
hit a permanent "limit: 0" quota wall, and SambaNova's free-tier quota
was also getting exhausted, wasting retry time on every chunk.
"""

import json
import math
import re
import time
import requests
from groq import Groq, RateLimitError, APIError

from config import (
    GROQ_API_KEY, CHANNEL_NAME, LANGUAGES, DEFAULT_LANGUAGE, DEFAULT_DURATION_MINUTES,
    VIDEO_STYLES, DEFAULT_VIDEO_STYLE,
)

# Optional fallback key — fallback is simply skipped if not configured.
try:
    from config import SAMBANOVA_API_KEY
except ImportError:
    SAMBANOVA_API_KEY = None

WORDS_PER_MINUTE = 150
MIN_ACCEPTABLE_RATIO = 0.85
CHUNK_TARGET_WORDS = 200  # bigger chunks = fewer sequential API round-trips per
                          # script (each one carries fixed overhead — network
                          # latency, provider fallback checks, etc. — regardless
                          # of chunk size, so fewer/larger chunks finish faster
                          # overall). Was 80; a 450-word "short" script now
                          # takes ~3 chunks instead of ~6.
SAMBANOVA_MODEL_NAME = "gpt-oss-120b"
SAMBANOVA_URL = "https://api.sambanova.ai/v1/chat/completions"
SAMBANOVA_MAX_TOKENS_CEILING = 8000  # hard cap so the +4000 retry growth in
                                      # _parse_llm_json can't run away

def _target_word_count(duration_minutes: float) -> int:
    return round(duration_minutes * WORDS_PER_MINUTE)


def _scene_count_for_words(word_budget: int) -> tuple:
    scene_low = max(2, round(word_budget / 90))
    scene_high = max(scene_low + 1, round(word_budget / 50))
    return scene_low, scene_high


def _build_metadata_prompt(language_name: str, style_key: str) -> str:
    narrator_style = VIDEO_STYLES.get(style_key, VIDEO_STYLES[DEFAULT_VIDEO_STYLE])["narrator_style"]
    return f"""You are a scriptwriter for a YouTube channel about history and finance \
(channel: {CHANNEL_NAME}). Write as {narrator_style}.

Write the ENTIRE response in {language_name}. Do not mix in another language unless \
{language_name} is English.

Return ONLY valid JSON, no markdown fences, no preamble, in this exact shape:
{{
  "title": "SEO-friendly YouTube title, under 100 characters, written in {language_name}",
  "description": "2-3 sentence YouTube description, written in {language_name}",
  "tags": ["tag1", "tag2", "..."]
}}

Do not include any text outside the JSON object."""


def _build_scenes_prompt(
    language_name: str, style_key: str, word_budget: int, is_first_chunk: bool
) -> str:
    scene_low, scene_high = _scene_count_for_words(word_budget)
    narrator_style = VIDEO_STYLES.get(style_key, VIDEO_STYLES[DEFAULT_VIDEO_STYLE])["narrator_style"]

    return f"""You are a scriptwriter for a YouTube channel about history and finance \
(channel: {CHANNEL_NAME}). Write as {narrator_style}.

Write the ENTIRE response in {language_name}. Do not mix in another language unless \
{language_name} is English.

Return ONLY valid JSON, no markdown fences, no preamble, in this exact shape:
{{
  "scenes": [
    {{
      "narration": "2-4 sentences of narration for this scene, written in {language_name}",
      "image_keywords": "2-4 words IN ENGLISH describing what image should show for this scene",
      "duration_seconds": 25
    }}
  ]
}}

CRITICAL: Do NOT include title, description, or tags. Only include the "scenes" array.
CRITICAL LENGTH REQUIREMENT: the combined narration across every scene in THIS response \
must add up to approximately {word_budget} words in total (aim for {word_budget}-\
{int(word_budget * 1.15)} words — never less). Use {scene_low}-{scene_high} scenes to hit \
that total, with each scene's narration long enough to read aloud in roughly its \
duration_seconds (~2.5 words per second). If you are unsure whether you have written \
enough, err on the side of writing more scenes and more narration per scene, not fewer. \
Do not include any text outside the JSON object."""


def _count_narration_words(scenes: list) -> int:
    return sum(len(scene.get("narration", "").split()) for scene in scenes)


def _repair_json(raw: str) -> str:
    if raw.count('"') % 2 != 0:
        raw = raw + '"'
    open_brackets = raw.count('[') - raw.count(']')
    open_braces = raw.count('{') - raw.count('}')
    for _ in range(open_brackets):
        raw = raw + ']'
    for _ in range(open_braces):
        raw = raw + '}'
    raw = re.sub(r',\s*([\}\]])', r'\1', raw)
    return raw


def _extract_scenes_from_truncated(raw: str) -> list:
    try:
        repaired = _repair_json(raw)
        data = json.loads(repaired)
        return data.get("scenes", [])
    except (json.JSONDecodeError, AttributeError):
        pass

    scenes = []
    narrations = re.findall(r'"narration"\s*:\s*"((?:\\.|[^"\\])*)"', raw)
    keywords = re.findall(r'"image_keywords"\s*:\s*"((?:\\.|[^"\\])*)"', raw)
    durations = re.findall(r'"duration_seconds"\s*:\s*(\d+)', raw)

    for i, narration in enumerate(narrations):
        if narration.strip():
            scenes.append({
                "narration": narration,
                "image_keywords": keywords[i] if i < len(keywords) else "historical scene",
                "duration_seconds": int(durations[i]) if i < len(durations) else 25,
            })
    return scenes


MAX_JSON_RETRY_DEPTH = 2  # hard ceiling on how many times we'll re-call the LLM
                          # for a bigger completion before giving up entirely.


def _parse_llm_json(raw: str, attempt_more_tokens_callback=None, max_tokens: int = 0, depth: int = 0):
    """Shared JSON parsing/repair logic used by every provider path.

    `depth` guards against unbounded recursion: if the LLM keeps returning
    unparseable JSON, attempt_more_tokens_callback would otherwise be called
    again and again forever (each call itself doing its own multi-attempt
    retry loop with sleeps), which is what caused the pipeline to hang.
    """
    if raw is None or not raw.strip():
        raise RuntimeError("LLM returned an empty response.")

    raw = raw.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        repaired = _repair_json(raw)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            scenes = _extract_scenes_from_truncated(raw)
            if scenes:
                return {"scenes": scenes}
            if attempt_more_tokens_callback and max_tokens and depth < MAX_JSON_RETRY_DEPTH:
                return attempt_more_tokens_callback(max_tokens + 4000, depth + 1)
            raise RuntimeError(
                f"LLM did not return valid JSON after {depth} retry attempt(s): "
                f"{e}\nRaw output:\n{raw[:500]}"
            )


# This Groq account is on the "on_demand" tier, which caps requests at 8000
# tokens per minute (prompt + max_completion_tokens counts against this,
# whether or not the model actually uses that many). Stay comfortably under
# that on every single call, no matter how large the request wanted to be.
GROQ_TPM_LIMIT = 8000
GROQ_SAFETY_MARGIN = 1500  # headroom for prompt tokens + estimation error


def _call_groq(client: Groq, system_prompt: str, user_content: str, max_tokens: int, depth: int = 0) -> dict:
    max_tokens = min(max_tokens, GROQ_TPM_LIMIT - GROQ_SAFETY_MARGIN)

    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            break
        except RateLimitError:
            if attempt < 3:
                time.sleep(8 + attempt * 4)
            else:
                raise
        except APIError as e:
            msg = str(e)
            if "rate_limit_exceeded" in msg or "tokens per minute" in msg or "413" in msg:
                # Request was too large for the TPM budget: shrink it and retry
                # rather than failing the whole job outright.
                if attempt < 3 and max_tokens > 1000:
                    max_tokens = max(1000, int(max_tokens * 0.6))
                    time.sleep(6 + attempt * 4)
                    continue
            raise RuntimeError(f"Groq API error: {e}")

    raw = response.choices[0].message.content

    def _retry_with_more_tokens(new_max_tokens, new_depth):
        return _call_groq(client, system_prompt, user_content, new_max_tokens, depth=new_depth)

    return _parse_llm_json(raw, _retry_with_more_tokens, max_tokens, depth=depth)


# Once SambaNova reports quota/credit exhaustion, there's no point retrying
# it on every subsequent chunk for the rest of this process's lifetime — that
# was costing ~15-20+ seconds of dead retries per chunk. Skip straight to
# Groq after the first confirmed exhaustion; a Repl restart re-checks it.
_samba_quota_exhausted = False


def _call_samba(system_prompt: str, user_content: str, max_tokens: int, depth: int = 0) -> dict:
    global _samba_quota_exhausted

    if not SAMBANOVA_API_KEY:
        raise RuntimeError(
            "SambaNova unavailable: set SAMBANOVA_API_KEY in "
            "config.py / Replit Secrets."
        )

    if _samba_quota_exhausted:
        raise RuntimeError("SambaNova skipped: quota already confirmed exhausted this run.")

    max_tokens = min(max_tokens, SAMBANOVA_MAX_TOKENS_CEILING)

    headers = {
        "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SAMBANOVA_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(SAMBANOVA_URL, headers=headers, json=payload, timeout=60)

            # Quota/credit exhaustion (402 Payment Required, or a 429 whose body
            # actually says "quota"/"credit" rather than a normal short-term rate
            # limit) won't recover by retrying a few seconds later — fail fast
            # and flag it so later chunks don't repeat the same dead attempt.
            body_lower = (resp.text or "").lower()
            if resp.status_code == 402 or (
                resp.status_code == 429 and ("quota" in body_lower or "credit" in body_lower)
            ):
                _samba_quota_exhausted = True
                raise RuntimeError(f"SambaNova quota/credits exhausted (HTTP {resp.status_code}): {resp.text[:200]}")

            if resp.status_code == 429:
                raise RuntimeError("SambaNova rate limited")
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]

            def _retry_with_more_tokens(new_max_tokens, new_depth):
                return _call_samba(system_prompt, user_content, new_max_tokens, depth=new_depth)

            return _parse_llm_json(raw, _retry_with_more_tokens, max_tokens, depth=depth)

        except Exception as e:
            last_error = e
            if _samba_quota_exhausted:
                # Fail immediately, no point sleeping and retrying a dead quota.
                raise RuntimeError(f"SambaNova call failed: {e}")
            if attempt < 2:
                time.sleep(4 + attempt * 4)
            else:
                raise RuntimeError(f"SambaNova call failed: {e}")

    raise RuntimeError(f"SambaNova failed after retries: {last_error}")


def _call_llm(client: Groq, system_prompt: str, user_content: str, max_tokens: int) -> dict:
    """Tries SambaNova first (primary); falls back to Groq if it fails."""
    errors = []

    if not _samba_quota_exhausted:
        or_start = time.time()
        try:
            result = _call_samba(system_prompt, user_content, max_tokens)
            print(f"[script_generator]   SambaNova succeeded in {time.time() - or_start:.1f}s")
            return result
        except Exception as e:
            errors.append(f"SambaNova: {e}")
            print(f"[script_generator]   SambaNova failed after {time.time() - or_start:.1f}s ({e}). Trying Groq...")
    else:
        print("[script_generator]   Skipping SambaNova (quota confirmed exhausted this run). Using Groq directly...")

    groq_start = time.time()
    try:
        result = _call_groq(client, system_prompt, user_content, max_tokens)
        print(f"[script_generator]   Groq succeeded in {time.time() - groq_start:.1f}s")
        return result
    except Exception as e:
        errors.append(f"Groq: {e}")
        print(f"[script_generator]   Groq also failed after {time.time() - groq_start:.1f}s ({e})")

    raise RuntimeError("All LLM providers failed.\n" + "\n".join(errors))


def _generate_metadata(client: Groq, topic: str, language_name: str, style: str) -> dict:
    system_prompt = _build_metadata_prompt(language_name, style)
    part = _call_llm(client, system_prompt, f"Topic: {topic}", max_tokens=4096)
    return {
        "title": part.get("title") or topic,
        "description": part.get("description") or "",
        "tags": part.get("tags") or [],
    }


def _generate_scenes_chunk(
    client: Groq, topic: str, language_name: str, style: str,
    word_budget: int, is_first_chunk: bool, previous_narration_tail: str,
) -> dict:
    system_prompt = _build_scenes_prompt(language_name, style, word_budget, is_first_chunk)
    # llama-3.3-70b-versatile has no hidden "reasoning" token overhead, so the
    # completion budget can track the actual chunk size (roughly 6 tokens per
    # word of narration, plus JSON overhead) instead of needing a huge flat
    # floor. Still capped well under the account's 8000 TPM limit in
    # _call_groq.
    max_tokens = min(3000, max(600, word_budget * 6))

    if is_first_chunk:
        user_content = f"Topic: {topic}"
    else:
        user_content = (
            f"Topic: {topic}\n\n"
            f"This continues a script already in progress. The most recent narration so far "
            f"was:\n\"{previous_narration_tail}\"\n\n"
            f"Write the NEXT part of the same script, continuing the story naturally from "
            f"there — do not repeat earlier content, do not restart the introduction."
        )

    part = _call_llm(client, system_prompt, user_content, max_tokens)

    if "scenes" not in part or not part["scenes"]:
        raise RuntimeError("Generated script chunk has no scenes.")

    actual_words = _count_narration_words(part["scenes"])

    if actual_words < word_budget * MIN_ACCEPTABLE_RATIO:
        try:
            retry_part = _call_llm(
                client, system_prompt,
                user_content + f"\n\n(Your last attempt only had about {actual_words} words — "
                f"write more this time, aiming for at least {word_budget} words.)",
                max_tokens,
            )
            if "scenes" in retry_part and _count_narration_words(retry_part["scenes"]) > actual_words:
                part = retry_part
        except RuntimeError:
            pass

    return part


def generate_script(
    topic: str,
    language: str = DEFAULT_LANGUAGE,
    duration_minutes: float = DEFAULT_DURATION_MINUTES,
    style: str = DEFAULT_VIDEO_STYLE,
    progress_callback=None,
) -> dict:
    """
    progress_callback, if given, is called as progress_callback(chunks_done, total_chunks)
    after every chunk (and once with (0, total_chunks) before the first chunk starts),
    so a caller (e.g. the Flask job status) can report fine-grained progress instead
    of just "script step in progress".
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in Replit Secrets.")

    language_name = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])
    total_target_words = _target_word_count(duration_minutes)
    num_chunks = max(1, math.ceil(total_target_words / CHUNK_TARGET_WORDS))

    client = Groq(api_key=GROQ_API_KEY)

    print(f"[script_generator] Starting: topic='{topic}', target={total_target_words} words "
          f"across {num_chunks} chunk(s)")

    metadata = _generate_metadata(client, topic, language_name, style)
    print(f"[script_generator] Metadata generated: \"{metadata['title']}\"")

    if progress_callback:
        progress_callback(0, num_chunks)

    all_scenes = []
    remaining_words = total_target_words

    for chunk_index in range(num_chunks):
        is_first = chunk_index == 0
        chunks_left = num_chunks - chunk_index
        word_budget = max(60, round(remaining_words / chunks_left)) if remaining_words > 0 else CHUNK_TARGET_WORDS

        previous_tail = ""
        if all_scenes:
            previous_tail = all_scenes[-1]["narration"][-300:]

        chunk_start = time.time()
        part = _generate_scenes_chunk(
            client, topic, language_name, style, word_budget, is_first, previous_tail,
        )
        chunk_elapsed = time.time() - chunk_start

        all_scenes.extend(part["scenes"])
        remaining_words -= _count_narration_words(part["scenes"])

        words_so_far = _count_narration_words(all_scenes)
        print(f"[script_generator] Chunk {chunk_index + 1}/{num_chunks} done in {chunk_elapsed:.1f}s "
              f"— {words_so_far}/{total_target_words} words, {len(all_scenes)} scenes so far")

        if progress_callback:
            progress_callback(chunk_index + 1, num_chunks)

    actual_total_words = _count_narration_words(all_scenes)
    print(f"[script_generator] target={total_target_words} words, actual={actual_total_words} words, "
          f"{len(all_scenes)} scenes across {num_chunks} chunk(s)")

    return {
        "title": metadata["title"],
        "description": metadata["description"],
        "tags": metadata["tags"],
        "scenes": all_scenes,
    }


if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "The Tulip Mania bubble of 1637"
    result = generate_script(topic)
    print(json.dumps(result, indent=2, ensure_ascii=False))
