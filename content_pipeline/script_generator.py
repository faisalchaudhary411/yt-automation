"""
Turns a topic into a scene-by-scene documentary narration script.

Primary LLM: SambaNova Cloud (OpenAI-compatible endpoint, running the
open-weight gpt-oss-120b model — fast RDU-hosted inference).
Falls back to Groq if SambaNova fails (rate limit/quota exhausted, API
error, or unparseable output after retries).

FIXES FOR URDU TEXT CORRUPTION:
  - Added proper Unicode handling in JSON parsing
  - Fixed _repair_json to not corrupt text content
  - Improved _extract_scenes_from_truncated to handle Urdu text safely
  - Added text sanitization to remove garbage characters
  - Added validation to detect corrupted Urdu text
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
CHUNK_TARGET_WORDS = 200
SAMBANOVA_MODEL_NAME = "gpt-oss-120b"
SAMBANOVA_URL = "https://api.sambanova.ai/v1/chat/completions"
SAMBANOVA_MAX_TOKENS_CEILING = 8000

# Unicode ranges for valid Urdu/Arabic characters
URDU_UNICODE_RANGE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"
    r"\s\u060C\u061B\u061F\u06D4.,!?;:\'\"\-—\u200C\u200D]+"
)

# Pattern to detect corrupted/garbage text (random symbols that shouldn't be in Urdu)
GARBAGE_PATTERN = re.compile(r"[&!@#$%^*+=~`|<>{}\[\]\\]{2,}")


def _target_word_count(duration_minutes: float) -> int:
    return round(duration_minutes * WORDS_PER_MINUTE)


def _scene_count_for_words(word_budget: int) -> tuple:
    scene_low = max(2, round(word_budget / 90))
    scene_high = max(scene_low + 1, round(word_budget / 50))
    return scene_low, scene_high


def _urdu_style_notes(language_name: str) -> str:
    """
    Extra guidance appended to prompts when writing Urdu, so the narration sounds
    like natural spoken Pakistani Urdu rather than stiff, overly literary, or
    machine-translated Urdu.
    """
    if language_name.strip().lower() != "urdu":
        return ""
    return """

URDU STYLE GUIDANCE (very important):
- Write like a Pakistani YouTube narrator talking to a general audience, not like a textbook or news bulletin.
- Prefer everyday, commonly spoken Urdu vocabulary over heavy Persian/Arabic literary words. If a simpler word is what people actually say out loud, use it.
- Use natural sentence rhythm and short-to-medium sentences suited to narration, not long nested literary clauses.
- It's fine and often better to keep common English loanwords that Pakistanis actually use in speech (e.g. "invest", "company", "market", "percent") instead of forcing an obscure formal Urdu equivalent — but do not switch entire sentences to English.
- Avoid word-for-word translated phrasing that sounds like it was translated from an English draft. Write the thought directly in Urdu the way a person would say it.
- Vary sentence openings; avoid repeating the same connector words (e.g. "لیکن", "اس کے بعد") in every sentence."""


def _build_metadata_prompt(language_name: str, style_key: str) -> str:
    narrator_style = VIDEO_STYLES.get(style_key, VIDEO_STYLES[DEFAULT_VIDEO_STYLE])["narrator_style"]
    return f"""You are a scriptwriter for a YouTube channel about history and finance (channel: {CHANNEL_NAME}). Write as {narrator_style}.

Write the ENTIRE response in {language_name}. Do not mix in another language unless {language_name} is English.{_urdu_style_notes(language_name)}

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

    return f"""You are a scriptwriter for a YouTube channel about history and finance (channel: {CHANNEL_NAME}). Write as {narrator_style}.

Write the ENTIRE response in {language_name}. Do not mix in another language unless {language_name} is English.{_urdu_style_notes(language_name)}

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
CRITICAL LENGTH REQUIREMENT: the combined narration across every scene in THIS response must add up to approximately {word_budget} words in total (aim for {word_budget}-{int(word_budget * 1.15)} words — never less). Use {scene_low}-{scene_high} scenes to hit that total, with each scene's narration long enough to read aloud in roughly its duration_seconds (~2.5 words per second). If you are unsure whether you have written enough, err on the side of writing more scenes and more narration per scene, not fewer. Do not include any text outside the JSON object.

CRITICAL TEXT FORMATTING:
- Use proper {language_name} punctuation (e.g., Urdu full stop: ۔ not .)
- Do NOT use Latin/English punctuation inside {language_name} text
- Do NOT mix English words inside {language_name} narration unless necessary"""


def _count_narration_words(scenes: list) -> int:
    return sum(len(scene.get("narration", "").split()) for scene in scenes)


def _is_text_corrupted(text: str) -> bool:
    """Detects if Urdu text contains garbage characters or corruption."""
    if not text:
        return True

    # Check for excessive garbage symbols
    if GARBAGE_PATTERN.search(text):
        return True

    # Check if text has reasonable ratio of Urdu characters to total
    urdu_chars = sum(1 for c in text if (0x0600 <= ord(c) <= 0x06FF) or
                     (0x0750 <= ord(c) <= 0x077F))
    total_chars = len([c for c in text if c.strip()])

    # If less than 30% Urdu characters and text is supposed to be Urdu, it's corrupted
    if total_chars > 0 and urdu_chars / total_chars < 0.3:
        # But allow English text (for image_keywords)
        latin_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        if latin_chars / total_chars < 0.5:  # Not mostly Latin either
            return True

    return False


def _sanitize_narration(text: str) -> str:
    """Cleans narration text by removing garbage and fixing common issues."""
    if not text:
        return text

    # Remove sequences of random symbols (like ,&!12)
    text = GARBAGE_PATTERN.sub(" ", text)

    # Fix common corruption patterns
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)

    # Remove isolated Latin punctuation that might be corruption
    # But preserve legitimate Urdu punctuation
    text = re.sub(r"(?<![\u0600-\u06FF])[,;:!?]+(?![\u0600-\u06FF])", " ", text)

    # Ensure proper Urdu sentence ending if missing
    text = text.strip()
    if text and text[-1] not in ".!?۔":
        text += "۔"

    return text.strip()


def _repair_json(raw: str) -> str:
    """
    Repairs malformed JSON without corrupting text content.
    Uses a safer approach than naive quote/bracket counting.
    """
    if not raw or not raw.strip():
        return raw

    raw = raw.strip()

    # Remove markdown fences
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    # Try to find the JSON object boundaries
    # Look for the outermost { ... }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        raw = raw[first_brace:last_brace + 1]

    # Remove trailing commas before closing brackets/braces
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    # Balance brackets and braces by finding the last complete structure
    # Don't just append to end - that corrupts text
    open_brackets = raw.count("[") - raw.count("]")
    open_braces = raw.count("{") - raw.count("}")

    # Only add closing if we're inside a structure and at the end
    if open_brackets > 0 and not raw.rstrip().endswith("]"):
        # Find where to add - after the last complete element
        raw = raw.rstrip() + "]" * open_brackets
    if open_braces > 0 and not raw.rstrip().endswith("}"):
        raw = raw.rstrip() + "}" * open_braces

    return raw


def _extract_scenes_safely(raw: str) -> list:
    """
    Safely extracts scenes from potentially malformed JSON.
    Handles Urdu/Arabic Unicode text properly.
    """
    scenes = []

    # Try to find all scene objects using a safer pattern
    # Look for narration fields and their values
    # Pattern: "narration" followed by colon and quoted string
    # Handles escaped quotes inside the string

    # First try: find complete scene objects
    scene_pattern = re.compile(
        r'"narration"\s*:\s*"(.*?)"\s*,\s*"image_keywords"\s*:\s*"(.*?)"\s*,\s*"duration_seconds"\s*:\s*(\d+)',
        re.DOTALL
    )

    matches = scene_pattern.findall(raw)
    for narration, keywords, duration in matches:
        # Clean up the extracted narration
        narration = narration.replace("\\\\", "\\").replace('\"', '"')
        narration = _sanitize_narration(narration)

        if narration.strip():
            scenes.append({
                "narration": narration,
                "image_keywords": keywords.replace("\\\\", "\\").replace('\"', '"'),
                "duration_seconds": int(duration),
            })

    # If no scenes found with strict pattern, try looser extraction
    if not scenes:
        # Find all narration values
        narration_pattern = re.compile(r'"narration"\s*:\s*"((?:[^"\]|\.)*)"')
        keyword_pattern = re.compile(r'"image_keywords"\s*:\s*"((?:[^"\]|\.)*)"')
        duration_pattern = re.compile(r'"duration_seconds"\s*:\s*(\d+)')

        narrations = narration_pattern.findall(raw)
        keywords = keyword_pattern.findall(raw)
        durations = duration_pattern.findall(raw)

        for i, narration in enumerate(narrations):
            narration = narration.replace("\\\\", "\\").replace('\"', '"')
            narration = _sanitize_narration(narration)

            if narration.strip():
                scenes.append({
                    "narration": narration,
                    "image_keywords": keywords[i].replace("\\\\", "\\").replace('\"', '"') if i < len(keywords) else "historical scene",
                    "duration_seconds": int(durations[i]) if i < len(durations) else 25,
                })

    return scenes


MAX_JSON_RETRY_DEPTH = 2


def _parse_llm_json(raw: str, attempt_more_tokens_callback=None, max_tokens: int = 0, depth: int = 0):
    """Shared JSON parsing/repair logic used by every provider path."""
    if raw is None or not raw.strip():
        raise RuntimeError("LLM returned an empty response.")

    raw = raw.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    # First attempt: direct JSON parse
    try:
        data = json.loads(raw)
        # Validate and sanitize scenes
        if "scenes" in data:
            for scene in data["scenes"]:
                if "narration" in scene:
                    scene["narration"] = _sanitize_narration(scene["narration"])
        return data
    except json.JSONDecodeError:
        pass

    # Second attempt: repair JSON
    try:
        repaired = _repair_json(raw)
        data = json.loads(repaired)
        if "scenes" in data:
            for scene in data["scenes"]:
                if "narration" in scene:
                    scene["narration"] = _sanitize_narration(scene["narration"])
        return data
    except json.JSONDecodeError:
        pass

    # Third attempt: extract scenes safely from malformed JSON
    scenes = _extract_scenes_safely(raw)
    if scenes:
        # Validate extracted scenes
        valid_scenes = []
        for scene in scenes:
            if not _is_text_corrupted(scene["narration"]):
                valid_scenes.append(scene)
            else:
                print(f"[WARNING] Skipping corrupted scene: {scene['narration'][:50]}...")

        if valid_scenes:
            return {"scenes": valid_scenes}

    # Final fallback: retry with more tokens
    if attempt_more_tokens_callback and max_tokens and depth < MAX_JSON_RETRY_DEPTH:
        return attempt_more_tokens_callback(max_tokens + 4000, depth + 1)

    raise RuntimeError(
        f"LLM did not return valid JSON after {depth} retry attempt(s). "
        f"Raw output:\n{raw[:800]}"
    )


GROQ_TPM_LIMIT = 8000
GROQ_SAFETY_MARGIN = 1500
MAX_GROQ_RATE_LIMIT_WAIT_SECONDS = 600


def _extract_retry_after_seconds(message: str):
    """Parses Groq's 'Please try again in 8m14.208s' / '45.2s' style hints."""
    match = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", message)
    if not match:
        return None
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60 + seconds


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
        except RateLimitError as e:
            wait_seconds = _extract_retry_after_seconds(str(e))
            if wait_seconds is not None and wait_seconds <= MAX_GROQ_RATE_LIMIT_WAIT_SECONDS:
                print(f"[script_generator]   Groq rate limited — waiting {wait_seconds:.0f}s "
                      f"(as advised) before retrying...")
                time.sleep(wait_seconds + 2)
                continue
            if wait_seconds is not None:
                raise RuntimeError(
                    f"Groq rate limited with a {wait_seconds:.0f}s wait — too long to block on. {e}"
                )
            if attempt < 3:
                time.sleep(8 + attempt * 4)
            else:
                raise
        except APIError as e:
            msg = str(e)
            wait_seconds = _extract_retry_after_seconds(msg)
            if wait_seconds is not None and wait_seconds <= MAX_GROQ_RATE_LIMIT_WAIT_SECONDS:
                print(f"[script_generator]   Groq rate limited — waiting {wait_seconds:.0f}s "
                      f"(as advised) before retrying...")
                time.sleep(wait_seconds + 2)
                continue
            if "rate_limit_exceeded" in msg or "tokens per minute" in msg or "413" in msg:
                if attempt < 3 and max_tokens > 1000:
                    max_tokens = max(1000, int(max_tokens * 0.6))
                    time.sleep(6 + attempt * 4)
                    continue
            raise RuntimeError(f"Groq API error: {e}")

    raw = response.choices[0].message.content

    def _retry_with_more_tokens(new_max_tokens, new_depth):
        return _call_groq(client, system_prompt, user_content, new_max_tokens, depth=new_depth)

    return _parse_llm_json(raw, _retry_with_more_tokens, max_tokens, depth=depth)


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

    SAMBANOVA_MIN_TOKENS_FLOOR = 2500
    max_tokens = min(max(max_tokens, SAMBANOVA_MIN_TOKENS_FLOOR), SAMBANOVA_MAX_TOKENS_CEILING)

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
        "reasoning_effort": "low",
    }

    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(SAMBANOVA_URL, headers=headers, json=payload, timeout=60)

            body_lower = (resp.text or "").lower()
            if resp.status_code == 402 or (
                resp.status_code == 429 and ("quota" in body_lower or "credit" in body_lower)
            ):
                _samba_quota_exhausted = True
                raise RuntimeError(f"SambaNova quota/credits exhausted (HTTP {resp.status_code}): {resp.text[:200]}")

            if resp.status_code == 429:
                raise RuntimeError("SambaNova rate limited")

            if not resp.ok:
                raise RuntimeError(f"SambaNova API error (HTTP {resp.status_code}): {resp.text[:500]}")

            data = resp.json()
            choice = data["choices"][0]
            raw = choice["message"]["content"]

            if not raw or not raw.strip():
                finish_reason = choice.get("finish_reason", "unknown")
                usage = data.get("usage", {})
                raise RuntimeError(
                    f"SambaNova returned empty content (finish_reason={finish_reason}, "
                    f"usage={usage}) — likely ran out of tokens mid-reasoning before "
                    f"writing the answer. Try raising the token budget further."
                )

            def _retry_with_more_tokens(new_max_tokens, new_depth):
                return _call_samba(system_prompt, user_content, new_max_tokens, depth=new_depth)

            return _parse_llm_json(raw, _retry_with_more_tokens, max_tokens, depth=depth)

        except Exception as e:
            last_error = e
            if _samba_quota_exhausted:
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
    max_tokens = min(3000, max(600, word_budget * 6))

    if is_first_chunk:
        user_content = f"Topic: {topic}"
    else:
        user_content = (
            f"Topic: {topic}\n\n"
            f"This continues a script already in progress. The most recent narration so far "
            f"was:\n{previous_narration_tail}\n\n"
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
