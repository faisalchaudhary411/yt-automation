"""
Turns a topic into a scene-by-scene documentary narration script.

Primary LLM: Groq. Falls back to Gemini, then OpenRouter, if the
previous provider fails (rate limit exhausted, API error, or
unparseable output after retries).
"""

import json
import math
import re
import time
import requests
from groq import Groq, RateLimitError, APIError

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from config import (
    GROQ_API_KEY, CHANNEL_NAME, LANGUAGES, DEFAULT_LANGUAGE, DEFAULT_DURATION_MINUTES,
    VIDEO_STYLES, DEFAULT_VIDEO_STYLE,
)

# Optional fallback keys — each fallback is simply skipped if not configured.
try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = None

try:
    from config import OPENROUTER_API_KEY
except ImportError:
    OPENROUTER_API_KEY = None

WORDS_PER_MINUTE = 150
MIN_ACCEPTABLE_RATIO = 0.85
CHUNK_TARGET_WORDS = 80
GEMINI_MODEL_NAME = "gemini-2.0-flash"
OPENROUTER_MODEL_NAME = "meta-llama/llama-3.1-70b-instruct:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


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


def _parse_llm_json(raw: str, attempt_more_tokens_callback=None, max_tokens: int = 0):
    """Shared JSON parsing/repair logic used by every provider path."""
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
            if attempt_more_tokens_callback and max_tokens:
                return attempt_more_tokens_callback(max_tokens + 4000)
            raise RuntimeError(f"LLM did not return valid JSON: {e}\nRaw output:\n{raw[:500]}")


def _call_groq(client: Groq, system_prompt: str, user_content: str, max_tokens: int) -> dict:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.6-27b",  # Groq recommended replacement (Aug 2026)
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
            if attempt < 2:
                time.sleep(8 + attempt * 4)
            else:
                raise
        except APIError as e:
            raise RuntimeError(f"Groq API error: {e}")

    raw = response.choices[0].message.content

    def _retry_with_more_tokens(new_max_tokens):
        return _call_groq(client, system_prompt, user_content, new_max_tokens)

    return _parse_llm_json(raw, _retry_with_more_tokens, max_tokens)


def _call_gemini(system_prompt: str, user_content: str, max_tokens: int) -> dict:
    if not (GEMINI_AVAILABLE and GEMINI_API_KEY):
        raise RuntimeError(
            "Gemini fallback unavailable: install google-generativeai and set "
            "GEMINI_API_KEY in config.py / Replit Secrets."
        )

    model = genai.GenerativeModel(
        GEMINI_MODEL_NAME,
        system_instruction=system_prompt,
    )

    last_error = None
    for attempt in range(3):
        try:
            response = model.generate_content(
                user_content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text

            def _retry_with_more_tokens(new_max_tokens):
                return _call_gemini(system_prompt, user_content, new_max_tokens)

            return _parse_llm_json(raw, _retry_with_more_tokens, max_tokens)

        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(4 + attempt * 4)
            else:
                raise RuntimeError(f"Gemini fallback also failed: {e}")

    raise RuntimeError(f"Gemini fallback failed after retries: {last_error}")


def _call_openrouter(system_prompt: str, user_content: str, max_tokens: int) -> dict:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OpenRouter fallback unavailable: set OPENROUTER_API_KEY in "
            "config.py / Replit Secrets."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL_NAME,
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
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                raise RuntimeError("OpenRouter rate limited")
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]

            def _retry_with_more_tokens(new_max_tokens):
                return _call_openrouter(system_prompt, user_content, new_max_tokens)

            return _parse_llm_json(raw, _retry_with_more_tokens, max_tokens)

        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(4 + attempt * 4)
            else:
                raise RuntimeError(f"OpenRouter fallback also failed: {e}")

    raise RuntimeError(f"OpenRouter fallback failed after retries: {last_error}")


def _call_llm(client: Groq, system_prompt: str, user_content: str, max_tokens: int) -> dict:
    """Tries Groq, then Gemini, then OpenRouter, in order."""
    errors = []

    try:
        return _call_groq(client, system_prompt, user_content, max_tokens)
    except Exception as e:
        errors.append(f"Groq: {e}")
        print(f"[script_generator] Groq failed ({e}). Trying Gemini...")

    try:
        return _call_gemini(system_prompt, user_content, max_tokens)
    except Exception as e:
        errors.append(f"Gemini: {e}")
        print(f"[script_generator] Gemini failed ({e}). Trying OpenRouter...")

    try:
        return _call_openrouter(system_prompt, user_content, max_tokens)
    except Exception as e:
        errors.append(f"OpenRouter: {e}")

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
    max_tokens = 12000

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
) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in Replit Secrets.")

    language_name = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])
    total_target_words = _target_word_count(duration_minutes)
    num_chunks = max(1, math.ceil(total_target_words / CHUNK_TARGET_WORDS))

    client = Groq(api_key=GROQ_API_KEY)

    metadata = _generate_metadata(client, topic, language_name, style)

    all_scenes = []
    remaining_words = total_target_words

    for chunk_index in range(num_chunks):
        is_first = chunk_index == 0
        chunks_left = num_chunks - chunk_index
        word_budget = max(60, round(remaining_words / chunks_left)) if remaining_words > 0 else CHUNK_TARGET_WORDS

        previous_tail = ""
        if all_scenes:
            previous_tail = all_scenes[-1]["narration"][-300:]

        part = _generate_scenes_chunk(
            client, topic, language_name, style, word_budget, is_first, previous_tail,
        )

        all_scenes.extend(part["scenes"])
        remaining_words -= _count_narration_words(part["scenes"])

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
