"""
Turns a topic into a scene-by-scene documentary narration script.

Provider chain (fastest → most reliable):
  1. Cerebras      (primary)   — gpt-oss-120b, 1M tokens/day free, ~2,600 tok/s
  2. SambaNova     (secondary) — gpt-oss-120b, RDU-hosted
  3. Groq          (tertiary)  — llama-3.3-70b-versatile, final fallback

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

# Optional provider keys — fallback is simply skipped if not configured.
try:
    from config import SAMBANOVA_API_KEY
except ImportError:
    SAMBANOVA_API_KEY = None

try:
    from config import CEREBRAS_API_KEY
except ImportError:
    CEREBRAS_API_KEY = None


# ─────────────────────────── Constants ───────────────────────────

WORDS_PER_MINUTE = 150
MIN_ACCEPTABLE_RATIO = 0.85
CHUNK_TARGET_WORDS = 200

# Cerebras
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL = "gpt-oss-120b"
CEREBRAS_MAX_TOKENS_CEILING = 8192          # Free-tier context ceiling
CEREBRAS_DAILY_TOKEN_LIMIT = 1_000_000      # Free-tier daily cap
CEREBRAS_MIN_TOKENS_FLOOR = 2500

# SambaNova
SAMBANOVA_MODEL_NAME = "gpt-oss-120b"
SAMBANOVA_URL = "https://api.sambanova.ai/v1/chat/completions"
SAMBANOVA_MAX_TOKENS_CEILING = 8000

# Groq
GROQ_TPM_LIMIT = 8000
GROQ_SAFETY_MARGIN = 1500
MAX_GROQ_RATE_LIMIT_WAIT_SECONDS = 600

MAX_JSON_RETRY_DEPTH = 2


# ─────────────────────────── State ───────────────────────────────

# Provider quota flags
_cerebras_quota_exhausted = False
_samba_quota_exhausted = False

# Cerebras daily token tracker (in-memory; persist to SQLite for multi-day accuracy)
_cerebras_tokens_today = 0
_cerebras_day_timestamp = time.strftime("%Y-%m-%d")


# ─────────────────────────── Urdu / Text Utils ─────────────────────

URDU_UNICODE_RANGE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"
    r"\s\u060C\u061B\u061F\u06D4.,!?;:\'\"\-—\u200C\u200D]+"
)

# Catches RUNS of 2+ symbols. Does NOT include $ or % (finance channel).
GARBAGE_PATTERN = re.compile(r"[&!@#^*+=~`|<>{}\[\]\\/]{2,}")

# Single-occurrence junk that never belongs in narration
_NEVER_LEGIT_CHARS = re.compile(r"[&@#^*+=~`|<>{}\[\]\\]")

# URL noise / repeated punctuation
_URL_NOISE_PATTERN = re.compile(r"https?:\/\/\S*|:\/\/|/{2,}|-{2,}|:{2,}")


def _target_word_count(duration_minutes: float) -> int:
    return round(duration_minutes * WORDS_PER_MINUTE)


def _scene_count_for_words(word_budget: int) -> tuple:
    scene_low = max(2, round(word_budget / 90))
    scene_high = max(scene_low + 1, round(word_budget / 50))
    return scene_low, scene_high


def _urdu_style_notes(language_name: str) -> str:
    if language_name.strip().lower() != "urdu":
        return ""
    return """

URDU STYLE GUIDANCE (very important):
- Write like a Pakistani YouTube narrator talking to a general audience, not like a textbook or news bulletin.
- Prefer everyday, commonly spoken Urdu vocabulary over heavy Persian/Arabic literary words.
- Use natural sentence rhythm and short-to-medium sentences suited to narration.
- It's fine to keep common English loanwords that Pakistanis actually use in speech (e.g. "invest", "company", "market", "percent") — but do not switch entire sentences to English.
- Avoid word-for-word translated phrasing. Write the thought directly in Urdu the way a person would say it.
- Vary sentence openings; avoid repeating the same connector words (e.g. "لیکن", "اس کے بعد") in every sentence."""


# ─────────────────────────── Prompt Builders ───────────────────────

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

    # YouTube-specific guidance for first and last scenes
    hook_guidance = ""
    if is_first_chunk:
        hook_guidance = (
            "\n- The FIRST scene must open with a STRONG HOOK in the first 5 seconds "
            "(a surprising fact, bold claim, or curiosity gap that stops the scroll)."
        )

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

CRITICAL YOUTUBE NARRATION RULES:
- Every scene narration must be written as spoken words for a voiceover, not as prose to be read silently.
- Use (PAUSE) and (EMPHASIS) markers where the narrator should pause or stress a word.
- Mark [B-ROLL: description] where background footage should appear while the narrator speaks.
- Keep sentences short and punchy. Avoid nested clauses.
- If this is the final chunk of the script, the LAST scene must end with a clear Call-To-Action (CTA): ask the viewer to subscribe, comment, or watch the next video.{hook_guidance}

CRITICAL TEXT FORMATTING:
- Use proper {language_name} punctuation (e.g., Urdu full stop: ۔ not .)
- Do NOT use Latin/English punctuation inside {language_name} text
- Do NOT mix English words inside {language_name} narration unless necessary"""


# ─────────────────────────── Word / Text Utils ───────────────────

def _count_narration_words(scenes: list) -> int:
    return sum(len(scene.get("narration", "").split()) for scene in scenes)


def _is_text_corrupted(text: str) -> bool:
    if not text:
        return True
    if GARBAGE_PATTERN.search(text):
        return True
    urdu_chars = sum(1 for c in text if (0x0600 <= ord(c) <= 0x06FF) or
                     (0x0750 <= ord(c) <= 0x077F))
    total_chars = len([c for c in text if c.strip()])
    if total_chars > 0 and urdu_chars / total_chars < 0.3:
        latin_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        if latin_chars / total_chars < 0.5:
            return True
    return False


def _sanitize_narration(text: str) -> str:
    if not text:
        return text
    text = GARBAGE_PATTERN.sub(" ", text)
    text = _NEVER_LEGIT_CHARS.sub(" ", text)
    text = _URL_NOISE_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<![\u0600-\u06FF])[,;:!?]+(?![\u0600-\u06FF])", " ", text)
    text = text.strip()
    if text and text[-1] not in ".!?۔":
        text += "۔"
    return text.strip()


def _repair_json(raw: str) -> str:
    if not raw or not raw.strip():
        return raw
    raw = raw.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        raw = raw[first_brace:last_brace + 1]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    open_brackets = raw.count("[") - raw.count("]")
    open_braces = raw.count("{") - raw.count("}")
    if open_brackets > 0 and not raw.rstrip().endswith("]"):
        raw = raw.rstrip() + "]" * open_brackets
    if open_braces > 0 and not raw.rstrip().endswith("}"):
        raw = raw.rstrip() + "}" * open_braces
    return raw


def _extract_scenes_safely(raw: str) -> list:
    scenes = []
    scene_pattern = re.compile(
        r'"narration"\s*:\s*"(.*?)"\s*,\s*"image_keywords"\s*:\s*"(.*?)"\s*,\s*"duration_seconds"\s*:\s*(\d+)',
        re.DOTALL
    )
    matches = scene_pattern.findall(raw)
    for narration, keywords, duration in matches:
        narration = narration.replace("\\\\", "\\").replace('\"', '"')
        narration = _sanitize_narration(narration)
        if narration.strip():
            scenes.append({
                "narration": narration,
                "image_keywords": keywords.replace("\\\\", "\\").replace('\"', '"'),
                "duration_seconds": int(duration),
            })
    if not scenes:
        narration_pattern = re.compile(r'"narration"\s*:\s*"((?:[^"\\]|\\.)*)"')
        keyword_pattern = re.compile(r'"image_keywords"\s*:\s*"((?:[^"\\]|\\.)*)"')
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


def _parse_llm_json(raw: str, attempt_more_tokens_callback=None, max_tokens: int = 0, depth: int = 0):
    if raw is None or not raw.strip():
        raise RuntimeError("LLM returned an empty response.")
    raw = raw.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
        if "scenes" in data:
            for scene in data["scenes"]:
                if "narration" in scene:
                    scene["narration"] = _sanitize_narration(scene["narration"])
        return data
    except json.JSONDecodeError:
        pass
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
    scenes = _extract_scenes_safely(raw)
    if scenes:
        valid_scenes = []
        for scene in scenes:
            if not _is_text_corrupted(scene["narration"]):
                valid_scenes.append(scene)
            else:
                print(f"[WARNING] Skipping corrupted scene: {scene['narration'][:50]}...")
        if valid_scenes:
            return {"scenes": valid_scenes}
    if attempt_more_tokens_callback and max_tokens and depth < MAX_JSON_RETRY_DEPTH:
        return attempt_more_tokens_callback(max_tokens + 4000, depth + 1)
    raise RuntimeError(
        f"LLM did not return valid JSON after {depth} retry attempt(s). "
        f"Raw output:\n{raw[:800]}"
    )


# ─────────────────────────── Cerebras ──────────────────────────────

def _call_cerebras(system_prompt: str, user_content: str, max_tokens: int, depth: int = 0) -> dict:
    """
    Primary provider. 1M tokens/day free. No credit card.
    Falls back on quota exhaustion, rate limits, or model unavailability.
    """
    global _cerebras_quota_exhausted, _cerebras_tokens_today, _cerebras_day_timestamp

    if not CEREBRAS_API_KEY:
        raise RuntimeError("Cerebras unavailable: set CEREBRAS_API_KEY in config.py.")

    if _cerebras_quota_exhausted:
        raise RuntimeError("Cerebras skipped: daily quota confirmed exhausted.")

    # Reset daily counter if day changed
    today = time.strftime("%Y-%m-%d")
    if today != _cerebras_day_timestamp:
        _cerebras_day_timestamp = today
        _cerebras_tokens_today = 0
        _cerebras_quota_exhausted = False

    # Rough estimate: prompt tokens + max output tokens
    prompt_tokens = len(system_prompt.split()) + len(user_content.split())
    estimated_cost = prompt_tokens + max_tokens

    if _cerebras_tokens_today + estimated_cost > CEREBRAS_DAILY_TOKEN_LIMIT:
        print(f"[script_generator]   Cerebras daily budget nearly exhausted "
              f"({_cerebras_tokens_today}/{CEREBRAS_DAILY_TOKEN_LIMIT}). Skipping to preserve quota.")
        _cerebras_quota_exhausted = True
        raise RuntimeError("Cerebras daily token budget nearly exhausted.")

    max_tokens = min(max(max_tokens, CEREBRAS_MIN_TOKENS_FLOOR), CEREBRAS_MAX_TOKENS_CEILING)

    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CEREBRAS_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(CEREBRAS_URL, headers=headers, json=payload, timeout=45)

            # Model silently removed from free tier?
            if resp.status_code == 404:
                body = resp.json() if resp.text else {}
                err_msg = body.get("error", {}).get("message", "")
                if "model" in err_msg.lower() or "not found" in err_msg.lower():
                    print(f"[script_generator]   Cerebras model '{CEREBRAS_MODEL}' unavailable "
                          f"(likely removed from free tier). Skipping Cerebras.")
                    _cerebras_quota_exhausted = True
                    raise RuntimeError(f"Cerebras model unavailable: {err_msg}")

            if resp.status_code == 429:
                body = resp.json() if resp.text else {}
                err_msg = body.get("error", {}).get("message", "")
                if "quota" in err_msg.lower() or "limit" in err_msg.lower():
                    _cerebras_quota_exhausted = True
                    raise RuntimeError(f"Cerebras quota/limit hit: {err_msg}")
                raise RuntimeError(f"Cerebras rate limited: {err_msg}")

            if resp.status_code == 402:
                _cerebras_quota_exhausted = True
                raise RuntimeError("Cerebras credits exhausted (HTTP 402).")

            if not resp.ok:
                raise RuntimeError(f"Cerebras API error (HTTP {resp.status_code}): {resp.text[:500]}")

            data = resp.json()
            choice = data["choices"][0]
            raw = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "unknown")

            # Track actual usage
            usage = data.get("usage", {})
            total_used = usage.get("total_tokens", estimated_cost)
            _cerebras_tokens_today += total_used

            if finish_reason == "length":
                print(
                    f"[script_generator]   ⚠ Cerebras hit the token limit (max_tokens={max_tokens}) "
                    f"before finishing — output may be truncated."
                )

            if not raw or not raw.strip():
                raise RuntimeError(
                    f"Cerebras returned empty content (finish_reason={finish_reason}) — "
                    f"likely ran out of tokens mid-generation."
                )

            def _retry_with_more_tokens(new_max_tokens, new_depth):
                return _call_cerebras(system_prompt, user_content, new_max_tokens, depth=new_depth)

            return _parse_llm_json(raw, _retry_with_more_tokens, max_tokens, depth=depth)

        except requests.exceptions.Timeout:
            last_error = "Cerebras request timed out"
            if attempt < 2:
                time.sleep(3 + attempt * 3)
            else:
                raise RuntimeError(f"Cerebras failed after retries: {last_error}")
        except Exception as e:
            last_error = e
            if _cerebras_quota_exhausted:
                raise RuntimeError(f"Cerebras call failed: {e}")
            if attempt < 2:
                time.sleep(3 + attempt * 3)
            else:
                raise RuntimeError(f"Cerebras call failed: {e}")

    raise RuntimeError(f"Cerebras failed after retries: {last_error}")


# ─────────────────────────── SambaNova ─────────────────────────────

def _call_samba(system_prompt: str, user_content: str, max_tokens: int, depth: int = 0) -> dict:
    global _samba_quota_exhausted

    if not SAMBANOVA_API_KEY:
        raise RuntimeError("SambaNova unavailable: set SAMBANOVA_API_KEY in config.py.")

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
            finish_reason = choice.get("finish_reason", "unknown")
            if finish_reason == "length":
                print(
                    f"[script_generator]   ⚠ SambaNova hit the token limit (max_tokens={max_tokens}) "
                    f"before finishing — output is truncated."
                )

            if not raw or not raw.strip():
                raise RuntimeError(
                    f"SambaNova returned empty content (finish_reason={finish_reason}) — "
                    f"likely ran out of tokens mid-generation."
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


# ─────────────────────────── Groq ────────────────────────────────

def _extract_retry_after_seconds(message: str):
    match = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", message)
    if not match:
        return None
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60 + seconds


def _extract_failed_generation(e: "APIError") -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        fg = (body.get("error") or {}).get("failed_generation")
        if fg:
            return fg
    match = re.search(r"'failed_generation':\s*'((?:[^'\\]|\\.)*)'", str(e))
    if match:
        try:
            return match.group(1).encode().decode("unicode_escape")
        except UnicodeDecodeError:
            return match.group(1)
    return ""


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
            if "json_validate_failed" in msg:
                failed_gen = _extract_failed_generation(e)
                if failed_gen:
                    print("[script_generator]   Groq's JSON-mode validator rejected the "
                          "completion — attempting to salvage the partial output...")
                    try:
                        return _parse_llm_json(failed_gen, None, max_tokens, depth=depth)
                    except Exception as salvage_error:
                        print(f"[script_generator]   Salvage attempt failed: {salvage_error}")
                if attempt < 3:
                    print(f"[script_generator]   Groq JSON validation failed "
                          f"(attempt {attempt + 1}/4) — retrying...")
                    time.sleep(4 + attempt * 3)
                    continue
            raise RuntimeError(f"Groq API error: {e}")

    raw = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    if finish_reason == "length":
        print(
            f"[script_generator]   ⚠ Groq hit the token limit (max_tokens={max_tokens}) before "
            f"finishing — output is truncated."
        )

    def _retry_with_more_tokens(new_max_tokens, new_depth):
        return _call_groq(client, system_prompt, user_content, new_max_tokens, depth=new_depth)

    return _parse_llm_json(raw, _retry_with_more_tokens, max_tokens, depth=depth)


# ─────────────────────────── Provider Router ─────────────────────

def _call_llm(client: Groq, system_prompt: str, user_content: str, max_tokens: int) -> dict:
    """
    Tries Cerebras first (primary), then SambaNova (secondary), then Groq (tertiary).
    """
    errors = []

    # 1️⃣ Cerebras (primary)
    if not _cerebras_quota_exhausted:
        cb_start = time.time()
        try:
            result = _call_cerebras(system_prompt, user_content, max_tokens)
            elapsed = time.time() - cb_start
            print(f"[script_generator]   Cerebras succeeded in {elapsed:.1f}s "
                  f"(tokens today: {_cerebras_tokens_today}/{CEREBRAS_DAILY_TOKEN_LIMIT})")
            return result
        except Exception as e:
            errors.append(f"Cerebras: {e}")
            print(f"[script_generator]   Cerebras failed after {time.time() - cb_start:.1f}s ({e}). "
                  f"Trying SambaNova...")
    else:
        print("[script_generator]   Skipping Cerebras (quota confirmed exhausted). "
              "Trying SambaNova...")

    # 2️⃣ SambaNova (secondary)
    if not _samba_quota_exhausted:
        or_start = time.time()
        try:
            result = _call_samba(system_prompt, user_content, max_tokens)
            print(f"[script_generator]   SambaNova succeeded in {time.time() - or_start:.1f}s")
            return result
        except Exception as e:
            errors.append(f"SambaNova: {e}")
            print(f"[script_generator]   SambaNova failed after {time.time() - or_start:.1f}s ({e}). "
                  f"Trying Groq...")
    else:
        print("[script_generator]   Skipping SambaNova (quota confirmed exhausted). Using Groq...")

    # 3️⃣ Groq (tertiary / final fallback)
    groq_start = time.time()
    try:
        result = _call_groq(client, system_prompt, user_content, max_tokens)
        print(f"[script_generator]   Groq succeeded in {time.time() - groq_start:.1f}s")
        return result
    except Exception as e:
        errors.append(f"Groq: {e}")
        print(f"[script_generator]   Groq also failed after {time.time() - groq_start:.1f}s ({e})")

    raise RuntimeError("All LLM providers failed.\n" + "\n".join(errors))


# ─────────────────────────── Metadata / Scenes ───────────────────

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
        retry_max_tokens = min(SAMBANOVA_MAX_TOKENS_CEILING, max_tokens + 2500)
        print(
            f"[script_generator]   Chunk came up short ({actual_words}/{word_budget} words) — "
            f"retrying with max_tokens {max_tokens} -> {retry_max_tokens} in case the shortfall "
            f"was truncation rather than the model just writing too little."
        )
        try:
            retry_part = _call_llm(
                client, system_prompt,
                user_content + f"\n\n(Your last attempt only had about {actual_words} words — "
                f"write more this time, aiming for at least {word_budget} words.)",
                retry_max_tokens,
            )
            if "scenes" in retry_part and _count_narration_words(retry_part["scenes"]) > actual_words:
                part = retry_part
        except RuntimeError as e:
            print(f"[script_generator]   Retry-with-more-tokens also failed: {e}")

    return part


# ─────────────────────────── Public API ──────────────────────────

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
