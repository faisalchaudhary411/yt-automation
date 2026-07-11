"""
Turns a topic into a scene-by-scene documentary narration script.
Each scene has: narration text, an image search keyword, and estimated duration.

The script is generated in chunks of a few scenes at a time (see CHUNK_TARGET_WORDS
below) rather than as one single Groq request. This matters a lot for two reasons:
1. Groq's llama-3.3-70b-versatile caps a single response at 8,192 tokens. A 6-10
   minute script easily needs more output than that in one shot.
2. Non-Latin scripts (Urdu, Hindi, Arabic, etc.) take roughly 3-5x more tokens per
   word to encode than English, so a request sized correctly for an English script
   can silently overflow the token cap for the same word count in Urdu, cutting the
   JSON off mid-string. Chunking keeps every individual request comfortably under
   that ceiling no matter which language or how long the target video is.
"""

import json
import math
import re
import time
from groq import Groq, RateLimitError
from config import (
    GROQ_API_KEY, CHANNEL_NAME, LANGUAGES, DEFAULT_LANGUAGE, DEFAULT_DURATION_MINUTES,
    VIDEO_STYLES, DEFAULT_VIDEO_STYLE,
)

# Average narration speaking pace used to size the script. This matters a lot:
# the actual final video length is whatever edge-tts takes to *speak* the
# narration text out loud — the "duration_seconds" field written per scene is
# never enforced anywhere downstream, so the total narration word count is
# what actually determines final video length.
WORDS_PER_MINUTE = 150
MIN_ACCEPTABLE_RATIO = 0.85  # retry a chunk once if it comes in under 85% of its target

# Each Groq request targets at most this many words of new narration. Kept
# small enough that even Urdu/Hindi/Arabic-level token density stays well
# under Groq's 8,192-token response cap, with room to spare for JSON syntax
# and the image_keywords/duration_seconds fields.
CHUNK_TARGET_WORDS = 180


def _target_word_count(duration_minutes: float) -> int:
    return round(duration_minutes * WORDS_PER_MINUTE)


def _scene_count_for_words(word_budget: int) -> tuple:
    # Roughly 50-90 words per scene (~20-35s of narration at ~150 words/min).
    scene_low = max(2, round(word_budget / 90))
    scene_high = max(scene_low + 1, round(word_budget / 50))
    return scene_low, scene_high


def _build_system_prompt(
    language_name: str, style_key: str, word_budget: int, is_first_chunk: bool
) -> str:
    scene_low, scene_high = _scene_count_for_words(word_budget)
    narrator_style = VIDEO_STYLES.get(style_key, VIDEO_STYLES[DEFAULT_VIDEO_STYLE])["narrator_style"]

    metadata_shape = ""
    if is_first_chunk:
        metadata_shape = f"""  "title": "SEO-friendly YouTube title, under 100 characters, written in {language_name}",
  "description": "2-3 sentence YouTube description, written in {language_name}",
  "tags": ["tag1", "tag2", "..."],
"""

    return f"""You are a scriptwriter for a YouTube channel about history and finance \
(channel: {CHANNEL_NAME}). Write as {narrator_style}.

Write the ENTIRE response — {"title, description, and " if is_first_chunk else ""}every \
scene's narration — in {language_name}. Do not mix in another language unless \
{language_name} is English.

Return ONLY valid JSON, no markdown fences, no preamble, in this exact shape:
{{
{metadata_shape}  "scenes": [
    {{
      "narration": "2-4 sentences of narration for this scene, written in {language_name}",
      "image_keywords": "2-4 words IN ENGLISH describing what image should show for this scene",
      "duration_seconds": 25
    }}
  ]
}}

CRITICAL LENGTH REQUIREMENT: the combined narration across every scene in THIS response \
must add up to approximately {word_budget} words in total (aim for {word_budget}-\
{int(word_budget * 1.15)} words — never less). Use {scene_low}-{scene_high} scenes to hit \
that total, with each scene's narration long enough to read aloud in roughly its \
duration_seconds (~2.5 words per second). If you are unsure whether you have written \
enough, err on the side of writing more scenes and more narration per scene, not fewer. \
Do not include any text outside the JSON object."""


def _count_narration_words(scenes: list) -> int:
    return sum(len(scene.get("narration", "").split()) for scene in scenes)


def _call_groq(client: Groq, system_prompt: str, user_content: str, max_tokens: int) -> dict:
    # Chunking means several requests per job in quick succession, which can
    # bump into Groq's free-tier per-minute token limit (shared across all
    # calls in that window) even though each individual chunk is small. A
    # short backoff and one retry handles that instead of failing the whole
    # job over a transient 429.
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.8,
                max_tokens=max_tokens,
            )
            break
        except RateLimitError:
            if attempt == 0:
                time.sleep(8)
            else:
                raise

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        part = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Groq did not return valid JSON: {e}\nRaw output:\n{raw[:500]}")

    if "scenes" not in part or not part["scenes"]:
        raise RuntimeError("Generated script chunk has no scenes.")

    return part


def _generate_chunk(
    client: Groq, topic: str, language_name: str, style: str,
    word_budget: int, is_first_chunk: bool, previous_narration_tail: str,
) -> dict:
    system_prompt = _build_system_prompt(language_name, style, word_budget, is_first_chunk)
    # Comfortably above what even a high token-density language needs for this
    # small a word budget, while staying well clear of Groq's 8,192 cap.
    max_tokens = min(6000, max(1200, word_budget * 12))

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

    part = _call_groq(client, system_prompt, user_content, max_tokens)
    actual_words = _count_narration_words(part["scenes"])

    if actual_words < word_budget * MIN_ACCEPTABLE_RATIO:
        try:
            retry_part = _call_groq(
                client, system_prompt,
                user_content + f"\n\n(Your last attempt only had about {actual_words} words — "
                f"write more this time, aiming for at least {word_budget} words.)",
                max_tokens,
            )
            if _count_narration_words(retry_part["scenes"]) > actual_words:
                part = retry_part
        except RuntimeError:
            pass  # keep the first attempt rather than fail the whole job over one retry hiccup

    return part


def generate_script(
    topic: str,
    language: str = DEFAULT_LANGUAGE,
    duration_minutes: float = DEFAULT_DURATION_MINUTES,
    style: str = DEFAULT_VIDEO_STYLE,
) -> dict:
    """
    topic: e.g. "Spain's 16th century silver defaults"
    language: a language code from config.LANGUAGES (e.g. "en", "es", "hi", "ur")
    duration_minutes: target length of the narration, in minutes
    style: a key from config.VIDEO_STYLES (e.g. "documentary", "cinematic")
    Returns a dict with title, description, tags, and a list of scenes.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in Replit Secrets.")

    language_name = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])
    total_target_words = _target_word_count(duration_minutes)
    num_chunks = max(1, math.ceil(total_target_words / CHUNK_TARGET_WORDS))

    client = Groq(api_key=GROQ_API_KEY)

    title, description, tags = None, None, []
    all_scenes = []
    remaining_words = total_target_words

    for chunk_index in range(num_chunks):
        is_first = chunk_index == 0
        chunks_left = num_chunks - chunk_index
        word_budget = max(60, round(remaining_words / chunks_left)) if remaining_words > 0 else CHUNK_TARGET_WORDS

        previous_tail = ""
        if all_scenes:
            previous_tail = all_scenes[-1]["narration"][-300:]

        part = _generate_chunk(
            client, topic, language_name, style, word_budget, is_first, previous_tail,
        )

        if is_first:
            title = part.get("title") or topic
            description = part.get("description") or ""
            tags = part.get("tags") or []

        all_scenes.extend(part["scenes"])
        remaining_words -= _count_narration_words(part["scenes"])

    actual_total_words = _count_narration_words(all_scenes)
    print(f"[script_generator] target={total_target_words} words, actual={actual_total_words} words, "
          f"{len(all_scenes)} scenes across {num_chunks} chunk(s)")

    return {
        "title": title,
        "description": description,
        "tags": tags,
        "scenes": all_scenes,
    }


if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "The Tulip Mania bubble of 1637"
    result = generate_script(topic)
    print(json.dumps(result, indent=2, ensure_ascii=False))
