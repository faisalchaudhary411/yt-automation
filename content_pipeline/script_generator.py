"""
Turns a topic into a scene-by-scene documentary narration script.
Each scene has: narration text, an image search keyword, and estimated duration.
"""

import json
import re
from groq import Groq
from config import (
    GROQ_API_KEY, CHANNEL_NAME, LANGUAGES, DEFAULT_LANGUAGE, DEFAULT_DURATION_MINUTES,
    VIDEO_STYLES, DEFAULT_VIDEO_STYLE,
)

# Average narration speaking pace used to size the script. This matters a lot:
# the actual final video length is whatever edge-tts takes to *speak* the
# narration text out loud — the "duration_seconds" field Groq writes per scene
# is never enforced anywhere downstream, so if the model writes short
# narration the video simply comes out shorter than the selected preset
# (e.g. a "Short (~3 min)" request rendering as 1:35). Anchoring the prompt to
# an explicit total word-count target, and checking the result against it,
# is what actually keeps the output close to the requested length.
WORDS_PER_MINUTE = 150
MIN_ACCEPTABLE_RATIO = 0.85  # retry once if actual narration comes in under 85% of target


def _target_word_count(duration_minutes: float) -> int:
    return round(duration_minutes * WORDS_PER_MINUTE)


def _build_system_prompt(language_name: str, duration_minutes: float, style_key: str, target_words: int) -> str:
    # Roughly 50-90 words per scene (~20-35s of narration at ~150 words/min).
    scene_low = max(4, round(target_words / 90))
    scene_high = max(scene_low + 2, round(target_words / 50))
    narrator_style = VIDEO_STYLES.get(style_key, VIDEO_STYLES[DEFAULT_VIDEO_STYLE])["narrator_style"]

    return f"""You are a scriptwriter for a YouTube channel about history and finance \
(channel: {CHANNEL_NAME}). Write as {narrator_style}.

Write the ENTIRE script — title, description, and every scene's narration — in \
{language_name}. Do not mix in another language unless {language_name} is English.

Return ONLY valid JSON, no markdown fences, no preamble, in this exact shape:
{{
  "title": "SEO-friendly YouTube title, under 100 characters, written in {language_name}",
  "description": "2-3 sentence YouTube description, written in {language_name}",
  "tags": ["tag1", "tag2", "..."],
  "scenes": [
    {{
      "narration": "2-4 sentences of narration for this scene, written in {language_name}",
      "image_keywords": "2-4 words IN ENGLISH describing what image should show for this scene",
      "duration_seconds": 25
    }}
  ]
}}

CRITICAL LENGTH REQUIREMENT: the combined narration across every scene must add up to \
approximately {target_words} words in total (aim for {target_words}-{int(target_words * 1.15)} \
words — never less). This is the single most important constraint: a script that is too \
short produces a video that is too short. Use {scene_low}-{scene_high} scenes to hit that \
total, with each scene's narration long enough to read aloud in roughly its duration_seconds \
(~2.5 words per second). If you are unsure whether you have written enough, err on the side \
of writing more scenes and more narration per scene, not fewer. Do not include any text \
outside the JSON object."""


def _count_narration_words(scenes: list) -> int:
    return sum(len(scene.get("narration", "").split()) for scene in scenes)


def _call_groq(client: Groq, system_prompt: str, user_content: str, max_tokens: int) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.8,
        max_tokens=max_tokens,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        script = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Groq did not return valid JSON: {e}\nRaw output:\n{raw[:500]}")

    if "scenes" not in script or not script["scenes"]:
        raise RuntimeError("Generated script has no scenes.")

    return script


def generate_script(
    topic: str,
    language: str = DEFAULT_LANGUAGE,
    duration_minutes: float = DEFAULT_DURATION_MINUTES,
    style: str = DEFAULT_VIDEO_STYLE,
) -> dict:
    """
    topic: e.g. "Spain's 16th century silver defaults"
    language: a language code from config.LANGUAGES (e.g. "en", "es", "hi")
    duration_minutes: target length of the narration, in minutes
    style: a key from config.VIDEO_STYLES (e.g. "documentary", "cinematic")
    Returns a dict with title, description, tags, and a list of scenes.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in Replit Secrets.")

    language_name = LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])
    target_words = _target_word_count(duration_minutes)
    system_prompt = _build_system_prompt(language_name, duration_minutes, style, target_words)
    # Scale the token budget with the target length so longer videos (more
    # scenes, more narration) can't get truncated mid-JSON by a fixed cap.
    max_tokens = min(8000, max(4000, int(target_words * 8)))

    client = Groq(api_key=GROQ_API_KEY)

    script = _call_groq(client, system_prompt, f"Topic: {topic}", max_tokens)
    actual_words = _count_narration_words(script["scenes"])
    print(f"[script_generator] target={target_words} words, first attempt={actual_words} words "
          f"({len(script['scenes'])} scenes)")

    # If the model undershot noticeably, retry once with an explicit callback
    # telling it exactly how short it was. This is what actually fixes the
    # "3-min preset renders as 1:35" symptom instead of just hoping the prompt
    # wording alone is enough.
    if actual_words < target_words * MIN_ACCEPTABLE_RATIO:
        shortfall = target_words - actual_words
        retry_prompt = (
            f"Your previous script only had about {actual_words} words of narration total, "
            f"but {target_words} words were required — that's {shortfall} words short. "
            f"Write it again from scratch on the same topic, this time adding more scenes "
            f"and/or longer narration per scene so the total reaches at least {target_words} "
            f"words. Same JSON shape, same language, same topic."
        )
        try:
            retry_script = _call_groq(
                client, system_prompt,
                f"Topic: {topic}\n\n{retry_prompt}",
                max_tokens,
            )
            retry_words = _count_narration_words(retry_script["scenes"])
            print(f"[script_generator] retry attempt={retry_words} words "
                  f"({len(retry_script['scenes'])} scenes)")
            if retry_words > actual_words:
                script, actual_words = retry_script, retry_words
        except RuntimeError as e:
            print(f"[script_generator] retry failed, keeping first attempt: {e}")

    if actual_words < target_words * MIN_ACCEPTABLE_RATIO:
        print(f"[script_generator] WARNING: final script still short — {actual_words}/{target_words} words, "
              f"video will run shorter than the selected preset.")

    return script


if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "The Tulip Mania bubble of 1637"
    result = generate_script(topic)
    print(json.dumps(result, indent=2, ensure_ascii=False))
