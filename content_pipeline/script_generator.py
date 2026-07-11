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


def _build_system_prompt(language_name: str, duration_minutes: float, style_key: str) -> str:
    # Roughly one scene per 22-35s of narration, clamped to a sane range.
    scene_low = max(4, round(duration_minutes * 60 / 35))
    scene_high = max(scene_low + 2, round(duration_minutes * 60 / 22))
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
      "narration": "1-3 sentences of narration for this scene, written in {language_name}",
      "image_keywords": "2-4 words IN ENGLISH describing what image should show for this scene",
      "duration_seconds": 8
    }}
  ]
}}

Aim for {scene_low}-{scene_high} scenes totaling approximately {duration_minutes:g} minutes \
of spoken narration in total. Keep each scene's narration tight enough to read aloud in \
roughly duration_seconds. Do not include any text outside the JSON object."""


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
    system_prompt = _build_system_prompt(language_name, duration_minutes, style)

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Topic: {topic}"},
        ],
        temperature=0.8,
        max_tokens=4000,
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


if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "The Tulip Mania bubble of 1637"
    result = generate_script(topic)
    print(json.dumps(result, indent=2, ensure_ascii=False))
