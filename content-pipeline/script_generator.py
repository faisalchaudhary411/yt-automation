"""
Turns a topic into a scene-by-scene documentary narration script.
Each scene has: narration text, an image search keyword, and estimated duration.
"""

import json
import re
from groq import Groq
from config import GROQ_API_KEY

SYSTEM_PROMPT = """You are a scriptwriter for a documentary-style YouTube channel about \
history and finance (channel: WealthThroughAges). Write in a clear, engaging, slightly \
dramatic narrator voice, similar to Real Stories or Business Casual style channels.

Return ONLY valid JSON, no markdown fences, no preamble, in this exact shape:
{
  "title": "SEO-friendly YouTube title, under 100 characters",
  "description": "2-3 sentence YouTube description",
  "tags": ["tag1", "tag2", "..."],
  "scenes": [
    {
      "narration": "1-3 sentences of narration for this scene",
      "image_keywords": "2-4 words describing what image should show for this scene",
      "duration_seconds": 8
    }
  ]
}

Aim for 12-20 scenes totaling 6-10 minutes of narration. Keep each scene's narration \
tight enough to read aloud in roughly duration_seconds. Do not include any text outside \
the JSON object."""


def generate_script(topic: str) -> dict:
    """
    topic: e.g. "Spain's 16th century silver defaults"
    Returns a dict with title, description, tags, and a list of scenes.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in Replit Secrets.")

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
