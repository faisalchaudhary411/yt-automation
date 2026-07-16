"""
SEO Optimizer (Stage 3)
=======================
Post-processes the LLM-generated metadata before upload.

Deliberately selective vs the donor system's seo_optimizer.py: the donor
rewrote titles with gimmicky power-words ("The Ultimate Guide to...") which
would actively damage the better, style-aware titles the live script chain
writes. So this module leaves the title alone and only ADDS discovery value:

  - chapter timestamps built from the real scene timings
  - hashtags (top tags appended to the description)
  - a subscribe/attribution block (incl. any Wikimedia image credits)
  - tag expansion with keywords extracted from title + description
"""

import re
from collections import Counter

STOPWORDS = {
    "this", "that", "with", "from", "have", "will", "they", "them", "their",
    "there", "where", "when", "what", "which", "about", "would", "could",
    "should", "been", "were", "was", "are", "and", "the", "for", "you",
    "your", "but", "not", "all", "can", "had", "his", "her", "its", "our",
    "out", "who", "how", "why", "did", "does",
}


def extract_keywords(text: str, limit: int = 12) -> list:
    """Most frequent meaningful words — used to broaden the tag list."""
    words = re.findall(r"\b[A-Za-z]{4,}\b", (text or "").lower())
    common = Counter(w for w in words if w not in STOPWORDS).most_common(limit)
    return [w for w, _ in common]


def optimize_tags(tags: list, title: str, description: str, max_tags: int = 15) -> list:
    """Keeps the LLM's specific tags first, then pads with extracted keywords."""
    seen = set()
    result = []
    for tag in (tags or []) + extract_keywords(f"{title} {description}"):
        cleaned = tag.strip()[:50]
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            result.append(cleaned)
        if len(result) >= max_tags:
            break
    return result


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_timestamps_block(scene_start_times: list, include_intro: bool) -> str:
    """YouTube chapter list from actual scene start times. YouTube requires
    the first chapter at 0:00 and at least 3 chapters to render them."""
    if not scene_start_times or len(scene_start_times) < 2:
        return ""
    lines = ["0:00 Intro"] if include_intro else []
    for i, start in enumerate(scene_start_times, 1):
        lines.append(f"{_format_timestamp(start)} Part {i}")
    # YouTube only renders chapters with >= 3 entries starting at 0:00.
    if len(lines) < 3 or not lines[0].startswith("0:00"):
        return ""
    return "\n".join(lines)


def enhance_metadata(content: dict, scene_start_times: list = None,
                     include_intro: bool = True, attributions_path: str = None,
                     channel_name: str = "") -> dict:
    """Returns a copy of content with description/tags enhanced for search.
    `content` needs title/description/tags; everything else is optional."""
    enhanced = dict(content)
    title = enhanced.get("title", "")
    description = enhanced.get("description", "") or ""
    tags = enhanced.get("tags", []) or []

    blocks = [description.strip()] if description.strip() else []

    timestamps = build_timestamps_block(scene_start_times or [], include_intro)
    if timestamps and "0:00" not in description:
        blocks.append("Chapters:\n" + timestamps)

    if attributions_path:
        try:
            with open(attributions_path, "r", encoding="utf-8") as f:
                credits = f.read().strip()
            if credits:
                blocks.append(credits)
        except OSError:
            pass

    if channel_name and "subscribe" not in description.lower():
        blocks.append(f"Subscribe to {channel_name} for more stories like this.")

    hashtags = ["#" + re.sub(r"[^A-Za-z0-9_]", "", t.replace(" ", ""))
                for t in tags[:3] if t.strip()]
    hashtags = [h for h in hashtags if len(h) > 1]
    if hashtags and not any(h in description for h in hashtags):
        blocks.append(" ".join(hashtags))

    enhanced["description"] = "\n\n".join(blocks)[:4900]  # YT limit is 5000
    enhanced["tags"] = optimize_tags(tags, title, description)
    return enhanced
