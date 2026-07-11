"""
Central config + GitHub-as-database helper.
Same pattern used across VoxCraft / QalamStudio: JSON state files
committed to a GitHub repo via the Contents API.

Required Replit Secrets:
  GROQ_API_KEY        - Groq API key (script generation)
  ELEVENLABS_API_KEY  - optional, for premium narration voice
  PEXELS_API_KEY      - free, for stock visuals
  GITHUB_TOKEN        - a fine-grained PAT with contents:write on your state repo
  GITHUB_REPO         - e.g. "yourusername/yt-automation-state"
  GITHUB_BRANCH       - defaults to "main"
"""

import os
import json
import base64
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

CHANNEL_NAME = os.environ.get("CHANNEL_NAME", "WealthThroughAges")

# Narration languages offered in the UI. Keys are gTTS language codes (also used
# to tell Groq which language to write the script in via their display name).
LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "hi": "Hindi",
    "ur": "Urdu",
    "ar": "Arabic",
    "tr": "Turkish",
    "ru": "Russian",
    "it": "Italian",
    "id": "Indonesian",
}
DEFAULT_LANGUAGE = "en"

# Video length presets shown in the UI (minutes of narration to aim for).
DURATION_PRESETS = {
    "short": 3,
    "medium": 6,
    "long": 10,
}
DEFAULT_DURATION_MINUTES = DURATION_PRESETS["medium"]

# Voice gender presets, mapped to ElevenLabs' public premade voice IDs (used when
# ELEVENLABS_API_KEY is set). gTTS has no gender control, so the free fallback
# voice is the same regardless of the selection below.
VOICE_PRESETS = {
    "female": "21m00Tcm4TlvDq8ikWAM",  # "Rachel" - premade ElevenLabs voice
    "male": "pNInz6obpgDQGcFmaJgB",    # "Adam" - premade ElevenLabs voice
}
DEFAULT_VOICE_GENDER = "female"

# Video style presets: each shapes the narrator's tone (via the script prompt)
# and the visual treatment (intro/outro color + Ken Burns zoom speed).
VIDEO_STYLES = {
    "documentary": {
        "name": "Documentary",
        "narrator_style": "a measured, informative documentary narrator building intrigue, "
                           "similar to Real Stories or History Channel specials",
        "bg_color": "0x141E30",
        "zoom_rate": 0.0008,
    },
    "cinematic": {
        "name": "Cinematic / Dramatic",
        "narrator_style": "a dramatic, suspenseful narrator slowly building tension, "
                           "similar to true-crime or mystery documentaries",
        "bg_color": "0x1A1A2E",
        "zoom_rate": 0.0005,
    },
    "motivational": {
        "name": "Motivational",
        "narrator_style": "an energetic, inspiring motivational voice, "
                           "similar to top self-improvement or success channels",
        "bg_color": "0x8A5A00",
        "zoom_rate": 0.0012,
    },
    "educational": {
        "name": "Educational / Explainer",
        "narrator_style": "a friendly, clear narrator explaining things simply, "
                           "similar to popular science explainer channels",
        "bg_color": "0x1B4332",
        "zoom_rate": 0.0007,
    },
}
DEFAULT_VIDEO_STYLE = "documentary"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

GITHUB_API_BASE = "https://api.github.com"

WORK_DIR = "output"  # local scratch folder on Replit (audio/images/video before upload)


def _gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def github_read_json(path, default=None):
    """Read a JSON file from the state repo. Returns `default` if it doesn't exist yet."""
    url = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/contents/{path}"
    resp = requests.get(url, headers=_gh_headers(), params={"ref": GITHUB_BRANCH})
    if resp.status_code == 404:
        return default
    resp.raise_for_status()
    content = base64.b64decode(resp.json()["content"]).decode("utf-8")
    return json.loads(content)


def github_write_json(path, data, message="update state"):
    """Write/overwrite a JSON file in the state repo (creates it if missing)."""
    url = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/contents/{path}"
    get_resp = requests.get(url, headers=_gh_headers(), params={"ref": GITHUB_BRANCH})
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

    payload = {
        "message": message,
        "content": base64.b64encode(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")).decode("utf-8"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    put_resp = requests.put(url, headers=_gh_headers(), json=payload)
    put_resp.raise_for_status()
    return put_resp.json()


def ensure_work_dir(job_id=None):
    work_dir = os.path.join(WORK_DIR, job_id) if job_id else WORK_DIR
    os.makedirs(work_dir, exist_ok=True)
    return work_dir
