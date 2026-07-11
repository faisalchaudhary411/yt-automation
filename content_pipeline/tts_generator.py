"""
Generates narration audio for each scene.
Tries ElevenLabs first (better quality, uses your existing voice clone if you have one).
Falls back to gTTS (free, no API key) if ElevenLabs isn't configured or fails.
"""

import os
import requests
from gtts import gTTS
from config import ELEVENLABS_API_KEY, DEFAULT_LANGUAGE

ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")  # your cloned voice ID, if any
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


def _tts_elevenlabs(text: str, out_path: str) -> bool:
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        return False

    url = ELEVENLABS_URL.format(voice_id=ELEVENLABS_VOICE_ID)
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        return False

    with open(out_path, "wb") as f:
        f.write(resp.content)
    return True


def _tts_gtts(text: str, out_path: str, language: str = DEFAULT_LANGUAGE):
    tts = gTTS(text=text, lang=language)
    tts.save(out_path)


def generate_scene_audio(text: str, out_path: str, language: str = DEFAULT_LANGUAGE):
    """Writes an mp3 to out_path. Tries ElevenLabs, falls back to gTTS.

    ElevenLabs' multilingual model auto-detects the language from the text itself,
    so `language` (a gTTS-style code) is only used for the gTTS fallback path.
    """
    ok = _tts_elevenlabs(text, out_path)
    if not ok:
        _tts_gtts(text, out_path, language)


def generate_all_scene_audio(scenes: list, work_dir: str, language: str = DEFAULT_LANGUAGE) -> list:
    """
    scenes: list of scene dicts from script_generator (each with "narration")
    language: gTTS-style language code (used for the free fallback voice)
    Returns the same list with an added "audio_path" key per scene.
    """
    audio_dir = os.path.join(work_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    for i, scene in enumerate(scenes):
        out_path = os.path.join(audio_dir, f"scene_{i:03d}.mp3")
        generate_scene_audio(scene["narration"], out_path, language)
        scene["audio_path"] = out_path

    return scenes
