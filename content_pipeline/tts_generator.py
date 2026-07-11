"""
Generates narration audio for each scene using edge-tts (Microsoft Edge's free,
no-API-key neural TTS). Supports one male + one female neural voice per
supported language, unlike gTTS which has no gender control.

Scenes are generated concurrently (I/O-bound network calls) to keep total
generation time low on longer videos.
"""

import os
import asyncio
import edge_tts
from concurrent.futures import ThreadPoolExecutor
from config import EDGE_VOICES, DEFAULT_LANGUAGE, DEFAULT_VOICE_GENDER

MAX_CONCURRENT_TTS = 6


def _resolve_voice(language: str, voice_gender: str) -> str:
    lang_voices = EDGE_VOICES.get(language, EDGE_VOICES[DEFAULT_LANGUAGE])
    return lang_voices.get(voice_gender) or lang_voices.get(DEFAULT_VOICE_GENDER)


async def _tts_edge_async(text: str, out_path: str, voice: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def generate_scene_audio(text: str, out_path: str, language: str = DEFAULT_LANGUAGE, voice_gender: str = DEFAULT_VOICE_GENDER):
    """Writes an mp3 to out_path using edge-tts."""
    voice = _resolve_voice(language, voice_gender)
    asyncio.run(_tts_edge_async(text, out_path, voice))


async def _generate_all_async(scenes: list, audio_dir: str, language: str, voice_gender: str):
    voice = _resolve_voice(language, voice_gender)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TTS)

    async def run_one(i, scene):
        out_path = os.path.join(audio_dir, f"scene_{i:03d}.mp3")
        async with semaphore:
            await _tts_edge_async(scene["narration"], out_path, voice)
        scene["audio_path"] = out_path

    await asyncio.gather(*(run_one(i, scene) for i, scene in enumerate(scenes)))


def generate_all_scene_audio(scenes: list, work_dir: str, language: str = DEFAULT_LANGUAGE, voice_gender: str = DEFAULT_VOICE_GENDER) -> list:
    """
    scenes: list of scene dicts from script_generator (each with "narration")
    language: language code from config.LANGUAGES
    voice_gender: "male" or "female" — picks an edge-tts neural voice for that language
    Returns the same list with an added "audio_path" key per scene, generated concurrently.
    """
    audio_dir = os.path.join(work_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Run the async batch on its own thread so this stays a plain sync function
    # callable from the pipeline thread (which already has no running event loop,
    # but this keeps it safe if that ever changes).
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(lambda: asyncio.run(_generate_all_async(scenes, audio_dir, language, voice_gender))).result()

    return scenes
