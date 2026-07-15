"""
Generates narration audio for each scene using edge-tts (Microsoft Edge's free,
no-API-key neural TTS). Supports one male + one female neural voice per
supported language, with specific support for Pakistani Urdu (ur-PK) voices.

Key improvements for Pakistani Urdu content:
  - Uses ur-PK-AsadNeural (male) and ur-PK-UzmaNeural (female) — genuine
    Pakistani-accented Urdu voices, NOT Indian Urdu (ur-IN) or Arabic voices
  - Pre-processes text to add natural pauses and fix common TTS issues with
    Urdu punctuation and number pronunciation
  - Supports SSML prosody for more natural, human-like speech pacing
  - Concurrent generation for speed

Scenes are generated concurrently (I/O-bound network calls) to keep total
generation time low on longer videos.

FIX: script_generator.py's prompts deliberately embed script/editing markers
in the narration text -- "(PAUSE)", "(EMPHASIS)", and "[B-ROLL: description]"
-- for pacing and editing purposes. Those are directorial notes, not words to
be spoken. Because edge-tts's Communicate class does not parse any markup in
its text input (it just speaks the literal string), those markers were being
read aloud verbatim: the narrator would literally say "PAUSE", "EMPHASIS", and
read out B-ROLL descriptions in English mid-Urdu-sentence. That abrupt
language/register switching is very likely what sounded like "the AI reading
words differently" -- it wasn't a voice-quality issue, it was literal stage
directions being spoken. This is now stripped/converted before any text
reaches TTS. Number/currency symbols (%, $) are also now spoken out in Urdu
words instead of being left for the voice to guess at in English.
"""

import os
import asyncio
import edge_tts
import re

# ---------------------------------------------------------------------------
# Voice configuration — Pakistani Urdu voices (edge-tts)
# ---------------------------------------------------------------------------

# Primary Pakistani Urdu voices — these are specifically trained on Pakistani
# Urdu pronunciation, NOT Indian Urdu or Arabic. Using ur-IN voices (Gul, Salman)
# will give an Indian accent which sounds wrong to Pakistani audiences.
URDU_PK_VOICES = {
    "male": "ur-PK-AsadNeural",      # Male, Pakistani Urdu accent
    "female": "ur-PK-UzmaNeural",    # Female, Pakistani Urdu accent
}

# Fallback voices in order of preference (if ur-PK is ever unavailable)
URDU_FALLBACK_VOICES = {
    "male": ["ur-PK-AsadNeural", "ur-IN-SalmanNeural"],
    "female": ["ur-PK-UzmaNeural", "ur-IN-GulNeural"],
}

MAX_CONCURRENT_TTS = 6


# ---------------------------------------------------------------------------
# Script-marker stripping (PAUSE / EMPHASIS / B-ROLL directives)
# ---------------------------------------------------------------------------

# Matches "[B-ROLL: anything]" (case-insensitive, tolerant of odd spacing) --
# these are footage directions for the video assembler, never meant to be
# spoken.
_BROLL_MARKER_RE = re.compile(r"\[\s*B-?ROLL\s*:.*?\]", re.IGNORECASE | re.DOTALL)

# Matches "(PAUSE)" -- converted to a real Urdu comma so edge-tts produces an
# actual audible pause, instead of speaking the literal word "pause".
_PAUSE_MARKER_RE = re.compile(r"\(\s*PAUSE\s*\)", re.IGNORECASE)

# Matches "(EMPHASIS)" -- edge-tts's Communicate class has no way to apply
# per-word stress/emphasis without SSML (which it doesn't parse anyway), so
# there's nothing meaningful to convert this to. It's simply removed rather
# than being read aloud as the word "emphasis".
_EMPHASIS_MARKER_RE = re.compile(r"\(\s*EMPHASIS\s*\)", re.IGNORECASE)


def _strip_narration_markers(text: str) -> str:
    """Removes/converts script-writing directives that were never meant to be
    spoken by TTS. Must run BEFORE sentence splitting/word-count logic so
    those don't get thrown off by leftover bracket text."""
    if not text:
        return text

    text = _BROLL_MARKER_RE.sub(" ", text)
    text = _PAUSE_MARKER_RE.sub("،", text)
    text = _EMPHASIS_MARKER_RE.sub(" ", text)

    # Clean up artifacts left behind: doubled punctuation, stray leading
    # commas, extra whitespace from the removals above.
    text = re.sub(r"[،,]{2,}", "،", text)
    text = re.sub(r"\s*،\s*", "، ", text)
    text = re.sub(r"^[،,\s]+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Number / currency preprocessing
# ---------------------------------------------------------------------------

# "50%" -> "50 فیصد" ("feesad" / percent). Left bare, edge-tts tends to read
# the symbol in English mid-Urdu-sentence, which is a jarring language switch.
_PERCENT_RE = re.compile(r"(\d[\d,.]*)\s*%")

# "$500" -> "500 ڈالر" (dollar). Same reasoning as above -- spelled out in
# Urdu so the whole sentence stays in one language/register.
_DOLLAR_PREFIX_RE = re.compile(r"\$\s*(\d[\d,.]*)")


def _normalize_numbers_and_currency(text: str) -> str:
    """Spells out %/$ in Urdu words so the voice doesn't switch languages
    mid-sentence to read a bare symbol."""
    if not text:
        return text
    text = _PERCENT_RE.sub(r"\1 فیصد", text)
    text = _DOLLAR_PREFIX_RE.sub(r"\1 ڈالر", text)
    return text


# ---------------------------------------------------------------------------
# Urdu text pre-processing for better TTS output
# ---------------------------------------------------------------------------

def _preprocess_urdu_text(text: str) -> str:
    """
    Cleans and prepares Urdu text for TTS to sound more natural.

    Fixes applied:
      1. Strips script/editing markers -- (PAUSE), (EMPHASIS), [B-ROLL: ...] --
         that are meant for the human editor/assembler, not to be spoken
      2. Normalizes Arabic/Urdu punctuation to standard forms TTS handles better
      3. Spells out %/$ in Urdu words instead of leaving bare symbols that get
         read in English mid-sentence
      4. Removes excessive whitespace
      5. Ensures proper sentence-ending punctuation for natural cadence

    NOTE: this used to have a use_ssml mode that injected literal <break> tags
    for pauses. edge-tts's Communicate class does NOT parse SSML tags embedded
    in its text input -- it just synthesizes whatever string you give it, tags
    included, so every "<break time=...>" was being read aloud word-for-word.
    Plain commas already give edge-tts a perfectly natural pause, so that's all
    this does now -- no XML is ever inserted into text that gets sent to TTS.
    """
    if not text:
        return text

    # Strip script-writing directives FIRST, before any other processing,
    # since they're not spoken content at all.
    text = _strip_narration_markers(text)

    # Normalize various Unicode space/punctuation variants
    text = text.replace("\u060c", ",")   # Arabic comma -> standard comma
    text = text.replace("\u061b", ";")   # Arabic semicolon -> standard
    text = text.replace("\u061f", "?")   # Arabic question mark -> standard
    text = text.replace("\u0640", "")    # Tatweel (kashida) -> remove (TTS chokes on it)

    # Spell out %/$ in Urdu words instead of bare symbols
    text = _normalize_numbers_and_currency(text)

    # Ensure sentence-ending punctuation for natural TTS cadence
    # TTS engines often run sentences together without clear ending marks
    text = text.strip()
    if text and text[-1] not in ".!?۔":
        text += "."

    # Replace multiple spaces/newlines with single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _rate_to_edge_format(rate: str) -> str:
    """Converts a friendly rate name to the percent string edge-tts's Communicate expects."""
    return {"slow": "-10%", "default": "+0%", "fast": "+10%"}.get(rate, "+0%")


def _pitch_to_edge_format(pitch: str) -> str:
    """
    Converts a friendly pitch name to the Hz string edge-tts's Communicate expects.
    NOTE: edge-tts's native pitch parameter uses Hz offsets (e.g. "+0Hz", "-5Hz"),
    not percentages -- percentages only apply to rate/volume.
    """
    return {"low": "-5Hz", "default": "+0Hz", "high": "+5Hz"}.get(pitch, "+0Hz")


def _split_long_sentences(text: str, max_words: int = 18) -> str:
    """
    Breaks very long sentences into shorter ones for more natural TTS pacing.
    Pakistani conversational Urdu rarely uses sentences longer than 15-20 words.
    This inserts breaks at conjunctions (اور، لیکن، کیونکہ، تو) when sentences
    exceed max_words.

    Chunks are joined with an Urdu comma pause -- edge-tts's Communicate never
    parses inline SSML like <break>, so a literal "<break time=...>" tag
    inserted here would just be read aloud as text (this was the actual bug
    causing narration to speak out its own pause tags). A comma is a real,
    audible pause that edge-tts already handles naturally.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    # Common Urdu conjunctions where we can safely split
    split_markers = ["اور", "لیکن", "کیونکہ", "تو", "پھر", "چنانچہ", "حالانکہ"]

    result = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words and word in split_markers:
            result.append(" ".join(current_chunk))
            current_chunk = []
        elif len(current_chunk) >= max_words + 5:
            # Force split even if no conjunction marker found
            result.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        result.append(" ".join(current_chunk))

    if len(result) <= 1:
        return text

    return "، ".join(result)


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------

def _resolve_voice(language: str, voice_gender: str) -> str:
    """
    Resolves the best edge-tts voice for the given language and gender.

    For Urdu (ur): ALWAYS prefers ur-PK (Pakistani) voices over ur-IN (Indian)
    or any Arabic fallback. This is critical — Indian Urdu voices have a 
    distinctly different accent that Pakistani audiences find jarring.
    """
    # Normalize language code
    lang = (language or "ur").lower().strip()
    gender = (voice_gender or "male").lower().strip()

    # For any Urdu variant, force Pakistani voices
    if lang in ("ur", "urd", "urdu", "ur-pk", "ur-in"):
        voice = URDU_PK_VOICES.get(gender)
        if voice:
            return voice
        # Fallback to opposite gender if preferred gender unavailable
        fallback_gender = "female" if gender == "male" else "male"
        voice = URDU_PK_VOICES.get(fallback_gender)
        if voice:
            return voice

    # If config has EDGE_VOICES, try that as fallback
    try:
        from config import EDGE_VOICES, DEFAULT_LANGUAGE, DEFAULT_VOICE_GENDER
        lang_voices = EDGE_VOICES.get(lang, EDGE_VOICES.get(DEFAULT_LANGUAGE, {}))
        voice = lang_voices.get(gender) or lang_voices.get(DEFAULT_VOICE_GENDER)
        if voice:
            return voice
    except ImportError:
        pass

    # Ultimate fallback
    return URDU_PK_VOICES.get("male", "ur-PK-AsadNeural")


# ---------------------------------------------------------------------------
# Async TTS generation
# ---------------------------------------------------------------------------

async def _tts_edge_async(text: str, out_path: str, voice: str, rate: str = "slow", pitch: str = "default"):
    """
    Generates TTS audio using edge-tts.

    rate: "slow" | "default" | "fast" — passed as a real Communicate parameter
    pitch: "low" | "default" | "high" — passed as a real Communicate parameter

    IMPORTANT: edge-tts's Communicate class does not parse SSML embedded in its
    `text` argument -- it treats that string as literal spoken text. Previous
    versions of this function wrapped text in <voice>/<prosody>/<break> tags,
    which edge-tts then read aloud verbatim (the actual cause of narration
    speaking its own markup). Prosody is now set the only way edge-tts actually
    supports it: as real constructor keyword arguments. Script-writing markers
    like (PAUSE)/(EMPHASIS)/[B-ROLL: ...] are stripped in _preprocess_urdu_text
    before any text reaches this function, for the same underlying reason --
    they are not spoken content.
    """
    processed_text = _preprocess_urdu_text(text)
    processed_text = _split_long_sentences(processed_text)

    communicate = edge_tts.Communicate(
        processed_text,
        voice,
        rate=_rate_to_edge_format(rate),
        pitch=_pitch_to_edge_format(pitch),
    )
    await communicate.save(out_path)


def generate_scene_audio(
    text: str, 
    out_path: str, 
    language: str = "ur", 
    voice_gender: str = "male",
    rate: str = "slow",
    pitch: str = "default",
):
    """Writes an mp3 to out_path using edge-tts with Pakistani Urdu voice."""
    voice = _resolve_voice(language, voice_gender)
    asyncio.run(_tts_edge_async(text, out_path, voice, rate, pitch))


async def _generate_all_async(
    scenes: list, 
    audio_dir: str, 
    language: str, 
    voice_gender: str, 
    progress_callback=None,
    rate: str = "slow",
    pitch: str = "default",
):
    voice = _resolve_voice(language, voice_gender)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TTS)
    total = len(scenes)
    done_count = 0

    async def run_one(i, scene):
        nonlocal done_count
        out_path = os.path.join(audio_dir, f"scene_{i:03d}.mp3")

        # Pre-process narration text before TTS
        narration = scene.get("narration", "")

        async with semaphore:
            await _tts_edge_async(narration, out_path, voice, rate, pitch)

        scene["audio_path"] = out_path
        done_count += 1
        if progress_callback:
            progress_callback(done_count, total)

    await asyncio.gather(*(run_one(i, scene) for i, scene in enumerate(scenes)))


def generate_all_scene_audio(
    scenes: list, 
    work_dir: str, 
    language: str = "ur",
    voice_gender: str = "male", 
    progress_callback=None,
    rate: str = "slow",
    pitch: str = "default",
) -> list:
    """
    scenes: list of scene dicts from script_generator (each with "narration")
    language: language code — for Urdu, use "ur" (automatically resolves to ur-PK)
    voice_gender: "male" or "female" — picks Pakistani Urdu neural voice
    rate: "slow" | "default" | "fast" — speech speed (real edge-tts parameter)
    pitch: "low" | "default" | "high" — pitch variation (real edge-tts parameter)
    progress_callback: called as progress_callback(done, total) per scene

    Returns the same list with an added "audio_path" key per scene.
    """
    audio_dir = os.path.join(work_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    asyncio.run(_generate_all_async(
        scenes, audio_dir, language, voice_gender, progress_callback, rate, pitch
    ))

    return scenes


# ---------------------------------------------------------------------------
# Utility: Test a voice
# ---------------------------------------------------------------------------

def test_voice(text: str = "اسلام علیکم۔ یہ ایک ٹیسٹ ہے۔", 
               out_path: str = "test_voice.mp3",
               voice_gender: str = "male") -> str:
    """
    Quick utility to test a voice without running the full pipeline.
    Returns the path to the generated test audio.
    """
    voice = _resolve_voice("ur", voice_gender)
    asyncio.run(_tts_edge_async(text, out_path, voice))
    return out_path
