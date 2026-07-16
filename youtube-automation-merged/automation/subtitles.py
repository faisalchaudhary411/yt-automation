"""
Subtitles (Stage 3)
===================
Generates a real .srt subtitle file for every video — without Whisper.

The donor system transcribed the finished audio with Whisper (a heavy model
download, slow on Replit). That's unnecessary here: the live pipeline already
knows every scene's exact narration text AND its exact audio duration, so the
SRT timeline is computed deterministically instead of guessed by a
transcriber — and it's always in sync, in the video's own language.

Timing replicates video_assembler's join logic exactly:
  - intro title card (if included) adds a fixed 3.5s offset
  - scene-to-scene transitions are hard cuts when BOTH intro and outro are
    included; otherwise the FIRST and/or LAST pair of scenes is crossfaded
    (0.6s overlap each), and with <= 3 scenes every transition is crossfaded
    (mirrors _join_mixed_transitions / _join_with_crossfades)
"""

import os

from content_pipeline.video_assembler import (
    _get_media_duration,
    _strip_narration_markers_for_captions,
    CROSSFADE_SECONDS,
)

# Must match the literal duration= passed to _build_title_card for the intro
# in video_assembler.assemble_video (the outro's length never affects scene
# start times, so only the intro constant is needed here).
INTRO_DURATION = 3.5


def _srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def compute_scene_start_times(durations: list, include_intro: bool, include_outro: bool) -> list:
    """Start time of each scene in the FINAL video, matching the assembler's
    transition logic (see module docstring)."""
    n = len(durations)
    fade_at_start = not include_intro
    fade_at_end = not include_outro

    # Which inter-scene transitions are crossfaded (overlap CROSSFADE_SECONDS)?
    crossfaded = []
    for i in range(1, n):
        if n <= 3:
            cf = fade_at_start or fade_at_end
        else:
            cf = (i == 1 and fade_at_start) or (i == n - 1 and fade_at_end)
        crossfaded.append(cf)

    offset = INTRO_DURATION if include_intro else 0.0
    starts = []
    cursor = offset
    for i in range(n):
        starts.append(cursor)
        cursor += durations[i]
        if i < len(crossfaded) and crossfaded[i]:
            cursor -= CROSSFADE_SECONDS
    return starts


def build_srt(scenes: list, include_intro: bool = True, include_outro: bool = True) -> str:
    """Builds SRT content from scene dicts (needs narration + audio_path)."""
    durations = [_get_media_duration(s["audio_path"]) for s in scenes]
    starts = compute_scene_start_times(durations, include_intro, include_outro)

    entries = []
    idx = 1
    for scene, start, dur in zip(scenes, starts, durations):
        text = _strip_narration_markers_for_captions(scene.get("narration", ""))
        if not text.strip():
            continue
        entries.append(
            f"{idx}\n{_srt_time(start)} --> {_srt_time(start + dur)}\n{text}\n"
        )
        idx += 1
    return "\n".join(entries)


def write_srt(scenes: list, work_dir: str, include_intro: bool = True,
              include_outro: bool = True) -> str:
    """Writes subtitles.srt into the job's work dir; returns the path."""
    srt_path = os.path.join(work_dir, "subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(build_srt(scenes, include_intro, include_outro))
    print(f"[subtitles] SRT written: {srt_path}")
    return srt_path
