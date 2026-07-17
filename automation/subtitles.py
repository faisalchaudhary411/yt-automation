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
  - named chapters (see generate_script()'s `chapters` list) split the scenes
    into groups, each separated by a CHAPTER_CARD_DURATION-long title card
    inserted with a hard cut on both sides (no crossfade eaten into it)
  - WITHIN each group: scene-to-scene transitions are hard cuts when BOTH
    intro and outro are included; otherwise the FIRST group's first pair
    and/or the LAST group's last pair is crossfaded (0.6s overlap each), and
    within any group of <= 3 scenes every transition is crossfaded (mirrors
    _join_mixed_transitions / _join_with_crossfades)
"""

import os

from content_pipeline.video_assembler import (
    _get_media_duration,
    _strip_narration_markers_for_captions,
    CROSSFADE_SECONDS,
    CHAPTER_CARD_DURATION,
    get_chapter_card_scene_indices,
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


def compute_scene_start_times(
    durations: list, include_intro: bool, include_outro: bool, chapters: list = None,
) -> list:
    """Start time of each scene in the FINAL video, matching the assembler's
    transition logic (see module docstring)."""
    n = len(durations)
    card_by_index = get_chapter_card_scene_indices(chapters or [], n)

    # Split into the same scene groups the assembler builds around chapter cards.
    groups = []
    current = []
    for i in range(n):
        if i in card_by_index and current:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)

    starts = [0.0] * n
    cursor = INTRO_DURATION if include_intro else 0.0

    for g_idx, group in enumerate(groups):
        is_first_group = g_idx == 0
        is_last_group = g_idx == len(groups) - 1
        fade_at_start = (not include_intro) if is_first_group else False
        fade_at_end = (not include_outro) if is_last_group else False

        gn = len(group)
        crossfaded = []
        for k in range(1, gn):
            if gn <= 3:
                cf = fade_at_start or fade_at_end
            else:
                cf = (k == 1 and fade_at_start) or (k == gn - 1 and fade_at_end)
            crossfaded.append(cf)

        for k, scene_i in enumerate(group):
            starts[scene_i] = cursor
            cursor += durations[scene_i]
            if k < len(crossfaded) and crossfaded[k]:
                cursor -= CROSSFADE_SECONDS

        if not is_last_group:
            cursor += CHAPTER_CARD_DURATION  # the chapter card that follows this group

    return starts


def build_srt(scenes: list, include_intro: bool = True, include_outro: bool = True,
              chapters: list = None) -> str:
    """Builds SRT content from scene dicts (needs narration + audio_path)."""
    durations = [_get_media_duration(s["audio_path"]) for s in scenes]
    starts = compute_scene_start_times(durations, include_intro, include_outro, chapters)

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
              include_outro: bool = True, chapters: list = None) -> str:
    """Writes subtitles.srt into the job's work dir; returns the path."""
    srt_path = os.path.join(work_dir, "subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(build_srt(scenes, include_intro, include_outro, chapters))
    print(f"[subtitles] SRT written: {srt_path}")
    return srt_path
