"""
Generates YouTube Shorts from an already-assembled long-form video.

Runs AFTER the main video is fully rendered and uploaded -- this cuts a few
highlight windows out of the FINISHED mp4 (not the raw scene clips), so it
doesn't touch or duplicate any of the main assembly logic. Each short is
reformatted from the main video's horizontal 16:9 frame into a vertical
9:16 frame using a blurred, scaled copy of the same footage as a background
fill -- the standard "reels conversion" look, not just a cropped/cut-off
picture.

Captions are NOT burned onto shorts (kept deliberately simple/safe, same
call as the main pipeline's non-Latin-script languages) -- shorts are
just video + the original narration audio.
"""

import os
import subprocess

from automation.subtitles import compute_scene_start_times
from content_pipeline.video_assembler import get_chapter_card_scene_indices

SHORT_WIDTH = 1080
SHORT_HEIGHT = 1920


def _get_media_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _pick_start_times(
    scenes: list, chapters: list, include_intro: bool, include_outro: bool,
    crossfade_seconds: float, logo_sting_seconds: float, count: int, clip_duration: float,
    total_video_duration: float,
) -> list:
    """Picks `count` good start points for short clips, spread across the
    video's chapter boundaries where possible (a chapter start is usually a
    natural "hook" moment), falling back to even spacing if there aren't
    enough chapters. Avoids starting inside the last `clip_duration`
    seconds of the video (nothing to cut) or inside the outro card."""
    durations = [_get_media_duration(s["audio_path"]) for s in scenes]
    scene_starts = compute_scene_start_times(
        durations, include_intro, include_outro, chapters, crossfade_seconds, logo_sting_seconds,
    )

    latest_allowed_start = max(0.0, total_video_duration - clip_duration - 1.0)

    candidates = []
    if chapters:
        card_indices = get_chapter_card_scene_indices(chapters, len(scenes))
        for scene_index in sorted(card_indices.keys()):
            if scene_index < len(scene_starts):
                candidates.append(scene_starts[scene_index])

    if len(candidates) < count:
        # Not enough chapter boundaries -- fall back to even spacing across
        # the whole video, skipping a little off the very start (intro card).
        span = max(latest_allowed_start, 1.0)
        for i in range(count):
            candidates.append(span * (i + 1) / (count + 1))

    # Dedup, clamp, sort, and take the first `count`.
    seen = []
    for t in sorted(candidates):
        t = min(max(t, 0.0), latest_allowed_start)
        if not seen or (t - seen[-1]) > (clip_duration * 0.5):
            seen.append(t)
    return seen[:count] if len(seen) >= count else seen


def generate_shorts(
    video_path: str, scenes: list, chapters: list, work_dir: str,
    include_intro: bool = True, include_outro: bool = True,
    crossfade_seconds: float = 0.6, logo_sting_seconds: float = 0.0,
    count: int = 3, clip_duration: float = 45.0,
) -> list:
    """Cuts `count` short highlight clips from the finished video, each
    `clip_duration` seconds, reformatted to vertical 9:16. Returns a list of
    absolute file paths to the generated shorts (fewer than `count` if the
    video is too short to fit that many non-overlapping clips)."""
    total_duration = _get_media_duration(video_path)
    if total_duration < clip_duration * 1.5:
        print(f"[shorts] Video is only {total_duration:.0f}s -- too short to cut "
              f"{count} non-overlapping {clip_duration:.0f}s shorts. Skipping.")
        return []

    start_times = _pick_start_times(
        scenes, chapters, include_intro, include_outro, crossfade_seconds, logo_sting_seconds,
        count, clip_duration, total_duration,
    )

    shorts_dir = os.path.join(work_dir, "shorts")
    os.makedirs(shorts_dir, exist_ok=True)
    output_paths = []

    for i, start in enumerate(start_times):
        out_path = os.path.join(shorts_dir, f"short_{i + 1}.mp4")
        actual_duration = min(clip_duration, total_duration - start)
        filter_complex = (
            f"[0:v]scale={SHORT_WIDTH}:{SHORT_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={SHORT_WIDTH}:{SHORT_HEIGHT},gblur=sigma=20[bg];"
            f"[0:v]scale={SHORT_WIDTH}:-2,setsar=1[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(actual_duration),
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "0:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            output_paths.append(out_path)
            print(f"[shorts] Generated short {i + 1}/{len(start_times)} "
                  f"(start={start:.0f}s, duration={actual_duration:.0f}s): {out_path}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: short {i + 1} failed to render ({e.stderr[-500:] if e.stderr else e}). Skipping just this one.")
        except subprocess.TimeoutExpired:
            print(f"Warning: short {i + 1} timed out rendering. Skipping just this one.")

    return output_paths
