"""
Assembles per-scene images + narration audio into one final MP4.
Each scene gets a slow Ken-Burns zoom on its still image, with the narration
audio and a burned-in caption of the narration text.

Requires ffmpeg to be installed on the Replit environment (add "ffmpeg" to
replit.nix or use the nix package manager in the Replit shell:
  `nix-env -iA nixpkgs.ffmpeg`
or simply enable it via Replit's "Nix" packages panel).
"""

import os
import subprocess
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor

# Scene clips are CPU-bound (ffmpeg encode); a couple of workers helps even on
# a 2-core box since ffmpeg itself doesn't saturate a core the whole time.
# Kept modest (not scaled up with scene count) so longer videos (more scenes)
# don't pile on more *simultaneous* encodes and risk OOM on a small Replit VM
# — they just take a proportionally longer total time instead.
MAX_CONCURRENT_CLIPS = 3

# x264 encode speed preset for clip/title-card rendering. "veryfast" trades a
# little file-size efficiency for a large wall-clock speedup versus the
# libx264 default ("medium") — this draft pipeline optimizes for turnaround.
X264_PRESET = "veryfast"

# Hard ceiling on any single ffmpeg/ffprobe call so a stuck process can never
# hang a job forever (longer videos have more clips, so more chances for one
# process to wedge on a bad input).
FFMPEG_TIMEOUT_SECONDS = 600


def _run(cmd: list, step_name: str) -> None:
    """
    Runs a subprocess command and raises a RuntimeError with the actual
    ffmpeg/ffprobe stderr on failure. subprocess.run(check=True) alone only
    surfaces the exit code ("returned non-zero exit status 1"), which makes
    real failures (bad filter graph, corrupt input, disk full, etc.) almost
    impossible to diagnose from the job's error field. This keeps the same
    behavior but attaches the last part of stderr so failures are readable
    directly in the /status/<job_id> response.
    """
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "").strip().splitlines()[-15:]
        raise RuntimeError(
            f"{step_name} failed (exit {e.returncode}): " + " | ".join(stderr_tail)
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"{step_name} timed out after {FFMPEG_TIMEOUT_SECONDS}s") from e


def _escape_for_drawtext(text: str) -> str:
    # ffmpeg drawtext needs these characters escaped
    text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _wrap_caption_lines(text: str, width: int, fontsize: int, max_chars_per_line: int = None) -> list:
    """
    Wraps narration text into a list of lines so drawtext never renders one
    giant line that overflows off-screen (previously the whole scene's
    narration was drawn as a single unbroken line, so only a fragment of it
    was ever visible on screen, cut off mid-word). ffmpeg's drawtext filter
    does not reliably honor an embedded "\\n" as a newline in this build, so
    each line is rendered as its own chained drawtext filter instead (see
    `_caption_filters`).
    """
    if max_chars_per_line is None:
        # Rough estimate: a bold-ish sans glyph averages ~0.55x fontsize wide.
        max_chars_per_line = max(20, int(width * 0.9 / (fontsize * 0.55)))
    lines = textwrap.wrap(text, width=max_chars_per_line, break_long_words=False)
    if not lines:
        lines = [text]
    # Cap at 3 lines so captions never eat too much of the frame; if the
    # narration is longer than that it still reads fine without every word.
    return lines[:3]


def _caption_filters(text: str, width: int, fontsize: int, bottom_margin: int) -> str:
    """
    Builds a chain of drawtext filters (one per wrapped line), stacked upward
    from `bottom_margin` pixels above the bottom edge, each with its own
    semi-transparent background box for readability over busy photos.
    """
    lines = _wrap_caption_lines(text, width, fontsize)
    line_height = fontsize + 22
    filters = []
    # Lines are drawn from the bottom line up, so the last wrapped line sits
    # closest to bottom_margin and earlier lines stack above it.
    for i, line in enumerate(reversed(lines)):
        y_from_bottom = bottom_margin + i * line_height
        filters.append(
            f"drawtext=text='{_escape_for_drawtext(line)}':fontcolor=white:fontsize={fontsize}:"
            f"box=1:boxcolor=black@0.55:boxborderw=12:"
            f"x=(w-text_w)/2:y=h-{y_from_bottom}"
        )
    return ",".join(reversed(filters))


def _get_audio_duration(audio_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
        ],
        capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS,
    )
    duration_str = (result.stdout or "").strip()
    if not duration_str:
        stderr_tail = (result.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError(
            f"ffprobe could not read duration for {audio_path}: " + " | ".join(stderr_tail)
        )
    return float(duration_str)


def _build_scene_clip(scene: dict, index: int, work_dir: str, width=1920, height=1080, zoom_rate=0.0008) -> str:
    clip_dir = os.path.join(work_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    out_path = os.path.join(clip_dir, f"clip_{index:03d}.mp4")

    if not scene.get("image_path") or not scene.get("audio_path"):
        raise RuntimeError(f"Scene {index} is missing an image or audio file.")

    duration = _get_audio_duration(scene["audio_path"])
    fps = 30
    total_frames = int(duration * fps)

    fontsize = 40
    caption_filters = _caption_filters(scene["narration"], width, fontsize, bottom_margin=90)

    # Ken Burns zoom (slow zoom-in) + boxed, word-wrapped captions at the bottom.
    # Scale is a modest 1.3x (not 2x) before zoompan — the zoom is subtle enough
    # that the extra resolution wasn't visibly needed, and halving the pixel
    # count here meaningfully cuts encode time.
    vf = (
        f"scale={int(width * 1.3)}:{int(height * 1.3)},"
        f"zoompan=z='min(zoom+{zoom_rate},1.15)':d={total_frames}:s={width}x{height}:fps={fps},"
        f"{caption_filters}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", scene["image_path"],
        "-i", scene["audio_path"],
        "-vf", vf,
        "-c:v", "libx264", "-preset", X264_PRESET, "-t", str(duration), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest",
        "-avoid_negative_ts", "make_zero", "-fflags", "+genpts",
        out_path,
    ]
    _run(cmd, f"Scene {index} render")
    return out_path


def _build_title_card(
    lines: list,
    out_path: str,
    duration: float = 3.5,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    bg_color: str = "0x141E30",
) -> str:
    """
    Renders a simple text-on-color title card (used for intro/outro) with a
    fade in/out, matching the codec/resolution/audio format of scene clips so
    it concatenates cleanly. `lines` is a list of 1-2 strings (title + subtitle).
    """
    title = _escape_for_drawtext(lines[0])
    filters = [
        f"drawtext=text='{title}':fontcolor=white:fontsize=56:borderw=3:"
        f"bordercolor=black@0.7:x=(w-text_w)/2:y=(h-text_h)/2-30"
    ]
    if len(lines) > 1 and lines[1]:
        subtitle = _escape_for_drawtext(lines[1])
        filters.append(
            f"drawtext=text='{subtitle}':fontcolor=white:fontsize=34:borderw=2:"
            f"bordercolor=black@0.7:x=(w-text_w)/2:y=(h-text_h)/2+40"
        )
    fade_out_start = max(0.0, duration - 0.6)
    filters.append(f"fade=t=in:st=0:d=0.5,fade=t=out:st={fade_out_start}:d=0.6")
    vf = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:r={fps}:d={duration}",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf", vf,
        "-c:v", "libx264", "-preset", X264_PRESET, "-t", str(duration), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest",
        out_path,
    ]
    _run(cmd, "Title card render")
    return out_path


def assemble_video(
    scenes: list,
    work_dir: str,
    output_name: str = "final_video.mp4",
    title: str = None,
    channel_name: str = None,
    include_intro: bool = True,
    include_outro: bool = True,
    style: str = "documentary",
) -> str:
    """
    Builds one clip per scene (plus optional intro/outro title cards for a more
    polished, human-produced feel) and concatenates them into the final MP4.
    `style` is a key from config.VIDEO_STYLES controlling the title-card color
    and Ken Burns zoom speed. Returns the path to the final video.
    """
    from config import VIDEO_STYLES, DEFAULT_VIDEO_STYLE
    style_conf = VIDEO_STYLES.get(style, VIDEO_STYLES[DEFAULT_VIDEO_STYLE])
    bg_color = style_conf["bg_color"]
    zoom_rate = style_conf["zoom_rate"]

    clip_dir = os.path.join(work_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)

    clip_paths = []

    if include_intro:
        subtitle = f"A {channel_name} Story" if channel_name else "A Documentary Story"
        clip_paths.append(
            _build_title_card(
                [title or "", subtitle],
                os.path.join(clip_dir, "intro.mp4"),
                duration=3.5,
                bg_color=bg_color,
            )
        )

    # Scene clips are independent of each other, so render them concurrently —
    # this is the single biggest win for total generation time on longer videos.
    # A failure in any one scene is surfaced immediately (future.result() below
    # re-raises it) rather than silently producing a broken/incomplete video.
    scene_clip_paths = [None] * len(scenes)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CLIPS) as executor:
        futures = {
            executor.submit(_build_scene_clip, scene, i, work_dir, zoom_rate=zoom_rate): i
            for i, scene in enumerate(scenes)
        }
        for future in futures:
            i = futures[future]
            scene_clip_paths[i] = future.result()
    clip_paths += scene_clip_paths

    if include_outro:
        subtitle = f"Subscribe to {channel_name} for more" if channel_name else "Subscribe for more stories like this"
        clip_paths.append(
            _build_title_card(
                ["Thanks for watching!", subtitle],
                os.path.join(clip_dir, "outro.mp4"),
                duration=4.0,
                bg_color=bg_color,
            )
        )

    concat_list_path = os.path.join(work_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for path in clip_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    final_path = os.path.join(work_dir, output_name)
    # Re-encode on concat instead of "-c copy". Stream-copy concat is brittle
    # once there are more than a handful of segments (this pipeline's own
    # symptom: a 3-scene ~3-min video concatenated fine, but a 10-16 scene
    # ~6-min video did not) — small per-clip timestamp/keyframe mismatches
    # between many independently-encoded segments can make the concat demuxer
    # produce a truncated or corrupt final file, or make ffmpeg exit non-zero,
    # with the failure only showing up once enough clips are chained together.
    # Re-encoding here guarantees one consistent timeline regardless of scene
    # count, at the cost of one extra (fast, "veryfast" preset) encode pass.
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c:v", "libx264", "-preset", X264_PRESET, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        final_path,
    ]
    _run(cmd, "Final concat/render")
    return final_path
