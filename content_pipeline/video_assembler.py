"""
Assembles per-scene images + narration audio into one final MP4.
Each scene gets a slow Ken-Burns zoom on its still image, with the narration
audio and a burned-in caption of the narration text. Scenes are joined with
short crossfade dissolves (instead of hard cuts) for a more polished,
less obviously auto-generated feel.

Requires ffmpeg to be installed on the Replit environment (add "ffmpeg" to
replit.nix or use the nix package manager in the Replit shell:
  `nix-env -iA nixpkgs.ffmpeg`
or simply enable it via Replit's "Nix" packages panel).
"""

import os
import subprocess
import re
import math
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

# Scene clips are CPU-bound (ffmpeg encode); a couple of workers helps even on
# a 2-core box since ffmpeg itself doesn't saturate a core the whole time.
# Kept modest (not scaled up with scene count) so longer videos (more scenes)
# don't pile on more *simultaneous* encodes and risk OOM on a small Replit VM
# — they just take a proportionally longer total time instead.
MAX_CONCURRENT_CLIPS = 2  # was 3 — reduced because the added color grade/
                          # vignette/watermark/timed-caption filters made each
                          # clip meaningfully more CPU-heavy, and Replit's
                          # shared/free-tier compute doesn't have much
                          # headroom for 3 full x264 encodes fighting at once.

# x264 encode speed preset for clip/title-card rendering. "veryfast" trades a
# little file-size efficiency for a large wall-clock speedup versus the
# libx264 default ("medium") — this draft pipeline optimizes for turnaround.
X264_PRESET = "veryfast"

# Hard ceiling on any single ffmpeg/ffprobe call so a stuck process can never
# hang a job forever (longer videos have more clips, so more chances for one
# process to wedge on a bad input). Raised from 600s -> 1200s: a genuinely
# long/text-heavy scene under Replit's shared CPU can legitimately take
# several minutes now that clips carry a heavier filter chain (color grade,
# vignette, watermark, timed multi-page captions) — the old 600s ceiling was
# killing renders that were still making real progress, not actually stuck.
FFMPEG_TIMEOUT_SECONDS = 1200

# Length of the dissolve between consecutive clips (intro/scenes/outro).
CROSSFADE_SECONDS = 0.6

# Faster preset used only for the per-scene/title-card intermediate renders —
# these get fully re-encoded again during the final crossfade join anyway, so
# spending extra encode time on them twice isn't worth it. The final join
# keeps the higher-quality X264_PRESET since that pass determines what the
# viewer actually sees, and is the one pass that can't be sped up this way.
INTERMEDIATE_PRESET = "ultrafast"


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


def _wrap_text_lines(text: str, width: int, fontsize: int, max_chars_per_line: int = None) -> list:
    """
    Wraps text into a list of lines sized to fit within `width` at `fontsize`,
    so drawtext never renders one giant line that overflows off-screen.
    Returns ALL wrapped lines (no cap) — callers decide how to paginate them.
    """
    if max_chars_per_line is None:
        # Rough estimate: a bold-ish sans glyph averages ~0.55x fontsize wide.
        max_chars_per_line = max(20, int(width * 0.9 / (fontsize * 0.55)))
    lines = textwrap.wrap(text, width=max_chars_per_line, break_long_words=False)
    if not lines:
        lines = [text]
    return lines


def _paginate_lines(lines: list, lines_per_page: int = 2, max_pages: int = 10) -> list:
    """
    Groups wrapped lines into pages of at most `lines_per_page` lines each.
    If that would produce more than `max_pages` (e.g. one scene ends up
    carrying an unusually large amount of narration — such as the LLM
    under-splitting a whole script chunk into a single scene), lines_per_page
    is increased just enough to fit within max_pages instead. This keeps the
    rendered drawtext filter chain bounded — each page is its own filter with
    a per-frame alpha/enable expression, so an unbounded page count was
    directly inflating encode time on unusually long scenes (each additional
    page adds real per-frame evaluation cost, not just a code-complexity
    concern). All text still gets shown; long scenes just get slightly denser
    pages (e.g. 3-4 lines instead of 2) rather than more of them.
    """
    if lines_per_page < 1:
        lines_per_page = 1
    n_pages = math.ceil(len(lines) / lines_per_page) if lines else 0
    if n_pages > max_pages:
        lines_per_page = math.ceil(len(lines) / max_pages)
    return [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)]


def _caption_filters(text: str, width: int, fontsize: int, bottom_margin: int, duration: float) -> str:
    """
    Builds a chain of drawtext filters covering the ENTIRE narration (no
    truncation), split into pages of up to 2 lines each that are shown one
    at a time across the clip's `duration` — each page's on-screen time is
    weighted by its share of the total word count, so pages advance roughly
    in step with how long that portion of narration takes to speak.

    Styled as bordered white text (no solid background box) with a quick
    fade in/out between pages, closer to a broadcast/streaming subtitle
    look than a flat boxed caption.
    """
    all_lines = _wrap_text_lines(text, width, fontsize)
    pages = _paginate_lines(all_lines, lines_per_page=2)
    if not pages:
        return ""

    word_counts = [max(1, sum(len(line.split()) for line in page)) for page in pages]
    total_words = sum(word_counts)
    line_height = fontsize + 22
    fade_seconds = 0.25

    filters = []
    t_cursor = 0.0
    for i, page in enumerate(pages):
        is_last = i == len(pages) - 1
        page_duration = duration * (word_counts[i] / total_words)
        start = t_cursor
        # Last page always extends to the exact end of duration, absorbing any
        # rounding drift instead of leaving a gap with no caption showing.
        end = duration if is_last else t_cursor + page_duration
        t_cursor = end

        # Fade window can't exceed half this page's own duration, so very
        # short pages still fade cleanly instead of the in/out overlapping.
        fade = min(fade_seconds, max(0.05, (end - start) / 2))
        alpha_expr = (
            f"if(lt(t,{start:.3f}+{fade:.3f}),(t-{start:.3f})/{fade:.3f},"
            f"if(lt(t,{end:.3f}-{fade:.3f}),1,({end:.3f}-t)/{fade:.3f}))"
        )

        # Lines within a page are drawn bottom-up, closest wrapped line nearest
        # bottom_margin, matching the original single-page stacking behavior.
        for j, line in enumerate(reversed(page)):
            y_from_bottom = bottom_margin + j * line_height
            filters.append(
                f"drawtext=text='{_escape_for_drawtext(line)}':fontcolor=white:fontsize={fontsize}:"
                f"borderw=3:bordercolor=black@0.85:"
                f"x=(w-text_w)/2:y=h-{y_from_bottom}:enable='between(t,{start:.3f},{end:.3f})':"
                f"alpha='{alpha_expr}'"
            )
    return ",".join(filters)


def _watermark_filter(channel_name: str, fontsize: int = 22) -> str:
    """
    Small, low-opacity channel name in the bottom-right corner, present for
    the whole clip (no timing/enable — unlike captions, this doesn't change).
    Kept subtle (0.55 alpha, no border/box) so it reads as a watermark rather
    than competing with the captions for attention.
    """
    if not channel_name:
        return ""
    return (
        f"drawtext=text='{_escape_for_drawtext(channel_name)}':"
        f"fontcolor=white@0.55:fontsize={fontsize}:x=w-text_w-24:y=h-text_h-20"
    )


def _get_media_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ],
        capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS,
    )
    duration_str = (result.stdout or "").strip()
    if not duration_str:
        stderr_tail = (result.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError(
            f"ffprobe could not read duration for {path}: " + " | ".join(stderr_tail)
        )
    return float(duration_str)


def _zoompan_expr(index: int, zoom_rate: float) -> str:
    """
    Alternates between a plain centered Ken Burns zoom (even scene indices)
    and a zoom with a slow lateral drift (odd indices), so consecutive
    scenes don't all move in exactly the same way — a small but noticeable
    step up from every clip using identical motion.
    """
    z_expr = f"min(zoom+{zoom_rate},1.15)"
    y_expr = "ih/2-(ih/zoom/2)"
    if index % 2 == 1:
        x_expr = "iw/2-(iw/zoom/2)+ceil(20*sin(on/40))"
    else:
        x_expr = "iw/2-(iw/zoom/2)"
    return f"z='{z_expr}':x='{x_expr}':y='{y_expr}'"


# Subtle color grade + vignette applied uniformly to every scene, for a
# consistent "graded" filmic look instead of raw, flat stock-photo colors.
COLOR_GRADE_FILTER = "eq=contrast=1.08:saturation=0.92:brightness=0.02,vignette=PI/5"


def _build_scene_clip(
    scene: dict, index: int, work_dir: str, width=1920, height=1080, zoom_rate=0.0008,
    channel_name: str = None,
) -> str:
    clip_dir = os.path.join(work_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    out_path = os.path.join(clip_dir, f"clip_{index:03d}.mp4")

    if not scene.get("image_path") or not scene.get("audio_path"):
        raise RuntimeError(f"Scene {index} is missing an image or audio file.")

    duration = _get_media_duration(scene["audio_path"])
    fps = 30
    total_frames = int(duration * fps)

    fontsize = 40
    caption_filters = _caption_filters(scene["narration"], width, fontsize, bottom_margin=90, duration=duration)
    watermark_filter = _watermark_filter(channel_name)

    # Ken Burns zoom (alternating plain/pan per scene) + color grade/vignette
    # for a consistent filmic look + timed, fading captions + a persistent
    # channel watermark, then a frozen-frame pad appended at the tail. That
    # pad exists purely so the crossfade dissolve into the next scene has
    # something silent/static to eat into — without it, the dissolve blended
    # directly into the last half-second of this scene's own narration/caption,
    # so the next scene's picture (and caption) visibly appeared while this
    # scene's voiceover was still speaking.
    # Scale is a modest 1.3x (not 2x) before zoompan — the zoom is subtle enough
    # that the extra resolution wasn't visibly needed, and halving the pixel
    # count here meaningfully cuts encode time.
    vf_parts = [
        f"scale={int(width * 1.3)}:{int(height * 1.3)}",
        f"zoompan={_zoompan_expr(index, zoom_rate)}:d={total_frames}:s={width}x{height}:fps={fps}",
        COLOR_GRADE_FILTER,
        caption_filters,
    ]
    if watermark_filter:
        vf_parts.append(watermark_filter)
    vf_parts.append(f"tpad=stop_mode=clone:stop_duration={CROSSFADE_SECONDS}")
    vf = ",".join(part for part in vf_parts if part)

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", scene["image_path"],
        "-i", scene["audio_path"],
        "-vf", vf,
        # loudnorm brings every scene's narration to the same target loudness,
        # so volume doesn't visibly jump between scenes voiced at slightly
        # different levels by the TTS engine. apad adds silence matching the
        # video's frozen-frame tail pad above, so the crossfade blends silence
        # into silence instead of clipping the tail of this scene's speech.
        "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=pad_dur={CROSSFADE_SECONDS}",
        "-c:v", "libx264", "-preset", INTERMEDIATE_PRESET,
        "-t", str(duration + CROSSFADE_SECONDS), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
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
    it joins cleanly. `lines` is a list of 1-2 strings (title + subtitle).
    """
    title_fontsize = 56
    subtitle_fontsize = 34
    line_height = title_fontsize + 16

    # Wrap the title (previously drawn as one unbroken line, which clipped
    # off both edges of the frame whenever it was too wide to fit) into as
    # many lines as it needs, capped at 3 so a very long title still fits
    # comfortably within a 3.5s card instead of overflowing vertically.
    title_lines = _wrap_text_lines(lines[0], width, title_fontsize)[:3]
    n_title_lines = len(title_lines)

    filters = []
    # Stack title lines centered as a block: with n lines, the first line's
    # vertical center sits (n-1)/2 line-heights above the card's true center,
    # and each subsequent line sits one line-height below the previous one.
    for i, line in enumerate(title_lines):
        offset = (i - (n_title_lines - 1) / 2) * line_height - 30
        filters.append(
            f"drawtext=text='{_escape_for_drawtext(line)}':fontcolor=white:fontsize={title_fontsize}:borderw=3:"
            f"bordercolor=black@0.7:x=(w-text_w)/2:y=(h-text_h)/2+{offset:.1f}"
        )

    if len(lines) > 1 and lines[1]:
        subtitle_lines = _wrap_text_lines(lines[1], width, subtitle_fontsize)[:2]
        subtitle_top = 40 + (n_title_lines - 1) * line_height
        for j, sub_line in enumerate(subtitle_lines):
            offset = subtitle_top + j * (subtitle_fontsize + 12)
            filters.append(
                f"drawtext=text='{_escape_for_drawtext(sub_line)}':fontcolor=white:fontsize={subtitle_fontsize}:borderw=2:"
                f"bordercolor=black@0.7:x=(w-text_w)/2:y=(h-text_h)/2+{offset:.1f}"
            )

    fade_out_start = max(0.0, duration - 0.6)
    filters.append(f"fade=t=in:st=0:d=0.5,fade=t=out:st={fade_out_start}:d=0.6")
    vf = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:r={fps}:d={duration}",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf", vf,
        "-c:v", "libx264", "-preset", INTERMEDIATE_PRESET, "-t", str(duration), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest",
        out_path,
    ]
    _run(cmd, "Title card render")
    return out_path


def _join_with_crossfades(clip_paths: list, final_path: str, crossfade_seconds: float = CROSSFADE_SECONDS) -> str:
    """
    Joins clips into one video using ffmpeg's xfade/acrossfade filters instead
    of the concat demuxer, so consecutive scenes dissolve into each other
    rather than hard-cutting — this alone is one of the biggest visible
    differences between an obviously auto-generated video and a polished one.
    Falls back to a straight re-encode (no dissolves) if there's only one clip.
    """
    n = len(clip_paths)
    if n == 0:
        raise RuntimeError("No clips to assemble into a final video.")

    if n == 1:
        cmd = [
            "ffmpeg", "-y", "-i", clip_paths[0],
            "-c:v", "libx264", "-preset", X264_PRESET, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart",
            final_path,
        ]
        _run(cmd, "Final render")
        return final_path

    durations = [_get_media_duration(p) for p in clip_paths]

    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]

    filter_parts = []

    prev_v_label = "0:v"
    running_duration = durations[0]
    for i in range(1, n):
        offset = max(0.0, running_duration - crossfade_seconds)
        out_label = f"v{i}"
        filter_parts.append(
            f"[{prev_v_label}][{i}:v]xfade=transition=fade:"
            f"duration={crossfade_seconds}:offset={offset:.3f}[{out_label}]"
        )
        prev_v_label = out_label
        running_duration = running_duration + durations[i] - crossfade_seconds

    prev_a_label = "0:a"
    for i in range(1, n):
        out_label = f"a{i}"
        filter_parts.append(
            f"[{prev_a_label}][{i}:a]acrossfade=d={crossfade_seconds}[{out_label}]"
        )
        prev_a_label = out_label

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{prev_v_label}]",
        "-map", f"[{prev_a_label}]",
        "-c:v", "libx264", "-preset", X264_PRESET, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        final_path,
    ]
    _run(cmd, "Final crossfade render")
    return final_path


def _mix_background_music(video_path: str, music_path: str, output_path: str, music_volume: float = 0.12) -> str:
    """
    Loops `music_path` under the video's existing narration audio, ducked to
    `music_volume` (a fraction of full volume — 0.12 sits well under narration
    without disappearing entirely), trimmed to match the video's own length.
    Video stream is copied (not re-encoded) since only the audio changes.
    Silently no-ops (returns video_path unchanged) if music_path is falsy or
    doesn't exist, so this feature is fully optional.
    """
    if not music_path or not os.path.isfile(music_path):
        return video_path

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex",
        f"[1:a]volume={music_volume}[music];[0:a][music]amix=inputs=2:duration=first:dropout_transition=3[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac",
        output_path,
    ]
    _run(cmd, "Background music mix")
    return output_path


def assemble_video(
    scenes: list,
    work_dir: str,
    output_name: str = "final_video.mp4",
    title: str = None,
    channel_name: str = None,
    include_intro: bool = True,
    include_outro: bool = True,
    style: str = "documentary",
    progress_callback=None,
    music_path: str = None,
) -> str:
    """
    Builds one clip per scene (plus optional intro/outro title cards for a more
    polished, human-produced feel) and joins them with short crossfade
    dissolves into the final MP4. `style` is a key from config.VIDEO_STYLES
    controlling the title-card color and Ken Burns zoom speed. Each scene
    carries a small `channel_name` watermark and a subtle color grade; if
    `music_path` points to an existing audio file, it's looped and ducked
    under the narration for the whole video. Returns the path to the final
    video.

    progress_callback, if given, is called as progress_callback(phase, done, total)
    where phase is "clips" while scene clips render (done/total = clips finished
    out of all scenes) and "join" once the final crossfade render starts/finishes
    (done/total = 0/1 then 1/1), so a caller can show fine-grained progress
    instead of just "video step in progress".
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
    total_scenes = len(scenes)
    scene_clip_paths = [None] * total_scenes
    clips_done = 0

    if progress_callback:
        progress_callback("clips", 0, total_scenes)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CLIPS) as executor:
        futures = {
            executor.submit(_build_scene_clip, scene, i, work_dir, zoom_rate=zoom_rate, channel_name=channel_name): i
            for i, scene in enumerate(scenes)
        }
        # as_completed (not iterating futures in submission order) so progress
        # reflects actual completions, not just submission order.
        for future in as_completed(futures):
            i = futures[future]
            scene_clip_paths[i] = future.result()
            clips_done += 1
            if progress_callback:
                progress_callback("clips", clips_done, total_scenes)
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

    if progress_callback:
        progress_callback("join", 0, 1)

    final_path = os.path.join(work_dir, output_name)
    _join_with_crossfades(clip_paths, final_path)

    if music_path and os.path.isfile(music_path):
        music_mixed_path = os.path.join(work_dir, "final_with_music.mp4")
        _mix_background_music(final_path, music_path, music_mixed_path)
        os.replace(music_mixed_path, final_path)
    elif music_path:
        print(f"[video_assembler] music_path '{music_path}' not found — skipping background music.")

    if progress_callback:
        progress_callback("join", 1, 1)

    return final_path