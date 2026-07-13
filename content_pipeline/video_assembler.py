"""
Assembles per-scene images + narration audio into one final MP4.
Each scene gets a slow Ken-Burns zoom on its still image, with the narration
audio and a caption of the narration text rendered via libass (not drawtext —
drawtext has no Arabic/Urdu text-shaping engine behind it, so joined Nastaliq
letterforms render as disconnected/isolated glyphs, and any glyph missing
from the fallback font shows as a "tofu" box). Scenes are joined with short
crossfade dissolves (instead of hard cuts) for a more polished, less
obviously auto-generated feel.

Requires ffmpeg to be installed on the Replit environment (add "ffmpeg" to
replit.nix or use the nix package manager in the Replit shell:
  `nix-env -iA nixpkgs.ffmpeg`
or simply enable it via Replit's "Nix" packages panel) — and specifically
requires an ffmpeg build with libass enabled (`ffmpeg -version` should list
`--enable-libass` under configuration; nearly all standard ffmpeg builds,
including Replit's nix package, include this).

Also requires a Nastaliq/Arabic-script font bundled at fonts/NotoNastaliqUrdu.ttf
(next to this file) — see FONTS_DIR/CAPTION_FONT_NAME below.
"""

import os
import subprocess
import re
import math
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

# Directory holding fonts bundled with the repo (not installed system-wide —
# passed to ffmpeg's `subtitles` filter via fontsdir=, so this works even in
# environments with no root/font-install access, like Replit).
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# Must exactly match the font's internal family name (not the filename) —
# libass resolves fonts by family name, looked up via fontsdir. Verify with
# `fc-scan fonts/NotoNastaliqUrdu.ttf | grep family` if you swap fonts.
CAPTION_FONT_NAME = "Noto Nastaliq Urdu"
CAPTION_FONT_FILENAME = "NotoNastaliqUrdu.ttf"

# Scene clips are CPU-bound (ffmpeg encode); a couple of workers helps even on
# a 2-core box since ffmpeg itself doesn't saturate a core the whole time.
# Kept modest (not scaled up with scene count) so longer videos (more scenes)
# don't pile on more *simultaneous* encodes and risk OOM on a small Replit VM
# — they just take a proportionally longer total time instead.
MAX_CONCURRENT_CLIPS = 3  # shared-CPU VM can usually handle 3 lightweight encodes — reduced because the added color grade/
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

# Frame rate used for BOTH scene clips and title cards. Must be a single
# shared constant, not duplicated per-function: scene clips and title cards
# get crossfaded/concatenated together, and ffmpeg's xfade filter (and
# stream-copy concat) both require every joined clip to share an identical
# timebase. A mismatch here (e.g. scenes at 15fps, title cards at 24fps)
# fails at join time with an xfade "timebase do not match" / "Invalid
# argument" error, not at render time — easy to miss until final assembly.
SCENE_FPS = 15  # was 24: ~37% fewer frames, visually identical for slow Ken Burns zooms — ~15-20% faster to encode with no perceptible quality difference for this style of motion.

# Length of the dissolve between consecutive clips (intro/scenes/outro).
CROSSFADE_SECONDS = 0.6

# Preset for per-scene/title-card renders. NOTE: despite the name, this is
# NOT a throwaway draft pass for most scenes — _join_mixed_transitions only
# re-encodes the intro/outro edge clips during the final join; every middle
# scene is joined via stream-copy concat (no re-encode at all). So whatever
# quality a scene clip is rendered at here is its final published quality.
# "faster" is a good middle ground: meaningfully better compression
# efficiency than "ultrafast" for a modest per-clip time cost.
INTERMEDIATE_PRESET = "faster"

# Explicit CRF (quality target, independent of preset) so visual quality
# stays consistent even if presets above are changed later. Lower = higher
# quality/larger file. 20 is a noticeable step up from libx264's default of
# 23, appropriate now that INTERMEDIATE_PRESET clips are final-quality output.
SCENE_CRF = "20"

# Cap each ffmpeg process's thread usage so MAX_CONCURRENT_CLIPS workers
# don't each try to claim every core and contend with each other. Leaves each
# worker a reasonable slice of a shared/limited Replit VM's CPU.
THREADS_PER_CLIP = max(1, (os.cpu_count() or 2) // MAX_CONCURRENT_CLIPS)


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
    # ffmpeg drawtext needs these characters escaped. Still used for the
    # watermark only (plain Latin text — no shaping issues there).
    text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _escape_ass_text(text: str) -> str:
    # ASS/SSA dialogue text needs backslashes and braces escaped (braces
    # start override tags like {\an5}). Actual line breaks are inserted
    # separately as literal "\N" tokens by the caller, not via this escape.
    text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return re.sub(r"\s+", " ", text).strip()


def _escape_for_filter_path(path: str) -> str:
    # ffmpeg filter option syntax uses ':' to separate key=value pairs and
    # wraps values in single quotes when needed — escape any embedded
    # single quotes/colons so a path containing either doesn't break parsing.
    return path.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _format_ass_time(seconds: float) -> str:
    """Formats seconds as ASS's H:MM:SS.cc timestamp (centiseconds)."""
    total_cs = max(0, int(round(seconds * 100)))
    hours, rem = divmod(total_cs, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _require_caption_font() -> str:
    """
    Returns the absolute path to the bundled caption font, raising a clear,
    actionable error immediately (not a cryptic ffmpeg filter failure deep in
    a subprocess call) if it hasn't been added to the repo yet.
    """
    font_path = os.path.join(FONTS_DIR, CAPTION_FONT_FILENAME)
    if not os.path.isfile(font_path):
        raise FileNotFoundError(
            f"Caption font not found at {font_path}. Download it (e.g. from "
            "https://github.com/google/fonts/raw/main/ofl/notonastaliqurdu/NotoNastaliqUrdu%5Bwght%5D.ttf) "
            f"and save it as fonts/{CAPTION_FONT_FILENAME} next to video_assembler.py — "
            "without it, Urdu/Arabic captions will render as boxes."
        )
    return font_path


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


def _write_caption_ass(
    text: str, width: int, height: int, fontsize: int, bottom_margin: int,
    duration: float, out_path: str,
) -> str:
    """
    Writes an ASS subtitle file covering the ENTIRE narration (no truncation),
    split into pages of up to 2 lines each shown one at a time across
    `duration` — each page's on-screen time weighted by its share of the
    total word count, same pacing logic as before.

    Rendered via ffmpeg's `subtitles` filter (libass -> HarfBuzz), NOT
    drawtext: this is what actually fixes Urdu/Arabic text — drawtext places
    glyphs one at a time with no shaping engine, so joined Nastaliq
    letterforms come out disconnected/isolated, and any glyph missing from
    the fallback font renders as a "tofu" box (□). libass does real script
    shaping and font fallback lookup via fontsdir.

    Style uses BorderStyle=3 (an opaque semi-transparent box behind the
    text) instead of a plain outline — this both fixes contrast against
    busy photo backgrounds and covers the "add a background bar for
    readability" request in one native filter option, no manual compositing.

    Returns out_path, or "" if there's no narration text to caption.
    """
    all_lines = _wrap_text_lines(text, width, fontsize)
    pages = _paginate_lines(all_lines, lines_per_page=2)
    if not pages:
        return ""

    word_counts = [max(1, sum(len(line.split()) for line in page)) for page in pages]
    total_words = sum(word_counts)

    events = []
    t_cursor = 0.0
    for i, page in enumerate(pages):
        is_last = i == len(pages) - 1
        page_duration = duration * (word_counts[i] / total_words)
        start = t_cursor
        end = duration if is_last else t_cursor + page_duration
        t_cursor = end

        page_text = "\\N".join(_escape_ass_text(line) for line in page)
        events.append(
            f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},"
            f"Caption,,0,0,0,,{page_text}"
        )

    # BackColour alpha 0x73 (~55% opaque black box). Colours are &HAABBGGRR.
    ass_content = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"  # 2 = no auto word-wrap; respect our manual \N breaks only
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,{CAPTION_FONT_NAME},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,"
        f"&H73000000,0,0,0,0,100,100,0,0,3,0,8,2,60,60,{bottom_margin},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "\n".join(events) + "\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    return out_path


def _watermark_filter(channel_name: str, fontsize: int = 26) -> str:
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
    fps = SCENE_FPS  # shared with _build_title_card — see SCENE_FPS comment for why this must match
    total_frames = int(duration * fps)

    fontsize = 40
    ass_path = os.path.join(clip_dir, f"scene_{index:03d}_captions.ass")
    caption_ass_path = _write_caption_ass(
        scene["narration"], width, height, fontsize, bottom_margin=90,
        duration=duration, out_path=ass_path,
    )
    caption_filter = ""
    if caption_ass_path:
        _require_caption_font()  # raises a clear error early if the font isn't bundled yet
        caption_filter = (
            f"subtitles=filename='{_escape_for_filter_path(caption_ass_path)}':"
            f"fontsdir='{_escape_for_filter_path(FONTS_DIR)}'"
        )
    watermark_filter = _watermark_filter(channel_name)

    # Ken Burns zoom (alternating plain/pan per scene) + color grade/vignette
    # for a consistent filmic look + timed captions (via libass — see
    # _write_caption_ass) + a persistent channel watermark, then a
    # frozen-frame pad appended at the tail. That pad exists purely so the
    # crossfade dissolve into the next scene has something silent/static to
    # eat into — without it, the dissolve blended directly into the last
    # half-second of this scene's own narration/caption, so the next scene's
    # picture (and caption) visibly appeared while this scene's voiceover was
    # still speaking.
    # Scale directly to output resolution before zoompan — no extra scale-up
    # buffer. Benchmarked: the 1.15x max zoom is subtle enough that the earlier
    # 1.3x buffer added no visible smoothness, just extra pixels to encode.
    vf_parts = [
        f"scale={width}:{height}",
        f"zoompan={_zoompan_expr(index, zoom_rate)}:d={total_frames}:s={width}x{height}:fps={fps}",
        COLOR_GRADE_FILTER,
        caption_filter,
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
        # single-pass dynaudnorm, no 2-pass analysis delay
        "-af", f"dynaudnorm=f=150:g=15,apad=pad_dur={CROSSFADE_SECONDS}",
        "-c:v", "libx264", "-preset", INTERMEDIATE_PRESET, "-crf", SCENE_CRF,
        "-threads", str(THREADS_PER_CLIP),
        "-t", str(duration + CROSSFADE_SECONDS), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-avoid_negative_ts", "make_zero", "-fflags", "+genpts",
        out_path,
    ]
    _run(cmd, f"Scene {index} render")
    return out_path


def _write_title_ass(
    lines: list, out_path: str, width: int, height: int, duration: float,
    title_fontsize: int = 56, subtitle_fontsize: int = 34,
) -> str:
    """
    Writes an ASS file for the intro/outro title card: title lines centered
    as a stacked block, with an optional subtitle line block below. Rendered
    via libass (subtitles filter), same as scene captions and for the same
    reason — the video's actual title is frequently Urdu too, and drawtext
    can't shape it correctly.

    Uses \\pos()+\\an5 (middle-center anchor) override tags per line to
    reproduce the original centered-stack layout math exactly: with an5,
    \\pos(x,y) places the *center* of that line at (x,y) regardless of the
    line's rendered width, so a plain width/2, height/2+offset reproduces
    the old (w-text_w)/2, (h-text_h)/2+offset drawtext positioning without
    needing to know each line's rendered width up front.
    """
    line_height = title_fontsize + 16
    title_lines = _wrap_text_lines(lines[0], width, title_fontsize)[:3]
    n_title_lines = len(title_lines)

    events = []
    end_ts = _format_ass_time(duration)
    for i, line in enumerate(title_lines):
        offset = (i - (n_title_lines - 1) / 2) * line_height - 30
        y = height / 2 + offset
        events.append(
            f"Dialogue: 0,0:00:00.00,{end_ts},Title,,0,0,0,,"
            f"{{\\an5\\pos({width / 2:.0f},{y:.0f})}}{_escape_ass_text(line)}"
        )

    if len(lines) > 1 and lines[1]:
        subtitle_lines = _wrap_text_lines(lines[1], width, subtitle_fontsize)[:2]
        subtitle_top = 40 + (n_title_lines - 1) * line_height
        for j, sub_line in enumerate(subtitle_lines):
            offset = subtitle_top + j * (subtitle_fontsize + 12)
            y = height / 2 + offset
            events.append(
                f"Dialogue: 0,0:00:00.00,{end_ts},Title,,0,0,0,,"
                f"{{\\an5\\pos({width / 2:.0f},{y:.0f})\\fs{subtitle_fontsize}\\bord2}}"
                f"{_escape_ass_text(sub_line)}"
            )

    # OutlineColour alpha 0x4D (~70% opaque black), matching the original
    # borderw=3/bordercolor=black@0.7 drawtext look. BorderStyle=1 keeps a
    # plain outline (not a box) since this sits on a solid color card, not a
    # busy photo — box background matters more for scene captions.
    ass_content = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Title,{CAPTION_FONT_NAME},{title_fontsize},&H00FFFFFF,&H000000FF,&H4D000000,"
        "&H00000000,0,0,0,0,100,100,0,0,1,3,0,5,20,20,20,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "\n".join(events) + "\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    return out_path


def _build_title_card(
    lines: list,
    out_path: str,
    duration: float = 3.5,
    width: int = 1920,
    height: int = 1080,
    fps: int = SCENE_FPS,  # shared with _build_scene_clip — see SCENE_FPS comment for why this must match
    bg_color: str = "0x141E30",
) -> str:
    """
    Renders a simple text-on-color title card (used for intro/outro) with a
    fade in/out, matching the codec/resolution/audio format of scene clips so
    it joins cleanly. `lines` is a list of 1-2 strings (title + subtitle).
    """
    title_fontsize = 56
    subtitle_fontsize = 34

    ass_path = out_path + ".ass"
    _require_caption_font()
    _write_title_ass(lines, ass_path, width, height, duration, title_fontsize, subtitle_fontsize)

    fade_out_start = max(0.0, duration - 0.6)
    vf = (
        f"subtitles=filename='{_escape_for_filter_path(ass_path)}':"
        f"fontsdir='{_escape_for_filter_path(FONTS_DIR)}',"
        f"fade=t=in:st=0:d=0.5,fade=t=out:st={fade_out_start}:d=0.6"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:r={fps}:d={duration}",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf", vf,
        "-c:v", "libx264", "-preset", INTERMEDIATE_PRESET, "-crf", SCENE_CRF,
        "-t", str(duration), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", "-shortest",
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
            "-c:v", "libx264", "-preset", X264_PRESET, "-crf", SCENE_CRF, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
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
        "-c:v", "libx264", "-preset", X264_PRESET, "-crf", SCENE_CRF, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        final_path,
    ]
    _run(cmd, "Final crossfade render")
    return final_path


def _concat_stream_copy(clip_paths: list, output_path: str) -> str:
    """
    Joins clips with hard cuts via ffmpeg's concat demuxer using -c copy —
    no re-encoding at all, just remuxing, so this is essentially instant
    regardless of how many/how long the clips are. Only valid because every
    clip in this pipeline is rendered with identical codec/resolution/fps
    settings; concat demuxer + stream copy requires that.
    """
    list_path = output_path + ".concat_list.txt"
    with open(list_path, "w") as f:
        for p in clip_paths:
            escaped = os.path.abspath(p).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy", "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
        output_path,
    ]
    try:
        _run(cmd, "Hard-cut concat")
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
    return output_path


def _join_mixed_transitions(
    clip_paths: list, final_path: str,
    fade_at_start: bool = True, fade_at_end: bool = True,
    crossfade_seconds: float = CROSSFADE_SECONDS,
) -> str:
    """
    Crossfades ONLY the intro->first-scene and/or last-scene->outro boundaries
    (whichever `fade_at_*` flags are set), and joins every other transition
    with an instant hard-cut stream-copy concat. This is the speed/polish
    middle ground: the crossfade re-encode — by far the most expensive part
    of assembly — only ever touches 2 clips at a time (the small edge pairs),
    never the full timeline, while still keeping the two transitions viewers
    notice most (the video's actual entrance and exit).

    Falls back to the original full-crossfade join (`_join_with_crossfades`)
    for small clip counts where this split wouldn't have a real "middle" run
    of hard-cut clips anyway (e.g. a single-scene video where the one scene
    clip has to fade both in from the intro and out to the outro) — in that
    case there's no cost advantage to the split, and the shared middle clip
    genuinely needs both fades applied to it, which only the original
    whole-timeline approach handles correctly.
    """
    n = len(clip_paths)
    if n <= 1:
        return _join_with_crossfades(clip_paths, final_path, crossfade_seconds)

    left_bound = 2 if (fade_at_start and n > 3) else 0
    right_bound = n - 2 if (fade_at_end and n > 3) else n

    if not (fade_at_start or fade_at_end) or left_bound >= right_bound or n <= 3:
        # No fades requested at all -> pure hard-cut concat. Or too few clips
        # for the split to make sense -> fall back to the original approach.
        if not (fade_at_start or fade_at_end):
            return _concat_stream_copy(clip_paths, final_path)
        return _join_with_crossfades(clip_paths, final_path, crossfade_seconds)

    tmp_dir = os.path.dirname(final_path) or "."
    pieces = []

    if fade_at_start:
        piece_start = os.path.join(tmp_dir, "_piece_start.mp4")
        _join_with_crossfades([clip_paths[0], clip_paths[1]], piece_start, crossfade_seconds)
        pieces.append(piece_start)

    middle_clips = clip_paths[left_bound:right_bound]
    if middle_clips:
        piece_middle = os.path.join(tmp_dir, "_piece_middle.mp4")
        if len(middle_clips) == 1:
            piece_middle = middle_clips[0]  # nothing to concat, use as-is
        else:
            _concat_stream_copy(middle_clips, piece_middle)
        pieces.append(piece_middle)

    if fade_at_end:
        piece_end = os.path.join(tmp_dir, "_piece_end.mp4")
        _join_with_crossfades([clip_paths[-2], clip_paths[-1]], piece_end, crossfade_seconds)
        pieces.append(piece_end)

    if len(pieces) == 1:
        os.replace(pieces[0], final_path) if os.path.dirname(pieces[0]) == tmp_dir else _concat_stream_copy(pieces, final_path)
    else:
        _concat_stream_copy(pieces, final_path)

    # Clean up intermediate pieces (concat already read them; safe to remove).
    for p in pieces:
        if p != final_path and os.path.exists(p) and os.path.basename(p).startswith("_piece_"):
            os.remove(p)

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
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
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
    _join_mixed_transitions(clip_paths, final_path, fade_at_start=include_intro, fade_at_end=include_outro)

    if music_path and os.path.isfile(music_path):
        music_mixed_path = os.path.join(work_dir, "final_with_music.mp4")
        _mix_background_music(final_path, music_path, music_mixed_path)
        os.replace(music_mixed_path, final_path)
    elif music_path:
        print(f"[video_assembler] music_path '{music_path}' not found — skipping background music.")

    if progress_callback:
        progress_callback("join", 1, 1)

    return final_path
