"""
Assembles per-scene images + narration audio into one final MP4.
Each scene gets a slow Ken-Burns zoom on its still image, with the narration
audio and a caption of the narration text rendered via libass (not drawtext —
drawtext has no Arabic/Urdu text-shaping engine behind it, so joined Nastaliq
letterforms render as disconnected/isolated glyphs, and any glyph missing
from the fallback font shows as a "tofu" box). Scenes are joined with short
crossfade dissolves (instead of hard cuts) for a more polished, less
obviously auto-generated feel.

Requires ffmpeg to be installed with libass enabled (`ffmpeg -version` should list
`--enable-libass` under configuration; nearly all standard ffmpeg builds include this).

FONT HANDLING:
  The code first checks for a bundled font at fonts/NotoNastaliqUrdu.ttf (relative
  to this script). If not found, it falls back to system-installed fonts detected
  via fc-list. This ensures Urdu/Arabic captions render correctly whether running
  in a container with bundled fonts or on a system with global font packages.
"""

import os
import subprocess
import re
import math
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Font resolution — try bundled first, then fall back to system fonts
# ---------------------------------------------------------------------------

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# Primary font for Urdu/Arabic text (Nastaliq style, preferred for Pakistani Urdu)
CAPTION_FONT_NAME = "Noto Nastaliq Urdu"
CAPTION_FONT_FILENAME = "NotoNastaliqUrdu.ttf"

# Fallback fonts in order of preference (system-wide, no bundling needed)
FALLBACK_FONT_NAMES = [
    "Noto Nastaliq Urdu",
    "Noto Naskh Arabic",
    "Noto Sans Arabic",
    "FreeSerif",
]
LATIN_FONT_NAME = "DejaVu Sans"
LATIN_FONT_FILENAME = "DejaVuSans.ttf"

# Latin fallback fonts (for English text - intro/outro/titles)
LATIN_FALLBACK_FONTS = [
    "DejaVu Sans",
    "Liberation Sans", 
    "Noto Sans",
    "FreeSans",
    "Arial",
]


def _resolve_font(for_latin_text: bool = False) -> tuple:
    """
    Returns (font_family_name, fonts_dir_path) for use with ffmpeg's subtitles filter.

    Priority:
      1. Bundled font in local fonts/ directory (for containers/Replit)
      2. System-installed font found via fc-list (for systems with font packages)

    for_latin_text: If True, resolves a Latin-capable font (DejaVu Sans, etc.) for
    English text. If False, resolves an Urdu/Arabic-shaping-capable font (Noto
    Nastaliq Urdu, etc.) so Urdu captions render properly instead of tofu boxes.

    If no suitable font is found, raises a clear error with installation instructions.
    """
    if for_latin_text:
        bundled_filename = LATIN_FONT_FILENAME
        bundled_family_name = LATIN_FONT_NAME
        fallback_names = LATIN_FALLBACK_FONTS
        label = "Latin"
    else:
        bundled_filename = CAPTION_FONT_FILENAME
        bundled_family_name = CAPTION_FONT_NAME
        fallback_names = FALLBACK_FONT_NAMES
        label = "Urdu/Arabic"

    # 1. Check bundled font first
    bundled_path = os.path.join(FONTS_DIR, bundled_filename)
    if os.path.isfile(bundled_path):
        # Verify the family name matches what libass expects
        try:
            result = subprocess.run(
                ["fc-scan", "--format=%{family}\n", bundled_path],
                capture_output=True, text=True, timeout=10
            )
            family = result.stdout.strip().split("\n")[0].strip()
            if family:
                return (family, FONTS_DIR)
        except Exception:
            pass  # fall through to system fonts
        return (bundled_family_name, FONTS_DIR)

    # 2. Fall back to system-installed fonts
    for font_name in fallback_names:
        try:
            result = subprocess.run(
                ["fc-list", font_name, ":file"],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip():
                # Font is installed system-wide; no need for fontsdir
                return (font_name, "")
        except Exception:
            continue

    # 3. Nothing found — raise helpful error
    raise FileNotFoundError(
        f"No {label}-capable font found. Tried: {fallback_names}\n\n"
        "To fix this, either:\n"
        "  a) Download the required font and place it at:\n"
        f"     {os.path.join(FONTS_DIR, bundled_filename)}\n"
        "  b) Install system fonts: apt-get install fonts-noto-core fonts-freefont-ttf"
    )


# Resolve fonts at module load (fails fast if missing)
(_RESOLVED_FONT_FAMILY, _RESOLVED_FONTS_DIR) = _resolve_font(for_latin_text=False)
(_RESOLVED_LATIN_FONT, _RESOLVED_LATIN_FONTS_DIR) = _resolve_font(for_latin_text=True)


def _get_fonts_dir_for_filter(for_latin: bool = False) -> str:
    """Returns the fontsdir path for ffmpeg subtitles filter, or empty string if using system fonts."""
    if for_latin:
        return _RESOLVED_LATIN_FONTS_DIR
    return _RESOLVED_FONTS_DIR


def _get_font_family(for_latin: bool = False) -> str:
    """Returns the resolved font family name for ASS styles."""
    if for_latin:
        return _RESOLVED_LATIN_FONT
    return _RESOLVED_FONT_FAMILY


def _is_latin_text(text: str) -> bool:
    """Detects whether a string is primarily Latin/English (vs. Urdu/Arabic)."""
    if not text:
        return True
    latin_chars = sum(1 for c in text if ord(c) < 128)
    return (latin_chars / max(len(text), 1)) > 0.5


def _combined_fonts_dir() -> str:
    """
    Returns a single fontsdir usable for an ASS file that mixes Urdu and Latin
    fonts in the same document (e.g. title cards with per-line \\fn overrides).
    Both bundled fonts live under the same FONTS_DIR, so one directory covers
    either/both; falls back to "" (system font cache only) if it doesn't exist.
    """
    if os.path.isdir(FONTS_DIR):
        return FONTS_DIR
    return ""


# ---------------------------------------------------------------------------
# Rest of the configuration
# ---------------------------------------------------------------------------

MAX_CONCURRENT_CLIPS = 3
X264_PRESET = "veryfast"
FFMPEG_TIMEOUT_SECONDS = 1200
SCENE_FPS = 15
CROSSFADE_SECONDS = 0.6
INTERMEDIATE_PRESET = "faster"
SCENE_CRF = "20"
THREADS_PER_CLIP = max(1, (os.cpu_count() or 2) // MAX_CONCURRENT_CLIPS)
COLOR_GRADE_FILTER = "eq=contrast=1.08:saturation=0.92:brightness=0.02,vignette=PI/5"


def _run(cmd: list, step_name: str) -> None:
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
    text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _escape_ass_text(text: str) -> str:
    text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return re.sub(r"\s+", " ", text).strip()


def _escape_for_filter_path(path: str) -> str:
    return path.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _format_ass_time(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    hours, rem = divmod(total_cs, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _wrap_text_lines(text: str, width: int, fontsize: int, max_chars_per_line: int = None) -> list:
    if max_chars_per_line is None:
        # Nastaliq is wider per character than Latin; use more conservative estimate
        max_chars_per_line = max(15, int(width * 0.85 / (fontsize * 0.65)))
    lines = textwrap.wrap(text, width=max_chars_per_line, break_long_words=False)
    if not lines:
        lines = [text]
    return lines


def _paginate_lines(lines: list, lines_per_page: int = 2, max_pages: int = 10) -> list:
    if lines_per_page < 1:
        lines_per_page = 1
    n_pages = math.ceil(len(lines) / lines_per_page) if lines else 0
    if n_pages > max_pages:
        lines_per_page = math.ceil(len(lines) / max_pages)
    return [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)]


def _write_caption_ass(
    text: str, width: int, height: int, fontsize: int, bottom_margin: int,
    duration: float, out_path: str, for_latin: bool = False,
) -> str:
    """
    Writes an ASS subtitle file with proper text shaping via libass.
    Uses the resolved font family (bundled or system).

    for_latin: If True, uses Latin-capable font for English text (prevents tofu boxes).
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

    font_family = _get_font_family(for_latin=for_latin)

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
        f"Style: Caption,{font_family},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,"
        f"&H73000000,0,0,0,0,100,100,0,0,3,0,8,2,60,60,{bottom_margin},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "\n".join(events) + "\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    return out_path


def _watermark_filter(channel_name: str, fontsize: int = 26) -> str:
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
    z_expr = f"min(zoom+{zoom_rate},1.15)"
    y_expr = "ih/2-(ih/zoom/2)"
    if index % 2 == 1:
        x_expr = "iw/2-(iw/zoom/2)+ceil(20*sin(on/40))"
    else:
        x_expr = "iw/2-(iw/zoom/2)"
    return f"z='{z_expr}':x='{x_expr}':y='{y_expr}'"


def _build_subtitles_filter(ass_path: str, for_latin: bool = False) -> str:
    """
    Builds the subtitles filter string. Only includes fontsdir when using bundled fonts.
    When using system fonts, fontsdir is omitted so libass uses the system font cache.
    """
    fonts_dir = _get_fonts_dir_for_filter(for_latin=for_latin)
    base = f"subtitles=filename='{_escape_for_filter_path(ass_path)}'"
    if fonts_dir:
        base += f":fontsdir='{_escape_for_filter_path(fonts_dir)}'"
    return base


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
    fps = SCENE_FPS
    total_frames = int(duration * fps)

    fontsize = 40
    ass_path = os.path.join(clip_dir, f"scene_{index:03d}_captions.ass")

    # Auto-detect if narration is primarily Latin/English text
    narration_text = scene.get("narration", "")
    is_latin = _is_latin_text(narration_text)

    caption_ass_path = _write_caption_ass(
        narration_text, width, height, fontsize, bottom_margin=90,
        duration=duration, out_path=ass_path, for_latin=is_latin,
    )

    caption_filter = ""
    if caption_ass_path:
        caption_filter = _build_subtitles_filter(caption_ass_path, for_latin=is_latin)

    watermark_filter = _watermark_filter(channel_name)

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
    title_fontsize: int = 92, subtitle_fontsize: int = 46,
) -> str:
    """
    Writes the title-card ASS file. Each line's font is chosen per-line based on
    whether that line's text is primarily Latin or Urdu/Arabic (via inline \\fn
    override), since a title card's main title (often Urdu) and its hardcoded
    subtitle (often English) can be different scripts within the same card.
    """
    line_height = title_fontsize + 20
    title_text = lines[0] if lines else ""
    title_font = _get_font_family(for_latin=_is_latin_text(title_text))
    title_lines = _wrap_text_lines(title_text, width, title_fontsize)[:3]
    n_title_lines = len(title_lines)

    events = []
    end_ts = _format_ass_time(duration)
    for i, line in enumerate(title_lines):
        offset = (i - (n_title_lines - 1) / 2) * line_height - 30
        y = height / 2 + offset
        events.append(
            f"Dialogue: 0,0:00:00.00,{end_ts},Title,,0,0,0,,"
            f"{{\\an5\\pos({width / 2:.0f},{y:.0f})\\fn{title_font}\\fs{title_fontsize}}}"
            f"{_escape_ass_text(line)}"
        )

    if len(lines) > 1 and lines[1]:
        subtitle_text = lines[1]
        subtitle_font = _get_font_family(for_latin=_is_latin_text(subtitle_text))
        subtitle_lines = _wrap_text_lines(subtitle_text, width, subtitle_fontsize)[:2]
        subtitle_top = 50 + (n_title_lines - 1) * line_height
        for j, sub_line in enumerate(subtitle_lines):
            offset = subtitle_top + j * (subtitle_fontsize + 14)
            y = height / 2 + offset
            events.append(
                f"Dialogue: 0,0:00:00.00,{end_ts},Title,,0,0,0,,"
                f"{{\\an5\\pos({width / 2:.0f},{y:.0f})\\fn{subtitle_font}\\fs{subtitle_fontsize}\\bord3}}"
                f"{_escape_ass_text(sub_line)}"
            )

    # Default/base style font doesn't matter much since every event carries its
    # own \fn override above, but set it to the title's font as a sane fallback.
    font_family = title_font

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
        f"Style: Title,{font_family},{title_fontsize},&H00FFFFFF,&H000000FF,&H00000000,"
        "&H00000000,0,0,0,0,100,100,0,0,1,4,1,5,20,20,20,1\n\n"
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
    fps: int = SCENE_FPS,
    bg_color: str = "0x141E30",
) -> str:
    title_fontsize = 92
    subtitle_fontsize = 46

    ass_path = out_path + ".ass"
    _write_title_ass(lines, ass_path, width, height, duration, title_fontsize, subtitle_fontsize)

    fade_out_start = max(0.0, duration - 0.6)
    fonts_dir = _combined_fonts_dir()
    subtitles_filter = f"subtitles=filename='{_escape_for_filter_path(ass_path)}'"
    if fonts_dir:
        subtitles_filter += f":fontsdir='{_escape_for_filter_path(fonts_dir)}'"
    vf = (
        f"{subtitles_filter},"
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
    list_path = output_path + ".concat_list.txt"
    with open(list_path, "w") as f:
        for p in clip_paths:
            escaped = os.path.abspath(p).replace("'", "'\''")
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
    n = len(clip_paths)
    if n <= 1:
        return _join_with_crossfades(clip_paths, final_path, crossfade_seconds)

    left_bound = 2 if (fade_at_start and n > 3) else 0
    right_bound = n - 2 if (fade_at_end and n > 3) else n

    if not (fade_at_start or fade_at_end) or n <= 3:
        if not (fade_at_start or fade_at_end):
            return _concat_stream_copy(clip_paths, final_path)
        return _join_with_crossfades(clip_paths, final_path, crossfade_seconds)

    if left_bound >= right_bound:
        # Too few clips for a dedicated middle section; crossfade all
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
            piece_middle = middle_clips[0]
        else:
            _concat_stream_copy(middle_clips, piece_middle)
        pieces.append(piece_middle)

    if fade_at_end:
        piece_end = os.path.join(tmp_dir, "_piece_end.mp4")
        _join_with_crossfades([clip_paths[-2], clip_paths[-1]], piece_end, crossfade_seconds)
        pieces.append(piece_end)

    if len(pieces) == 1:
        if os.path.dirname(pieces[0]) == tmp_dir:
            os.replace(pieces[0], final_path)
        else:
            _concat_stream_copy(pieces, final_path)
    else:
        _concat_stream_copy(pieces, final_path)

    for p in pieces:
        if p != final_path and os.path.exists(p) and os.path.basename(p).startswith("_piece_"):
            os.remove(p)

    return final_path


def _mix_background_music(video_path: str, music_path: str, output_path: str, music_volume: float = 0.12) -> str:
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


def _sanitize_filename(name: str, max_length: int = 80) -> str:
    """
    Converts a topic/title string into a safe, readable filename.

    Rules:
      - Removes/replaces characters illegal in filenames (/, \\, :, *, ?, ", <, >, |, !, etc.)
      - Collapses multiple spaces/hyphens into single ones
      - Truncates to max_length to avoid overly long filenames
      - Preserves Urdu/Arabic Unicode characters (they are valid in filenames on Linux/Mac)
      - Falls back to "final_video" if the result is empty
    """
    if not name:
        return "final_video.mp4"

    # Ensure string and strip whitespace
    name = str(name).strip()

    if not name:
        return "final_video.mp4"

    # Replace illegal filename characters with hyphens
    # Windows illegal: < > : " / \ | ? * 
    # Also common problematic chars: ! @ # $ % ^ & ( ) [ ] { } + = ` ~ '
    # And whitespace/newlines
    illegal_chars = r'''[<>:"/\\|?*!@#$%^&()\[\]{}+=`~'\n\r\t]'''
    safe = re.sub(illegal_chars, "-", name)

    # Collapse multiple spaces/hyphens/underscores into single hyphen
    safe = re.sub(r"[ \s_]+", "-", safe)
    safe = re.sub(r"-+", "-", safe)

    # Remove leading/trailing hyphens
    safe = safe.strip("-")

    # Truncate if too long (keep room for .mp4 extension)
    if len(safe) > max_length:
        safe = safe[:max_length].rsplit("-", 1)[0]  # cut at last hyphen to avoid mid-word

    # If somehow empty after sanitization, fallback
    if not safe:
        return "final_video.mp4"

    # Ensure .mp4 extension
    if not safe.lower().endswith(".mp4"):
        safe += ".mp4"

    return safe


def assemble_video(
    scenes: list,
    work_dir: str,
    output_name: str = None,   # explicit override; if None, falls back to title, then topic
    topic: str = None,         # fallback for auto-naming when no title/output_name is given
    title: str = None,         # generated video title; used for both the intro card and the output filename
    channel_name: str = None,
    include_intro: bool = True,
    include_outro: bool = True,
    style: str = "documentary",
    progress_callback=None,
    music_path: str = None,
) -> str:
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

    # Resolve output filename: explicit override > video title > topic > fallback
    if output_name:
        resolved_name = output_name
    elif title:
        resolved_name = _sanitize_filename(title)
    elif topic:
        resolved_name = _sanitize_filename(topic)
    else:
        resolved_name = "final_video.mp4"

    final_path = os.path.join(work_dir, resolved_name)
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
