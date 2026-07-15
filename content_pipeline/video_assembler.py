"""
Assembles per-scene images + narration audio into one final MP4.
Each scene gets a slow Ken-Burns zoom on its still image, with the narration
audio and a caption rendered via Pillow (not libass) to avoid Replit's
HarfBuzz-ng 10.2.0 regression that turns Nastaliq into tofu boxes.
Scenes are joined with short crossfade dissolves.
"""

import os
import subprocess
import re
import math
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

# Pillow is required for the PNG text-overlay fix
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    Image = ImageDraw = ImageFont = None

# ---------------------------------------------------------------------------
# Font resolution -- try bundled first, then fall back to system fonts
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

_FONT_MAGIC_BYTES = (
    b"\x00\x01\x00\x00",  # TrueType
    b"OTTO",                   # OpenType/CFF
    b"true",                   # TrueType (older mac)
    b"ttcf",                   # TrueType collection
)


def _validate_font_file(path: str) -> str:
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return f"could not stat file ({e})"

    if size == 0:
        return "file is 0 bytes (empty -- upload/commit likely failed)"

    if size < 2048:
        try:
            with open(path, "rb") as f:
                head = f.read(200)
            if head.startswith(b"version https://git-lfs"):
                return (
                    f"file is only {size} bytes and is a Git LFS pointer, not the "
                    "actual font binary."
                )
            return f"file is only {size} bytes -- too small to be a real font ({head[:50]!r})"
        except OSError as e:
            return f"could not read file to inspect it ({e})"

    try:
        with open(path, "rb") as f:
            magic = f.read(4)
    except OSError as e:
        return f"could not read file header ({e})"

    if magic not in _FONT_MAGIC_BYTES:
        return f"file header {magic!r} doesn't match any known font format (corrupted upload?)"

    return ""


def _resolve_font(for_latin_text: bool = False) -> tuple:
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

    bundled_path = os.path.join(FONTS_DIR, bundled_filename)
    print(f"[font_resolve:{label}] looking for bundled font at {bundled_path}")

    if os.path.isfile(bundled_path):
        problem = _validate_font_file(bundled_path)
        if problem:
            print(f"[font_resolve:{label}] FOUND bundled file but INVALID: {problem}")
        else:
            size = os.path.getsize(bundled_path)
            print(f"[font_resolve:{label}] bundled file looks valid ({size} bytes)")
            try:
                result = subprocess.run(
                    ["fc-scan", "--format=%{family}\n", bundled_path],
                    capture_output=True, text=True, timeout=10
                )
                family = result.stdout.strip().split("\n")[0].strip()
                if family:
                    print(f"[font_resolve:{label}] fc-scan reports family='{family}' -- using it")
                    return (family, FONTS_DIR)
                print(f"[font_resolve:{label}] fc-scan returned no family name; using hardcoded '{bundled_family_name}'")
            except Exception as e:
                print(f"[font_resolve:{label}] fc-scan failed ({e}); using hardcoded '{bundled_family_name}'")
            return (bundled_family_name, FONTS_DIR)
    else:
        print(f"[font_resolve:{label}] no bundled file found at that path")

    for font_name in fallback_names:
        try:
            result = subprocess.run(
                ["fc-list", font_name, ":file"],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip():
                print(f"[font_resolve:{label}] using system font '{font_name}' (fc-list found: {result.stdout.strip().splitlines()[0]})")
                return (font_name, "")
        except Exception:
            continue

    raise FileNotFoundError(
        f"No {label}-capable font found. Tried: {fallback_names}\n\n"
        "To fix this, either:\n"
        "  a) Place a REAL, valid font file at:\n"
        f"     {os.path.join(FONTS_DIR, bundled_filename)}\n"
        "     (check the file size on GitHub -- it should be hundreds of KB, not a few bytes)\n"
        "  b) Install system fonts: apt-get install fonts-noto-core fonts-freefont-ttf"
    )


# Resolve fonts at module load (fails fast if missing)
(_RESOLVED_FONT_FAMILY, _RESOLVED_FONTS_DIR) = _resolve_font(for_latin_text=False)
(_RESOLVED_LATIN_FONT, _RESOLVED_LATIN_FONTS_DIR) = _resolve_font(for_latin_text=True)


# Unicode blocks covering Arabic/Urdu script (including presentation forms used by
# some fonts/renderers for joined letterforms)
_ARABIC_SCRIPT_RANGES = (
    (0x0600, 0x06FF),  # Arabic (includes Urdu-specific letters)
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


def _is_arabic_script_char(c: str) -> bool:
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_SCRIPT_RANGES)


def _is_latin_text(text: str) -> bool:
    if not text:
        return True
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    arabic_count = sum(1 for c in letters if _is_arabic_script_char(c))
    return (arabic_count / len(letters)) < 0.15


# ---------------------------------------------------------------------------
# Pillow-based text rendering (replaces libass/ASS entirely)
# ---------------------------------------------------------------------------

def _resolve_font_path(for_latin: bool = False) -> str:
    """Returns an actual filesystem path to a TTF for Pillow."""
    filename = LATIN_FONT_FILENAME if for_latin else CAPTION_FONT_FILENAME
    bundled = os.path.join(FONTS_DIR, filename)
    if os.path.isfile(bundled) and not _validate_font_file(bundled):
        return bundled

    names = LATIN_FALLBACK_FONTS if for_latin else FALLBACK_FONT_NAMES
    for name in names:
        try:
            r = subprocess.run(
                ["fc-list", name, ":file"],
                capture_output=True, text=True, timeout=10
            )
            if r.stdout.strip():
                path = r.stdout.strip().splitlines()[0].split(":")[0]
                if os.path.isfile(path):
                    return path
        except Exception:
            continue
    return ""


def _render_text_page(lines, width, height, fontsize, bottom_margin, out_path, for_latin=False):
    """Renders a page of caption lines to a full-size transparent PNG."""
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required. Install it: pip install pillow")

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_path = _resolve_font_path(for_latin=for_latin)
    try:
        font = ImageFont.truetype(font_path, fontsize) if font_path else ImageFont.load_default()
    except Exception as e:
        print(f"[render_text] Font load failed ({e}), using default")
        font = ImageFont.load_default()

    direction = "ltr" if for_latin else "rtl"
    line_spacing = int(fontsize * 0.35)
    line_height = fontsize + line_spacing

    measured = []
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font, direction=direction)
        measured.append({
            "text": line,
            "w": bb[2] - bb[0],
            "h": bb[3] - bb[1],
            "left": bb[0],
            "top": bb[1],
        })

    max_w = max((m["w"] for m in measured), default=0)
    total_h = len(lines) * line_height - line_spacing

    x_start = (width - max_w) // 2
    y_start = height - bottom_margin - total_h

    pad_x, pad_y = 48, 24
    box = [
        max(0, x_start - pad_x),
        max(0, y_start - pad_y),
        min(width, x_start + max_w + pad_x),
        min(height, y_start + total_h + pad_y),
    ]
    draw.rectangle(box, fill=(0, 0, 0, 115))

    for i, m in enumerate(measured):
        x = (width - m["w"]) // 2 - m["left"]
        y = y_start + i * line_height - m["top"]

        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            draw.text((x + dx, y + dy), m["text"], font=font, fill=(0, 0, 0, 160), direction=direction)
        draw.text((x, y), m["text"], font=font, fill=(255, 255, 255, 255), direction=direction)

    img.save(out_path)
    return out_path


def _write_caption_pngs(text, width, height, fontsize, bottom_margin, duration, out_dir, prefix="cap", for_latin=False):
    """Paginates text and renders each page to a transparent PNG with timing info."""
    all_lines = _wrap_text_lines(text, width, fontsize)
    pages = _paginate_lines(all_lines, lines_per_page=2)
    if not pages:
        return []

    word_counts = [max(1, sum(len(line.split()) for line in page)) for page in pages]
    total_words = sum(word_counts)

    png_infos = []
    t_cursor = 0.0
    for i, page in enumerate(pages):
        is_last = i == len(pages) - 1
        page_duration = duration * (word_counts[i] / total_words)
        start = t_cursor
        end = duration if is_last else t_cursor + page_duration
        t_cursor = end

        out_path = os.path.join(out_dir, f"{prefix}_page_{i:03d}.png")
        _render_text_page(page, width, height, fontsize, bottom_margin, out_path, for_latin=for_latin)
        png_infos.append({"path": out_path, "start": start, "end": end})
    return png_infos


def _render_title_card_png(lines, width, height, out_path, title_fontsize=92, subtitle_fontsize=46):
    """Renders a title card to a full-size transparent PNG."""
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required. Install it: pip install pillow")

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    title_text = lines[0] if lines else ""
    is_title_latin = _is_latin_text(title_text)
    title_path = _resolve_font_path(for_latin=is_title_latin)
    try:
        title_font = ImageFont.truetype(title_path, title_fontsize) if title_path else ImageFont.load_default()
    except Exception:
        title_font = ImageFont.load_default()
    title_dir = "ltr" if is_title_latin else "rtl"
    title_lines = _wrap_text_lines(title_text, width, title_fontsize)[:3]

    subtitle_text = lines[1] if len(lines) > 1 else ""
    is_sub_latin = _is_latin_text(subtitle_text) if subtitle_text else True
    sub_path = _resolve_font_path(for_latin=is_sub_latin)
    try:
        sub_font = ImageFont.truetype(sub_path, subtitle_fontsize) if sub_path else ImageFont.load_default()
    except Exception:
        sub_font = ImageFont.load_default()
    sub_dir = "ltr" if is_sub_latin else "rtl"
    subtitle_lines = _wrap_text_lines(subtitle_text, width, subtitle_fontsize)[:2] if subtitle_text else []

    title_line_h = title_fontsize + 20
    sub_line_h = subtitle_fontsize + 14
    gap = 30 if subtitle_lines else 0
    total_h = len(title_lines) * title_line_h + gap + len(subtitle_lines) * sub_line_h
    y_cursor = (height - total_h) // 2

    for line in title_lines:
        bb = draw.textbbox((0, 0), line, font=title_font, direction=title_dir)
        text_w = bb[2] - bb[0]
        x = (width - text_w) // 2 - bb[0]
        y = y_cursor - bb[1]
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            draw.text((x + dx, y + dy), line, font=title_font, fill=(0, 0, 0, 180), direction=title_dir)
        draw.text((x, y), line, font=title_font, fill=(255, 255, 255, 255), direction=title_dir)
        y_cursor += title_line_h

    if subtitle_lines:
        y_cursor += gap - 10
        for line in subtitle_lines:
            bb = draw.textbbox((0, 0), line, font=sub_font, direction=sub_dir)
            text_w = bb[2] - bb[0]
            x = (width - text_w) // 2 - bb[0]
            y = y_cursor - bb[1]
            for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                draw.text((x + dx, y + dy), line, font=sub_font, fill=(0, 0, 0, 160), direction=sub_dir)
            draw.text((x, y), line, font=sub_font, fill=(255, 255, 255, 255), direction=sub_dir)
            y_cursor += sub_line_h

    img.save(out_path)
    return out_path


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
    narration_text = scene.get("narration", "")
    is_latin = _is_latin_text(narration_text)

    caption_pngs = _write_caption_pngs(
        narration_text, width, height, fontsize, bottom_margin=90,
        duration=duration, out_dir=clip_dir, prefix=f"scene_{index:03d}", for_latin=is_latin,
    )

    watermark_filter = _watermark_filter(channel_name)

    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene["image_path"], "-i", scene["audio_path"]]
    for info in caption_pngs:
        cmd += ["-loop", "1", "-i", info["path"]]

    base_filters = [
        f"scale={width}:{height}",
        f"zoompan={_zoompan_expr(index, zoom_rate)}:d={total_frames}:s={width}x{height}:fps={fps}",
        COLOR_GRADE_FILTER,
    ]
    filters = [f"[0:v]{','.join(base_filters)}[base]"]

    current_label = "[base]"
    for i, info in enumerate(caption_pngs):
        input_idx = 2 + i
        out_label = f"[cap{i}]" if i < len(caption_pngs) - 1 else "[vcap]"
        start = info["start"]
        end = info["end"]
        filters.append(
            f"{current_label}[{input_idx}:v]overlay=0:0:enable='between(t\\,{start:.3f}\\,{end:.3f})'{out_label}"
        )
        current_label = out_label

    final_filters = []
    if watermark_filter:
        final_filters.append(watermark_filter)
    final_filters.append(f"tpad=stop_mode=clone:stop_duration={CROSSFADE_SECONDS}")

    if final_filters:
        filters.append(f"{current_label}{','.join(final_filters)}[vout]")
        current_label = "[vout]"

    filter_complex = ";".join(filters)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", current_label,
        "-map", "1:a",
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


def _build_title_card(
    lines: list,
    out_path: str,
    duration: float = 3.5,
    width: int = 1920,
    height: int = 1080,
    fps: int = SCENE_FPS,
    bg_color: str = "0x141E30",
) -> str:
    png_path = out_path + ".title.png"
    _render_title_card_png(lines, width, height, png_path)

    fade_out_start = max(0.0, duration - 0.6)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:r={fps}:d={duration}",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-loop", "1", "-i", png_path,
        "-filter_complex", f"[0:v][2:v]overlay=0:0,fade=t=in:st=0:d=0.5,fade=t=out:st={fade_out_start}:d=0.6",
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
    if not name:
        return "final_video.mp4"

    name = str(name).strip()
    if not name:
        return "final_video.mp4"

    illegal_chars = "[<>:\"/\\\\|?*!@#$%^&()\\[\\]{}+=`~'\\n\\r\\t]"
    safe = re.sub(illegal_chars, "-", name)
    safe = re.sub(r"[ \s_]+", "-", safe)
    safe = re.sub(r"-+", "-", safe)
    safe = safe.strip("-")

    if len(safe) > max_length:
        safe = safe[:max_length].rsplit("-", 1)[0]

    if not safe:
        return "final_video.mp4"

    if not safe.lower().endswith(".mp4"):
        safe += ".mp4"

    return safe


def assemble_video(
    scenes: list,
    work_dir: str,
    output_name: str = None,
    topic: str = None,
    title: str = None,
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
        print(f"[video_assembler] music_path '{music_path}' not found -- skipping background music.")

    if progress_callback:
        progress_callback("join", 1, 1)

    return final_path
