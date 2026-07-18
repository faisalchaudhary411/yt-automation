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

# Optional: pure-Python Arabic shaping & BiDi reordering (no libraqm needed)
try:
    import arabic_reshaper
    _HAS_ARABIC_RESHAPER = True
except ImportError:
    _HAS_ARABIC_RESHAPER = False

# ---------------------------------------------------------------------------
# Script-marker stripping (PAUSE / EMPHASIS / B-ROLL directives)
# ---------------------------------------------------------------------------
# script_generator.py deliberately embeds "(PAUSE)", "(EMPHASIS)", and
# "[B-ROLL: description]" in narration text -- these are directorial notes for
# pacing/editing, not words meant to be shown to the viewer. Without
# stripping them here too, on-screen captions would literally display
# "(PAUSE)" and "[B-ROLL: old newspaper archive footage]" as caption text,
# out of sync with the audio (which strips these in voice_generator.py).
# Duplicated locally rather than imported from voice_generator.py so this
# module keeps working standalone regardless of how it's deployed.

_BROLL_MARKER_RE = re.compile(r"[\[\(]\s*B[\s\-]?ROLL\b[^\]\)]*[\]\)]", re.IGNORECASE | re.DOTALL)
_BROLL_BARE_MARKER_RE = re.compile(r"\bB[\s\-]?ROLL\b\s*:?\s*[^۔.!?\n]*", re.IGNORECASE)
_PAUSE_MARKER_RE = re.compile(r"\(\s*PAUSE\s*\)", re.IGNORECASE)
_EMPHASIS_MARKER_RE = re.compile(r"\(\s*EMPHASIS\s*\)", re.IGNORECASE)


def _strip_narration_markers_for_captions(text: str) -> str:
    """Removes script-writing directives before text is shown as an on-screen
    caption. Unlike the TTS-side version, this does NOT spell out %/$ in
    words -- captions should keep the literal symbols for reading."""
    if not text:
        return text

    text = _BROLL_MARKER_RE.sub(" ", text)
    text = _BROLL_BARE_MARKER_RE.sub(" ", text)
    # For captions (unlike audio) there's no need for an actual pause -- just
    # drop the marker rather than inserting a visible comma.
    text = _PAUSE_MARKER_RE.sub(" ", text)
    text = _EMPHASIS_MARKER_RE.sub(" ", text)

    text = re.sub(r"[,،]{2,}", "،", text)
    text = re.sub(r"^[,،\s]+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Lower-third stat extraction (Phase 5 polish)
# ---------------------------------------------------------------------------
# Purely rule-based (no LLM call): looks for the kind of concrete number a
# documentary would want to stamp on screen -- a currency amount, a
# percentage, a year, or an "N-year(s)" span. Digits/currency symbols read
# fine regardless of the narration's language, so this works even inside
# Urdu narration with embedded Latin numerals. Only the FIRST match (by the
# priority order below) is used per scene, so a scene never gets more than
# one stamp even if it mentions several numbers.

_STAT_CURRENCY_RE = re.compile(
    r"[£$€]\s?\d[\d,]*(?:\.\d+)?\s?(?:thousand|million|billion|trillion|k|m|bn)?",
    re.IGNORECASE,
)
_STAT_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
_STAT_YEAR_SPAN_RE = re.compile(r"\b\d+[\s-]?years?[\s-]?old\b", re.IGNORECASE)
_STAT_BIG_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_STAT_MULTIPLIER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:thousand|million|billion|trillion)\b", re.IGNORECASE
)
_STAT_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

_STAT_PATTERNS = [
    _STAT_CURRENCY_RE, _STAT_PERCENT_RE, _STAT_YEAR_SPAN_RE,
    _STAT_BIG_NUMBER_RE, _STAT_MULTIPLIER_RE, _STAT_YEAR_RE,
]


def _extract_stat(text: str) -> str:
    """Returns the first documentary-worthy stat found in `text`, or "" if
    none. `text` should already have B-ROLL/PAUSE/EMPHASIS markers stripped."""
    if not text:
        return ""
    for pattern in _STAT_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0).strip()
    return ""

try:
    from bidi.algorithm import get_display
    _HAS_BIDI = True
except ImportError:
    _HAS_BIDI = False

# ---------------------------------------------------------------------------
# HARD-CODED FONT PATHS — points to content_pipeline/fonts/
# ---------------------------------------------------------------------------

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# NOTE ON FONT CHOICE: NotoNastaliqUrdu.ttf is kept here and still tried
# first, but it will almost always FAIL verification and be skipped --
# arabic_reshaper converts text into legacy "Arabic Presentation Forms"
# codepoints (so this file can shape text WITHOUT a full OpenType shaping
# engine, working around a HarfBuzz/libass regression in Replit's ffmpeg
# build). Nastaliq fonts are built for real OpenType shaping and typically
# have ZERO cmap entries for those legacy codepoints -- confirmed directly:
# 30 of 37 characters in a real test sentence came back with no glyph at
# all, i.e. tofu boxes, regardless of how valid the file itself is. This
# isn't a font-file problem, it's a fundamental mismatch between this
# rendering approach and how Nastaliq fonts are built. NotoSansArabic.ttf
# (Naskh-style) is added below specifically because it DOES have full
# presentation-forms coverage and was confirmed to render correctly with
# this exact pipeline. If proper raqm/HarfBuzz shaping becomes available on
# Replit later, Nastaliq can go back to working with no code change needed
# here -- it's left in the candidate list rather than deleted.
URDU_FONT_PATH = os.path.join(FONTS_DIR, "NotoNastaliqUrdu.ttf")
NASKH_FONT_PATH = os.path.join(FONTS_DIR, "NotoSansArabic.ttf")
LATIN_FONT_PATH = os.path.join(FONTS_DIR, "DejaVuSans.ttf")

# Fallback fonts in order of preference (system-wide, no bundling needed).
# Naskh/Sans Arabic listed before Nastaliq since they're the ones that
# actually work with this file's current (non-raqm) rendering approach --
# see the note on URDU_FONT_PATH above.
FALLBACK_FONT_NAMES = [
    "Noto Naskh Arabic",
    "Noto Sans Arabic",
    "Noto Nastaliq Urdu",
    "FreeSerif",
    "DejaVu Sans",
]
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


def _find_font_in_dirs(filenames, dirs):
    """Walk common font directories looking for a matching font file."""
    for d in dirs:
        if not os.path.isdir(d):
            continue
        try:
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf", ".ttc")):
                        for target in filenames:
                            if target.lower().replace(" ", "") in f.lower().replace(" ", ""):
                                path = os.path.join(root, f)
                                if not _validate_font_file(path):
                                    return path
        except Exception:
            continue
    return ""


def _has_real_glyph_coverage(font_path: str, sample_text: str) -> bool:
    """Checks whether EVERY non-whitespace character in sample_text has a
    REAL, drawable glyph in the font at font_path -- not just any cmap
    entry, and not a missing/.notdef placeholder.

    This replaces an earlier version of this check that rendered the sample
    with Pillow and looked at whether ANY ink came out. That approach was
    fundamentally broken: a MISSING glyph still renders something (a visible
    tofu-box placeholder) when drawn, so a render-based test can never tell
    'real glyph' apart from 'tofu box' -- confirmed directly: a font with
    zero real Arabic support still produced non-empty rendered pixels for
    Arabic text. This version instead inspects the font's actual character
    map and glyph outline data via fontTools, which can tell the difference.

    IMPORTANT: sample_text must be the ACTUAL text Pillow will be asked to
    draw at render time. For Arabic/Urdu that means text already run through
    _prepare_text_for_rendering (reshaping + BiDi reordering) -- reshaping
    converts text into Arabic Presentation Forms codepoints that some fonts
    (e.g. Noto Nastaliq Urdu) have ZERO cmap entries for, even though they
    render the unshaped base-Arabic-block text perfectly fine. Testing with
    unshaped text would pass fonts that then fail at actual render time.
    """
    try:
        from fontTools.ttLib import TTFont
        from fontTools.pens.boundsPen import BoundsPen
    except ImportError:
        # fontTools isn't installed -- can't do a reliable check here.
        # _load_font_for_rendering will still catch outright load failures,
        # so this isn't a silent failure mode, just a less precise one.
        print("[font_resolve] fontTools not installed -- cannot verify real "
              "glyph coverage (pip install fonttools for a reliable check). "
              "Assuming this candidate is OK; it may still tofu at render time.")
        return True

    try:
        tt = TTFont(font_path, lazy=True, fontNumber=0)
        cmap = tt.getBestCmap()
        glyph_set = tt.getGlyphSet()
    except Exception:
        return False

    for ch in sample_text:
        if not ch.strip():
            continue
        cp = ord(ch)
        if cp not in cmap:
            return False
        try:
            pen = BoundsPen(glyph_set)
            glyph_set[cmap[cp]].draw(pen)
        except Exception:
            return False
        if pen.bounds is None:
            return False

    return True


# A real Urdu phrase (base Arabic-block characters) used to verify font
# candidates. Deliberately includes a range of common letters.
_URDU_VERIFICATION_PHRASE = "اردو ہے جس کا نام ہم جانتے ہیں داغ"


# ---------------------------------------------------------------------------
# Unicode blocks covering Arabic/Urdu script (including presentation forms used by
# some fonts/renderers for joined letterforms)
# ---------------------------------------------------------------------------

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


def _prepare_text_for_rendering(text: str, is_latin: bool) -> str:
    """
    For RTL scripts (Urdu/Arabic), reshape letters and apply BiDi reordering
    so Pillow can render them correctly WITHOUT needing libraqm.
    Falls back to raw text if arabic-reshaper / python-bidi are not installed.
    """
    if is_latin or not text:
        return text

    # If we have the optional libraries, reshape + reorder
    if _HAS_ARABIC_RESHAPER and _HAS_BIDI:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)

    # Otherwise just return raw text (Pillow will render LTR; not ideal but won't crash)
    print("[WARN] arabic-reshaper and/or python-bidi not installed. "
          "Urdu/Arabic text will render unjoined and left-to-right. "
          "Install: pip install arabic-reshaper python-bidi")
    return text


# ---------------------------------------------------------------------------
# Pillow-based text rendering (replaces libass/ASS entirely)
# ---------------------------------------------------------------------------

# Cache verified font paths so we don't re-run fc-list/font-loading/glyph
# tests on every single caption page (this gets called many times per video).
_VERIFIED_FONT_CACHE = {}


def _candidate_font_paths(for_latin: bool) -> list:
    """Builds an ordered list of font paths worth trying: the bundled
    file(s) first, then every system font matching any of the fallback
    family names (via fc-list), in preference order. Does NOT validate any
    of them yet -- that happens in _resolve_font_path so each candidate can
    be tested for real glyph coverage before being trusted."""
    candidates = []

    if for_latin:
        bundled_paths = [LATIN_FONT_PATH]
    else:
        # NotoSansArabic.ttf (Naskh-style) listed first since it's the one
        # confirmed to actually work with this file's reshaping approach.
        # NotoNastaliqUrdu.ttf is still included -- it'll almost certainly
        # fail verification and be skipped, but costs nothing to try, and
        # means Nastaliq starts working automatically for free if raqm
        # shaping is ever added later.
        bundled_paths = [NASKH_FONT_PATH, URDU_FONT_PATH]

    for bundled in bundled_paths:
        if os.path.isfile(bundled):
            candidates.append(bundled)

    names = LATIN_FALLBACK_FONTS if for_latin else FALLBACK_FONT_NAMES
    for name in names:
        try:
            # NOTE: the element argument must be "file", NOT ":file". A
            # leading colon there is invalid fc-list syntax and silently
            # returns empty output no matter what fonts are installed --
            # this was quietly disabling the ENTIRE system-font fallback
            # search, regardless of what was actually on the system.
            r = subprocess.run(
                ["fc-list", name, "file"],
                capture_output=True, text=True, timeout=10
            )
            for line in r.stdout.strip().splitlines():
                path = line.split(":")[0].strip()
                if path and path not in candidates:
                    candidates.append(path)
        except Exception:
            continue

    return candidates


def _resolve_font_path(for_latin: bool = False) -> str:
    """Returns a font path that has actually been verified to (a) load
    successfully with Pillow, and (b) for Urdu/Arabic, actually contain
    REAL (non-placeholder) glyphs for the exact reshaped/reordered text this
    file sends to Pillow at render time -- not merely a path that happens to
    exist on disk, and not merely "renders SOME ink for raw Urdu letters"
    (a font with zero real Arabic support still renders a visible tofu-box
    placeholder, which is non-empty -- rendering-based checks can't tell the
    difference; fontTools cmap/outline inspection can).
    """
    cache_key = for_latin
    if cache_key in _VERIFIED_FONT_CACHE:
        return _VERIFIED_FONT_CACHE[cache_key]

    verification_sample = (
        "Sample Text 123" if for_latin
        else _prepare_text_for_rendering(_URDU_VERIFICATION_PHRASE, is_latin=False)
    )

    tried = []
    for path in _candidate_font_paths(for_latin):
        problem = _validate_font_file(path)
        if problem:
            tried.append(f"{path} (rejected: {problem})")
            continue
        try:
            ImageFont.truetype(path, 40)
        except Exception as e:
            tried.append(f"{path} (rejected: Pillow could not load it -- {e})")
            continue

        if for_latin or _has_real_glyph_coverage(path, verification_sample):
            _VERIFIED_FONT_CACHE[cache_key] = path
            print(f"[font_resolve:{'Latin' if for_latin else 'Urdu/Arabic'}] "
                  f"using verified font: {path}")
            return path
        else:
            tried.append(f"{path} (rejected: loads fine, but missing real glyphs for "
                         f"the actual shaped/reordered text used at render time -- "
                         f"likely needs raqm/HarfBuzz shaping, or lacks Arabic "
                         f"Presentation Forms compatibility glyphs)")

    label = "Latin" if for_latin else "Urdu/Arabic"
    print(f"[font_resolve:{label}] NO valid {label} font found. Candidates tried:")
    for t in tried:
        print(f"    - {t}")
    if not tried:
        print(f"    - (no candidates found at all -- bundled path(s) don't exist, "
              f"and fc-list found no system matches for "
              f"{LATIN_FALLBACK_FONTS if for_latin else FALLBACK_FONT_NAMES})")

    _VERIFIED_FONT_CACHE[cache_key] = ""
    return ""


def _load_font_for_rendering(font_path: str, fontsize: int, for_latin: bool):
    """Load a font at the given (already-verified) path and size.

    font_path is expected to come from _resolve_font_path, which only ever
    returns paths that have already passed real glyph-coverage verification.
    If font_path is empty, NO fallback silently substitutes a broken font --
    this raises a loud, actionable error instead, because a silent fallback
    here is exactly what caused tofu boxes to keep showing up invisibly in
    Railway logs nobody was watching.
    """
    if not font_path:
        label = "Latin" if for_latin else "Urdu/Arabic"
        bundled = LATIN_FONT_PATH if for_latin else URDU_FONT_PATH
        raise RuntimeError(
            f"No verified {label} font is available -- refusing to render with a "
            f"guessed/broken font, since that's what was silently producing tofu boxes "
            f"before. Checked bundled path:\n"
            f"    {bundled}\n"
            f"and system fonts matching: {LATIN_FALLBACK_FONTS if for_latin else FALLBACK_FONT_NAMES}\n\n"
            f"To fix this:\n"
            f"  1. On Railway, open a shell and run:\n"
            f"       ls -la {os.path.dirname(bundled)}\n"
            f"     Confirm the file is really there and its size is in the hundreds of KB "
            f"(a few bytes or ~130 bytes usually means it's a Git LFS pointer, not the real font).\n"
            f"  2. If it's missing or tiny, re-upload it to GitHub using the 'Add file > Upload "
            f"files' button (not pasting/editing as text) -- editing a binary .ttf as text in "
            f"GitHub's mobile web editor corrupts it.\n"
            f"  3. As a more reliable belt-and-suspenders option, install the font system-wide "
            f"instead of relying on the repo copy: add 'fonts-noto-core' (or 'fonts-noto-ttf') "
            f"as an apt package in your Railway build (nixpacks.toml / apt.txt), which installs "
            f"a real Noto Nastaliq Urdu / Noto Naskh Arabic font regardless of repo file state."
        )

    try:
        font = ImageFont.truetype(font_path, fontsize)
    except Exception as e:
        raise RuntimeError(
            f"Font at '{font_path}' passed verification earlier but failed to load now "
            f"({e}). This shouldn't normally happen -- the file may have changed between "
            f"checks."
        ) from e

    return font


def log_font_diagnostics():
    """Prints exactly what font will be used for Latin and Urdu/Arabic text,
    or exactly why none could be found. Safe to call anytime (never raises).
    Runs automatically once at module import so this always shows up in
    Railway logs without needing any extra steps."""
    for for_latin in (True, False):
        label = "Latin" if for_latin else "Urdu/Arabic"
        try:
            path = _resolve_font_path(for_latin=for_latin)
            if not path:
                print(f"[font_diagnostics] {label}: NO valid font found -- "
                      f"captions/titles in this script will FAIL LOUDLY (not silently "
                      f"show tofu) until this is fixed. See messages above for exact reasons.")
        except Exception as e:
            print(f"[font_diagnostics] {label}: diagnostic check itself failed: {e}")


log_font_diagnostics()


def _render_text_page(lines, width, height, fontsize, bottom_margin, out_path, for_latin=False):
    """Renders a page of caption lines to a full-size transparent PNG."""
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required. Install it: pip install pillow")

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_path = _resolve_font_path(for_latin=for_latin)
    font = _load_font_for_rendering(font_path, fontsize, for_latin)

    line_spacing = int(fontsize * 0.35)
    line_height = fontsize + line_spacing

    # Prepare lines for rendering (reshape/reorder RTL if needed)
    prepared_lines = [_prepare_text_for_rendering(line, for_latin) for line in lines]

    measured = []
    for line in prepared_lines:
        bb = draw.textbbox((0, 0), line, font=font)
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
            draw.text((x + dx, y + dy), m["text"], font=font, fill=(0, 0, 0, 160))
        draw.text((x, y), m["text"], font=font, fill=(255, 255, 255, 255))

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


def _render_title_card_png(
    lines, width, height, out_path, title_fontsize=92, subtitle_fontsize=46,
    accent_color: tuple = (198, 164, 84),
):
    """Renders a title card to a full-size transparent PNG. `accent_color` draws
    a short bar under the title -- gives each video style (see VIDEO_STYLES) its
    own on-screen color identity instead of every style's cards looking alike."""
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required. Install it: pip install pillow")

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    title_text = lines[0] if lines else ""
    is_title_latin = _is_latin_text(title_text)
    title_path = _resolve_font_path(for_latin=is_title_latin)
    title_font = _load_font_for_rendering(title_path, title_fontsize, is_title_latin)
    title_lines = _wrap_text_lines(title_text, width, title_fontsize)[:3]

    subtitle_text = lines[1] if len(lines) > 1 else ""
    is_sub_latin = _is_latin_text(subtitle_text) if subtitle_text else True
    sub_path = _resolve_font_path(for_latin=is_sub_latin)
    sub_font = _load_font_for_rendering(sub_path, subtitle_fontsize, is_sub_latin)
    subtitle_lines = _wrap_text_lines(subtitle_text, width, subtitle_fontsize)[:2] if subtitle_text else []

    # Prepare text for rendering
    title_lines = [_prepare_text_for_rendering(line, is_title_latin) for line in title_lines]
    subtitle_lines = [_prepare_text_for_rendering(line, is_sub_latin) for line in subtitle_lines]

    bar_h = 6
    bar_gap = 22  # space above and below the accent bar
    title_line_h = title_fontsize + 20
    sub_line_h = subtitle_fontsize + 14
    show_bar = bool(title_lines)  # every card with a title gets the style's accent bar
    bar_block_h = (bar_gap * 2 + bar_h) if show_bar else 0
    total_h = len(title_lines) * title_line_h + bar_block_h + len(subtitle_lines) * sub_line_h
    y_cursor = (height - total_h) // 2

    for line in title_lines:
        bb = draw.textbbox((0, 0), line, font=title_font)
        text_w = bb[2] - bb[0]
        x = (width - text_w) // 2 - bb[0]
        y = y_cursor - bb[1]
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            draw.text((x + dx, y + dy), line, font=title_font, fill=(0, 0, 0, 180))
        draw.text((x, y), line, font=title_font, fill=(255, 255, 255, 255))
        y_cursor += title_line_h

    if show_bar:
        bar_y = y_cursor + bar_gap
        draw.rectangle(
            [width // 2 - 60, bar_y, width // 2 + 60, bar_y + bar_h],
            fill=(*accent_color, 255),
        )
        y_cursor += bar_block_h

    if subtitle_lines:
        for line in subtitle_lines:
            bb = draw.textbbox((0, 0), line, font=sub_font)
            text_w = bb[2] - bb[0]
            x = (width - text_w) // 2 - bb[0]
            y = y_cursor - bb[1]
            for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                draw.text((x + dx, y + dy), line, font=sub_font, fill=(0, 0, 0, 160))
            draw.text((x, y), line, font=sub_font, fill=(255, 255, 255, 255))
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

# Mid-video named chapter title cards (e.g. "The Warning Signs"). Shorter than
# the intro/outro cards since they're a brief beat, not a full pause.
# automation/subtitles.py mirrors this constant to keep .srt timing in sync.
CHAPTER_CARD_DURATION = 2.2


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


def _render_lower_third_png(
    stat_text: str, width: int, height: int, out_path: str,
    accent_color: tuple = (198, 164, 84), fontsize: int = 54,
) -> str:
    """Renders a brief stat-callout graphic (a year, currency amount, a
    percentage) as a transparent PNG: a dark box with a colored accent
    stripe on its left edge. Positioned bottom-LEFT specifically so it never
    collides with the bottom-CENTER captions or the bottom-RIGHT channel
    watermark (see _watermark_filter)."""
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required. Install it: pip install pillow")

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    is_latin = _is_latin_text(stat_text)
    font_path = _resolve_font_path(for_latin=is_latin)
    font = _load_font_for_rendering(font_path, fontsize, is_latin)
    text = _prepare_text_for_rendering(stat_text, is_latin)

    bb = draw.textbbox((0, 0), text, font=font)
    text_w = bb[2] - bb[0]
    text_h = bb[3] - bb[1]

    pad_x, pad_y = 28, 18
    stripe_w = 6
    box_w = text_w + pad_x * 2 + stripe_w
    box_h = text_h + pad_y * 2

    box_x0 = 60
    box_y0 = height - 210 - box_h  # well above the caption band near the bottom
    box_x1 = box_x0 + box_w
    box_y1 = box_y0 + box_h

    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(10, 12, 18, 200))
    draw.rectangle([box_x0, box_y0, box_x0 + stripe_w, box_y1], fill=(*accent_color, 255))

    text_x = box_x0 + stripe_w + pad_x - bb[0]
    text_y = box_y0 + pad_y - bb[1]
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))

    img.save(out_path)
    return out_path


def _build_scene_clip(
    scene: dict, index: int, work_dir: str, width=1920, height=1080, zoom_rate=0.0008,
    channel_name: str = None, accent_color: tuple = (198, 164, 84),
) -> str:
    clip_dir = os.path.join(work_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    out_path = os.path.join(clip_dir, f"clip_{index:03d}.mp4")

    if not scene.get("image_path") or not scene.get("audio_path"):
        raise RuntimeError(f"Scene {index} is missing an image or audio file.")

    media_type = scene.get("media_type", "photo")
    duration = _get_media_duration(scene["audio_path"])
    fps = SCENE_FPS
    total_frames = int(duration * fps)

    fontsize = 40
    narration_text = _strip_narration_markers_for_captions(scene.get("narration", ""))
    is_latin = _is_latin_text(narration_text)

    caption_pngs = _write_caption_pngs(
        narration_text, width, height, fontsize, bottom_margin=90,
        duration=duration, out_dir=clip_dir, prefix=f"scene_{index:03d}", for_latin=is_latin,
    )

    from config import LOWER_THIRDS_ENABLED
    lower_third_info = None
    if LOWER_THIRDS_ENABLED and duration >= 2.5:
        stat_text = _extract_stat(narration_text)
        if stat_text:
            lt_start = 0.3
            lt_end = min(duration - 0.2, lt_start + 3.2)
            if lt_end > lt_start:
                lt_path = os.path.join(clip_dir, f"scene_{index:03d}_stat.png")
                _render_lower_third_png(stat_text, width, height, lt_path, accent_color=accent_color)
                lower_third_info = {"path": lt_path, "start": lt_start, "end": lt_end}

    watermark_filter = _watermark_filter(channel_name)

    if media_type == "video":
        # Real motion b-roll: loop the source clip indefinitely (-stream_loop
        # -1) and let the final -t duration cut it to exactly the length we
        # need, whether the source is shorter OR longer than that -- no need
        # to probe its length first. It already has motion, so no Ken Burns
        # zoompan here; just fill the frame without distorting the footage
        # (force_original_aspect_ratio=increase + crop, rather than a
        # stretch-to-fit scale) and normalize to our fixed output fps.
        cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", scene["image_path"], "-i", scene["audio_path"]]
        base_filters = [
            f"scale={width}:{height}:force_original_aspect_ratio=increase",
            f"crop={width}:{height}",
            f"fps={fps}",
            COLOR_GRADE_FILTER,
        ]
    else:
        cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene["image_path"], "-i", scene["audio_path"]]
        base_filters = [
            f"scale={width}:{height}",
            f"zoompan={_zoompan_expr(index, zoom_rate)}:d={total_frames}:s={width}x{height}:fps={fps}",
            COLOR_GRADE_FILTER,
        ]

    for info in caption_pngs:
        cmd += ["-loop", "1", "-i", info["path"]]
    if lower_third_info:
        cmd += ["-loop", "1", "-i", lower_third_info["path"]]

    filters = [f"[0:v]{','.join(base_filters)}[base]"]

    current_label = "[base]"
    for i, info in enumerate(caption_pngs):
        input_idx = 2 + i
        is_last_overlay = (i == len(caption_pngs) - 1) and not lower_third_info
        out_label = f"[cap{i}]" if not is_last_overlay else "[vcap]"
        start = info["start"]
        end = info["end"]
        filters.append(
            f"{current_label}[{input_idx}:v]overlay=0:0:enable='between(t\\,{start:.3f}\\,{end:.3f})'{out_label}"
        )
        current_label = out_label

    if lower_third_info:
        lt_input_idx = 2 + len(caption_pngs)
        start = lower_third_info["start"]
        end = lower_third_info["end"]
        filters.append(
            f"{current_label}[{lt_input_idx}:v]overlay=0:0:enable='between(t\\,{start:.3f}\\,{end:.3f})'[vstat]"
        )
        current_label = "[vstat]"

    # NOTE: we deliberately do NOT pad the clip's tail with tpad/apad here.
    # xfade/acrossfade (used later when crossfading consecutive clips) blend
    # the existing tail of one clip with the existing head of the next -- they
    # don't need extra padding frames to work with. Adding a frozen-frame +
    # silent-audio tail here used to cause a visible stutter/freeze on every
    # scene transition that was NOT crossfaded (i.e. every hard-cut in the
    # middle of the video, since only the first and last transitions actually
    # get crossfaded).
    final_filters = []
    if watermark_filter:
        final_filters.append(watermark_filter)

    if final_filters:
        filters.append(f"{current_label}{','.join(final_filters)}[vout]")
        current_label = "[vout]"

    filter_complex = ";".join(filters)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", current_label,
        "-map", "1:a",
        "-af", "dynaudnorm=f=150:g=15",
        "-c:v", "libx264", "-preset", INTERMEDIATE_PRESET, "-crf", SCENE_CRF,
        "-threads", str(THREADS_PER_CLIP),
        "-t", str(duration), "-pix_fmt", "yuv420p",
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
    accent_color: tuple = (198, 164, 84),
) -> str:
    png_path = out_path + ".title.png"
    _render_title_card_png(lines, width, height, png_path, accent_color=accent_color)

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


def _render_logo_sting_png(
    logo_path: str, width: int, height: int, out_path: str,
    accent_color: tuple = (198, 164, 84), channel_name: str = None,
) -> str:
    """Composites the logo + a colored accent bar + the channel name into one
    PNG (same technique as the title cards), scaling the logo to fit within
    ~34% of the frame while preserving its aspect ratio. This composited
    frame then gets a gentle zoom-in via ffmpeg (see _build_logo_sting) --
    the same Ken Burns technique already used for photo scenes, applied here
    at a much subtler rate so a plain static logo doesn't feel like a dead
    frame."""
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required. Install it: pip install pillow")

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    logo = Image.open(logo_path).convert("RGBA")
    max_w, max_h = int(width * 0.34), int(height * 0.34)
    scale = min(max_w / logo.width, max_h / logo.height, 1.0)
    logo = logo.resize((max(1, round(logo.width * scale)), max(1, round(logo.height * scale))), Image.LANCZOS)

    logo_x = (width - logo.width) // 2
    logo_y = (height - logo.height) // 2 - (30 if channel_name else 0)
    canvas.paste(logo, (logo_x, logo_y), logo)

    draw = ImageDraw.Draw(canvas)
    bar_y = logo_y + logo.height + 26
    draw.rectangle([width // 2 - 50, bar_y, width // 2 + 50, bar_y + 5], fill=(*accent_color, 255))

    if channel_name:
        font_path = _resolve_font_path(for_latin=True)
        if font_path:
            font = ImageFont.truetype(font_path, 34)
            text = channel_name.upper()
            bb = draw.textbbox((0, 0), text, font=font)
            text_w = bb[2] - bb[0]
            tx = (width - text_w) // 2 - bb[0]
            ty = bar_y + 22 - bb[1]
            for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                draw.text((tx + dx, ty + dy), text, font=font, fill=(0, 0, 0, 160))
            draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))

    canvas.save(out_path)
    return out_path


def _build_logo_sting(
    logo_path: str, out_path: str, duration: float, bg_color: str,
    accent_color: tuple = (198, 164, 84), channel_name: str = None,
    width: int = 1920, height: int = 1080, fps: int = SCENE_FPS,
) -> str:
    """A brief (~1-1.5s) branded opener shown once before the intro title
    card: the composited logo/accent-bar/channel-name frame from
    _render_logo_sting_png, given a gentle zoom-in (same Ken Burns technique
    as photo scenes, at a subtler rate) and the same vignette/color-grade
    treatment as everything else, rather than sitting as a flat static card."""
    clip_dir = os.path.dirname(out_path)
    png_path = out_path + ".sting.png"
    _render_logo_sting_png(logo_path, width, height, png_path, accent_color=accent_color, channel_name=channel_name)

    fade_out_start = max(0.0, duration - 0.4)
    total_frames = int(duration * fps)
    logo_zoom_rate = 0.0006  # gentle -- this is a brief brand beat, not a Ken Burns scene

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", png_path,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-filter_complex",
        f"color=c={bg_color}:s={width}x{height}:r={fps}[bg];"
        f"[bg][0:v]overlay=(W-w)/2:(H-h)/2:format=auto,"
        f"zoompan={_zoompan_expr(0, logo_zoom_rate)}:d={total_frames}:s={width}x{height}:fps={fps},"
        f"{COLOR_GRADE_FILTER},"
        f"fade=t=in:st=0:d=0.3,fade=t=out:st={fade_out_start}:d=0.4",
        "-c:v", "libx264", "-preset", INTERMEDIATE_PRESET, "-crf", SCENE_CRF,
        "-t", str(duration), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", "-shortest",
        out_path,
    ]
    _run(cmd, "Logo sting render")
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
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        _run(cmd, "Hard-cut concat")
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
    return output_path


def _remux_single(path: str, final_path: str) -> str:
    """Copies a single already-encoded clip straight to final_path, adding
    +faststart so the deliverable is web-playback-friendly."""
    if os.path.abspath(path) == os.path.abspath(final_path):
        return final_path
    cmd = ["ffmpeg", "-y", "-i", path, "-c", "copy", "-movflags", "+faststart", final_path]
    _run(cmd, "Final remux")
    return final_path


def get_chapter_card_scene_indices(chapters: list, total_scenes: int) -> dict:
    """Turns generate_script()'s `chapters` list into {scene_index: title} for
    scenes that should get a chapter title card immediately before them.

    Scene index 0 is deliberately excluded — a chapter card right after the
    intro card would be redundant. Both video_assembler (real ffmpeg join)
    and automation/subtitles.py (timestamp math) call this so they can never
    disagree about where the cards go.
    """
    result = {}
    for c in chapters or []:
        i = c.get("scene_index")
        title = (c.get("title") or "").strip()
        if title and isinstance(i, int) and 0 < i < total_scenes and i not in result:
            result[i] = title
    return result


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
    chapters: list = None,
) -> str:
    from config import (
        VIDEO_STYLES, DEFAULT_VIDEO_STYLE, LOGO_STING_ENABLED, CHANNEL_LOGO_PATH, LOGO_STING_DURATION,
    )
    style_conf = VIDEO_STYLES.get(style, VIDEO_STYLES[DEFAULT_VIDEO_STYLE])
    bg_color = style_conf["bg_color"]
    zoom_rate = style_conf["zoom_rate"]
    accent_color = style_conf.get("accent_color", (198, 164, 84))
    crossfade_seconds = style_conf.get("crossfade_seconds", CROSSFADE_SECONDS)

    clip_dir = os.path.join(work_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)

    logo_sting_path = None
    intro_path = None
    outro_path = None

    if include_intro and LOGO_STING_ENABLED and CHANNEL_LOGO_PATH and os.path.isfile(CHANNEL_LOGO_PATH):
        logo_sting_path = _build_logo_sting(
            CHANNEL_LOGO_PATH, os.path.join(clip_dir, "logo_sting.mp4"),
            duration=LOGO_STING_DURATION, bg_color=bg_color,
            accent_color=accent_color, channel_name=channel_name,
        )

    if include_intro:
        subtitle = f"A {channel_name} Story" if channel_name else "A Documentary Story"
        intro_path = _build_title_card(
            [title or "", subtitle],
            os.path.join(clip_dir, "intro.mp4"),
            duration=3.5,
            bg_color=bg_color, accent_color=accent_color,
        )

    total_scenes = len(scenes)
    scene_clip_paths = [None] * total_scenes
    clips_done = 0

    if progress_callback:
        progress_callback("clips", 0, total_scenes)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CLIPS) as executor:
        futures = {
            executor.submit(_build_scene_clip, scene, i, work_dir, zoom_rate=zoom_rate, channel_name=channel_name, accent_color=accent_color): i
            for i, scene in enumerate(scenes)
        }
        for future in as_completed(futures):
            i = futures[future]
            scene_clip_paths[i] = future.result()
            clips_done += 1
            if progress_callback:
                progress_callback("clips", clips_done, total_scenes)

    if include_outro:
        subtitle = f"Subscribe to {channel_name} for more" if channel_name else "Subscribe for more stories like this"
        outro_path = _build_title_card(
            ["Thanks for watching!", subtitle],
            os.path.join(clip_dir, "outro.mp4"),
            duration=4.0,
            bg_color=bg_color, accent_color=accent_color,
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

    # Join the scene clips among themselves first. Title cards already fade
    # in/out on their own (see _build_title_card), so we deliberately do NOT
    # crossfade a title card against the adjacent scene -- that would produce
    # a double-fade (fade to bg color, then crossfade through it again).
    # Instead, scene-to-scene crossfading only happens at the edges that
    # AREN'T already covered by an intro/outro/chapter title card.
    #
    # Named chapters (from generate_script()'s `chapters` list) split the
    # scenes into groups, with a chapter title card hard-cut in between each
    # group -- the same "no double-fade against a card" rule as intro/outro,
    # just applied at however many internal points the script calls for.
    card_by_index = get_chapter_card_scene_indices(chapters, total_scenes)

    if total_scenes == 0:
        body_pieces = []
    else:
        groups = []
        current_group = []
        for i, clip_path in enumerate(scene_clip_paths):
            if i in card_by_index and current_group:
                groups.append(("scenes", current_group))
                current_group = []
                card_path = os.path.join(clip_dir, f"chapter_{i}.mp4")
                _build_title_card(
                    [card_by_index[i]], card_path,
                    duration=CHAPTER_CARD_DURATION, bg_color=bg_color, accent_color=accent_color,
                )
                groups.append(("card", card_path))
            current_group.append(clip_path)
        if current_group:
            groups.append(("scenes", current_group))

        scene_group_positions = [idx for idx, (kind, _) in enumerate(groups) if kind == "scenes"]
        first_scene_group = scene_group_positions[0] if scene_group_positions else None
        last_scene_group = scene_group_positions[-1] if scene_group_positions else None

        body_pieces = []
        for idx, (kind, payload) in enumerate(groups):
            if kind == "card":
                body_pieces.append(payload)
                continue
            fade_start = (not include_intro) if idx == first_scene_group else False
            fade_end = (not include_outro) if idx == last_scene_group else False
            if len(payload) == 1:
                body_pieces.append(payload[0])
            else:
                joined_sub_path = os.path.join(clip_dir, f"_group_{idx}.mp4")
                _join_mixed_transitions(
                    payload, joined_sub_path, fade_at_start=fade_start, fade_at_end=fade_end,
                    crossfade_seconds=crossfade_seconds,
                )
                body_pieces.append(joined_sub_path)

    outer_pieces = []
    if logo_sting_path:
        outer_pieces.append(logo_sting_path)
    if intro_path:
        outer_pieces.append(intro_path)
    outer_pieces.extend(body_pieces)
    if outro_path:
        outer_pieces.append(outro_path)

    if not outer_pieces:
        raise RuntimeError("No clips to assemble into a final video.")
    elif len(outer_pieces) == 1:
        _remux_single(outer_pieces[0], final_path)
    else:
        _concat_stream_copy(outer_pieces, final_path)

    if music_path and os.path.isfile(music_path):
        music_mixed_path = os.path.join(work_dir, "final_with_music.mp4")
        _mix_background_music(final_path, music_path, music_mixed_path)
        os.replace(music_mixed_path, final_path)
    elif music_path:
        print(f"[video_assembler] music_path '{music_path}' not found -- skipping background music.")

    if progress_callback:
        progress_callback("join", 1, 1)

    return final_path
