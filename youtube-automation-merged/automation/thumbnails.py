"""
Thumbnail Generator (Stage 3)
=============================
Creates a 1280x720 YouTube thumbnail for each generated video and uploads it
via thumbnails.set right after the video upload.

The donor system's thumbnail_generator.py was broken (it passed an invalid
`font_size=` kwarg to PIL's draw.text, drew on solid random colors, and never
uploaded anything). This rewrite leans on the live system's infrastructure
instead:

  - background: the video's own first scene image, cover-cropped to 16:9 —
    instantly on-topic, unlike a random solid color
  - text: rendered with content_pipeline.video_assembler's verified font
    resolution + RTL shaping, so Urdu/Arabic titles render correctly
  - styling: dark gradient for readability, gold accent bar + channel name
    matching the channel's visual identity
"""

import os
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from config import CHANNEL_NAME
from content_pipeline.video_assembler import (
    _resolve_font_path,
    _prepare_text_for_rendering,
    _is_latin_text,
)

THUMB_SIZE = (1280, 720)
GOLD = (198, 164, 84, 255)


def _cover_crop(img: Image.Image, size=THUMB_SIZE) -> Image.Image:
    """Scales and center-crops an image to exactly fill `size`."""
    target_w, target_h = size
    scale = max(target_w / img.width, target_h / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)),
                     Image.LANCZOS)
    left = (img.width - target_w) // 2
    top = (img.height - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _add_gradient(img: Image.Image) -> Image.Image:
    """Dark bottom-up gradient so the title stays readable over any photo."""
    overlay = Image.new("RGBA", THUMB_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(THUMB_SIZE[1]):
        alpha = int(200 * max(0.0, (y - THUMB_SIZE[1] * 0.35) / (THUMB_SIZE[1] * 0.65)))
        draw.line([(0, y), (THUMB_SIZE[0], y)], fill=(0, 0, 0, alpha))
    return Image.alpha_composite(img, overlay)


def _draw_title(img: Image.Image, title: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    is_latin = _is_latin_text(title)
    font_path = _resolve_font_path(for_latin=is_latin)
    if not font_path:
        print("[thumbnails] No verified font — skipping title text.")
        return img

    fontsize = 64
    max_chars = max(12, int(THUMB_SIZE[0] * 0.9 / (fontsize * 0.6)))
    lines = textwrap.wrap(title, width=max_chars, break_long_words=False)[:3]
    if not lines:
        return img

    font = ImageFont.truetype(font_path, fontsize)
    line_h = int(fontsize * 1.25)
    total_h = len(lines) * line_h
    y = THUMB_SIZE[1] - total_h - 70

    for line in lines:
        prepared = _prepare_text_for_rendering(line, is_latin)
        bb = draw.textbbox((0, 0), prepared, font=font)
        w = bb[2] - bb[0]
        x = (THUMB_SIZE[0] - w) // 2 - bb[0]
        for dx, dy in [(-3, -3), (-3, 3), (3, -3), (3, 3), (0, 4), (4, 0), (-4, 0), (0, -4)]:
            draw.text((x + dx, y + dy - bb[1]), prepared, font=font, fill=(0, 0, 0, 200))
        draw.text((x, y - bb[1]), prepared, font=font, fill=(255, 255, 255, 255))
        y += line_h

    # Gold accent bar above the title — the channel's brand color.
    bar_y = THUMB_SIZE[1] - total_h - 92
    draw.rectangle([THUMB_SIZE[0] // 2 - 60, bar_y, THUMB_SIZE[0] // 2 + 60, bar_y + 6],
                   fill=GOLD)
    return img


def _draw_channel_badge(img: Image.Image, channel_name: str) -> Image.Image:
    if not channel_name:
        return img
    draw = ImageDraw.Draw(img)
    font_path = _resolve_font_path(for_latin=True)
    if not font_path:
        return img
    font = ImageFont.truetype(font_path, 30)
    name = channel_name.upper()
    bb = draw.textbbox((0, 0), name, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    pad_x, pad_y = 22, 12
    x0, y0 = THUMB_SIZE[0] - w - pad_x * 2 - 28, 28
    draw.rounded_rectangle([x0, y0, x0 + w + pad_x * 2, y0 + h + pad_y * 2],
                           radius=8, fill=(11, 18, 32, 215), outline=GOLD, width=2)
    draw.text((x0 + pad_x - bb[0], y0 + pad_y - bb[1]), name, font=font, fill=GOLD)
    return img


def generate_thumbnail(title: str, background_image_path: str, out_path: str,
                       channel_name: str = CHANNEL_NAME) -> str:
    """Builds the thumbnail and saves it as a JPEG (YouTube requires < 2MB)."""
    if background_image_path and os.path.isfile(background_image_path):
        img = Image.open(background_image_path).convert("RGB")
        img = _cover_crop(img).convert("RGBA")
        img = ImageEnhance.Contrast(img).enhance(1.05)
        img = ImageEnhance.Color(img).enhance(1.1)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    else:
        # Fallback: brand-colored backdrop if no scene image is available.
        img = Image.new("RGBA", THUMB_SIZE, (20, 30, 48, 255))

    img = _add_gradient(img)
    img = _draw_title(img, title)
    img = _draw_channel_badge(img, channel_name)

    img.convert("RGB").save(out_path, "JPEG", quality=90)
    print(f"[thumbnails] Thumbnail saved: {out_path}")
    return out_path
