---
name: ffmpeg drawtext multi-line captions
description: How to render wrapped, multi-line burned-in captions with ffmpeg's drawtext filter without them overflowing or silently breaking.
---

Embedding a literal `\n` inside a single `drawtext=text='...'` value does not
reliably produce a line break — in practice ffmpeg drops the backslash and
leaves a stray "n" character stuck to the surrounding words instead of
wrapping. Passing the text through as a single unwrapped string also means
long captions run off the edge of the frame with only a fragment visible.

**Why:** confirmed by direct testing — `text='line one\nline two'` renders as
one line reading "line onenline two", both when the `\n` came from a raw
shell string and from a Python-escaped `"\\n"` built programmatically.

**How to apply:** word-wrap the caption text yourself (e.g. Python
`textwrap.wrap`) into a small number of lines sized to fit the frame width at
the chosen font size, then render each line as its own `drawtext` filter
chained with commas, stacked at increasing `y=h-<margin>` offsets from the
bottom. Add `box=1:boxcolor=black@0.5:boxborderw=...` per line for
readability over busy backgrounds. This is the pattern used in
`content_pipeline/video_assembler.py`'s `_caption_filters`.
