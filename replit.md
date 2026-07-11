# YouTube Automation — Stage 1 (Content Generation)

Topic → Script (Groq) → Voiceover (ElevenLabs/gTTS) → Images (Pexels) → Final MP4 (FFmpeg).

Flask app that turns a text topic into a ready-to-review documentary-style video draft.
Nothing here uploads to YouTube yet — Stage 2 (upload) comes later.

## Stack
- Python 3.12, Flask (web UI + `/generate` endpoint)
- Groq (`llama-3.3-70b-versatile`) for script generation
- gTTS (free) or ElevenLabs (optional, premium) for narration
- Pexels for stock images
- FFmpeg (via Nix) for final video assembly
- Optional: a separate GitHub repo used as a simple JSON "database" to log drafts

## Running it
- Workflow **Start application** runs `PORT=5000 python3 main.py` and serves the web UI (visit `/`, type a topic, click Generate).
- CLI mode also works: `python3 main.py "Some topic"` — prints the result JSON and writes `output/final_video.mp4`.
- Full pipeline verified end-to-end on 2026-07-11 (script → 12 scenes of narration/images → final MP4, ~32MB).

## Secrets
Required:
- `GROQ_API_KEY` — script generation (configured)
- `PEXELS_API_KEY` — stock images (configured)

Optional (app degrades gracefully without these):
- `ELEVENLABS_API_KEY` / `ELEVENLABS_VOICE_ID` — premium narration voice; falls back to free gTTS if unset
- `GITHUB_TOKEN` / `GITHUB_REPO` / `GITHUB_BRANCH` — logs each draft as a row in `drafts.json` in a GitHub repo; without these the app just prints a warning and continues (no draft history is kept)

## Notes / fixes made during import setup
- The `content_pipeline` module directory was imported as `content-pipeline` (hyphen), which doesn't match Python import syntax used in `main.py`. Renamed to `content_pipeline` (underscore).
- `groq==0.9.0` (pinned in requirements.txt) was incompatible with the current `httpx` release (`Client.__init__() got an unexpected keyword argument 'proxies'`). Upgraded to `groq==1.5.0`.
- `/generate` originally ran the whole pipeline (script + narration + images + video, a few minutes) synchronously inside one HTTP request. Replit's preview proxy times out long-lived requests, so the browser reported "could not reach app" even though the backend was still working. Reworked into a background-job model: `POST /generate` returns a `job_id` immediately, a background thread runs the pipeline into `output/<job_id>/`, and the page polls `GET /status/<job_id>` every 2s until `done`/`error`. The finished video is served from `/output/<job_id>/final_video.mp4`.
- GitHub draft-history logging is wired up: `GITHUB_PERSONAL_ACCESS_TOKEN` secret (read as a `GITHUB_TOKEN` fallback in `config.py`) + `GITHUB_REPO=faisalchaudhary411/yt-lite` env var. Verified a draft actually lands in `drafts.json` in that repo.
- Generated files (audio clips, images, final MP4) land in `output/<job_id>/`, which is gitignored scratch space — not committed.

## User preferences
None recorded yet.
