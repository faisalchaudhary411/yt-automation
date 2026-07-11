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
- Workflow **Start application** runs `PORT=5000 python3 main.py` and serves the web UI (visit `/`, fill in the options, click Generate).
- CLI mode also works: `python3 main.py "Some topic"` — prints the result JSON and writes `output/final_video.mp4` (uses default language/duration/style/voice).
- Full pipeline verified end-to-end on 2026-07-11 (script → scenes of narration/images → final MP4 with intro/outro, multiple styles and voice genders tested).

## Generation options (web UI)
- **Language** — narration language for the script + voiceover (English, Spanish, French, German, Portuguese, Hindi, Urdu, Arabic, Turkish, Russian, Italian, Indonesian). See `config.LANGUAGES`.
- **Video length** — Short (~3 min) / Medium (~6 min) / Long (~10 min); steers scene count and pacing in the script prompt.
- **Voiceover gender** — Male/Female. Only affects narration when `ELEVENLABS_API_KEY` is set (maps to ElevenLabs' premade "Adam"/"Rachel" voices in `config.VOICE_PRESETS`); the free gTTS fallback has no gender control, so both sound the same without ElevenLabs.
- **Video style** — Documentary / Cinematic / Motivational / Educational (`config.VIDEO_STYLES`). Changes the narrator's tone in the script prompt, the intro/outro title-card color, and the Ken Burns zoom speed.
- **Intro/outro toggles** — add a title-card intro (video title) and an outro ("Thanks for watching, subscribe to `CHANNEL_NAME`") for a more polished, less obviously AI-generated feel. `CHANNEL_NAME` env var controls the name used (defaults to "WealthThroughAges").

## Secrets
Required:
- `GROQ_API_KEY` — script generation (configured)
- `PEXELS_API_KEY` — stock images (configured)

Optional (app degrades gracefully without these):
- `ELEVENLABS_API_KEY` / `ELEVENLABS_VOICE_ID` — premium narration voice; falls back to free gTTS if unset. When set, the UI's male/female selection picks ElevenLabs' preset voice IDs rather than `ELEVENLABS_VOICE_ID`.
- `GITHUB_TOKEN` / `GITHUB_REPO` / `GITHUB_BRANCH` — logs each draft as a row in `drafts.json` in a GitHub repo; without these the app just prints a warning and continues (no draft history is kept)
- `CHANNEL_NAME` — used in intro/outro cards and the script prompt; defaults to "WealthThroughAges"

## Notes / fixes made during import setup
- The `content_pipeline` module directory was imported as `content-pipeline` (hyphen), which doesn't match Python import syntax used in `main.py`. Renamed to `content_pipeline` (underscore).
- `groq==0.9.0` (pinned in requirements.txt) was incompatible with the current `httpx` release (`Client.__init__() got an unexpected keyword argument 'proxies'`). Upgraded to `groq==1.5.0`.
- `/generate` originally ran the whole pipeline (script + narration + images + video, a few minutes) synchronously inside one HTTP request. Replit's preview proxy times out long-lived requests, so the browser reported "could not reach app" even though the backend was still working. Reworked into a background-job model: `POST /generate` returns a `job_id` immediately, a background thread runs the pipeline into `output/<job_id>/`, and the page polls `GET /status/<job_id>` every 2s until `done`/`error`. The finished video is served from `/output/<job_id>/final_video.mp4`.
- GitHub draft-history logging is wired up: `GITHUB_PERSONAL_ACCESS_TOKEN` secret (read as a `GITHUB_TOKEN` fallback in `config.py`) + `GITHUB_REPO=faisalchaudhary411/yt-lite` env var. Verified a draft actually lands in `drafts.json` in that repo.
- Generated files (audio clips, images, final MP4) land in `output/<job_id>/`, which is gitignored scratch space — not committed.
- Deployment was initially configured as `autoscale` (Cloud Run), which doesn't fit this app: it keeps job state in an in-memory dict and writes videos to local disk, both of which break on autoscale's stateless, spin-up/spin-down instances. Switched `deploymentTarget` to `vm` (Reserved VM), which stays running continuously.
- Publish failed with `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'` because ffmpeg was only available via the dev shell's runtime path, not as an explicit Nix dependency picked up by the deployment build. Fixed by installing `ffmpeg` as a system dependency (`installSystemDependencies`), which persists it for both dev and deployment.

## User preferences
None recorded yet.
