# YouTube Automation — Stage 1 (Content Generation)

Topic → Script (Groq) → Voiceover (ElevenLabs/gTTS) → Images (Pexels) → Final MP4 (FFmpeg).

## Setup on Replit

1. Create a new **Python** Repl.
2. Paste all these files in with the same folder structure:
   ```
   main.py
   config.py
   requirements.txt
   content_pipeline/script_generator.py
   content_pipeline/tts_generator.py
   content_pipeline/image_fetcher.py
   content_pipeline/video_assembler.py
   ```
3. In the Repl's **Secrets** tab (padlock icon), add:
   - `GROQ_API_KEY` — from console.groq.com
   - `PEXELS_API_KEY` — free, from pexels.com/api
   - `ELEVENLABS_API_KEY` — optional, only if you want premium narration
   - `ELEVENLABS_VOICE_ID` — optional, your cloned voice ID
   - `GITHUB_TOKEN` — a fine-grained personal access token with contents:read/write
     on a small repo you create just for this (e.g. `yt-automation-state`)
   - `GITHUB_REPO` — e.g. `yourusername/yt-automation-state`
4. Install FFmpeg: open the **Shell** tab in Replit and run:
   ```
   nix-env -iA nixpkgs.ffmpeg
   ```
   (or enable it via Replit's Nix packages panel if available on your plan)
5. Click **Run** — this starts a small web page where you can type a topic and
   click Generate. Or use the Shell to run it directly:
   ```
   python main.py "The Tulip Mania bubble of 1637"
   ```

## What you get
- A finished MP4 in the `output/` folder, ready to watch/review.
- Title, description, and tags generated alongside it.
- A running log of every draft saved to your `GITHUB_REPO` as `drafts.json`,
  so nothing is lost if the Repl restarts.

## Notes
- First run will take a few minutes (script + ~15 audio clips + ~15 images + video render).
- If ElevenLabs isn't configured, narration automatically falls back to free Google TTS (gTTS) —
  lower quality but works with zero extra setup.
- Nothing here uploads to YouTube yet. Review the MP4 first — Stage 2 (upload with your
  approval step) comes next once you're happy with the video quality/style.

---

## Stage 2 — YouTube Upload + Approval Gate

New files: `youtube_auth.py`, `youtube_uploader.py`, `telegram_notifier.py` (updated `main.py`).

### 1. Google Cloud OAuth setup (one-time)
1. Go to console.cloud.google.com → create/select a project.
2. Enable the **YouTube Data API v3** (APIs & Services → Library → search it → Enable).
3. APIs & Services → Credentials → **Create Credentials → OAuth client ID**.
   - Application type: **Web application**
   - Authorized redirect URI: `https://<your-repl-name>.<your-username>.repl.co/oauth2callback`
     (use your actual Replit public URL — check the webview tab for the exact domain)
4. Copy the Client ID and Client Secret.
5. If prompted, configure the OAuth consent screen as **External** + add yourself as a test user
   (or verify the app later if you want it fully public — test mode works fine for personal use).

### 2. Telegram bot setup (one-time)
1. Message **@BotFather** on Telegram → `/newbot` → follow prompts → copy the token.
2. Send your new bot any message (e.g. "hi") so it can message you back.
3. In a browser, visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Find `"chat":{"id":123456789,...}` in the response — that number is your chat ID.

### 3. New Replit Secrets to add
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `REPL_URL` — your Repl's public https URL, no trailing slash
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### 4. Connect your channel (one-time)
Visit `<REPL_URL>/authorize`, sign in with the Google account that owns your YouTube
channel, approve access. That's it — the refresh token is saved to your GitHub state
repo, so you won't need to do this again.

### 5. How it works from here
1. You generate a video (web form or `python main.py "topic"`).
2. It's uploaded to YouTube as **private** automatically.
3. You get a Telegram message with the title, a private preview link, and an
   **Approve & Publish** button.
4. Tap it → the video flips to public. Ignore it → it just stays private forever.

Nothing publishes without your tap. If you ever want a video fully public later, you can
always approve it whenever you get to reviewing it — it doesn't expire.
