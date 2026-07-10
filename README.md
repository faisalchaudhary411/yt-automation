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
