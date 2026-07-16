# YouTube Automation — Merged System (Stage 1 + 2 + 3)

Your live system (Topic → Script → Voiceover → Images → MP4 → private upload →
Telegram approval → publish) now merged with the missing features from the
companion automation suite — adapted to this codebase's architecture instead of
bolted on as-is.

## What's new (Stage 3, merged from the companion system)

| Feature | How it works here |
|---|---|
| **Comment auto-replies** | Background loop checks your published videos every 30 min, filters spam/offensive comments, and replies — AI-written via your existing Cerebras→SambaNova→Groq chain (no OpenAI key needed), with template fallback. State-tracked so it never double-replies, capped at 20 replies/hour. |
| **Welcome comment** | Posted automatically as the first comment when you approve a video. (The old system's "pin" trick actually reported your own comment as spam — removed. YouTube's API can't pin; pin manually in Studio if you want.) |
| **Subtitles (.srt)** | Every video gets a real subtitle file computed from exact scene timings — no Whisper, always in sync. Downloadable from the result card, and auto-uploaded as YouTube captions when you publish. |
| **Branded thumbnails** | Generated per video (scene image + title in your verified fonts, gold accent, channel badge), auto-uploaded to YouTube after each upload. Note: YouTube only accepts custom thumbnails from phone-verified channels. |
| **SEO enhancement** | Descriptions now get chapter timestamps (from real timings), hashtags, subscribe block, and any Wikimedia image credits. Titles stay LLM-written — the old system's "Ultimate Guide to…" rewriter was dropped on purpose. |
| **Trending ideas** | `/trending` page: YouTube mostPopular + Reddit (r/history, r/Economics, r/finance, r/documentaries), scored and refreshed daily. "Use this topic →" pre-fills the generator form. |
| **Analytics** | Daily snapshots of every published video + channel totals, stored in your GitHub state repo. `/analytics` page shows views/likes/comments with deltas, plus an optional daily Telegram digest. |
| **Playlists** | Set `PLAYLIST_ID` in Secrets and every published video is auto-added. (The old system only logged this — it's implemented for real here.) |
| **Scheduler** | APScheduler inside the Flask process (fits the Reserved VM deployment). See `/scheduler` for job status. Optional `AUTO_DAILY_VIDEO=true` generates one video/day from the top trending topic — still requires your Telegram approval to publish. |

## File layout

```
main.py                     # orchestrator (Stages 1+2+3 wired together)
config.py                   # all settings incl. Stage 3 feature flags
youtube_auth.py             # OAuth (unchanged — scopes already cover everything)
youtube_uploader.py         # + set_thumbnail(), upload_captions()
telegram_notifier.py        # + send_message() for digests/alerts
content_pipeline/           # script / tts / images / video assembly (unchanged)
automation/
  youtube_client.py         # shared minimal YouTube API client (OAuth + requests)
  comments.py               # comment fetch/reply/filter + welcome comment
  analytics.py              # video/channel snapshots + reports
  trending.py               # trending topics (YouTube + Reddit)
  seo.py                    # description/tag enhancement
  thumbnails.py             # branded thumbnail generator
  subtitles.py              # exact-timing .srt builder
  playlists.py              # real playlist create/add
  scheduler.py              # APScheduler jobs
```

## New Replit Secrets (all optional — everything degrades gracefully)

- `PLAYLIST_ID` — auto-add published videos to this playlist
- `AUTO_DAILY_VIDEO` — `true` to enable one auto-generated video/day (default off)
- `SCHEDULER_ENABLED` — `false` to turn off all background jobs (default on)
- `AUTO_REPLY_ENABLED` / `AI_REPLIES_ENABLED` — comment automation controls
- `COMMENT_CHECK_INTERVAL_MINUTES` (default 30), `MAX_REPLIES_PER_HOUR` (default 20)
- `TRENDING_REGION` (default US), `TRENDING_SUBREDDITS` (comma-separated)
- `ANALYTICS_TELEGRAM_DIGEST` — daily channel stats via Telegram (default on)
- `THUMBNAILS_ENABLED` / `SUBTITLES_ENABLED` / `CAPTIONS_AUTO_UPLOAD` / `WELCOME_COMMENT_ENABLED`
- `WELCOME_COMMENT_TEXT` — override the default welcome comment

## Setup

Same as before — plus one new package:

```bash
pip install -r requirements.txt   # adds apscheduler
```

No re-authorization needed: your existing YouTube OAuth grant
(`youtube.upload` + `youtube.force-ssl`) already covers comments, captions,
thumbnails, and playlists.

## State files (all in your GitHub state repo)

- `drafts.json` — every generated draft + approval tokens (as before)
- `youtube_token.json` — OAuth refresh token (as before)
- `automation_comments.json` — replied-comment IDs, welcome-comment log, rate-limit window
- `automation_analytics.json` — daily metrics snapshots
- `automation_trending.json` — cached trending topics

## Deliberately NOT merged from the companion system

- **SQLite database** — everything syncs into your GitHub state repo instead (one state system, survives Repl restarts)
- **Community-post bot** — the YouTube Data API has no community-posts endpoint; it was a stub
- **Monetization tracker** — was a stub returning zeros; real revenue data needs the Analytics API + a monetized channel
- **OpenAI content generator** — your Cerebras/Groq chain is strictly better; AI comment replies reuse it
- **REST API server / dashboard** — your Flask app already covers both
- **Email notifications** — you already have Telegram, which is strictly faster for approvals
