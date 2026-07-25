"""
Sends you a Telegram message with a one-tap approve link whenever a video
is ready. Telegram is used here (rather than email) because it's instant
and works great from a phone.

One-time setup:
  1. Message @BotFather on Telegram, /newbot, follow the prompts -> get a token.
  2. Message your new bot anything once (so it can message you back).
  3. Visit https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates in a browser
     to find your numeric chat_id.
  4. Put TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Replit Secrets.
"""

import os
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_message(text: str):
    """Sends a plain Telegram message. Used by the Stage 3 automation modules
    (comment summaries, daily analytics digest, alerts). Silently no-ops when
    Telegram isn't configured, so automation never crashes on it."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[Telegram not configured] {text}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    resp.raise_for_status()


def send_approval_request_actions(title: str, video_id: str, youtube_preview_url: str, app_repo: str):
    """Notifies you a video is ready and tells you how to publish it via the
    'Approve & Publish Video' GitHub Actions workflow. This works even when
    Replit is offline or asleep -- GitHub Actions is what actually flips the
    video to public (see scripts/approve_publish.py), not this Replit app."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[Telegram not configured] Publish via GitHub Actions with video_id: {video_id}")
        return

    actions_url = f"https://github.com/{app_repo}/actions/workflows/approve-publish.yml"
    text = (
        f"🎬 New video ready for review:\n\n"
        f"*{title}*\n\n"
        f"Preview (private, only you can see it): {youtube_preview_url}\n\n"
        f"To publish: open Actions, run *Approve & Publish Video*, and paste this video ID:\n"
        f"`{video_id}`\n\n"
        f"Ignore this message to leave it private."
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": {
            "inline_keyboard": [[{"text": "▶️ Open Actions to Approve", "url": actions_url}]]
        },
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()


def send_approval_request(title: str, approve_url: str, youtube_preview_url: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[Telegram not configured] Approve here: {approve_url}")
        return

    text = (
        f"🎬 New video ready for review:\n\n"
        f"*{title}*\n\n"
        f"Preview (private, only you can see it): {youtube_preview_url}\n\n"
        f"Tap below to publish it publicly, or ignore to leave it private."
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": {
            "inline_keyboard": [[{"text": "✅ Approve & Publish", "url": approve_url}]]
        },
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
