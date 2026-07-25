"""
Standalone "approve & publish" script, meant to run inside GitHub Actions
(workflow_dispatch), NOT on Replit.

Why this exists: the original /approve route in main.py only works while the
Replit workspace/deployment is up and reachable at whatever URL Telegram's
Approve button points to. On a free/preview-only Replit setup that URL is
unstable and the workspace sleeps, so approvals were silently failing.

This script does the exact same job (flip video to public, update
drafts.json in the GitHub state repo, run the Stage 3 post-publish hooks,
send a Telegram confirmation) but runs entirely inside a GitHub Actions
runner, which needs no Replit uptime at all -- only the secrets below.

Usage (this is what the workflow calls):
    python scripts/approve_publish.py <video_id>

Required environment variables (set as GitHub Actions repo secrets, see the
matching .github/workflows/approve-publish.yml):
    GOOGLE_CLIENT_ID
    GOOGLE_CLIENT_SECRET
    YOUTUBE_REFRESH_TOKEN
    GITHUB_PERSONAL_ACCESS_TOKEN   (fine-grained PAT, contents:write on the STATE repo)
    GITHUB_REPO                    (the STATE repo, e.g. "yourname/yt-lite")
    GITHUB_BRANCH                  (defaults to "main")
    TELEGRAM_BOT_TOKEN             (optional but recommended)
    TELEGRAM_CHAT_ID               (optional but recommended)

Note on captions: caption upload is intentionally NOT attempted here. The
.srt file lives on Replit's local disk (output/<job_id>/subtitles.srt),
which this Actions runner has no access to. If you want captions to keep
auto-uploading, either run that step while Replit happens to be online, or
extend the pipeline to also commit .srt files into the GitHub state repo so
this script can fetch and upload them too.
"""

import os
import sys

# Make sibling modules (config.py, youtube_auth.py, youtube_uploader.py,
# telegram_notifier.py, automation/) importable when this script is run as
# `python scripts/approve_publish.py` from the repo root or via Actions.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import github_read_json, github_write_json
from youtube_auth import get_access_token
from youtube_uploader import publish_video
from telegram_notifier import send_message


def main():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: python scripts/approve_publish.py <video_id>")
        sys.exit(1)

    video_id = sys.argv[1].strip()
    print(f"Looking up draft for video_id={video_id}")

    history = github_read_json("drafts.json", default=[])
    matching = [d for d in history if d.get("video_id") == video_id]
    if not matching:
        print(f"ERROR: No draft found for video_id={video_id} in drafts.json")
        sys.exit(1)

    draft = matching[-1]
    if draft.get("status") == "published":
        print(f"Video {video_id} is already published. Nothing to do.")
        return

    print("Fetching YouTube access token...")
    access_token = get_access_token()

    print(f"Publishing video {video_id} ({draft.get('title', '')!r}) as public...")
    publish_video(video_id, access_token)
    print("Publish call succeeded.")

    draft["status"] = "published"
    github_write_json("drafts.json", history, message=f"Mark published: {draft.get('title', video_id)}")
    print("drafts.json updated in state repo.")

    # --- Stage 3 post-publish hooks (each independent and non-fatal) ---
    notes = []

    try:
        from automation import comments as comment_automation
        if comment_automation.maybe_post_welcome_comment(video_id):
            notes.append("welcome comment posted")
    except Exception as e:
        print(f"Warning: welcome comment failed ({e})")

    try:
        from automation import playlists as playlist_automation
        if playlist_automation.maybe_add_on_publish(video_id):
            notes.append("added to playlist")
    except Exception as e:
        print(f"Warning: playlist add failed ({e})")

    try:
        from automation import analytics as analytics_automation
        analytics_automation.collect_snapshot([video_id])
    except Exception as e:
        print(f"Warning: analytics baseline failed ({e})")

    try:
        extra = f" ({', '.join(notes)})" if notes else ""
        send_message(f"✅ Published: {draft.get('title', video_id)}{extra}\nhttps://youtube.com/watch?v={video_id}")
    except Exception as e:
        print(f"Warning: Telegram confirmation failed ({e})")

    print(f"Done: https://youtube.com/watch?v={video_id}")


if __name__ == "__main__":
    main()
