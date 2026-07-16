"""
Comment Manager (Stage 3)
=========================
Auto-replies to comments on published videos, with spam/offensive filtering.

Ported from the donor system's comment_manager.py and adapted to the live
architecture:

  - YouTube access via OAuth (automation.youtube_client) instead of an API
    key + googleapiclient. Posting replies needs youtube.force-ssl, which the
    live app's OAuth grant already includes.
  - "Which videos do we watch?" comes from drafts.json in the GitHub state
    repo (videos with status "published") instead of a SQLite database.
  - Replied-comment dedupe is persisted to the GitHub state repo
    (automation_comments.json), so a Repl restart never double-replies.
  - AI replies go through the live content_pipeline provider chain
    (Cerebras -> SambaNova -> Groq) with template fallback — no OpenAI key.
  - BUG FIX from donor: the old pin_welcome_comment() "pinned" the welcome
    comment by calling comments.markAsSpam on it — which reports your OWN
    comment as spam. The YouTube Data API has no pin endpoint at all, so the
    welcome comment is simply posted as the first top-level comment on
    publish; pin it manually in YouTube Studio if you want it pinned.
"""

import re
import time
from datetime import datetime, timedelta

from config import (
    github_read_json, github_write_json, CHANNEL_NAME,
    AUTO_REPLY_ENABLED, MAX_REPLIES_PER_HOUR, AI_REPLIES_ENABLED,
    WELCOME_COMMENT_ENABLED, WELCOME_COMMENT_TEXT,
)
from automation import youtube_client

STATE_PATH = "automation_comments.json"

SPAM_PATTERNS = [
    r"check (out )?my channel",
    r"sub\s*4\s*sub",
    r"subscribe to me",
    r"free money",
    r"click (the )?link",
    r"make money fast",
    r"\b(viagra|cialis|casino|crypto scam|whatsapp me|telegram me)\b",
]

OFFENSIVE_PATTERNS = [
    r"\b(hate you|stupid|idiot|dumb|kill yourself)\b",
]

POSITIVE_WORDS = [
    "love", "amazing", "awesome", "great", "fantastic", "excellent",
    "perfect", "best", "helpful", "thanks", "thank you", "brilliant",
]

TEMPLATE_REPLIES = [
    "Thanks for watching! What was your favorite part?",
    "Glad you enjoyed it — more stories like this are on the way!",
    "Thanks for the support! What topic should we cover next?",
    "Appreciate you taking the time to comment!",
    "Thanks for being part of the community!",
]

REPLY_TEMPLATES = {
    "question": "Great question! We'll try to cover that in a future video.",
    "compliment": "Thank you so much! Comments like this keep the channel going.",
}


# ---------------------------------------------------------------------------
# State (GitHub state repo)
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    state = github_read_json(STATE_PATH, default=None)
    if not isinstance(state, dict):
        state = {}
    state.setdefault("replied_comment_ids", [])
    state.setdefault("welcome_comment_video_ids", [])
    state.setdefault("reply_timestamps", [])  # ISO strings, rolling 24h window
    return state


def _save_state(state: dict):
    # Cap lists so the state file stays small over months of use.
    state["replied_comment_ids"] = state["replied_comment_ids"][-5000:]
    state["welcome_comment_video_ids"] = state["welcome_comment_video_ids"][-1000:]
    state["reply_timestamps"] = state["reply_timestamps"][-2000:]
    github_write_json(STATE_PATH, state, message="Update comment automation state")


def _replies_in_last_hour(state: dict) -> int:
    cutoff = datetime.utcnow() - timedelta(hours=1)
    count = 0
    for ts in state["reply_timestamps"]:
        try:
            if datetime.fromisoformat(ts) > cutoff:
                count += 1
        except ValueError:
            continue
    return count


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _matches_any(text: str, patterns) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_spam(text: str) -> bool:
    return _matches_any(text.lower(), SPAM_PATTERNS)


def _is_offensive(text: str) -> bool:
    return _matches_any(text.lower(), OFFENSIVE_PATTERNS)


def _is_positive(text: str) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in POSITIVE_WORDS)


def _is_question(text: str) -> bool:
    return "?" in text


def _template_reply(text: str) -> str:
    if _is_question(text):
        return REPLY_TEMPLATES["question"]
    if _is_positive(text):
        return REPLY_TEMPLATES["compliment"]
    return TEMPLATE_REPLIES[len(text) % len(TEMPLATE_REPLIES)]


# ---------------------------------------------------------------------------
# AI reply generation (live provider chain — Cerebras -> SambaNova -> Groq)
# ---------------------------------------------------------------------------

def _ai_reply(comment_text: str, video_title: str):
    """Generates a short, human reply via the live system's LLM chain.
    Returns None if every provider fails (caller falls back to templates)."""
    try:
        from groq import Groq
        from config import GROQ_API_KEY
        from content_pipeline.script_generator import _call_llm

        if not GROQ_API_KEY:
            return None

        client = Groq(api_key=GROQ_API_KEY)
        system_prompt = (
            f"You run the history/finance YouTube channel {CHANNEL_NAME}. "
            "Reply to a viewer's comment. Rules: 1-2 short sentences max, warm and "
            "genuine, no emojis unless the comment uses them, never generic "
            "'thanks for watching' filler, answer questions directly if you can, "
            "never promise anything specific. Match the comment's language. "
            'Return ONLY valid JSON: {"reply": "..."}'
        )
        data = _call_llm(
            client, system_prompt,
            f'Video title: "{video_title}"\nViewer comment: "{comment_text}"',
            max_tokens=300,
        )
        reply = (data.get("reply") or "").strip()
        return reply or None
    except Exception as e:
        print(f"[comments] AI reply failed, using template instead: {e}")
        return None


# ---------------------------------------------------------------------------
# YouTube operations
# ---------------------------------------------------------------------------

def fetch_comments(video_id: str, max_results: int = 50) -> list:
    """Fetches the newest top-level comments for a video."""
    data = youtube_client.get("commentThreads", params={
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max_results,
        "order": "time",
        "textFormat": "plainText",
    })
    comments = []
    for item in data.get("items", []):
        top = item["snippet"]["topLevelComment"]
        snippet = top["snippet"]
        comments.append({
            "id": top["id"],
            "text": snippet.get("textDisplay", ""),
            "author": snippet.get("authorDisplayName", ""),
            "like_count": snippet.get("likeCount", 0),
            "published_at": snippet.get("publishedAt", ""),
        })
    return comments


def post_reply(comment_id: str, reply_text: str):
    youtube_client.post("comments", params={"part": "snippet"}, body={
        "snippet": {"parentId": comment_id, "textOriginal": reply_text},
    })


def post_welcome_comment(video_id: str) -> bool:
    """Posts the channel's welcome comment as the first top-level comment.

    NOTE: YouTube's Data API cannot pin comments (the donor system faked it by
    calling markAsSpam, which actually reports your own comment as spam — do
    NOT reintroduce that). Pin manually in YouTube Studio if desired.
    """
    youtube_client.post("commentThreads", params={"part": "snippet"}, body={
        "snippet": {
            "videoId": video_id,
            "topLevelComment": {"snippet": {"textOriginal": WELCOME_COMMENT_TEXT}},
        },
    })
    return True


def maybe_post_welcome_comment(video_id: str) -> bool:
    """Posts the welcome comment exactly once per video (state-tracked)."""
    if not WELCOME_COMMENT_ENABLED:
        return False
    state = _load_state()
    if video_id in state["welcome_comment_video_ids"]:
        return False
    try:
        post_welcome_comment(video_id)
        state["welcome_comment_video_ids"].append(video_id)
        _save_state(state)
        print(f"[comments] Welcome comment posted on {video_id}")
        return True
    except Exception as e:
        # Comments are disabled on the video until it has processed, and can
        # also be blocked on brand-new uploads — non-fatal either way.
        print(f"[comments] Welcome comment failed on {video_id} (non-fatal): {e}")
        return False


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def published_videos() -> list:
    """Videos we should watch comments on: published entries from drafts.json."""
    drafts = github_read_json("drafts.json", default=[]) or []
    return [d for d in drafts if d.get("video_id") and d.get("status") == "published"]


def run_once(max_replies_per_run: int = None) -> dict:
    """One pass over all published videos: filter + reply to new comments.
    This is what the scheduler calls — bounded, no infinite loop."""
    summary = {"videos_checked": 0, "comments_seen": 0, "replies_posted": 0,
               "spam_skipped": 0, "errors": []}
    state = _load_state()
    replied = set(state["replied_comment_ids"])

    for video in published_videos():
        video_id = video["video_id"]
        title = video.get("title", "")
        summary["videos_checked"] += 1
        try:
            comments = fetch_comments(video_id)
        except Exception as e:
            summary["errors"].append(f"{video_id}: fetch failed: {e}")
            continue

        for comment in comments:
            summary["comments_seen"] += 1
            cid = comment["id"]
            if cid in replied:
                continue
            text = comment["text"]

            if _is_spam(text) or _is_offensive(text):
                # Don't reply — but remember we've seen it so it's not rechecked.
                replied.add(cid)
                summary["spam_skipped"] += 1
                continue

            if len(text.strip()) < 3:
                replied.add(cid)
                continue

            if not AUTO_REPLY_ENABLED:
                continue

            if _replies_in_last_hour(state) >= MAX_REPLIES_PER_HOUR:
                summary["errors"].append("Hourly reply cap reached — remaining comments skipped.")
                state["replied_comment_ids"] = list(replied)
                _save_state(state)
                return summary

            if max_replies_per_run and summary["replies_posted"] >= max_replies_per_run:
                state["replied_comment_ids"] = list(replied)
                _save_state(state)
                return summary

            reply = _ai_reply(text, title) if AI_REPLIES_ENABLED else None
            if not reply:
                reply = _template_reply(text)

            try:
                post_reply(cid, reply)
                replied.add(cid)
                state["replied_comment_ids"] = list(replied)
                state["reply_timestamps"].append(datetime.utcnow().isoformat())
                summary["replies_posted"] += 1
                print(f"[comments] Replied to {comment['author']} on {video_id}")
                time.sleep(2)  # be gentle with write-rate limits
            except Exception as e:
                summary["errors"].append(f"{video_id}/{cid}: reply failed: {e}")

    state["replied_comment_ids"] = list(replied)
    _save_state(state)
    return summary


def get_comment_stats(video_id: str) -> dict:
    """Comment breakdown for one video (used by the analytics page)."""
    comments = fetch_comments(video_id, max_results=100)
    total = len(comments)
    questions = sum(1 for c in comments if _is_question(c["text"]))
    positive = sum(1 for c in comments if _is_positive(c["text"]))
    spam = sum(1 for c in comments if _is_spam(c["text"]))
    return {
        "total": total, "questions": questions, "positive": positive,
        "spam": spam,
        "engagement_score": round((positive + questions) / max(total, 1), 3),
    }
