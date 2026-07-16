"""
Analytics (Stage 3)
===================
Collects per-video and channel statistics from the YouTube Data API and keeps
a daily snapshot history in the GitHub state repo (automation_analytics.json),
so trends survive Repl restarts. Replaces the donor system's SQLite-backed
AnalyticsManager.
"""

from datetime import datetime

from config import github_read_json, github_write_json
from automation import youtube_client

STATE_PATH = "automation_analytics.json"
MAX_SNAPSHOTS_PER_VIDEO = 400  # ~13 months of daily snapshots


def _load_state() -> dict:
    state = github_read_json(STATE_PATH, default=None)
    if not isinstance(state, dict):
        state = {}
    state.setdefault("videos", {})       # video_id -> [snapshots]
    state.setdefault("channel", [])      # channel-level snapshots
    return state


def _save_state(state: dict):
    for vid, snaps in state["videos"].items():
        state["videos"][vid] = snaps[-MAX_SNAPSHOTS_PER_VIDEO:]
    state["channel"] = state["channel"][-MAX_SNAPSHOTS_PER_VIDEO:]
    github_write_json(STATE_PATH, state, message="Update analytics snapshots")


def get_video_metrics(video_id: str) -> dict:
    """Current view/like/comment counts for one video."""
    data = youtube_client.get("videos", params={
        "part": "statistics,snippet", "id": video_id,
    })
    items = data.get("items", [])
    if not items:
        return {}
    stats = items[0].get("statistics", {})
    return {
        "title": items[0].get("snippet", {}).get("title", ""),
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0)),
    }


def get_channel_stats() -> dict:
    """Channel-level totals for the connected channel (mine=true)."""
    data = youtube_client.get("channels", params={
        "part": "statistics,snippet", "mine": "true",
    })
    items = data.get("items", [])
    if not items:
        return {}
    stats = items[0].get("statistics", {})
    return {
        "channel_title": items[0].get("snippet", {}).get("title", ""),
        "subscribers": int(stats.get("subscriberCount", 0)),
        "total_views": int(stats.get("viewCount", 0)),
        "total_videos": int(stats.get("videoCount", 0)),
    }


def collect_snapshot(video_ids: list = None) -> dict:
    """Takes a metrics snapshot for every known video + the channel, and
    appends it to the state history. Called daily by the scheduler."""
    from automation.comments import published_videos

    if video_ids is None:
        video_ids = [d["video_id"] for d in published_videos()]

    state = _load_state()
    now = datetime.utcnow().isoformat()

    for video_id in video_ids:
        try:
            metrics = get_video_metrics(video_id)
        except Exception as e:
            print(f"[analytics] metrics failed for {video_id}: {e}")
            continue
        if not metrics:
            continue
        metrics["recorded_at"] = now
        state["videos"].setdefault(video_id, []).append(metrics)

    try:
        channel = get_channel_stats()
        if channel:
            channel["recorded_at"] = now
            state["channel"].append(channel)
    except Exception as e:
        print(f"[analytics] channel stats failed: {e}")

    _save_state(state)
    return state


def latest_report() -> dict:
    """Builds the report shown on the /analytics page: latest numbers plus
    the delta vs the previous snapshot for each video."""
    state = _load_state()

    videos = []
    for video_id, snaps in state["videos"].items():
        if not snaps:
            continue
        latest = snaps[-1]
        prev = snaps[-2] if len(snaps) > 1 else None
        videos.append({
            "video_id": video_id,
            "title": latest.get("title", ""),
            "views": latest["views"],
            "likes": latest["likes"],
            "comments": latest["comments"],
            "views_delta": latest["views"] - prev["views"] if prev else 0,
            "likes_delta": latest["likes"] - prev["likes"] if prev else 0,
            "recorded_at": latest.get("recorded_at", ""),
        })
    videos.sort(key=lambda v: v["views"], reverse=True)

    channel = state["channel"][-1] if state["channel"] else {}
    channel_delta = {}
    if len(state["channel"]) > 1:
        prev = state["channel"][-2]
        channel_delta = {
            "subscribers_delta": channel.get("subscribers", 0) - prev.get("subscribers", 0),
            "views_delta": channel.get("total_views", 0) - prev.get("total_views", 0),
        }

    return {
        "channel": channel,
        "channel_delta": channel_delta,
        "videos": videos,
        "snapshots_taken": len(state["channel"]),
    }
