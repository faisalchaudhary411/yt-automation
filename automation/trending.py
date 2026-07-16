"""
Trending Topics (Stage 3)
=========================
Finds video ideas from two sources:

  1. YouTube's mostPopular chart (via OAuth — no separate API key needed)
  2. Reddit hot posts from configurable subreddits (keyless JSON endpoint)

Results are scored, merged, and cached to the GitHub state repo
(automation_trending.json) so the /trending page loads instantly and the
scheduler can refresh it daily. Replaces the donor system's trending_topics.py
(which used a YouTube API key + googleapiclient, and a barely-active subreddit).
"""

from datetime import datetime

import requests

from config import (
    github_read_json, github_write_json,
    TRENDING_REGION, TRENDING_SUBREDDITS,
)
from automation import youtube_client

STATE_PATH = "automation_trending.json"


def _youtube_trending(count: int = 15) -> list:
    """Most-popular videos right now, scored by engagement like the donor did."""
    data = youtube_client.get("videos", params={
        "part": "snippet,statistics",
        "chart": "mostPopular",
        "regionCode": TRENDING_REGION,
        "maxResults": min(count, 50),
    })
    topics = []
    for item in data.get("items", []):
        stats = item.get("statistics", {})
        score = (
            int(stats.get("viewCount", 0)) * 0.5
            + int(stats.get("likeCount", 0)) * 2
            + int(stats.get("commentCount", 0)) * 3
        )
        topics.append({
            "title": item["snippet"]["title"],
            "source": "youtube_trending",
            "score": score,
            "url": f"https://youtube.com/watch?v={item['id']}",
        })
    return topics


def _reddit_trending(count_per_sub: int = 8) -> list:
    """Hot posts from the configured subreddits (defaults tuned for a
    history/finance channel)."""
    topics = []
    for sub in TRENDING_SUBREDDITS:
        try:
            resp = requests.get(
                f"https://www.reddit.com/r/{sub}/hot.json?limit={count_per_sub}",
                headers={"User-Agent": "WealthThroughAges-Automation/1.0"},
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            for post in resp.json().get("data", {}).get("children", []):
                pdata = post.get("data", {})
                if pdata.get("stickied"):
                    continue
                topics.append({
                    "title": pdata.get("title", ""),
                    "source": f"reddit/r/{sub}",
                    "score": pdata.get("score", 0),
                    "url": f"https://reddit.com{pdata.get('permalink', '')}",
                })
        except Exception as e:
            print(f"[trending] Reddit r/{sub} failed: {e}")
    return topics


def refresh_trending(count: int = 25) -> list:
    """Fetches fresh topics from all sources, caches them to the state repo,
    and returns the ranked list."""
    topics = []
    try:
        topics.extend(_youtube_trending(count))
    except Exception as e:
        print(f"[trending] YouTube trending failed: {e}")
    topics.extend(_reddit_trending())

    # Normalize scores to 0-100 across the merged list for a comparable ranking.
    max_score = max((t["score"] for t in topics), default=1) or 1
    for t in topics:
        t["score"] = round(100 * t["score"] / max_score, 1)

    topics.sort(key=lambda t: t["score"], reverse=True)
    topics = topics[:count]

    github_write_json(STATE_PATH, {
        "refreshed_at": datetime.utcnow().isoformat(),
        "topics": topics,
    }, message="Refresh trending topics")
    return topics


def get_trending(max_age_hours: int = 24) -> list:
    """Returns cached topics; refreshes only if the cache is missing or stale."""
    state = github_read_json(STATE_PATH, default=None)
    if state and state.get("topics"):
        try:
            age = datetime.utcnow() - datetime.fromisoformat(state["refreshed_at"])
            if age.total_seconds() < max_age_hours * 3600:
                return state["topics"]
        except (ValueError, KeyError):
            pass
    try:
        return refresh_trending()
    except Exception as e:
        print(f"[trending] refresh failed, serving stale cache: {e}")
        return (state or {}).get("topics", [])
