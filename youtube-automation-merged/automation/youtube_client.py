"""
Shared minimal YouTube Data API v3 client.

Every ported automation module talks to YouTube through here so there's exactly
one place that handles auth. Uses the live system's OAuth refresh token (stored
in the GitHub state repo) via youtube_auth.get_access_token(), and plain
requests — no google-api-python-client dependency, matching the live uploader.

The OAuth scopes already granted by the live app cover everything used here:
  - youtube.upload     -> thumbnails.set
  - youtube.force-ssl  -> comments, commentThreads, playlists, playlistItems,
                          captions, channels.list(mine), videos.list
"""

import requests

from youtube_auth import get_access_token

API_BASE = "https://www.googleapis.com/youtube/v3"


def _headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


def get(path: str, params: dict = None, access_token: str = None) -> dict:
    """Authenticated GET against the YouTube Data API. Returns parsed JSON."""
    token = access_token or get_access_token()
    resp = requests.get(
        f"{API_BASE}/{path}",
        headers=_headers(token),
        params=params or {},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def post(path: str, params: dict = None, body: dict = None, access_token: str = None) -> dict:
    """Authenticated POST with a JSON body. Returns parsed JSON ({} if empty)."""
    token = access_token or get_access_token()
    resp = requests.post(
        f"{API_BASE}/{path}",
        headers={**_headers(token), "Content-Type": "application/json"},
        params=params or {},
        json=body or {},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json() if resp.text else {}


def put(path: str, params: dict = None, body: dict = None, access_token: str = None) -> dict:
    """Authenticated PUT with a JSON body. Returns parsed JSON ({} if empty)."""
    token = access_token or get_access_token()
    resp = requests.put(
        f"{API_BASE}/{path}",
        headers={**_headers(token), "Content-Type": "application/json"},
        params=params or {},
        json=body or {},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json() if resp.text else {}
