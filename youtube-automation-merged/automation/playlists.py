"""
Playlist Manager (Stage 3)
==========================
Real playlist management via the YouTube Data API + the live OAuth grant —
the donor system's playlist_manager.py was a stub that only logged intent.

Scope note: playlists.insert and playlistItems.insert are both covered by the
youtube.force-ssl scope the live app already has, so no re-authorization is
needed.
"""

from config import PLAYLIST_ID, AUTO_ADD_TO_PLAYLIST
from automation import youtube_client


def create_playlist(title: str, description: str = "", privacy: str = "public") -> str:
    """Creates a playlist on the connected channel. Returns its ID."""
    data = youtube_client.post("playlists", params={"part": "snippet,status"}, body={
        "snippet": {"title": title, "description": description},
        "status": {"privacyStatus": privacy},
    })
    playlist_id = data["id"]
    print(f"[playlists] Created playlist '{title}': {playlist_id}")
    return playlist_id


def add_video_to_playlist(video_id: str, playlist_id: str = None) -> bool:
    """Adds a video to the configured (or given) playlist. Returns success."""
    playlist_id = playlist_id or PLAYLIST_ID
    if not playlist_id:
        print("[playlists] No PLAYLIST_ID configured — skipping playlist add.")
        return False
    try:
        youtube_client.post("playlistItems", params={"part": "snippet"}, body={
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {"kind": "youtube#video", "videoId": video_id},
            },
        })
        print(f"[playlists] Added {video_id} to playlist {playlist_id}")
        return True
    except Exception as e:
        # Duplicate-in-playlist and similar errors shouldn't break publishing.
        print(f"[playlists] Could not add {video_id} to playlist (non-fatal): {e}")
        return False


def maybe_add_on_publish(video_id: str) -> bool:
    """Called when a video is approved & published."""
    if not AUTO_ADD_TO_PLAYLIST:
        return False
    return add_video_to_playlist(video_id)
