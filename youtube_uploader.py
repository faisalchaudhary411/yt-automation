"""
Uploads a finished video to YouTube as PRIVATE, and separately flips it to
PUBLIC once you approve. Uses the resumable upload protocol directly via
requests (no heavy google-api-python-client dependency needed).
"""

import json
import requests

UPLOAD_INIT_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


def upload_video(video_path: str, title: str, description: str, tags: list, access_token: str) -> str:
    """Uploads video_path to YouTube as private. Returns the new video's ID."""
    metadata = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": tags,
            "categoryId": "27",  # Education; change to "22" (People & Blogs) etc. if preferred
        },
        "status": {
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": False,
        },
    }

    # Step 1: initiate resumable upload session
    init_resp = requests.post(
        UPLOAD_INIT_URL,
        params={"uploadType": "resumable", "part": "snippet,status"},
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "video/mp4",
        },
        data=json.dumps(metadata),
    )
    init_resp.raise_for_status()
    upload_url = init_resp.headers["Location"]

    # Step 2: upload the actual video bytes
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    upload_resp = requests.put(
        upload_url,
        headers={"Content-Type": "video/mp4"},
        data=video_bytes,
    )
    upload_resp.raise_for_status()

    video_id = upload_resp.json()["id"]
    return video_id


def publish_video(video_id: str, access_token: str):
    """Flips an existing private video to public. Call this only after your approval tap."""
    resp = requests.put(
        VIDEOS_URL,
        params={"part": "status"},
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        data=json.dumps({"id": video_id, "status": {"privacyStatus": "public"}}),
    )
    resp.raise_for_status()
    return resp.json()
