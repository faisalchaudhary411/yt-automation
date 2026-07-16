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


# ---------------------------------------------------------------------------
# Stage 3 additions
# ---------------------------------------------------------------------------

THUMBNAILS_SET_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
CAPTIONS_URL = "https://www.googleapis.com/upload/youtube/v3/captions"


def set_thumbnail(video_id: str, image_path: str, access_token: str):
    """Uploads a custom thumbnail for a video (covered by the youtube.upload
    scope the app already has). NOTE: YouTube only honors this on channels
    that have verified their phone number — unverified channels get a 403,
    which callers should treat as non-fatal."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    resp = requests.post(
        THUMBNAILS_SET_URL,
        params={"videoId": video_id},
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "image/jpeg",
        },
        data=image_bytes,
    )
    resp.raise_for_status()
    return resp.json()


def upload_captions(video_id: str, srt_path: str, language: str, access_token: str,
                    name: str = "Subtitles"):
    """Uploads an SRT file as the video's caption track (covered by the
    youtube.force-ssl scope). YouTube may reject captions on a video that
    hasn't finished processing yet — callers should treat failure as
    non-fatal; the .srt remains downloadable from the job's output folder."""
    metadata = {
        "snippet": {
            "videoId": video_id,
            "language": language,
            "name": name,
            "isDraft": False,
        }
    }

    with open(srt_path, "rb") as f:
        srt_bytes = f.read()

    boundary = "caption_upload_boundary_7f3a9c"
    body = (
        f"--{boundary}\r\n"
        'Content-Type: application/json; charset=UTF-8\r\n\r\n'
        f"{json.dumps(metadata)}\r\n"
        f"--{boundary}\r\n"
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + srt_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    resp = requests.post(
        CAPTIONS_URL,
        params={"uploadType": "multipart", "part": "snippet"},
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": f"multipart/related; boundary={boundary}",
        },
        data=body,
    )
    resp.raise_for_status()
    return resp.json()
