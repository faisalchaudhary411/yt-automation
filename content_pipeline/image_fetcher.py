"""
Fetches one stock photo per scene from Pexels (free API, no cost, generous limits).
Scenes are fetched concurrently (network I/O) to keep total generation time low.
"""

import os
import requests
from concurrent.futures import ThreadPoolExecutor
from config import PEXELS_API_KEY

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
MAX_CONCURRENT_FETCHES = 6


def fetch_image_for_scene(keywords: str, out_path: str) -> bool:
    """Downloads the top matching landscape photo for `keywords` to `out_path`."""
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY is not set in Replit Secrets.")

    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": keywords, "per_page": 1, "orientation": "landscape"}

    resp = requests.get(PEXELS_SEARCH_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    photos = data.get("photos", [])
    if not photos:
        return False

    image_url = photos[0]["src"]["large2x"]
    img_resp = requests.get(image_url, timeout=30)
    img_resp.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(img_resp.content)
    return True


def fetch_all_scene_images(scenes: list, work_dir: str) -> list:
    """
    scenes: list of scene dicts (each with "image_keywords")
    Returns the same list with an added "image_path" key per scene.
    Falls back to the nearest earlier scene's image (or None) if a search returns nothing.
    Fetches run concurrently since each is an independent network call.
    """
    image_dir = os.path.join(work_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    results = [None] * len(scenes)

    def fetch_one(i, scene):
        out_path = os.path.join(image_dir, f"scene_{i:03d}.jpg")
        try:
            found = fetch_image_for_scene(scene["image_keywords"], out_path)
        except requests.RequestException:
            found = False
        results[i] = out_path if found else None

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FETCHES) as executor:
        list(executor.map(lambda args: fetch_one(*args), enumerate(scenes)))

    last_good_path = None
    for i, scene in enumerate(scenes):
        if results[i]:
            scene["image_path"] = results[i]
            last_good_path = results[i]
        else:
            scene["image_path"] = last_good_path  # may be None for scene 0, handled downstream

    return scenes
