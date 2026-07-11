"""
Fetches one stock photo per scene from Pexels (free API, no cost, generous limits).
Scenes are searched concurrently (network I/O) to keep total generation time low,
then images are assigned sequentially so no two scenes in the same video end up
with the same photo unless there's truly no other option.
"""

import os
import requests
from concurrent.futures import ThreadPoolExecutor
from config import PEXELS_API_KEY

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
MAX_CONCURRENT_FETCHES = 6
CANDIDATES_PER_SCENE = 8


def _search_pexels(query: str, per_page: int = CANDIDATES_PER_SCENE) -> list:
    """Returns a list of {id, url} dicts for the given query, or [] on no results/error."""
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY is not set in Replit Secrets.")

    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": per_page, "orientation": "landscape"}

    try:
        resp = requests.get(PEXELS_SEARCH_URL, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    photos = data.get("photos", [])
    return [{"id": p["id"], "url": p["src"]["large2x"]} for p in photos]


def _candidates_for_scene(keywords: str) -> list:
    """
    Searches with the scene's own keywords first; if that returns nothing (a
    too-specific or oddly-phrased query), falls back to a shorter, broader
    version of the same keywords so a scene is far less likely to end up with
    no usable image at all.
    """
    candidates = _search_pexels(keywords)
    if candidates:
        return candidates

    # Broaden: keep only the last 1-2 words (usually the actual subject noun,
    # e.g. "17th century Amsterdam tulip market" -> "tulip market").
    words = keywords.split()
    if len(words) > 2:
        broader = " ".join(words[-2:])
        candidates = _search_pexels(broader)

    return candidates


def _download(url: str, out_path: str) -> bool:
    try:
        img_resp = requests.get(url, timeout=30)
        img_resp.raise_for_status()
    except requests.RequestException:
        return False
    with open(out_path, "wb") as f:
        f.write(img_resp.content)
    return True


def fetch_all_scene_images(scenes: list, work_dir: str) -> list:
    """
    scenes: list of scene dicts (each with "image_keywords")
    Returns the same list with an added "image_path" key per scene.

    Runs the Pexels *searches* concurrently (fast network calls), then assigns
    images to scenes sequentially so each scene gets the first candidate photo
    ID not already used elsewhere in this video — this is what stops longer
    videos from re-showing the same handful of photos over and over. Only
    falls back to reusing a photo if a scene's whole candidate pool is
    already exhausted by earlier scenes, and only falls back to the previous
    scene's image entirely if a scene's search returned nothing at all.
    """
    image_dir = os.path.join(work_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    candidate_lists = [None] * len(scenes)

    def search_one(i, scene):
        candidate_lists[i] = _candidates_for_scene(scene["image_keywords"])

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FETCHES) as executor:
        list(executor.map(lambda args: search_one(*args), enumerate(scenes)))

    # Sequential assignment so "already used" can be tracked without races.
    used_ids = set()
    chosen_urls = [None] * len(scenes)
    for i, candidates in enumerate(candidate_lists):
        if not candidates:
            continue
        pick = next((c for c in candidates if c["id"] not in used_ids), candidates[0])
        used_ids.add(pick["id"])
        chosen_urls[i] = pick["url"]

    results = [None] * len(scenes)

    def download_one(i):
        if not chosen_urls[i]:
            return
        out_path = os.path.join(image_dir, f"scene_{i:03d}.jpg")
        if _download(chosen_urls[i], out_path):
            results[i] = out_path

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FETCHES) as executor:
        list(executor.map(download_one, range(len(scenes))))

    last_good_path = None
    for i, scene in enumerate(scenes):
        if results[i]:
            scene["image_path"] = results[i]
            last_good_path = results[i]
        else:
            scene["image_path"] = last_good_path  # may be None for scene 0, handled downstream

    return scenes
