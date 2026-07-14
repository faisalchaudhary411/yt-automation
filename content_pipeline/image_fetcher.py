"""
Fetches one image per scene, primarily from Pexels (free API, no cost,
generous limits, no attribution required). Falls back to Wikimedia Commons
(public domain / openly-licensed archival photos, paintings, and maps) for
historical/archival subjects that Pexels — a modern stock-photo library —
has little or no coverage of.

Scenes are searched concurrently (network I/O) to keep total generation time
low, then images are assigned sequentially so no two scenes in the same video
end up with the same photo unless there's truly no other option.

ATTRIBUTION NOTE: Pexels images don't require credit. Wikimedia Commons images
can be public domain OR under a license (CC-BY, CC-BY-SA, etc.) that legally
requires attribution. Whenever a Commons image is used, its credit line is
recorded on the scene and written to an `attributions.txt` file in the work
directory so it can be included in the video description if required.
"""

import os
import re
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from config import PEXELS_API_KEY

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
MAX_CONCURRENT_FETCHES = 6
CANDIDATES_PER_SCENE = 8
MIN_WIKIMEDIA_WIDTH = 800  # skip tiny thumbnails/icons that sneak into Commons search


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
    except requests.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", "?")
        print(f"[image_fetcher] Pexels search failed for '{query}' (HTTP {status}): {e}")
        return []

    photos = data.get("photos", [])
    return [{"id": f"px-{p['id']}", "url": p["src"]["large2x"], "credit": None} for p in photos]


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _search_wikimedia_commons(query: str, per_page: int = CANDIDATES_PER_SCENE) -> list:
    """
    Searches Wikimedia Commons for openly-licensed images — used as a fallback
    for historical/archival subjects that Pexels has little or no coverage of.
    Returns [{id, url, credit}] or [] on no results/error. `credit` is None for
    clearly public-domain works and a "Name / Wikimedia Commons (License)"
    string otherwise, so attribution-requiring images can be tracked.
    """
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,  # File: namespace only
        "gsrlimit": per_page,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1920,
        "format": "json",
        "origin": "*",
    }
    try:
        resp = requests.get(WIKIMEDIA_API_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[image_fetcher] Wikimedia Commons search failed for '{query}': {e}")
        return []

    pages = (data.get("query") or {}).get("pages") or {}
    candidates = []
    for page in pages.values():
        info_list = page.get("imageinfo") or []
        if not info_list:
            continue
        info = info_list[0]
        mime = info.get("mime", "")
        width = info.get("width", 0) or 0
        if not mime.startswith("image/") or mime == "image/svg+xml":
            continue
        if width and width < MIN_WIKIMEDIA_WIDTH:
            continue
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue

        extmeta = info.get("extmetadata") or {}
        license_short = (extmeta.get("LicenseShortName") or {}).get("value", "")
        artist = _strip_html((extmeta.get("Artist") or {}).get("value", ""))

        credit = None
        if license_short and "public domain" not in license_short.lower() and "pd" not in license_short.lower():
            if artist:
                credit = f"{artist} / Wikimedia Commons ({license_short})"
            else:
                credit = f"Wikimedia Commons ({license_short})"

        candidates.append({
            "id": f"wm-{page.get('pageid')}",
            "url": url,
            "credit": credit,
        })
    return candidates


def _candidates_for_scene(keywords: str) -> list:
    """
    Searches with the scene's own keywords first; if that returns nothing (a
    too-specific or oddly-phrased query), falls back to a shorter, broader
    version of the same keywords on Pexels, then — since Pexels is a modern
    stock-photo library with little/no archival or historical imagery — falls
    back to Wikimedia Commons (openly licensed, has real archival material)
    for historical/archival subjects Pexels can't cover at all.
    """
    words = keywords.split()
    broader = " ".join(words[-2:]) if len(words) > 2 else None

    candidates = _search_pexels(keywords)
    if candidates:
        return candidates

    if broader:
        candidates = _search_pexels(broader)
        if candidates:
            return candidates

    candidates = _search_wikimedia_commons(keywords)
    if candidates:
        return candidates

    if broader:
        candidates = _search_wikimedia_commons(broader)

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


def _write_attributions(scenes: list, work_dir: str) -> None:
    """Writes a plain-text attribution list for any scene whose image requires credit."""
    credited = [
        (i, scene["image_credit"])
        for i, scene in enumerate(scenes)
        if scene.get("image_credit")
    ]
    if not credited:
        return
    path = os.path.join(work_dir, "attributions.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Image credits required for this video (Wikimedia Commons):\n\n")
        for i, credit in credited:
            f.write(f"Scene {i}: {credit}\n")
    print(f"[image_fetcher] {len(credited)} image(s) require attribution — see {path}")


def fetch_all_scene_images(scenes: list, work_dir: str, progress_callback=None) -> list:
    """
    scenes: list of scene dicts (each with "image_keywords")
    Returns the same list with added "image_path" and "image_credit" keys per scene.
    "image_credit" is None unless the image came from Wikimedia Commons under a
    license that requires attribution.

    Runs the image *searches* concurrently (fast network calls), then assigns
    images to scenes sequentially so each scene gets the first candidate image
    ID not already used elsewhere in this video — this is what stops longer
    videos from re-showing the same handful of photos over and over. Only
    falls back to reusing a photo if a scene's whole candidate pool is
    already exhausted by earlier scenes.

    If a scene's own search returns nothing at all, its image is filled in
    from a nearby scene that did succeed (checking earlier scenes first, then
    later ones) — so no scene is ever left without SOME image, avoiding a hard
    crash downstream in video_assembler. A RuntimeError is raised only if
    every single scene failed to get any image at all.

    progress_callback, if given, is called as progress_callback(phase, done, total)
    where phase is "search" during the concurrent searches and "download"
    during the concurrent image downloads (each phase counts 0..len(scenes)).
    """
    image_dir = os.path.join(work_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    total = len(scenes)

    candidate_lists = [None] * total
    search_done = 0
    search_lock = threading.Lock()

    def search_one(i, scene):
        nonlocal search_done
        candidate_lists[i] = _candidates_for_scene(scene["image_keywords"])
        if progress_callback:
            with search_lock:
                search_done += 1
                done_snapshot = search_done
            progress_callback("search", done_snapshot, total)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FETCHES) as executor:
        list(executor.map(lambda args: search_one(*args), enumerate(scenes)))

    # Sequential assignment so "already used" can be tracked without races.
    used_ids = set()
    chosen = [None] * total  # each entry: the picked candidate dict, or None
    for i, candidates in enumerate(candidate_lists):
        if not candidates:
            continue
        pick = next((c for c in candidates if c["id"] not in used_ids), candidates[0])
        used_ids.add(pick["id"])
        chosen[i] = pick

    results = [None] * total
    download_done = 0
    download_lock = threading.Lock()

    def download_one(i):
        nonlocal download_done
        if chosen[i]:
            out_path = os.path.join(image_dir, f"scene_{i:03d}.jpg")
            if _download(chosen[i]["url"], out_path):
                results[i] = out_path
        if progress_callback:
            with download_lock:
                download_done += 1
                done_snapshot = download_done
            progress_callback("download", done_snapshot, total)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FETCHES) as executor:
        list(executor.map(download_one, range(total)))

    # Forward-fill: reuse the most recent successfully-downloaded image (and
    # its credit) for any scene whose own fetch failed.
    last_good_path = None
    last_good_credit = None
    for i, scene in enumerate(scenes):
        if results[i]:
            scene["image_path"] = results[i]
            scene["image_credit"] = chosen[i].get("credit") if chosen[i] else None
            last_good_path = scene["image_path"]
            last_good_credit = scene["image_credit"]
        else:
            scene["image_path"] = last_good_path
            scene["image_credit"] = last_good_credit

    # Backward-fill: if the *leading* scene(s) failed and had nothing earlier
    # to reuse, fall back to the nearest later scene's image instead of
    # leaving image_path as None (which would crash video_assembler entirely).
    next_good_path = None
    next_good_credit = None
    for i in range(total - 1, -1, -1):
        if results[i]:
            next_good_path = scenes[i]["image_path"]
            next_good_credit = scenes[i]["image_credit"]
        elif not scenes[i].get("image_path"):
            scenes[i]["image_path"] = next_good_path
            scenes[i]["image_credit"] = next_good_credit

    missing = [i for i, scene in enumerate(scenes) if not scene.get("image_path")]
    if missing:
        raise RuntimeError(
            f"[image_fetcher] No image could be found or downloaded for any scene "
            f"(failed indices: {missing}). Check PEXELS_API_KEY, network connectivity, "
            "and rate limits — see console output above for the specific search errors."
        )

    _write_attributions(scenes, work_dir)

    print(f"[image_fetcher] Images ready for {total} scene(s) "
          f"({len(used_ids)} unique image(s) used).")

    return scenes
