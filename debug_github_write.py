import os
import base64
import json
import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
GITHUB_API_BASE = "https://api.github.com"

print("=" * 60)
print("CONFIG CHECK")
print("=" * 60)
print(f"GITHUB_REPO   = {GITHUB_REPO!r}")
print(f"GITHUB_BRANCH = {GITHUB_BRANCH!r}")
if not GITHUB_TOKEN:
    print("GITHUB_TOKEN  = ** NOT SET AT ALL ** <-- this alone would explain everything")
else:
    print(f"GITHUB_TOKEN  = set, starts with '{GITHUB_TOKEN[:7]}...', length {len(GITHUB_TOKEN)}")

if not GITHUB_REPO or not GITHUB_TOKEN:
    print("\nStopping here -- fix the missing Secret above first.")
    raise SystemExit(1)

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}
path = "debug_write_test.json"
url = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/contents/{path}"

print("\n" + "=" * 60)
print("STEP 1: Can we even SEE this repo? (GET repo info)")
print("=" * 60)
repo_resp = requests.get(f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}", headers=headers)
print(f"Status: {repo_resp.status_code}")
if repo_resp.status_code != 200:
    print(f"Body: {repo_resp.text[:500]}")
else:
    info = repo_resp.json()
    print(f"Repo found: {info.get('full_name')}, private={info.get('private')}, default_branch={info.get('default_branch')}")

print("\n" + "=" * 60)
print("STEP 2: GET the test file (expected: 404, it doesn't exist yet)")
print("=" * 60)
get_resp = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
print(f"Status: {get_resp.status_code}")
print(f"Body: {get_resp.text[:500]}")
sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

print("\n" + "=" * 60)
print("STEP 3: PUT (write) the test file")
print("=" * 60)
payload = {
    "message": "Debug write test",
    "content": base64.b64encode(json.dumps({"test": "hello"}).encode("utf-8")).decode("utf-8"),
    "branch": GITHUB_BRANCH,
}
if sha:
    payload["sha"] = sha

put_resp = requests.put(url, headers=headers, json=payload)
print(f"Status: {put_resp.status_code}")
print(f"Body: {put_resp.text[:1000]}")

if put_resp.status_code not in (200, 201):
    print("\n^^^ THIS is the real error.")
else:
    print("\nWrite succeeded! Reading it back immediately...")
    verify_resp = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
    print(f"Read-back status: {verify_resp.status_code}")
    if verify_resp.status_code == 200:
        content = base64.b64decode(verify_resp.json()["content"]).decode("utf-8")
        print(f"Read-back content: {content}")
