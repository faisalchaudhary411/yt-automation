"""
Handles Google OAuth2 for YouTube Data API access.

One-time setup (see README Stage 2 section):
  1. Create an OAuth client in Google Cloud Console (type: Web application).
  2. Add redirect URI: <your-repl-public-url>/oauth2callback
  3. Put GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REPL_URL in Replit Secrets.
  4. Visit <your-repl-public-url>/authorize once in your browser, sign in with
     the Google account that owns your YouTube channel, and approve.
  5. The refresh token is then stored in your GitHub state repo automatically —
     you never have to do this again unless you revoke access.
"""

import os
import requests
from config import github_read_json, github_write_json, GITHUB_REPO, GITHUB_BRANCH

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
REPL_URL = os.environ.get("REPL_URL", "").rstrip("/")  # e.g. https://yt-automation.yourname.repl.co

SCOPES = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.force-ssl"

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"

TOKEN_STATE_PATH = "youtube_token.json"


def build_authorize_url() -> str:
    redirect_uri = f"{REPL_URL}/oauth2callback"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent",  # forces a refresh_token to be issued every time
    }
    query = "&".join(f"{k}={requests.utils.quote(v)}" for k, v in params.items())
    return f"{AUTH_URL}?{query}"


def exchange_code_for_tokens(code: str):
    redirect_uri = f"{REPL_URL}/oauth2callback"
    resp = requests.post(TOKEN_URL, data={
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    })
    resp.raise_for_status()
    tokens = resp.json()

    if "refresh_token" not in tokens:
        raise RuntimeError(
            "Google didn't return a refresh_token. Revoke prior access at "
            "https://myaccount.google.com/permissions and try /authorize again."
        )

    write_result = github_write_json(TOKEN_STATE_PATH, {"refresh_token": tokens["refresh_token"]},
                                      message="Store YouTube OAuth refresh token")

    # Don't just trust the write call — read it back from GitHub to confirm
    # the token is actually there before telling the caller it succeeded.
    # This is what makes it possible to tell "it silently failed" apart from
    # "it worked but you were looking in the wrong place/repo/branch".
    verify = github_read_json(TOKEN_STATE_PATH)
    if not verify or verify.get("refresh_token") != tokens["refresh_token"]:
        raise RuntimeError(
            f"GitHub write call returned {write_result.get('content', {}).get('sha', 'no sha')} "
            f"but reading '{TOKEN_STATE_PATH}' back from {GITHUB_REPO} (branch: {GITHUB_BRANCH}) "
            "did not show the new token. Check GITHUB_REPO/GITHUB_BRANCH/GITHUB_TOKEN in Secrets "
            "match the repo you're actually looking at."
        )

    commit_url = write_result.get("commit", {}).get("html_url", "(no commit URL returned)")
    print(f"[youtube_auth] Token verified written to {GITHUB_REPO}/{TOKEN_STATE_PATH} "
          f"(branch: {GITHUB_BRANCH}). Commit: {commit_url}")
    return tokens


def get_access_token() -> str:
    """Uses the stored refresh_token to mint a fresh access_token for this request."""
    stored = github_read_json(TOKEN_STATE_PATH)
    if not stored or "refresh_token" not in stored:
        raise RuntimeError(
            f"No YouTube auth on file. Visit {REPL_URL}/authorize once to connect your channel."
        )

    resp = requests.post(TOKEN_URL, data={
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": stored["refresh_token"],
        "grant_type": "refresh_token",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]
