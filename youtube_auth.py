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
import json
import requests
from config import github_read_json, github_write_json

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

    # Try GitHub first, fall back to local file
    try:
        github_write_json(TOKEN_STATE_PATH, {"refresh_token": tokens["refresh_token"]},
                           message="Store YouTube OAuth refresh token")
        print("Refresh token saved to GitHub.")
    except Exception as e:
        print(f"Warning: Could not save token to GitHub ({e}). Saving locally.")
        local_path = os.path.join("output", TOKEN_STATE_PATH)
        os.makedirs("output", exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump({"refresh_token": tokens["refresh_token"]}, f)
        print(f"Refresh token saved locally to {local_path}")

    return tokens


def get_access_token() -> str:
    """Uses the stored refresh_token to mint a fresh access_token for this request."""
    stored = None

    # Try GitHub first
    try:
        stored = github_read_json(TOKEN_STATE_PATH)
    except Exception as e:
        print(f"GitHub read failed ({e}), trying local fallback")

    # Try local fallback
    if not stored or "refresh_token" not in stored:
        local_path = os.path.join("output", TOKEN_STATE_PATH)
        if os.path.isfile(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            print("Loaded refresh token from local fallback.")

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
