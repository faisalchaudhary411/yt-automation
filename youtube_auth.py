"""
Handles Google OAuth2 for YouTube Data API access.

Setup:
  1. Put GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REPL_URL in Replit Secrets.
  2. Visit <your-repl-public-url>/authorize once, sign in, and approve.
  3. Copy the refresh_token from the callback page into Replit Secrets
     as YOUTUBE_REFRESH_TOKEN.
  4. Restart the repl. Done — you never have to authorize again unless revoked.
"""

import os
import json
import requests

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
REPL_URL = os.environ.get("REPL_URL", "").rstrip("/")

SCOPES = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.force-ssl"

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"

LOCAL_TOKEN_PATH = "output/youtube_token.json"


def _read_local_token():
    try:
        with open(LOCAL_TOKEN_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_local_token(data):
    os.makedirs(os.path.dirname(LOCAL_TOKEN_PATH), exist_ok=True)
    with open(LOCAL_TOKEN_PATH, "w") as f:
        json.dump(data, f, indent=2)


def build_authorize_url() -> str:
    redirect_uri = f"{REPL_URL}/oauth2callback"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent",
    }
    query = "&".join(f"{k}={requests.utils.quote(v)}" for k, v in params.items())
    return f"{AUTH_URL}?{query}"


def exchange_code_for_tokens(code: str):
    """Exchanges the OAuth code for tokens and saves locally."""
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

    refresh_token = tokens["refresh_token"]
    _write_local_token({"refresh_token": refresh_token})

    # Also print to console/logs as backup
    print(f"\n[AUTH] Refresh token: {refresh_token}\n")

    return tokens


def build_callback_html(tokens: dict) -> str:
    """
    Returns a beautiful HTML page showing the refresh token.
    Use this as the response body for your /oauth2callback route.
    """
    refresh_token = tokens.get("refresh_token", "NOT_FOUND")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Auth Success</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            color: #e0e0e0;
        }}
        .container {{
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 40px;
            max-width: 700px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        .success-icon {{
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #00b894, #00cec9);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 25px;
            font-size: 40px;
        }}
        h1 {{
            text-align: center;
            font-size: 28px;
            margin-bottom: 10px;
            color: #fff;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 15px;
        }}
        .token-box {{
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            position: relative;
        }}
        .token-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #58a6ff;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .token-value {{
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            color: #7ee787;
            word-break: break-all;
            line-height: 1.6;
            user-select: all;
        }}
        .copy-btn {{
            position: absolute;
            top: 15px;
            right: 15px;
            background: #238636;
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .copy-btn:hover {{ background: #2ea043; }}
        .steps {{
            margin-top: 25px;
            padding: 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            border-left: 3px solid #f39c12;
        }}
        .steps h3 {{
            color: #f39c12;
            font-size: 14px;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .steps ol {{
            padding-left: 20px;
            color: #bbb;
            font-size: 14px;
            line-height: 2;
        }}
        .steps li strong {{
            color: #fff;
        }}
        .warning {{
            margin-top: 20px;
            padding: 15px;
            background: rgba(231, 76, 60, 0.1);
            border: 1px solid rgba(231, 76, 60, 0.3);
            border-radius: 10px;
            color: #e74c3c;
            font-size: 13px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="success-icon">✓</div>
        <h1>Authorization Successful!</h1>
        <p class="subtitle">Your YouTube channel is now connected.</p>

        <div class="token-box">
            <div class="token-label">YOUTUBE_REFRESH_TOKEN</div>
            <div class="token-value" id="token">{refresh_token}</div>
            <button class="copy-btn" onclick="copyToken()">Copy</button>
        </div>

        <div class="steps">
            <h3>Next Steps</h3>
            <ol>
                <li>Click <strong>Copy</strong> above (or select the green text)</li>
                <li>Go to your Replit project → <strong>🔒 Secrets</strong> tab</li>
                <li>Click <strong>New Secret</strong></li>
                <li><strong>Key:</strong> <code>YOUTUBE_REFRESH_TOKEN</code></li>
                <li><strong>Value:</strong> Paste the copied token</li>
                <li>Click <strong>Add Secret</strong>, then <strong>Restart</strong> your repl</li>
            </ol>
        </div>

        <div class="warning">
            ⚠️ Do NOT share this token or commit it to GitHub. Keep it in Replit Secrets only.
        </div>
    </div>

    <script>
        function copyToken() {{
            const token = document.getElementById('token').innerText;
            navigator.clipboard.writeText(token).then(() => {{
                const btn = document.querySelector('.copy-btn');
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = 'Copy', 2000);
            }});
        }}
    </script>
</body>
</html>"""


def _get_refresh_token() -> str:
    """Reads refresh token from Replit Secrets (preferred) or local file."""
    env_token = os.environ.get("YOUTUBE_REFRESH_TOKEN")
    if env_token:
        return env_token

    local = _read_local_token()
    if local and "refresh_token" in local:
        return local["refresh_token"]

    raise RuntimeError(
        f"No YouTube auth found. Visit {REPL_URL}/authorize once, then copy "
        f"the refresh_token into Replit Secrets as YOUTUBE_REFRESH_TOKEN."
    )


def get_access_token() -> str:
    """Uses the stored refresh_token to mint a fresh access_token."""
    refresh_token = _get_refresh_token()

    resp = requests.post(TOKEN_URL, data={
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]
