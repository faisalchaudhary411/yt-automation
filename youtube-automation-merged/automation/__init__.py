"""
Automation package — Stage 3 features merged into the live system.

All modules here follow the live system's conventions:
  - YouTube access via youtube_auth.get_access_token() (OAuth2, token stored
    in the GitHub state repo) + plain `requests` — no google-api-python-client.
  - State persisted to the GitHub state repo via config.github_read_json /
    github_write_json, so nothing is lost when the Repl restarts.
  - AI text generation via the live content_pipeline provider chain
    (Cerebras -> SambaNova -> Groq), never a separate OpenAI dependency.
"""
