# Memory Index

- [Groq SDK / httpx pin mismatch](groq-httpx-pin.md) — old pinned `groq` versions break with current `httpx`; upgrade `groq` rather than downgrading `httpx`.
- [Production path and deployment](production-path-and-deployment.md) — WORK_DIR must be absolute (os.path.dirname(__file__)); deploy as `vm` not `cloudrun` (in-memory JOBS + threading + local files).
- [ffmpeg drawtext multi-line captions](ffmpeg-drawtext-multiline-captions.md) — embedded `\n` in drawtext text doesn't reliably wrap; chain one drawtext filter per line instead.
- [LLM provider fallback quota gotchas](llm-provider-quotas.md) — Groq TPD cap and Gemini free-tier-limit-0 are both account-level 429s, not code bugs; know which is which before retrying.
