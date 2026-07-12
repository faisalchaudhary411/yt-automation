---
name: LLM provider fallback quota gotchas
description: Groq/Gemini free-tier quota behaviors relevant to this project's script_generator LLM fallback chain.
---

This project's script generator calls Groq first, then falls back to Gemini on any Groq failure.

- Groq's "on_demand" free tier enforces both a per-minute (TPM) and a per-day (TPD) token cap on the same API key. Hitting the daily cap returns a 429 with a `retry in Nm` hint even though it's labeled "per day" — the hint is real (short rolling top-ups), but a fully exhausted day can require tens of minutes to free up meaningful budget.
- A Gemini free-tier API key can have `generate_content_free_tier_requests` **limit: 0** for a given model — this is a Google-side project/billing enablement issue, not a code bug. When this happens the fallback chain correctly attempts Gemini and correctly reports the 429, but every request fails until the user enables billing/API access in Google AI Studio / Cloud console for that key's project.

**Why:** both are hard account-level constraints external to the app; retrying or tuning token budgets in code cannot fix either one.

**How to apply:** when both Groq and Gemini fail with 429s in the same error message, don't keep retrying — check whether it's a Groq daily/per-minute cap (wait it out or upgrade tier) vs. a Gemini free-tier-limit-0 error (user needs to enable billing/API access for that Gemini key's project).
