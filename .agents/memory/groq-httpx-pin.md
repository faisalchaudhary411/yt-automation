---
name: Groq SDK / httpx pin mismatch
description: Old pinned groq package versions break with current httpx; fix direction.
---

Old pinned `groq` package versions can break when `httpx` is upgraded by the environment/lockfile.

**Why:** the `groq` SDK's internal client wiring changed to match newer `httpx` internals; downgrading `httpx` re-triggers other dependency conflicts.

**How to apply:** if you see `groq` client errors that look like an httpx API mismatch, upgrade `groq` to a current version rather than pinning `httpx` down.
