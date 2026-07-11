---
name: Groq SDK / httpx pin mismatch
description: Old pinned groq package versions raise TypeError on Client init due to httpx API changes.
---

Projects that pin an old `groq` Python package (e.g. `groq==0.9.0`) fail at runtime with
`TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` once a current
`httpx` is installed alongside it.

**Why:** the old `groq` SDK passes a `proxies` kwarg into `httpx.Client()` that newer `httpx`
releases no longer accept. Since `httpx` isn't independently pinned in these requirements
files, pip installs the latest `httpx`, creating the mismatch.

**How to apply:** when this error appears, upgrade `groq` to latest (`pip install -U groq`)
rather than trying to downgrade `httpx` — the newer SDK is compatible with current `httpx`
and keeps other dependents happy too.
