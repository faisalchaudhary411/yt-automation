---
name: Production path and deployment lessons
description: Why WORK_DIR must be absolute and why this app must deploy as vm not cloudrun/autoscale
---

## Rule: WORK_DIR must be an absolute path

`WORK_DIR = "output"` (relative) breaks in production because gunicorn's `cwd` can differ from the project root.  
Fixed with: `WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")`  
`main.py`'s `/output/<job_id>/<filename>` route must use this absolute `WORK_DIR`, never `os.path.join(os.getcwd(), "output", job_id)`.

**Why:** `os.getcwd()` in a gunicorn worker is not guaranteed to be the source root. Absolute paths anchored to `__file__` are portable across every run mode (flask dev, gunicorn, CLI).

**How to apply:** Any new module that needs to write/read from the shared output dir should import `WORK_DIR` from `config.py` rather than constructing its own relative path.

## Rule: Deploy as `vm`, not `cloudrun`/autoscale

This app uses:
- In-memory `JOBS` dict (lost on every autoscale instance spin-up)
- `threading` for background job execution (state bound to one process)
- A background `SCHEDULER` (comment-reply loop, analytics, trending refresh)
- Local filesystem output files served from `output/`

Autoscale (cloudrun) spins instances up/down on demand — all of the above is wiped on each new instance. `vm` keeps one process always running, preserving all state.

`.replit` deployment section must read `deploymentTarget = "vm"`.

**Why:** Videos would silently generate but be inaccessible — the instance that created the file and the JOBS entry could differ from the instance serving `/status/` or `/output/` requests.
