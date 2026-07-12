"""
Stage 1 orchestrator: Topic -> Script -> Voiceover -> Images -> Final MP4.

Run modes:
  1. Command line:  python main.py "Your topic here"
  2. Web:           run this file, then POST /generate with {"topic": "..."}
                     or use the simple form at GET /

Nothing here uploads to YouTube or touches comments yet — that's Stage 2/3.

Web generation runs as a background job: /generate kicks it off and returns
immediately with a job_id, and the page polls /status/<job_id> for progress.
This avoids the browser/proxy timing out on a request that can take several
minutes (script + narration + images + video render).
"""

import os
import sys
import json
import time
import uuid
import threading
import secrets as pysecrets
import traceback
from flask import Flask, request, jsonify, render_template_string, send_from_directory, redirect

from config import (
    ensure_work_dir, github_write_json, github_read_json,
    CHANNEL_NAME, LANGUAGES, DEFAULT_LANGUAGE, DURATION_PRESETS, DEFAULT_DURATION_MINUTES,
    EDGE_VOICES, DEFAULT_VOICE_GENDER, VIDEO_STYLES, DEFAULT_VIDEO_STYLE,
)
from content_pipeline.script_generator import generate_script
from content_pipeline.tts_generator import generate_all_scene_audio
from content_pipeline.image_fetcher import fetch_all_scene_images
from content_pipeline.video_assembler import assemble_video
from youtube_auth import build_authorize_url, exchange_code_for_tokens, get_access_token, REPL_URL
from youtube_uploader import upload_video, publish_video
from telegram_notifier import send_approval_request

app = Flask(__name__)

# In-memory job store. Fine for a single-process dev server; jobs are lost on restart.
JOBS = {}
JOBS_LOCK = threading.Lock()

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WealthThroughAges — Video Studio</title>
<style>
  :root {
    --ink: #0B1220;
    --panel: #141B2D;
    --panel-raised: #1B2436;
    --hairline: rgba(198,164,84,0.22);
    --gold: #C6A454;
    --gold-bright: #DFC078;
    --text: #EDEAE2;
    --muted: #8B93A6;
    --success: #34D399;
    --error: #F87171;
    --error-bg: #2A1212;
    --error-border: #7F1D1D;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--ink);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.5;
    padding: 0 0 48px 0;
  }
  .masthead {
    padding: 22px 20px 18px 20px;
    border-bottom: 1px solid var(--hairline);
    background: linear-gradient(180deg, #101828 0%, var(--ink) 100%);
  }
  .eyebrow {
    font-size: 12px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 600;
    margin: 0 0 6px 0;
  }
  h1 {
    font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
    font-size: 26px;
    font-weight: 400;
    margin: 0 0 12px 0;
    color: var(--text);
  }
  .connect-link {
    display: inline-block;
    font-size: 13px;
    color: var(--gold-bright);
    text-decoration: none;
    border: 1px solid var(--hairline);
    padding: 6px 12px;
    border-radius: 20px;
  }
  .connect-link:hover { border-color: var(--gold); }

  .container {
    max-width: 560px;
    margin: 0 auto;
    padding: 20px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-radius: 12px;
    padding: 20px;
  }
  .field { margin-bottom: 18px; }
  .field:last-of-type { margin-bottom: 0; }
  label.field-label {
    display: block;
    font-size: 12px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }
  input[type=text], input:not([type]), select {
    width: 100%;
    background: var(--panel-raised);
    border: 1px solid var(--hairline);
    border-radius: 8px;
    padding: 11px 12px;
    color: var(--text);
    font-size: 15px;
    font-family: inherit;
  }
  input:focus, select:focus {
    outline: none;
    border-color: var(--gold);
    box-shadow: 0 0 0 3px rgba(198,164,84,0.15);
  }
  .row-2 { display: flex; gap: 12px; }
  .row-2 .field { flex: 1; }
  .hint {
    display: block;
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
  }
  .checkbox-row {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }
  .checkbox-row label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    color: var(--text);
  }
  input[type=checkbox] {
    width: 18px; height: 18px;
    accent-color: var(--gold);
  }
  .divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--hairline), transparent);
    margin: 18px 0;
  }
  button#genBtn {
    width: 100%;
    background: var(--gold);
    color: #1A1305;
    border: none;
    border-radius: 8px;
    padding: 13px 16px;
    font-size: 15px;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    margin-top: 20px;
  }
  button#genBtn:hover:not(:disabled) { background: var(--gold-bright); }
  button#genBtn:disabled {
    background: #3A3F4A;
    color: var(--muted);
    cursor: not-allowed;
  }

  #progressWrap { margin-top: 18px; }
  .progress-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
  }
  #progressLabel { font-size: 13px; color: var(--text); }
  #progressPercent { font-size: 13px; color: var(--gold-bright); font-weight: 600; }
  .progress-track {
    background: var(--panel-raised);
    border: 1px solid var(--hairline);
    border-radius: 6px;
    overflow: hidden;
    height: 10px;
    width: 100%;
  }
  #progressFill {
    background: linear-gradient(90deg, var(--gold) 0%, var(--gold-bright) 100%);
    height: 100%;
    width: 0%;
    transition: width 0.4s ease;
  }

  #errorBox {
    margin-top: 18px;
    padding: 14px 16px;
    border: 1px solid var(--error-border);
    border-left: 4px solid var(--error);
    border-radius: 8px;
    background: var(--error-bg);
  }
  #errorBox strong {
    color: var(--error);
    font-size: 14px;
  }
  #errorStep {
    margin: 6px 0 2px 0;
    color: var(--muted);
    font-size: 13px;
  }
  #errorMessage {
    white-space: pre-wrap;
    word-break: break-word;
    margin: 8px 0 0 0;
    font-size: 12.5px;
    color: #FCA5A5;
    font-family: ui-monospace, "SF Mono", Consolas, monospace;
  }

  #result {
    margin-top: 20px;
  }
  #result .card h3 {
    font-family: Georgia, "Iowan Old Style", serif;
    font-weight: 400;
    font-size: 20px;
    margin: 0 0 8px 0;
    color: var(--text);
  }
  #result .card p { color: var(--muted); font-size: 14px; }
  #result video {
    width: 100%;
    border-radius: 8px;
    margin: 10px 0;
    border: 1px solid var(--hairline);
  }
  #result a.action-link {
    display: inline-block;
    margin-top: 8px;
    margin-right: 12px;
    font-size: 13px;
    color: var(--gold-bright);
    text-decoration: none;
    border: 1px solid var(--hairline);
    padding: 8px 14px;
    border-radius: 20px;
  }
  #result a.action-link:hover { border-color: var(--gold); }
</style>
</head>
<body>

<div class="masthead">
  <div class="container" style="padding:0;">
    <p class="eyebrow">{{ channel_name }} · Video Studio</p>
    <h1>Generate a video draft</h1>
    <a class="connect-link" href="/authorize" target="_blank">Connect / reconnect YouTube channel</a>
  </div>
</div>

<div class="container">
  <div class="card">
    <form id="genForm">
      <div class="field">
        <label class="field-label" for="topic">Topic</label>
        <input id="topic" name="topic" placeholder="e.g. The Tulip Mania bubble of 1637" required>
      </div>

      <div class="field">
        <label class="field-label" for="language">Voiceover language</label>
        <select id="language" name="language">
          {% for code, name in languages.items() %}
          <option value="{{ code }}" {% if code == default_language %}selected{% endif %}>{{ name }}</option>
          {% endfor %}
        </select>
      </div>

      <div class="row-2">
        <div class="field">
          <label class="field-label" for="duration">Video length</label>
          <select id="duration" name="duration">
            {% for key, minutes in duration_presets.items() %}
            <option value="{{ minutes }}" {% if minutes == default_duration %}selected{% endif %}>
              {{ key|capitalize }} (~{{ minutes }} min)
            </option>
            {% endfor %}
          </select>
        </div>
        <div class="field">
          <label class="field-label" for="voiceGender">Voiceover</label>
          <select id="voiceGender" name="voiceGender">
            {% for gender in voice_genders %}
            <option value="{{ gender }}" {% if gender == default_voice_gender %}selected{% endif %}>{{ gender|capitalize }}</option>
            {% endfor %}
          </select>
        </div>
      </div>
      <span class="hint">Free edge-tts neural voice — male and female both fully supported.</span>

      <div class="field" style="margin-top:18px;">
        <label class="field-label" for="videoStyle">Video style</label>
        <select id="videoStyle" name="videoStyle">
          {% for key, style in video_styles.items() %}
          <option value="{{ key }}" {% if key == default_video_style %}selected{% endif %}>{{ style.name }}</option>
          {% endfor %}
        </select>
      </div>

      <div class="divider"></div>

      <div class="field checkbox-row">
        <label><input type="checkbox" id="includeIntro" checked> Add intro title card</label>
        <label><input type="checkbox" id="includeOutro" checked> Add outro / subscribe card</label>
      </div>

      <button type="submit" id="genBtn">Generate</button>
    </form>

    <div id="progressWrap" style="display:none;">
      <div class="progress-header">
        <span id="progressLabel">Starting…</span>
        <span id="progressPercent">0%</span>
      </div>
      <div class="progress-track">
        <div id="progressFill"></div>
      </div>
    </div>

    <div id="errorBox" style="display:none;">
      <strong>Generation failed</strong>
      <p id="errorStep"></p>
      <pre id="errorMessage"></pre>
    </div>
  </div>

  <div id="result"></div>
</div>

<script>
const form = document.getElementById("genForm");
const progressLabel = document.getElementById("progressLabel");
const progressPercent = document.getElementById("progressPercent");
const resultEl = document.getElementById("result");
const btn = document.getElementById("genBtn");

const STEP_LABELS = {
  queued: "Queued…",
  script: "Generating script…",
  audio: "Generating narration audio…",
  images: "Fetching scene images…",
  video: "Assembling final video (incl. intro/outro)…",
  uploading: "Uploading to YouTube (private)…",
  pending_approval: "Uploaded! Check Telegram to approve publishing.",
  done: "Done!",
  error: "Failed."
};

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const topic = document.getElementById("topic").value;
  const language = document.getElementById("language").value;
  const duration = document.getElementById("duration").value;
  const voiceGender = document.getElementById("voiceGender").value;
  const videoStyle = document.getElementById("videoStyle").value;
  const includeIntro = document.getElementById("includeIntro").checked;
  const includeOutro = document.getElementById("includeOutro").checked;
  btn.disabled = true;
  resultEl.innerHTML = "";
  document.getElementById("errorBox").style.display = "none";
  document.getElementById("progressWrap").style.display = "block";
  document.getElementById("progressFill").style.width = "0%";
  progressPercent.textContent = "0%";
  progressLabel.textContent = "Starting…";

  const resp = await fetch("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic, language,
      duration_minutes: duration,
      voice_gender: voiceGender,
      style: videoStyle,
      include_intro: includeIntro,
      include_outro: includeOutro
    })
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({error: "Unknown error"}));
    document.getElementById("progressWrap").style.display = "none";
    document.getElementById("errorBox").style.display = "block";
    document.getElementById("errorStep").textContent = "Failed at: request validation";
    document.getElementById("errorMessage").textContent = err.error;
    btn.disabled = false;
    return;
  }
  const { job_id } = await resp.json();
  poll(job_id);
});

async function poll(jobId) {
  try {
    const resp = await fetch("/status/" + jobId);
    const data = await resp.json();

    const pct = typeof data.progress === "number" ? Math.max(0, Math.min(100, data.progress)) : 0;
    document.getElementById("progressFill").style.width = pct + "%";
    progressPercent.textContent = Math.round(pct) + "%";
    progressLabel.textContent = data.detail || STEP_LABELS[data.step] || data.step;

    if (data.step === "done" || data.step === "pending_approval") {
      const r = data.result;
      let extra = "";
      if (r.preview_url) {
        extra = "<p>Uploaded to YouTube as <b>private</b>. A Telegram message with an " +
                "Approve &amp; Publish button has been sent.<br>" +
                "Private preview available.</p>" +
                "<a class='action-link' href='" + r.preview_url + "' target='_blank'>View on YouTube</a>";
      }
      resultEl.innerHTML =
        "<div class='card'>" +
        "<h3>" + r.title + "</h3><p>" + r.description + "</p>" +
        "<video controls src='" + r.video_url + "'></video>" +
        "<a class='action-link' href='" + r.video_url + "' download>Download MP4</a>" + extra +
        "</div>";
      btn.disabled = false;
      return;
    }
    if (data.step === "error") {
      document.getElementById("progressWrap").style.display = "none";
      document.getElementById("errorBox").style.display = "block";
      const failedStepLabel = STEP_LABELS[data.failed_step] || data.failed_step || "unknown step";
      document.getElementById("errorStep").textContent = "Failed during: " + failedStepLabel;
      document.getElementById("errorMessage").textContent = data.error || "Unknown error.";
      btn.disabled = false;
      return;
    }
    setTimeout(() => poll(jobId), 2000);
  } catch (e) {
    progressLabel.textContent = "Lost connection, retrying…";
    setTimeout(() => poll(jobId), 3000);
  }
}
</script>
"""


def _set_job(job_id: str, **fields):
    with JOBS_LOCK:
        JOBS[job_id].update(fields)


# Overall progress (0-100) is split across pipeline stages by rough relative
# cost. Script generation and video assembly get fine-grained sub-progress
# (per chunk / per clip); audio and image steps currently only jump between
# their start/end bounds since those modules don't expose a progress hook yet.
STEP_RANGES = {
    "script": (0, 35),
    "audio": (35, 55),
    "images": (55, 70),
    "video": (70, 95),
    "uploading": (95, 100),
}


def _set_progress(job_id: str, step: str, fraction: float, detail: str = None):
    start, end = STEP_RANGES.get(step, (0, 100))
    fraction = max(0.0, min(1.0, fraction))
    progress = round(start + fraction * (end - start), 1)
    fields = {"step": step, "progress": progress}
    if detail is not None:
        fields["detail"] = detail
    _set_job(job_id, **fields)


def run_pipeline_job(
    job_id: str,
    topic: str,
    language: str = DEFAULT_LANGUAGE,
    duration_minutes: float = DEFAULT_DURATION_MINUTES,
    voice_gender: str = DEFAULT_VOICE_GENDER,
    style: str = DEFAULT_VIDEO_STYLE,
    include_intro: bool = True,
    include_outro: bool = True,
):
    """Runs the full Stage 1 pipeline for one topic inside a background thread."""
    try:
        work_dir = ensure_work_dir(job_id)

        _set_progress(job_id, "script", 0.0, detail="Starting script generation…")
        print(f"[1/4] Generating script for: {topic} (lang={language}, ~{duration_minutes}min, style={style})")

        def _script_progress(chunks_done, total_chunks):
            frac = chunks_done / total_chunks if total_chunks else 0.0
            _set_progress(job_id, "script", frac, detail=f"Writing script: chunk {chunks_done}/{total_chunks}")

        script = generate_script(
            topic, language=language, duration_minutes=duration_minutes, style=style,
            progress_callback=_script_progress,
        )

        _set_progress(job_id, "audio", 0.0, detail=f"Generating narration audio ({len(script['scenes'])} scenes)…")
        print(f"[2/4] Generating narration audio ({len(script['scenes'])} scenes, voice={voice_gender})")

        def _audio_progress(done, total):
            frac = done / total if total else 0.0
            _set_progress(job_id, "audio", frac, detail=f"Narration audio: {done}/{total} scenes")

        script["scenes"] = generate_all_scene_audio(
            script["scenes"], work_dir, language=language, voice_gender=voice_gender,
            progress_callback=_audio_progress,
        )
        _set_progress(job_id, "audio", 1.0, detail="Narration audio complete")

        _set_progress(job_id, "images", 0.0, detail="Fetching scene images…")
        print("[3/4] Fetching scene images")

        def _images_progress(phase, done, total):
            # Search and download are each half of this step.
            frac = (0.5 * (done / total if total else 0.0)) if phase == "search" else (0.5 + 0.5 * (done / total if total else 0.0))
            label = "Searching images" if phase == "search" else "Downloading images"
            _set_progress(job_id, "images", frac, detail=f"{label}: {done}/{total} scenes")

        script["scenes"] = fetch_all_scene_images(script["scenes"], work_dir, progress_callback=_images_progress)
        _set_progress(job_id, "images", 1.0, detail="Scene images complete")

        _set_progress(job_id, "video", 0.0, detail="Assembling final video…")
        print("[4/4] Assembling final video")

        def _video_progress(phase, done, total):
            if phase == "clips":
                # Clip rendering is the bulk of the video step (0-90% of this stage);
                # the final crossfade join gets the remaining 90-100%.
                frac = 0.9 * (done / total if total else 0.0)
                detail = f"Rendering scene clip {done}/{total}"
            else:  # "join"
                frac = 0.9 + 0.1 * done  # done is 0 or 1
                detail = "Joining final video…" if done == 0 else "Final video assembled"
            _set_progress(job_id, "video", frac, detail=detail)

        video_path = assemble_video(
            script["scenes"], work_dir,
            title=script["title"],
            channel_name=CHANNEL_NAME,
            include_intro=include_intro,
            include_outro=include_outro,
            style=style,
            progress_callback=_video_progress,
        )

        result = {
            "topic": topic,
            "title": script["title"],
            "description": script["description"],
            "tags": script["tags"],
            "video_url": f"/output/{job_id}/final_video.mp4",
            "status": "ready_for_review",
        }

        # --- Stage 2: upload as private + Telegram approval gate ---
        try:
            _set_progress(job_id, "uploading", 0.0, detail="Uploading to YouTube as private…")
            print("[5/5] Uploading to YouTube as private")
            access_token = get_access_token()
            video_id = upload_video(
                video_path=video_path,
                title=script["title"],
                description=script["description"],
                tags=script["tags"],
                access_token=access_token,
            )

            approval_token = pysecrets.token_urlsafe(16)
            approve_url = f"{REPL_URL}/approve/{video_id}?token={approval_token}"
            preview_url = f"https://youtube.com/watch?v={video_id}"

            result.update({
                "video_id": video_id,
                "preview_url": preview_url,
                "status": "pending_approval",
                "approval_token": approval_token,
            })

            send_approval_request(script["title"], approve_url, preview_url)
            job_step = "pending_approval"
            _set_progress(job_id, "uploading", 1.0, detail="Uploaded, awaiting approval")
        except Exception as e:
            # Upload failing shouldn't hide the fact that the video itself rendered fine —
            # the local file is still downloadable from the UI either way.
            print(f"Warning: YouTube upload/notify failed ({e}). Video is still available locally.")
            result["upload_error"] = str(e)
            job_step = "done"

        # Log this draft to the GitHub state repo so nothing is lost between runs
        try:
            history = github_read_json("drafts.json", default=[])
            history.append({k: v for k, v in result.items() if k != "video_url"})
            github_write_json("drafts.json", history, message=f"Add draft: {script['title']}")
        except Exception as e:
            print(f"Warning: could not log draft to GitHub ({e}). Continuing anyway.")

        _set_job(job_id, step=job_step, result=result, progress=100.0, detail="Complete")
    except Exception as e:
        traceback.print_exc()
        with JOBS_LOCK:
            failed_step = JOBS.get(job_id, {}).get("step", "unknown")
        _set_job(job_id, step="error", error=str(e), failed_step=failed_step)


@app.route("/")
def index():
    return render_template_string(
        PAGE,
        channel_name=CHANNEL_NAME,
        languages=LANGUAGES,
        default_language=DEFAULT_LANGUAGE,
        duration_presets=DURATION_PRESETS,
        default_duration=DEFAULT_DURATION_MINUTES,
        voice_genders=["female", "male"],
        default_voice_gender=DEFAULT_VOICE_GENDER,
        video_styles=VIDEO_STYLES,
        default_video_style=DEFAULT_VIDEO_STYLE,
    )


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    body = request.get_json(silent=True) or {}
    topic = request.form.get("topic") or body.get("topic")
    if not topic:
        return jsonify({"error": "Missing 'topic'"}), 400

    language = request.form.get("language") or body.get("language") or DEFAULT_LANGUAGE
    if language not in LANGUAGES:
        return jsonify({"error": f"Unsupported language '{language}'"}), 400

    voice_gender = request.form.get("voice_gender") or body.get("voice_gender") or DEFAULT_VOICE_GENDER
    if voice_gender not in ("female", "male"):
        return jsonify({"error": f"Unsupported voice_gender '{voice_gender}'"}), 400

    style = request.form.get("style") or body.get("style") or DEFAULT_VIDEO_STYLE
    if style not in VIDEO_STYLES:
        return jsonify({"error": f"Unsupported style '{style}'"}), 400

    try:
        duration_minutes = float(request.form.get("duration_minutes") or body.get("duration_minutes") or DEFAULT_DURATION_MINUTES)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid duration_minutes"}), 400
    duration_minutes = max(2, min(20, duration_minutes))  # sane guardrails

    def _as_bool(value, default=True):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).lower() not in ("false", "0", "no")

    include_intro = _as_bool(request.form.get("include_intro") if "include_intro" in request.form else body.get("include_intro"))
    include_outro = _as_bool(request.form.get("include_outro") if "include_outro" in request.form else body.get("include_outro"))

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"step": "queued", "topic": topic, "created_at": time.time(), "progress": 0.0, "detail": "Queued…"}

    thread = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, topic, language, duration_minutes, voice_gender, style, include_intro, include_outro),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status_endpoint(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify(job)


@app.route("/output/<job_id>/<path:filename>")
def output_file(job_id, filename):
    directory = os.path.join(os.getcwd(), "output", job_id)
    return send_from_directory(directory, filename)


@app.route("/authorize")
def authorize():
    return redirect(build_authorize_url())


@app.route("/oauth2callback")
def oauth2callback():
    code = request.args.get("code")
    if not code:
        return "Missing 'code' from Google — authorization may have failed.", 400
    exchange_code_for_tokens(code)
    return "YouTube channel connected. You can close this tab and return to the app."


@app.route("/approve/<video_id>")
def approve(video_id):
    submitted_token = request.args.get("token", "")

    history = github_read_json("drafts.json", default=[])
    matching = [d for d in history if d.get("video_id") == video_id]
    if not matching:
        return "No pending draft found for this video ID.", 404

    draft = matching[-1]
    if not submitted_token or submitted_token != draft.get("approval_token"):
        return "Invalid or expired approval link.", 403

    access_token = get_access_token()
    publish_video(video_id, access_token)

    draft["status"] = "published"
    github_write_json("drafts.json", history, message=f"Mark published: {draft['title']}")

    return f"Published: {draft['title']} - https://youtube.com/watch?v={video_id}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode: python main.py "topic here" — runs synchronously, no job queue.
        topic_arg = " ".join(sys.argv[1:])
        job_id = "cli"
        run_pipeline_job(job_id, topic_arg)
        print(json.dumps(JOBS[job_id], indent=2, ensure_ascii=False))
    else:
        # Web mode (default on Replit — click Run)
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
