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
import traceback
from flask import Flask, request, jsonify, render_template_string, send_from_directory

from config import ensure_work_dir, github_write_json, github_read_json
from content_pipeline.script_generator import generate_script
from content_pipeline.tts_generator import generate_all_scene_audio
from content_pipeline.image_fetcher import fetch_all_scene_images
from content_pipeline.video_assembler import assemble_video

app = Flask(__name__)

# In-memory job store. Fine for a single-process dev server; jobs are lost on restart.
JOBS = {}
JOBS_LOCK = threading.Lock()

PAGE = """
<!doctype html>
<title>YT Automation - Stage 1</title>
<h2>Generate a video draft</h2>
<form id="genForm">
  <input id="topic" name="topic" style="width:400px" placeholder="e.g. The Tulip Mania bubble of 1637" required>
  <button type="submit" id="genBtn">Generate</button>
</form>
<p id="status"></p>
<div id="result"></div>

<script>
const form = document.getElementById("genForm");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const btn = document.getElementById("genBtn");

const STEP_LABELS = {
  queued: "Queued…",
  script: "Generating script…",
  audio: "Generating narration audio…",
  images: "Fetching scene images…",
  video: "Assembling final video…",
  done: "Done!",
  error: "Failed."
};

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const topic = document.getElementById("topic").value;
  btn.disabled = true;
  resultEl.innerHTML = "";
  statusEl.textContent = "Starting…";

  const resp = await fetch("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic })
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({error: "Unknown error"}));
    statusEl.textContent = "Error: " + err.error;
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
    statusEl.textContent = STEP_LABELS[data.step] || data.step;

    if (data.step === "done") {
      const r = data.result;
      resultEl.innerHTML =
        "<h3>" + r.title + "</h3><p>" + r.description + "</p>" +
        "<video controls width='480' src='" + r.video_url + "'></video><br>" +
        "<a href='" + r.video_url + "' download>Download MP4</a>";
      btn.disabled = false;
      return;
    }
    if (data.step === "error") {
      statusEl.textContent = "Error: " + data.error;
      btn.disabled = false;
      return;
    }
    setTimeout(() => poll(jobId), 2000);
  } catch (e) {
    statusEl.textContent = "Lost connection, retrying…";
    setTimeout(() => poll(jobId), 3000);
  }
}
</script>
"""


def _set_job(job_id: str, **fields):
    with JOBS_LOCK:
        JOBS[job_id].update(fields)


def run_pipeline_job(job_id: str, topic: str):
    """Runs the full Stage 1 pipeline for one topic inside a background thread."""
    try:
        work_dir = ensure_work_dir(job_id)

        _set_job(job_id, step="script")
        print(f"[1/4] Generating script for: {topic}")
        script = generate_script(topic)

        _set_job(job_id, step="audio")
        print(f"[2/4] Generating narration audio ({len(script['scenes'])} scenes)")
        script["scenes"] = generate_all_scene_audio(script["scenes"], work_dir)

        _set_job(job_id, step="images")
        print("[3/4] Fetching scene images")
        script["scenes"] = fetch_all_scene_images(script["scenes"], work_dir)

        _set_job(job_id, step="video")
        print("[4/4] Assembling final video")
        video_path = assemble_video(script["scenes"], work_dir)

        result = {
            "topic": topic,
            "title": script["title"],
            "description": script["description"],
            "tags": script["tags"],
            "video_url": f"/output/{job_id}/final_video.mp4",
            "status": "ready_for_review",
        }

        # Log this draft to the GitHub state repo so nothing is lost between runs
        try:
            history = github_read_json("drafts.json", default=[])
            history.append({k: v for k, v in result.items() if k != "video_url"})
            github_write_json("drafts.json", history, message=f"Add draft: {script['title']}")
        except Exception as e:
            print(f"Warning: could not log draft to GitHub ({e}). Continuing anyway.")

        _set_job(job_id, step="done", result=result)
    except Exception as e:
        traceback.print_exc()
        _set_job(job_id, step="error", error=str(e))


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    topic = request.form.get("topic") or (request.get_json(silent=True) or {}).get("topic")
    if not topic:
        return jsonify({"error": "Missing 'topic'"}), 400

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"step": "queued", "topic": topic, "created_at": time.time()}

    thread = threading.Thread(target=run_pipeline_job, args=(job_id, topic), daemon=True)
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
