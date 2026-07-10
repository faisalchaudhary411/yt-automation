"""
Stage 1 orchestrator: Topic -> Script -> Voiceover -> Images -> Final MP4.

Run modes:
  1. Command line:  python main.py "Your topic here"
  2. Web:           run this file, then POST /generate with {"topic": "..."}
                     or use the simple form at GET /

Nothing here uploads to YouTube or touches comments yet — that's Stage 2/3.
"""

import os
import sys
import json
import traceback
from flask import Flask, request, jsonify, render_template_string

from config import ensure_work_dir, github_write_json, github_read_json
from content_pipeline.script_generator import generate_script
from content_pipeline.tts_generator import generate_all_scene_audio
from content_pipeline.image_fetcher import fetch_all_scene_images
from content_pipeline.video_assembler import assemble_video

app = Flask(__name__)

PAGE = """
<!doctype html>
<title>YT Automation - Stage 1</title>
<h2>Generate a video draft</h2>
<form action="/generate" method="post">
  <input name="topic" style="width:400px" placeholder="e.g. The Tulip Mania bubble of 1637" required>
  <button type="submit">Generate</button>
</form>
<p>{{ status }}</p>
"""


def run_pipeline(topic: str) -> dict:
    """Runs the full Stage 1 pipeline for one topic. Returns metadata + video path."""
    work_dir = ensure_work_dir()

    print(f"[1/4] Generating script for: {topic}")
    script = generate_script(topic)

    print(f"[2/4] Generating narration audio ({len(script['scenes'])} scenes)")
    script["scenes"] = generate_all_scene_audio(script["scenes"], work_dir)

    print("[3/4] Fetching scene images")
    script["scenes"] = fetch_all_scene_images(script["scenes"], work_dir)

    print("[4/4] Assembling final video")
    video_path = assemble_video(script["scenes"], work_dir)

    result = {
        "topic": topic,
        "title": script["title"],
        "description": script["description"],
        "tags": script["tags"],
        "video_path": video_path,
        "status": "ready_for_review",
    }

    # Log this draft to the GitHub state repo so nothing is lost between runs
    try:
        history = github_read_json("drafts.json", default=[])
        history.append({k: v for k, v in result.items() if k != "video_path"})
        github_write_json("drafts.json", history, message=f"Add draft: {script['title']}")
    except Exception as e:
        print(f"Warning: could not log draft to GitHub ({e}). Continuing anyway.")

    return result


@app.route("/")
def index():
    return render_template_string(PAGE, status="")


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    topic = request.form.get("topic") or (request.json or {}).get("topic")
    if not topic:
        return jsonify({"error": "Missing 'topic'"}), 400

    try:
        result = run_pipeline(topic)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode: python main.py "topic here"
        topic_arg = " ".join(sys.argv[1:])
        output = run_pipeline(topic_arg)
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Web mode (default on Replit — click Run)
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
