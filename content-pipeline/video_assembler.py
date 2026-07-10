"""
Assembles per-scene images + narration audio into one final MP4.
Each scene gets a slow Ken-Burns zoom on its still image, with the narration
audio and a burned-in caption of the narration text.

Requires ffmpeg to be installed on the Replit environment (add "ffmpeg" to
replit.nix or use the nix package manager in the Replit shell:
  `nix-env -iA nixpkgs.ffmpeg`
or simply enable it via Replit's "Nix" packages panel).
"""

import os
import subprocess
import re


def _escape_for_drawtext(text: str) -> str:
    # ffmpeg drawtext needs these characters escaped
    text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_audio_duration(audio_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _build_scene_clip(scene: dict, index: int, work_dir: str, width=1920, height=1080) -> str:
    clip_dir = os.path.join(work_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    out_path = os.path.join(clip_dir, f"clip_{index:03d}.mp4")

    if not scene.get("image_path") or not scene.get("audio_path"):
        raise RuntimeError(f"Scene {index} is missing an image or audio file.")

    duration = _get_audio_duration(scene["audio_path"])
    fps = 30
    total_frames = int(duration * fps)

    caption = _escape_for_drawtext(scene["narration"])

    # Ken Burns zoom (slow zoom-in) + burned-in caption at the bottom
    vf = (
        f"scale={width * 2}:{height * 2},"
        f"zoompan=z='min(zoom+0.0008,1.15)':d={total_frames}:s={width}x{height}:fps={fps},"
        f"drawtext=text='{caption}':fontcolor=white:fontsize=42:borderw=3:bordercolor=black@0.8:"
        f"x=(w-text_w)/2:y=h-160"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", scene["image_path"],
        "-i", scene["audio_path"],
        "-vf", vf,
        "-c:v", "libx264", "-t", str(duration), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def assemble_video(scenes: list, work_dir: str, output_name: str = "final_video.mp4") -> str:
    """
    Builds one clip per scene, then concatenates them into the final MP4.
    Returns the path to the final video.
    """
    clip_paths = [
        _build_scene_clip(scene, i, work_dir) for i, scene in enumerate(scenes)
    ]

    concat_list_path = os.path.join(work_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for path in clip_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    final_path = os.path.join(work_dir, output_name)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c", "copy",
        final_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return final_path
