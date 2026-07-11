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


def _build_scene_clip(scene: dict, index: int, work_dir: str, width=1920, height=1080, zoom_rate=0.0008) -> str:
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
        f"zoompan=z='min(zoom+{zoom_rate},1.15)':d={total_frames}:s={width}x{height}:fps={fps},"
        f"drawtext=text='{caption}':fontcolor=white:fontsize=42:borderw=3:bordercolor=black@0.8:"
        f"x=(w-text_w)/2:y=h-160"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", scene["image_path"],
        "-i", scene["audio_path"],
        "-vf", vf,
        "-c:v", "libx264", "-t", str(duration), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def _build_title_card(
    lines: list,
    out_path: str,
    duration: float = 3.5,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    bg_color: str = "0x141E30",
) -> str:
    """
    Renders a simple text-on-color title card (used for intro/outro) with a
    fade in/out, matching the codec/resolution/audio format of scene clips so
    it concatenates cleanly. `lines` is a list of 1-2 strings (title + subtitle).
    """
    title = _escape_for_drawtext(lines[0])
    filters = [
        f"drawtext=text='{title}':fontcolor=white:fontsize=56:borderw=3:"
        f"bordercolor=black@0.7:x=(w-text_w)/2:y=(h-text_h)/2-30"
    ]
    if len(lines) > 1 and lines[1]:
        subtitle = _escape_for_drawtext(lines[1])
        filters.append(
            f"drawtext=text='{subtitle}':fontcolor=white:fontsize=34:borderw=2:"
            f"bordercolor=black@0.7:x=(w-text_w)/2:y=(h-text_h)/2+40"
        )
    fade_out_start = max(0.0, duration - 0.6)
    filters.append(f"fade=t=in:st=0:d=0.5,fade=t=out:st={fade_out_start}:d=0.6")
    vf = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:r={fps}:d={duration}",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf", vf,
        "-c:v", "libx264", "-t", str(duration), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def assemble_video(
    scenes: list,
    work_dir: str,
    output_name: str = "final_video.mp4",
    title: str = None,
    channel_name: str = None,
    include_intro: bool = True,
    include_outro: bool = True,
    style: str = "documentary",
) -> str:
    """
    Builds one clip per scene (plus optional intro/outro title cards for a more
    polished, human-produced feel) and concatenates them into the final MP4.
    `style` is a key from config.VIDEO_STYLES controlling the title-card color
    and Ken Burns zoom speed. Returns the path to the final video.
    """
    from config import VIDEO_STYLES, DEFAULT_VIDEO_STYLE
    style_conf = VIDEO_STYLES.get(style, VIDEO_STYLES[DEFAULT_VIDEO_STYLE])
    bg_color = style_conf["bg_color"]
    zoom_rate = style_conf["zoom_rate"]

    clip_dir = os.path.join(work_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)

    clip_paths = []

    if include_intro:
        subtitle = f"A {channel_name} Story" if channel_name else "A Documentary Story"
        clip_paths.append(
            _build_title_card(
                [title or "", subtitle],
                os.path.join(clip_dir, "intro.mp4"),
                duration=3.5,
                bg_color=bg_color,
            )
        )

    clip_paths += [
        _build_scene_clip(scene, i, work_dir, zoom_rate=zoom_rate) for i, scene in enumerate(scenes)
    ]

    if include_outro:
        subtitle = f"Subscribe to {channel_name} for more" if channel_name else "Subscribe for more stories like this"
        clip_paths.append(
            _build_title_card(
                ["Thanks for watching!", subtitle],
                os.path.join(clip_dir, "outro.mp4"),
                duration=4.0,
                bg_color=bg_color,
            )
        )

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
