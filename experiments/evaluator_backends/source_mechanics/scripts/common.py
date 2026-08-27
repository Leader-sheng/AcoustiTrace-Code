from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha1_short(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def run_cmd(
    cmd: Sequence[str] | str,
    cwd: str | Path | None = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        shell = True
        args = cmd
    else:
        shell = False
        args = list(cmd)
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        shell=shell,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def tool_cmd(tool: str) -> List[str]:
    direct = shutil.which(tool)
    if direct:
        return [direct]
    return [tool]


def list_recursive_files(root: str | Path, exts: Sequence[str]) -> List[Path]:
    root = Path(root)
    exts = {e.lower() for e in exts}
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort()
    return out


def read_csv_dicts(path: str | Path) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: str | Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_ffprobe_json(video_path: str | Path) -> dict:
    cmd = tool_cmd("ffprobe") + [
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    cp = run_cmd(cmd, capture_output=True, check=True)
    return json.loads(cp.stdout)


def _fraction_to_float(value: str) -> float:
    if not value or value in {"0/0", "N/A"}:
        return float("nan")
    if "/" in value:
        a, b = value.split("/", 1)
        try:
            return float(a) / float(b)
        except Exception:
            return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def probe_video(video_path: str | Path) -> dict:
    info = {
        "duration_sec": float("nan"),
        "has_audio": False,
        "width": None,
        "height": None,
        "fps": float("nan"),
        "probe_failed": False,
        "probe_error": "",
    }
    try:
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if fps and fps > 0:
                    info["fps"] = float(fps)
                    if frames and frames > 0:
                        info["duration_sec"] = float(frames / fps)
                if width and width > 0:
                    info["width"] = int(width)
                if height and height > 0:
                    info["height"] = int(height)
                cap.release()
            else:
                raise RuntimeError("OpenCV could not open video")
        except Exception:
            try:
                from moviepy.editor import VideoFileClip

                clip = VideoFileClip(str(video_path))
                info["duration_sec"] = float(clip.duration or float("nan"))
                info["fps"] = float(clip.fps or float("nan"))
                if clip.size and len(clip.size) == 2:
                    info["width"], info["height"] = int(clip.size[0]), int(clip.size[1])
                info["has_audio"] = clip.audio is not None
                clip.close()
            except Exception:
                obj = parse_ffprobe_json(video_path)
                fmt = obj.get("format", {})
                if fmt.get("duration") is not None:
                    info["duration_sec"] = float(fmt["duration"])
                for stream in obj.get("streams", []):
                    if stream.get("codec_type") == "video" and info["width"] is None:
                        info["width"] = int(stream.get("width") or 0) or None
                        info["height"] = int(stream.get("height") or 0) or None
                        fps = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""
                        info["fps"] = _fraction_to_float(fps)
                    if stream.get("codec_type") == "audio":
                        info["has_audio"] = True
    except Exception as e:
        info["probe_failed"] = True
        info["probe_error"] = str(e)
    return info


def guess_audio_path(video_path: str | Path) -> Optional[Path]:
    p = Path(video_path)
    parts = list(p.parts)
    if "videos" in parts:
        idx = parts.index("videos")
        candidate = Path(*parts[:idx], "audio", *parts[idx + 1 :]).with_suffix(".wav")
        return candidate
    if p.parent.name == "video":
        candidate = p.parent.parent / "audio" / p.name
        return candidate.with_suffix(".wav")
    return p.with_suffix(".wav")


def ensure_relative_audio_for_video(video_path: str | Path) -> Optional[Path]:
    cand = guess_audio_path(video_path)
    return cand if cand and cand.exists() else None


def safe_float(value, default=float("nan")):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def parse_detection_lines(text: str) -> List[dict]:
    """Parse OV-AVEL text exports without relying on localized log labels."""
    rows = []
    current_window = None
    foreground_frames = 0

    def flush_current_window():
        nonlocal current_window, foreground_frames
        if current_window and foreground_frames > 0:
            start_sec, end_sec = current_window
            rows.append(
                {
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                    "score": min(1.0, foreground_frames / 10.0),
                }
            )
        current_window = None
        foreground_frames = 0

    for line in text.splitlines():
        mwin = re.search(r"window[_ ]([0-9.]+)s[-_]([0-9.]+)s", line, re.I)
        if mwin:
            flush_current_window()
            current_window = (float(mwin.group(1)), float(mwin.group(2)))
            foreground_frames = 0
            continue

        normalized = line.lower()
        if current_window and any(
            token in normalized
            for token in ("foreground", "physical impact", "positive")
        ):
            foreground_frames += 1

        m = re.search(r"\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*$", line)
        if m and "Physical Impact" in line:
            rows.append(
                {
                    "start_sec": float(m.group(1)),
                    "end_sec": float(m.group(2)),
                    "score": float(m.group(3)),
                }
            )
    flush_current_window()
    return rows
