from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_dicts(path: str | Path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: str | Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"_json_error": line})
    return rows


def run_cmd(cmd: Sequence[str] | str, cwd: str | Path | None = None, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        return subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=capture_output)
    return subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=capture_output)


def tool_cmd(tool: str) -> List[str]:
    direct = shutil.which(tool)
    if direct:
        return [direct]
    return [tool]


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def probe_video(video_path: str | Path) -> dict:
    info = {
        "duration_sec": float("nan"),
        "has_audio": False,
        "width": None,
        "height": None,
        "fps": float("nan"),
    }
    try:
        cmd = tool_cmd("ffprobe") + [
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]
        cp = run_cmd(cmd, capture_output=True)
        if cp.returncode != 0 or not cp.stdout:
            return info
        obj = json.loads(cp.stdout)
        fmt = obj.get("format", {})
        if fmt.get("duration") is not None:
            info["duration_sec"] = float(fmt["duration"])
        for stream in obj.get("streams", []):
            if stream.get("codec_type") == "video" and info["width"] is None:
                info["width"] = safe_int(stream.get("width"), None)
                info["height"] = safe_int(stream.get("height"), None)
                fps = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""
                if fps and fps != "0/0" and "/" in fps:
                    a, b = fps.split("/", 1)
                    try:
                        info["fps"] = float(a) / float(b)
                    except Exception:
                        pass
                else:
                    info["fps"] = safe_float(fps, float("nan"))
            if stream.get("codec_type") == "audio":
                info["has_audio"] = True
    except Exception:
        pass
    return info


def extract_frame(video_path: str | Path, sec: float, out_path: str | Path) -> bool:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    cmd = tool_cmd("ffmpeg") + [
        "-y",
        "-ss",
        f"{max(0.0, sec):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_path),
    ]
    cp = run_cmd(cmd, check=False)
    return cp.returncode == 0 and out_path.exists()


def extract_audio(video_path: str | Path, out_path: str | Path, sr: int = 16000) -> bool:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    cmd = tool_cmd("ffmpeg") + [
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        str(out_path),
    ]
    cp = run_cmd(cmd, check=False)
    return cp.returncode == 0 and out_path.exists()


def select_detection_targets(row: dict, target_cfg: dict) -> tuple[str, List[str]]:
    title = clean_text(row.get("title"), "").lower()
    seed_category = clean_text(row.get("seed_category"), "").strip()
    matched_key = "default"
    if seed_category in target_cfg:
        matched_key = seed_category
    elif any(k in title for k in ["keyboard", "typing", "switch"]):
        matched_key = "keyboard_switch_click"
    elif any(k in title for k in ["ball", "basketball", "tennis", "ping pong"]):
        matched_key = "ball_bounce_sports"
    elif any(k in title for k in ["hammer", "wood", "metal", "forge", "chisel"]):
        matched_key = "woodworking_metalworking"
    elif any(k in title for k in ["door", "drawer", "latch", "click"]):
        matched_key = "door_mechanical_events"
    elif any(k in title for k in ["car", "motorcycle", "airplane", "drone", "siren", "vehicle"]):
        matched_key = "receiver_observer_vehicle"
    targets = target_cfg.get(matched_key) or target_cfg.get("default") or []
    return matched_key, list(targets)


def json_dump(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def jsonl_rows_from_csv(path: str | Path) -> List[dict]:
    return read_csv_dicts(path)


def ensure_symlink_or_hardlink(src: Path, dst: Path) -> str:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        os.symlink(src, dst)
        return "symlink"
