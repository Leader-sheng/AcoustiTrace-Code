#!/usr/bin/env python3
"""Shared helpers for the STARSS clap RT60 proxy calibration pipeline."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "inputs"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "work"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}
META_EXTS = {".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".txt"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_id(text: str) -> str:
    text = Path(str(text)).stem
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text.strip("_") or "sample"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_for_csv(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten_for_csv(row.get(key, "")) for key in fields})


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def file_url(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        if value is None or value == "":
            return ""
        f = float(value)
        if not math.isfinite(f):
            return ""
        return f"{f:.{digits}f}"
    except Exception:
        return ""


def run_cmd(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def existing_path(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text if text and Path(text).exists() else ""


def rt60_class(value: float | None) -> str:
    if value is None:
        return "invalid"
    if value < 0.25:
        return "very_short"
    if value < 0.45:
        return "short"
    if value < 0.75:
        return "medium"
    if value < 1.20:
        return "long"
    return "very_long"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
