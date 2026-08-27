"""Shared helpers for the public backend adapter protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def native_id(sample_id: str) -> str:
    prefix = sample_id.split(":", 1)[0].replace("-", "_")
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def evaluator_inputs(job: Mapping[str, Any], evaluator_or_group: str) -> dict[str, Any]:
    raw = job.get("evaluator_inputs", {})
    if not isinstance(raw, Mapping):
        return {}
    nested = raw.get(evaluator_or_group, {})
    if isinstance(nested, Mapping):
        return dict(raw) | dict(nested)
    return dict(raw)


def run_stage(command: Sequence[str], *, cwd: Path) -> None:
    result = subprocess.run(
        list(command), cwd=str(cwd), text=True, capture_output=True, check=False
    )
    if result.returncode:
        rendered = " ".join(command)
        raise RuntimeError(
            f"stage failed ({result.returncode}): {rendered}\n"
            f"STDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}"
        )


def python_stage(script: Path, *arguments: str) -> list[str]:
    return [sys.executable, str(script), *map(str, arguments)]


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def ensure_symlink(source: str | Path, destination: str | Path) -> Path:
    """Expose one central checkpoint at an upstream repository's fixed path."""

    source_path = Path(source).resolve()
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source_path}")
    if destination_path.exists() or destination_path.is_symlink():
        try:
            if destination_path.resolve() == source_path:
                return destination_path
        except OSError:
            pass
        if destination_path.is_file():
            # An upstream install may already have downloaded the same named
            # weight. Preserve it; the adapter never overwrites user files.
            return destination_path
        raise FileExistsError(f"checkpoint destination is not a file: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination_path.symlink_to(source_path)
    except OSError as exc:
        raise RuntimeError(
            f"could not link {source_path} to {destination_path}; on the supported "
            "Linux evaluator platform, ensure the filesystem permits symlinks"
        ) from exc
    return destination_path
