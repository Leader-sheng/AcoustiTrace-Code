"""One-command orchestration for the public AcoustiTrace release suite.

The orchestrator intentionally speaks a small JSONL protocol to evaluator
backends. Heavy evaluator environments can therefore run through ``conda run``
without leaking their dependency conflicts into the portable scoring package.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from .aggregate import ScoreRecord, bootstrap_cell
from .manifests import PromptRecord, load_output_manifest, load_prompt_manifest
from .release import load_release_profile, validate_release_suite
from .scores import (
    ScoreResult,
    causality_violation,
    impact_decay,
    log_attack_time,
    motion_loudness,
    native_receiver_score,
    range_attenuation,
    rt60_consistency,
)


EVALUATOR_GROUPS: dict[str, tuple[str, ...]] = {
    "receiver_observer": (
        "range_attenuation",
        "approach_gain",
        "lateral_stability",
    ),
    "source_mechanics": ("motion_loudness", "impact_decay"),
    "causality": ("causality_violation",),
    "rt60": ("rt60_consistency",),
    "log_attack_time": ("log_attack_time",),
}
ALL_EVALUATORS = tuple(
    evaluator for values in EVALUATOR_GROUPS.values() for evaluator in values
)
CANONICAL_VIDEO_EXTENSION = ".mp4"
KNOWN_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")


class EvaluationError(RuntimeError):
    """Raised for a release-suite or backend-protocol error."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise EvaluationError(f"{path}:{line_number} is not a JSON object")
            rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _score(evaluator: str, evidence: Mapping[str, Any]):
    if evaluator == "motion_loudness":
        return motion_loudness(evidence["level_pairs_db"])
    if evaluator == "log_attack_time":
        return log_attack_time(
            evidence["generated_attack_seconds"],
            evidence["reference_attack_seconds"],
        )
    if evaluator == "impact_decay":
        if isinstance(evidence.get("events"), list):
            event_results = [
                impact_decay(
                    item["fit_r2"],
                    item["tail_residual_mae_db"],
                    item["peak_to_floor_db"],
                )
                for item in evidence["events"]
                if isinstance(item, Mapping)
            ]
            valid_scores = [
                float(result.score)
                for result in event_results
                if result.valid and result.score is not None
            ]
            if not valid_scores:
                return ScoreResult(False, None, "no valid impact-decay event")
            return ScoreResult(True, sum(valid_scores) / len(valid_scores))
        return impact_decay(
            evidence["fit_r2"],
            evidence["tail_residual_mae_db"],
            evidence["peak_to_floor_db"],
        )
    if evaluator == "rt60_consistency":
        return rt60_consistency(evidence["audio_rt60"], evidence["visual_rt60"])
    if evaluator == "causality_violation":
        return causality_violation(evidence["onset_delays_seconds"])
    if evaluator == "range_attenuation":
        return range_attenuation(evidence["window_r2_values"])
    if evaluator in {"approach_gain", "lateral_stability"}:
        return native_receiver_score(evidence["native_score"], evaluator=evaluator)
    raise EvaluationError(f"unknown evaluator: {evaluator}")


def parse_dimensions(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [part.strip() for part in value.replace(";", ",").split(",")]
    else:
        raw = [str(part).strip() for part in value]
    selected = [part for part in raw if part]
    if not selected or selected == ["all"]:
        return ALL_EVALUATORS
    unknown = sorted(set(selected) - set(ALL_EVALUATORS))
    if unknown:
        raise EvaluationError(f"unknown dimensions: {', '.join(unknown)}")
    return tuple(evaluator for evaluator in ALL_EVALUATORS if evaluator in selected)


def load_prompts(paths: Sequence[str | Path]) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for path in paths:
        records.extend(load_prompt_manifest(path))
    return records


def select_smoke_prompts(
    prompts: Sequence[PromptRecord],
    selected: Sequence[str],
    limit_per_dimension: int,
) -> list[PromptRecord]:
    """Deterministically select at most ``limit`` assignments per evaluator.

    A prompt may belong to more than one evaluator.  For smoke testing we keep
    only memberships that have not reached their limit, so overlapping pools
    cannot silently push another dimension above the requested cap.
    """

    if limit_per_dimension <= 0:
        return list(prompts)
    selected_set = set(selected)
    counts = {evaluator: 0 for evaluator in selected}
    subset: list[PromptRecord] = []
    for prompt in sorted(prompts, key=lambda row: (row.task, row.prompt_id)):
        memberships = tuple(
            evaluator
            for evaluator in prompt.evaluator_membership
            if evaluator in selected_set and counts[evaluator] < limit_per_dimension
        )
        if not memberships:
            continue
        subset.append(replace(prompt, evaluator_membership=memberships))
        for evaluator in memberships:
            counts[evaluator] += 1
        if all(count >= limit_per_dimension for count in counts.values()):
            break
    missing = [
        f"{evaluator}={count}/{limit_per_dimension}"
        for evaluator, count in counts.items()
        if count < limit_per_dimension
    ]
    if missing:
        raise EvaluationError(
            "smoke-test selection could not satisfy every dimension: "
            + ", ".join(missing)
        )
    return subset


def _safe_prompt_filename(prompt_id: str) -> str:
    if Path(prompt_id).name != prompt_id or prompt_id in {".", ".."}:
        raise EvaluationError(
            f"prompt_id {prompt_id!r} is not safe as a video filename; use --outputs"
        )
    return prompt_id


def discover_video_outputs(
    prompts: Sequence[PromptRecord],
    videos_dir: str | Path,
    model: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Discover the canonical ``<videos>/<task>/<prompt_id>.mp4`` layout."""

    root = Path(videos_dir).resolve()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    expected_paths: set[Path] = set()
    for prompt in prompts:
        prompt_id = _safe_prompt_filename(prompt.prompt_id)
        subdir = "i2av_lat" if prompt.task == "i2av" else prompt.task
        expected_paths.add((root / subdir / f"{prompt_id}.mp4").resolve())

    observed_paths = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in KNOWN_VIDEO_EXTENSIONS
    } if root.is_dir() else set()
    noncanonical = sorted(observed_paths - expected_paths)
    if noncanonical:
        preview = [str(path.relative_to(root)) for path in noncanonical[:10]]
        raise EvaluationError(
            f"found {len(noncanonical)} non-canonical or unknown video files; "
            f"expected exactly <task>/<prompt_id>.mp4; first: {preview}"
        )

    for prompt in prompts:
        prompt_id = _safe_prompt_filename(prompt.prompt_id)
        subdir = "i2av_lat" if prompt.task == "i2av" else prompt.task
        expected_path = root / subdir / f"{prompt_id}{CANONICAL_VIDEO_EXTENSION}"
        sample_id = f"{prompt.task}:{prompt.prompt_id}"
        if expected_path.is_file():
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "task": prompt.task,
                    "model": model,
                    "video_path": str(expected_path.resolve()),
                    "status": "success",
                    "error": "",
                }
            )
        else:
            missing.append(sample_id)
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "task": prompt.task,
                    "model": model,
                    "video_path": "",
                    "status": "missing",
                    "error": "generated video not found",
                }
            )
    return rows, missing


def _normalize_outputs(
    rows: list[dict[str, Any]],
    prompts: Sequence[PromptRecord],
    *,
    source_dir: Path,
    model: str | None,
    check_files: bool,
) -> tuple[list[dict[str, Any]], str]:
    expected = {(row.task, row.prompt_id) for row in prompts}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    models: set[str] = set()
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        task = str(row.get("task", "")).strip().lower()
        prompt_id = str(row.get("prompt_id", row.get("sample_id", ""))).strip()
        key = (task, prompt_id)
        if not task or not prompt_id:
            errors.append(f"output row {index} is missing task or prompt_id")
            continue
        if key in seen:
            errors.append(f"duplicate output row: {task}:{prompt_id}")
        seen.add(key)
        row_model = str(row.get("model", model or "")).strip()
        if not row_model:
            errors.append(f"{task}:{prompt_id} has no model name")
        else:
            models.add(row_model)
        status = str(row.get("status", "success")).strip().lower() or "success"
        video_text = str(row.get("video_path", "")).strip()
        video_path = Path(video_text).expanduser() if video_text else None
        if video_path and not video_path.is_absolute():
            video_path = (source_dir / video_path).resolve()
        if status == "success" and not video_path:
            errors.append(f"{task}:{prompt_id} is successful but has no video_path")
        if check_files and status == "success" and video_path and not video_path.is_file():
            errors.append(f"{task}:{prompt_id} video does not exist: {video_path}")
        normalized.append(
            {
                **row,
                "task": task,
                "prompt_id": prompt_id,
                "model": row_model,
                "video_path": str(video_path) if video_path else "",
                "status": status,
                "error": str(row.get("error", "")).strip(),
            }
        )
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        errors.append(f"missing {len(missing)} output rows; first: {missing[:5]}")
    if extra:
        errors.append(f"found {len(extra)} unknown output rows; first: {extra[:5]}")
    if len(models) > 1:
        errors.append(f"one evaluation run must contain one model; found {sorted(models)}")
    if model and models and models != {model}:
        errors.append(f"--model {model!r} does not match output model {sorted(models)!r}")
    if errors:
        raise EvaluationError("\n".join(errors))
    inferred_model = next(iter(models)) if models else str(model or "")
    return normalized, inferred_model


def _resolve_manifest_path(value: str, manifest_base: Path, repo_root: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    repo_candidate = (repo_root / path).resolve()
    manifest_candidate = (manifest_base / path).resolve()
    if repo_candidate.exists() or not manifest_candidate.exists():
        return str(repo_candidate)
    return str(manifest_candidate)


def _resolve_paths(value: Any, manifest_base: Path, repo_root: Path) -> Any:
    if isinstance(value, Mapping):
        resolved: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, str) and item and (
                str(key).endswith("_path") or str(key).endswith("_file")
            ):
                resolved[str(key)] = _resolve_manifest_path(
                    item,
                    manifest_base,
                    repo_root,
                )
            else:
                resolved[str(key)] = _resolve_paths(item, manifest_base, repo_root)
        return resolved
    if isinstance(value, list):
        return [_resolve_paths(item, manifest_base, repo_root) for item in value]
    return value


def _backend_jobs(
    prompts: Sequence[PromptRecord],
    outputs: Sequence[Mapping[str, Any]],
    selected: Sequence[str],
    *,
    repo_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    output_map = {(str(row["task"]), str(row["prompt_id"])): row for row in outputs}
    groups = {name: [] for name in EVALUATOR_GROUPS}
    immediate: list[dict[str, Any]] = []
    selected_set = set(selected)
    for prompt in prompts:
        output = output_map[(prompt.task, prompt.prompt_id)]
        sample_id = f"{prompt.task}:{prompt.prompt_id}"
        memberships = [e for e in prompt.evaluator_membership if e in selected_set]
        if not memberships:
            continue
        if str(output["status"]) != "success":
            for evaluator in memberships:
                immediate.append(
                    {
                        "sample_id": sample_id,
                        "prompt_id": prompt.prompt_id,
                        "task": prompt.task,
                        "model": output["model"],
                        "evaluator": evaluator,
                        "status": "invalid",
                        "reason": f"generation_{output['status']}: {output.get('error', '')}".rstrip(),
                        "evidence": {},
                        "backend": "generation",
                    }
                )
            continue
        manifest_base = Path(prompt.manifest_path).parent if prompt.manifest_path else Path.cwd()
        common = {
            "sample_id": sample_id,
            "prompt_id": prompt.prompt_id,
            "task": prompt.task,
            "model": output["model"],
            "prompt_text": prompt.prompt_text,
            "video_path": output["video_path"],
            "category": prompt.category,
            "conditioning_asset_id": prompt.conditioning_asset_id,
            "conditioning_asset_path": _resolve_manifest_path(
                prompt.conditioning_asset_path,
                manifest_base,
                repo_root,
            )
            if prompt.conditioning_asset_path
            else "",
            "evaluator_inputs": _resolve_paths(
                prompt.evaluator_inputs,
                manifest_base,
                repo_root,
            ),
        }
        for group, group_evaluators in EVALUATOR_GROUPS.items():
            requested = [e for e in memberships if e in group_evaluators]
            if requested:
                groups[group].append({**common, "evaluators": requested})
    return groups, immediate


def load_backend_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("backends"), dict):
        raise EvaluationError(f"invalid backend config: {source}")
    data["_source"] = str(source)
    return data


def _format_command(
    tokens: Sequence[str], placeholders: Mapping[str, str]
) -> list[str]:
    command: list[str] = []
    for token in tokens:
        rendered = str(token).format_map(placeholders)
        if rendered:
            command.append(rendered)
    return command


def _fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _run_backend(
    *,
    group: str,
    jobs: list[dict[str, Any]],
    spec: Mapping[str, Any],
    output_root: Path,
    repo_root: Path,
    resume: bool,
    dry_run: bool,
    allow_mock: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    work_dir = output_root / "work" / group
    input_path = work_dir / "input.jsonl"
    evidence_path = work_dir / "evidence.jsonl"
    state_path = work_dir / "state.json"
    log_path = output_root / "logs" / f"{group}.log"
    work_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(input_path, jobs)
    input_hash = _fingerprint(jobs)

    is_mock = bool(spec.get("mock", False))
    if is_mock and not allow_mock:
        raise EvaluationError(
            f"backend {group} is marked mock; pass --allow-mock only for protocol testing"
        )
    command_template = spec.get("command")
    if not isinstance(command_template, list) or not command_template:
        raise EvaluationError(f"backend {group} has no command configured")

    placeholders = {
        "python": sys.executable,
        "repo_root": str(repo_root),
        "input_manifest": str(input_path),
        "output_evidence": str(evidence_path),
        "work_dir": str(work_dir),
        "output_root": str(output_root),
        "resume_flag": "--resume" if resume else "",
    }
    command = _format_command(command_template, placeholders)
    state = {
        "group": group,
        "input_sha256": input_hash,
        "command": command,
        "mock": is_mock,
        "status": "planned" if dry_run else "running",
    }
    if resume and state_path.is_file() and evidence_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("status") == "success" and previous.get("input_sha256") == input_hash:
            return _read_jsonl(evidence_path), previous | {"resumed": True}

    if dry_run:
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return [], state

    cwd_text = str(spec.get("cwd", "{repo_root}")).format_map(placeholders)
    cwd = Path(cwd_text)
    if not cwd.is_absolute():
        cwd = (repo_root / cwd).resolve()
    env = os.environ.copy()
    for key, value in dict(spec.get("env", {})).items():
        env[str(key)] = str(value).format_map(placeholders)
    timeout = int(spec.get("timeout_seconds", 0)) or None
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"COMMAND: {shlex.join(command)}\nRETURN CODE: {result.returncode}\n\n"
        f"STDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}\n",
        encoding="utf-8",
    )
    state.update(
        {
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "log": str(log_path),
        }
    )
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if result.returncode != 0:
        raise EvaluationError(f"backend {group} failed; see {log_path}")
    if not evidence_path.is_file():
        raise EvaluationError(f"backend {group} did not create {evidence_path}")
    return _read_jsonl(evidence_path), state


def _expected_keys(jobs: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(row["sample_id"]), str(evaluator))
        for row in jobs
        for evaluator in row["evaluators"]
    }


def _normalize_backend_rows(
    group: str,
    jobs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = _expected_keys(jobs)
    metadata = {str(row["sample_id"]): row for row in jobs}
    observed: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows, start=1):
        sample_id = str(raw.get("sample_id", ""))
        evaluator = str(raw.get("evaluator", ""))
        key = (sample_id, evaluator)
        if key not in expected:
            raise EvaluationError(f"backend {group} returned unexpected row {index}: {key}")
        if key in observed:
            raise EvaluationError(f"backend {group} returned duplicate row: {key}")
        observed.add(key)
        job = metadata[sample_id]
        evidence = raw.get("evidence", {})
        if not isinstance(evidence, Mapping):
            raise EvaluationError(f"backend {group} row {index} evidence is not an object")
        normalized.append(
            {
                "sample_id": sample_id,
                "prompt_id": job["prompt_id"],
                "task": job["task"],
                "model": job["model"],
                "evaluator": evaluator,
                "status": str(raw.get("status", "success")),
                "reason": str(raw.get("reason", "")),
                "evidence": dict(evidence),
                "backend": group,
                "backend_version": str(raw.get("backend_version", "")),
                "artifacts": raw.get("artifacts", {}),
            }
        )
    for sample_id, evaluator in sorted(expected - observed):
        job = metadata[sample_id]
        normalized.append(
            {
                "sample_id": sample_id,
                "prompt_id": job["prompt_id"],
                "task": job["task"],
                "model": job["model"],
                "evaluator": evaluator,
                "status": "error",
                "reason": f"backend {group} returned no row",
                "evidence": {},
                "backend": group,
                "backend_version": "",
                "artifacts": {},
            }
        )
    return normalized


def _score_rows(evidence_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in evidence_rows:
        valid = False
        score: float | None = None
        reason = str(row.get("reason", ""))
        if str(row.get("status", "success")) == "success":
            try:
                result = _score(str(row["evaluator"]), row.get("evidence", {}))
                valid = bool(result.valid)
                score = float(result.score) if result.score is not None else None
                reason = result.reason or ""
            except Exception as exc:  # backend evidence must remain auditable
                reason = f"score_mapping_error: {exc}"
        scored.append(
            {
                "sample_id": row["sample_id"],
                "prompt_id": row.get("prompt_id", ""),
                "model": row["model"],
                "task": row["task"],
                "evaluator": row["evaluator"],
                "valid": str(valid).lower(),
                "score": "" if score is None else score,
                "reason": reason,
                "backend": row.get("backend", ""),
            }
        )
    return scored


def _aggregate_rows(
    scored: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str], list[ScoreRecord]] = {}
    for row in scored:
        record = ScoreRecord(
            sample_id=str(row["sample_id"]),
            model=str(row["model"]),
            task=str(row["task"]),
            evaluator=str(row["evaluator"]),
            valid=str(row["valid"]).lower() == "true",
            score=float(row["score"]) if str(row.get("score", "")).strip() else None,
        )
        cells.setdefault((record.model, record.task, record.evaluator), []).append(record)
    output: list[dict[str, Any]] = []
    for index, (key, rows) in enumerate(sorted(cells.items())):
        n_valid = sum(row.valid for row in rows)
        if n_valid:
            stats = bootstrap_cell(rows, n_resamples=resamples, seed=seed + index)
        else:
            stats = {
                "n_total": len(rows),
                "n_valid": 0,
                "valid_rate": 0.0,
                "mean": "",
                "ci_low": "",
                "ci_high": "",
                "n_resamples": resamples,
                "seed": seed + index,
            }
        output.append(dict(zip(("model", "task", "evaluator"), key)) | stats)
    return output


def evaluate_release(
    *,
    prompt_paths: Sequence[str | Path],
    backend_config_path: str | Path,
    output_dir: str | Path,
    outputs_path: str | Path | None = None,
    videos_dir: str | Path | None = None,
    model: str | None = None,
    dimensions: str | Sequence[str] = "all",
    profile_path: str | Path | None = None,
    allow_subset: bool = False,
    allow_missing: bool = False,
    allow_mock: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    check_files: bool = True,
    resamples: int = 10_000,
    seed: int = 2027,
    smoke_limit_per_dimension: int = 0,
) -> dict[str, Any]:
    if bool(outputs_path) == bool(videos_dir):
        raise EvaluationError("provide exactly one of outputs_path or videos_dir")
    full_prompts = load_prompts(prompt_paths)
    profile = load_release_profile(profile_path)
    suite_errors = validate_release_suite(full_prompts, profile, allow_subset=allow_subset)
    if suite_errors:
        raise EvaluationError("release suite validation failed:\n" + "\n".join(suite_errors))
    if smoke_limit_per_dimension < 0:
        raise EvaluationError("--smoke-limit-per-dimension must be non-negative")
    selected = parse_dimensions(dimensions)
    prompts = select_smoke_prompts(
        full_prompts,
        selected,
        smoke_limit_per_dimension,
    )
    prompt_keys = {(prompt.task, prompt.prompt_id) for prompt in prompts}

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if videos_dir:
        if not model:
            raise EvaluationError("--model is required with --videos-dir")
        all_outputs, _ = discover_video_outputs(full_prompts, videos_dir, model)
        raw_outputs = [
            row
            for row in all_outputs
            if (str(row["task"]), str(row["prompt_id"])) in prompt_keys
        ]
        missing = [
            f"{row['task']}:{row['prompt_id']}"
            for row in raw_outputs
            if str(row["status"]) != "success"
        ]
        if missing and not allow_missing:
            raise EvaluationError(
                f"{len(missing)} generated videos are missing; first: {missing[:10]}"
            )
        source_dir = Path(videos_dir).resolve()
        _write_csv(
            output_root / "generated_outputs.csv",
            raw_outputs,
            ("prompt_id", "task", "model", "video_path", "status", "error"),
        )
    else:
        source = Path(outputs_path).resolve()  # type: ignore[arg-type]
        raw_outputs = [
            row
            for row in load_output_manifest(source)
            if (str(row.get("task", "")).strip().lower(), str(row.get("prompt_id", row.get("sample_id", ""))).strip())
            in prompt_keys
        ]
        source_dir = source.parent
    outputs, inferred_model = _normalize_outputs(
        raw_outputs,
        prompts,
        source_dir=source_dir,
        model=model,
        check_files=check_files,
    )

    config = load_backend_config(backend_config_path)
    repo_root = Path(config.get("repo_root", Path(config["_source"]).parent.parent))
    if not repo_root.is_absolute():
        repo_root = (Path(config["_source"]).parent / repo_root).resolve()
    jobs, immediate = _backend_jobs(
        prompts,
        outputs,
        selected,
        repo_root=repo_root,
    )

    evidence_rows = list(immediate)
    backend_states: dict[str, Any] = {}
    run_errors: list[str] = []
    mock_groups: list[str] = []
    for group, group_jobs in jobs.items():
        if not group_jobs:
            continue
        spec = config["backends"].get(group)
        if not isinstance(spec, Mapping) or not bool(spec.get("enabled", True)):
            run_errors.append(f"backend {group} is not configured or disabled")
            for job in group_jobs:
                for evaluator in job["evaluators"]:
                    immediate_row = {
                        "sample_id": job["sample_id"],
                        "prompt_id": job["prompt_id"],
                        "task": job["task"],
                        "model": job["model"],
                        "evaluator": evaluator,
                        "status": "error",
                        "reason": f"backend {group} is not configured",
                        "evidence": {},
                        "backend": group,
                    }
                    evidence_rows.append(immediate_row)
            continue
        if bool(spec.get("mock", False)):
            mock_groups.append(group)
        try:
            raw_backend_rows, state = _run_backend(
                group=group,
                jobs=group_jobs,
                spec=spec,
                output_root=output_root,
                repo_root=repo_root,
                resume=resume,
                dry_run=dry_run,
                allow_mock=allow_mock,
            )
            backend_states[group] = state
            if not dry_run:
                evidence_rows.extend(
                    _normalize_backend_rows(group, group_jobs, raw_backend_rows)
                )
        except Exception as exc:
            run_errors.append(str(exc))
            backend_states[group] = {"status": "failed", "error": str(exc)}
            for job in group_jobs:
                for evaluator in job["evaluators"]:
                    evidence_rows.append(
                        {
                            "sample_id": job["sample_id"],
                            "prompt_id": job["prompt_id"],
                            "task": job["task"],
                            "model": job["model"],
                            "evaluator": evaluator,
                            "status": "error",
                            "reason": str(exc),
                            "evidence": {},
                            "backend": group,
                        }
                    )

    run_summary: dict[str, Any] = {
        "schema_version": 1,
        "release_profile": profile["name"],
        "model": inferred_model,
        "selected_dimensions": list(selected),
        "prompt_count": len(prompts),
        "backend_states": backend_states,
        "mock_backends": mock_groups,
        "official_scores": (
            not mock_groups
            and not dry_run
            and smoke_limit_per_dimension == 0
            and not allow_subset
            and not allow_missing
            and tuple(selected) == ALL_EVALUATORS
        ),
        "smoke_test": smoke_limit_per_dimension > 0,
        "smoke_limit_per_dimension": smoke_limit_per_dimension,
        "status": "planned" if dry_run else ("partial" if run_errors else "success"),
        "errors": run_errors,
    }
    if dry_run:
        (output_root / "run.json").write_text(
            json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return run_summary

    evidence_rows.sort(key=lambda row: (str(row["task"]), str(row["sample_id"]), str(row["evaluator"])))
    _write_jsonl(output_root / "evidence.jsonl", evidence_rows)
    scored = _score_rows(evidence_rows)
    scored.sort(key=lambda row: (str(row["task"]), str(row["sample_id"]), str(row["evaluator"])))
    score_fields = (
        "sample_id",
        "prompt_id",
        "model",
        "task",
        "evaluator",
        "valid",
        "score",
        "reason",
        "backend",
    )
    _write_csv(output_root / "scores.csv", scored, score_fields)
    failures = [row for row in scored if row["valid"] != "true"]
    _write_jsonl(output_root / "failures.jsonl", failures)
    summary_rows = _aggregate_rows(scored, resamples=resamples, seed=seed)
    summary_fields = (
        "model",
        "task",
        "evaluator",
        "n_total",
        "n_valid",
        "valid_rate",
        "mean",
        "ci_low",
        "ci_high",
        "n_resamples",
        "seed",
    )
    _write_csv(output_root / "summary.csv", summary_rows, summary_fields)
    run_summary.update(
        {
            "score_rows": len(scored),
            "valid_score_rows": sum(row["valid"] == "true" for row in scored),
            "invalid_score_rows": len(failures),
            "outputs": {
                "evidence": str(output_root / "evidence.jsonl"),
                "scores": str(output_root / "scores.csv"),
                "summary": str(output_root / "summary.csv"),
                "failures": str(output_root / "failures.jsonl"),
            },
        }
    )
    (output_root / "run.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_summary
