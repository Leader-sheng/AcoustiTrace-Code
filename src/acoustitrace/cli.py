"""Command-line interface for paper-aligned scoring and aggregation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .aggregate import ScoreRecord, bootstrap_cell
from .contract import load_contract, validate_contract
from .end2end import EvaluationError, evaluate_release
from .manifests import (
    generation_request,
    load_output_manifest,
    load_prompt_manifest,
    validate_output_manifest,
    validate_prompt_manifest,
)
from .scores import (
    causality_violation,
    impact_decay,
    log_attack_time,
    motion_loudness,
    native_receiver_score,
    range_attenuation,
    rt60_consistency,
)
from .release import load_release_profile, validate_release_suite


def _default_contract() -> Path:
    return Path(__file__).with_name("paper_contract.json")


def _score(evaluator: str, evidence: dict[str, Any]):
    if evaluator == "motion_loudness":
        return motion_loudness(evidence["level_pairs_db"])
    if evaluator == "log_attack_time":
        return log_attack_time(
            evidence["generated_attack_seconds"], evidence["reference_attack_seconds"]
        )
    if evaluator == "impact_decay":
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
    raise ValueError(f"unknown evaluator: {evaluator}")


def cmd_score(args: argparse.Namespace) -> int:
    result = _score(args.evaluator, json.loads(args.evidence))
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0 if result.valid else 2


def cmd_score_batch(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    with Path(args.input).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            result = _score(item["evaluator"], item["evidence"])
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "model": item["model"],
                    "task": item["task"],
                    "evaluator": item["evaluator"],
                    "valid": str(result.valid).lower(),
                    "score": "" if result.score is None else result.score,
                    "reason": result.reason or "",
                    "source_line": line_number,
                }
            )
    if not rows:
        raise ValueError("input JSONL contains no evidence rows")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def cmd_aggregate(args: argparse.Namespace) -> int:
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    records = [
        ScoreRecord(
            sample_id=row["sample_id"],
            model=row["model"],
            task=row["task"],
            evaluator=row["evaluator"],
            valid=_parse_bool(row["valid"]),
            score=float(row["score"]) if row.get("score", "").strip() else None,
        )
        for row in raw
    ]
    cells: dict[tuple[str, str, str], list[ScoreRecord]] = {}
    for row in records:
        cells.setdefault((row.model, row.task, row.evaluator), []).append(row)
    output: list[dict[str, Any]] = []
    for index, (key, rows) in enumerate(sorted(cells.items())):
        stats = bootstrap_cell(
            rows, n_resamples=args.resamples, seed=args.seed + index
        )
        output.append(dict(zip(("model", "task", "evaluator"), key)) | stats)
    if not output:
        raise ValueError("input CSV contains no score rows")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return 0


def cmd_validate_contract(args: argparse.Namespace) -> int:
    path = Path(args.contract) if args.contract else _default_contract()
    errors = validate_contract(load_contract(path))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {path}")
    return 0


def _print_errors(errors: list[str]) -> int:
    if not errors:
        return 0
    for error in errors:
        print(f"ERROR: {error}")
    return 1


def cmd_validate_prompts(args: argparse.Namespace) -> int:
    contract_path = Path(args.contract) if args.contract else _default_contract()
    records = load_prompt_manifest(args.input)
    errors = validate_prompt_manifest(
        records, load_contract(contract_path), allow_subset=args.allow_subset
    )
    if errors:
        return _print_errors(errors)
    by_task = {
        task: sum(record.task == task for record in records) for task in ("t2av", "i2av")
    }
    print(f"OK: {args.input} (t2av={by_task['t2av']}, i2av={by_task['i2av']})")
    return 0


def cmd_validate_outputs(args: argparse.Namespace) -> int:
    prompts = load_prompt_manifest(args.prompts)
    outputs = load_output_manifest(args.input)
    errors = validate_output_manifest(outputs, prompts, check_files=args.check_files)
    if errors:
        return _print_errors(errors)
    print(f"OK: {args.input} ({len(outputs)} outputs)")
    return 0


def cmd_export_generation_requests(args: argparse.Namespace) -> int:
    records = load_prompt_manifest(args.input)
    if args.task:
        records = [record for record in records if record.task == args.task]
    if not records:
        raise ValueError("no prompts match the requested task")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda row: (row.task, row.prompt_id)):
            handle.write(json.dumps(generation_request(record), ensure_ascii=False) + "\n")
    print(f"OK: wrote {len(records)} requests to {destination}")
    return 0


def cmd_validate_release_suite(args: argparse.Namespace) -> int:
    records = []
    for path in args.prompts:
        records.extend(load_prompt_manifest(path))
    errors = validate_release_suite(
        records,
        load_release_profile(args.profile),
        allow_subset=args.allow_subset,
    )
    if errors:
        return _print_errors(errors)
    by_task = {
        task: sum(record.task == task for record in records) for task in ("t2av", "i2av")
    }
    print(
        f"OK: public release suite (t2av={by_task['t2av']}, "
        f"i2av_lat={by_task['i2av']})"
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        result = evaluate_release(
            prompt_paths=args.prompts,
            backend_config_path=args.backend_config,
            output_dir=args.output_dir,
            outputs_path=args.outputs,
            videos_dir=args.videos_dir,
            model=args.model,
            dimensions=args.dimensions,
            profile_path=args.profile,
            allow_subset=args.allow_subset,
            allow_missing=args.allow_missing,
            allow_mock=args.allow_mock,
            resume=args.resume,
            dry_run=args.dry_run,
            check_files=not args.no_check_files,
            resamples=args.resamples,
            seed=args.seed,
            smoke_limit_per_dimension=args.smoke_limit_per_dimension,
        )
    except (EvaluationError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if result["status"] == "partial" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acoustitrace")
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="map structured evidence to a score")
    score.add_argument("evaluator")
    score.add_argument("evidence", help="JSON object")
    score.set_defaults(func=cmd_score)
    score_batch = subparsers.add_parser(
        "score-batch", help="score structured-evidence JSONL"
    )
    score_batch.add_argument("input")
    score_batch.add_argument("output")
    score_batch.set_defaults(func=cmd_score_batch)
    aggregate = subparsers.add_parser(
        "aggregate", help="aggregate a validity-aware score CSV"
    )
    aggregate.add_argument("input")
    aggregate.add_argument("output")
    aggregate.add_argument("--resamples", type=int, default=10_000)
    aggregate.add_argument("--seed", type=int, default=2027)
    aggregate.set_defaults(func=cmd_aggregate)
    contract = subparsers.add_parser("validate-contract")
    contract.add_argument("--contract")
    contract.set_defaults(func=cmd_validate_contract)
    prompt_manifest = subparsers.add_parser(
        "validate-prompts", help="validate frozen prompt counts and pool overlaps"
    )
    prompt_manifest.add_argument("input")
    prompt_manifest.add_argument("--contract")
    prompt_manifest.add_argument(
        "--allow-subset", action="store_true", help="skip exact final-count checks"
    )
    prompt_manifest.set_defaults(func=cmd_validate_prompts)
    output_manifest = subparsers.add_parser(
        "validate-outputs", help="check one generated output row per frozen prompt"
    )
    output_manifest.add_argument("input")
    output_manifest.add_argument("--prompts", required=True)
    output_manifest.add_argument("--check-files", action="store_true")
    output_manifest.set_defaults(func=cmd_validate_outputs)
    export = subparsers.add_parser(
        "export-generation-requests",
        help="export a model-neutral JSONL request file from frozen prompts",
    )
    export.add_argument("input")
    export.add_argument("output")
    export.add_argument("--task", choices=("t2av", "i2av"))
    export.set_defaults(func=cmd_export_generation_requests)
    release_suite = subparsers.add_parser(
        "validate-release-suite",
        help="validate the public 605 T2AV + 143 I2AV-LAT suite",
    )
    release_suite.add_argument("prompts", nargs="+")
    release_suite.add_argument("--profile")
    release_suite.add_argument("--allow-subset", action="store_true")
    release_suite.set_defaults(func=cmd_validate_release_suite)
    evaluate = subparsers.add_parser(
        "evaluate",
        help="run the public release suite from generated videos to summary scores",
    )
    evaluate.add_argument("--prompts", nargs="+", required=True)
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--videos-dir")
    source.add_argument("--outputs", help="generated-output CSV/JSONL")
    evaluate.add_argument("--model")
    evaluate.add_argument("--backend-config", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--profile")
    evaluate.add_argument("--dimensions", default="all")
    evaluate.add_argument("--allow-subset", action="store_true")
    evaluate.add_argument("--allow-missing", action="store_true")
    evaluate.add_argument("--allow-mock", action="store_true")
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--no-check-files", action="store_true")
    evaluate.add_argument("--resamples", type=int, default=10_000)
    evaluate.add_argument("--seed", type=int, default=2027)
    evaluate.add_argument(
        "--smoke-limit-per-dimension",
        type=int,
        default=0,
        metavar="N",
        help="deterministically run at most N assignments per selected dimension",
    )
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
