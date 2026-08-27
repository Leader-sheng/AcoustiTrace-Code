from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from time_causality_eval.scripts.tc_common import (
    collect_numeric,
    dataframe_to_records,
    ensure_dir,
    file_exists,
    first_existing,
    groupby_rows,
    kendall_spearman_from_orders,
    label_category,
    label_compatibility,
    load_label_mapping,
    load_yaml,
    mean_median_std,
    normalize_label,
    read_csv_df,
    read_json,
    save_histogram_png,
    save_scatter_png,
    safe_bool,
    safe_float,
    safe_int,
    sequence_edit_distance,
    sort_df,
    write_csv_df,
    write_json,
)

LABEL_MAPPING = None


def _base_paths(cfg: dict, project_root: str | Path | None = None, output_root: str | Path | None = None):
    pr = Path(project_root or cfg["input"]["project_root"]).resolve()
    out = Path(output_root or cfg["output"]["output_root"]).resolve()
    return pr, out


def _canonical_path(project_root: Path, rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _find_file_candidates(project_root: Path, rel_pattern: str) -> List[Path]:
    return sorted(project_root.glob(rel_pattern))


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _pick_priority_file(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def inspect_existing_outputs(cfg: dict, output_root: str | Path | None = None) -> dict:
    project_root, out_root = _base_paths(cfg, output_root=output_root)
    inspection_dir = ensure_dir(out_root)
    candidates = {
        "visual_event_files": [
            project_root / "outputs_full" / "ov_avel" / "visual_events.csv",
            project_root / "outputs" / "ov_avel" / "visual_events.csv",
            project_root / "tmp_small_out" / "ov_avel" / "visual_events.csv",
        ],
        "audio_event_files": [
            project_root / "outputs_full" / "flexsed" / "audio_events.csv",
            project_root / "outputs" / "flexsed" / "audio_events.csv",
            project_root / "tmp_small_out" / "flexsed" / "audio_events.csv",
        ],
        "matched_event_files": [
            project_root / "outputs_full" / "matched_events" / "matched_av_events.csv",
            project_root / "outputs" / "matched_events" / "matched_av_events.csv",
            project_root / "tmp_small_out" / "matched_events" / "matched_av_events.csv",
        ],
        "unmatched_visual_event_files": [
            project_root / "outputs_full" / "matched_events" / "unmatched_visual_events.csv",
            project_root / "outputs" / "matched_events" / "unmatched_visual_events.csv",
            project_root / "tmp_small_out" / "matched_events" / "unmatched_visual_events.csv",
        ],
        "unmatched_audio_event_files": [
            project_root / "outputs_full" / "matched_events" / "unmatched_audio_events.csv",
            project_root / "outputs" / "matched_events" / "unmatched_audio_events.csv",
            project_root / "tmp_small_out" / "matched_events" / "unmatched_audio_events.csv",
        ],
        "usable_ab_manifest_files": [
            project_root.parent.parent / "soundphysics_youtube_crawler" / "data" / "source_causality" / "new_batch_20260515" / "review_package" / "usable_ab_manifest.csv",
        ],
        "review_decay_manifests": [
            project_root / "outputs_full" / "metrics" / "review_decay_r2_ge_0p9" / "selected_decay_r2_ge_0p9_manifest.csv",
            project_root / "outputs" / "metrics" / "review_decay_r2_ge_0p9" / "selected_decay_r2_ge_0p9_manifest.csv",
            project_root / "tmp_small_out" / "metrics" / "review_decay_r2_ge_0p9" / "selected_decay_r2_ge_0p9_manifest.csv",
        ],
        "review_decay_audio_errors": [
            project_root / "outputs_full" / "metrics" / "review_decay_r2_ge_0p9" / "audio_errors.csv",
            project_root / "outputs" / "metrics" / "review_decay_r2_ge_0p9" / "audio_errors.csv",
            project_root / "tmp_small_out" / "metrics" / "review_decay_r2_ge_0p9" / "audio_errors.csv",
        ],
    }

    report = {
        "project_root": str(project_root),
        "output_root": str(out_root),
        "found_files": {},
        "recommended_inputs": {},
        "field_inventory": {},
        "missing_fields": {},
        "can_directly_compute_time_causality_metrics": False,
        "notes": [],
    }

    visual_df = pd.DataFrame()
    audio_df = pd.DataFrame()
    matched_df = pd.DataFrame()
    unmatched_v_df = pd.DataFrame()
    unmatched_a_df = pd.DataFrame()

    for key, paths in candidates.items():
        found = [str(p) for p in paths if p.exists()]
        report["found_files"][key] = found
        if found:
            report["recommended_inputs"][key] = found[0]
        if not found:
            report["field_inventory"][key] = []
            continue
        sample = pd.read_csv(found[0], nrows=5)
        report["field_inventory"][key] = list(sample.columns)

    if report["found_files"].get("visual_event_files"):
        visual_df = pd.read_csv(report["found_files"]["visual_event_files"][0])
    if report["found_files"].get("audio_event_files"):
        audio_df = pd.read_csv(report["found_files"]["audio_event_files"][0])
    if report["found_files"].get("matched_event_files"):
        matched_df = pd.read_csv(report["found_files"]["matched_event_files"][0])
    if report["found_files"].get("unmatched_visual_event_files"):
        unmatched_v_df = pd.read_csv(report["found_files"]["unmatched_visual_event_files"][0])
    if report["found_files"].get("unmatched_audio_event_files"):
        unmatched_a_df = pd.read_csv(report["found_files"]["unmatched_audio_event_files"][0])

    req_visual = {"video_id", "visual_event_id", "visual_label", "start_sec", "end_sec", "peak_sec", "confidence"}
    req_audio = {"video_id", "audio_event_id", "audio_label", "start_sec", "end_sec", "peak_sec", "confidence"}
    req_matched = {
        "video_id",
        "visual_event_id",
        "audio_event_id",
        "visual_label",
        "audio_label",
        "visual_start_sec",
        "visual_end_sec",
        "visual_peak_sec",
        "audio_start_sec",
        "audio_end_sec",
        "audio_peak_sec",
        "av_offset_sec",
        "match_confidence",
    }
    vis_missing = sorted(req_visual - set(visual_df.columns))
    aud_missing = sorted(req_audio - set(audio_df.columns))
    mat_missing = sorted(req_matched - set(matched_df.columns))
    report["missing_fields"] = {
        "visual_events": vis_missing,
        "audio_events": aud_missing,
        "matched_events": mat_missing,
    }

    report["can_directly_compute_time_causality_metrics"] = (
        len(vis_missing) == 0
        and len(aud_missing) == 0
        and len(mat_missing) == 0
        and not matched_df.empty
        and not visual_df.empty
        and not audio_df.empty
    )
    if vis_missing or aud_missing or mat_missing:
        report["notes"].append("Raw event CSVs need adapter fields before direct causal metric computation.")
    if report["found_files"].get("review_decay_audio_errors"):
        report["notes"].append("Audio extraction failures are available for review_decay_r2_ge_0p9.")

    write_json(inspection_dir / "inspection_report.json", report)
    return report


def build_time_causality_manifest(
    cfg: dict,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    limit: int | None = None,
    skip_existing: bool = False,
) -> pd.DataFrame:
    project_root, out_root = _base_paths(cfg, project_root, output_root)
    manifest_dir = ensure_dir(out_root / "manifests")
    out_csv = manifest_dir / "time_causality_manifest.csv"
    if skip_existing and out_csv.exists():
        return pd.read_csv(out_csv)

    base_manifest_path = _canonical_path(
        project_root.parent.parent, "soundphysics_youtube_crawler/data/source_causality/new_batch_20260515/review_package/usable_ab_manifest.csv"
    )
    if not base_manifest_path.exists():
        raise FileNotFoundError(f"Missing usable_ab_manifest.csv: {base_manifest_path}")
    base = pd.read_csv(base_manifest_path)
    if "chunk_id" not in base.columns:
        raise ValueError("usable_ab_manifest.csv must contain chunk_id")

    selected_decay_path = project_root / "outputs_full" / "metrics" / "review_decay_r2_ge_0p9" / "selected_decay_r2_ge_0p9_manifest.csv"
    audio_errors_path = project_root / "outputs_full" / "metrics" / "review_decay_r2_ge_0p9" / "audio_errors.csv"
    matched_path = project_root / "outputs_full" / "matched_events" / "matched_av_events.csv"
    visual_path = project_root / "outputs_full" / "ov_avel" / "visual_events.csv"
    audio_path = project_root / "outputs_full" / "flexsed" / "audio_events.csv"

    selected_decay = read_csv_df(selected_decay_path) if selected_decay_path.exists() else pd.DataFrame()
    audio_errors = read_csv_df(audio_errors_path) if audio_errors_path.exists() else pd.DataFrame()
    matched = read_csv_df(matched_path) if matched_path.exists() else pd.DataFrame()
    visual = read_csv_df(visual_path) if visual_path.exists() else pd.DataFrame()
    audio = read_csv_df(audio_path) if audio_path.exists() else pd.DataFrame()

    visual_samples = set(str(v) for v in visual["video_id"].dropna().astype(str).tolist()) if not visual.empty and "video_id" in visual.columns else set()
    audio_samples = set(str(v) for v in audio["video_id"].dropna().astype(str).tolist()) if not audio.empty and "video_id" in audio.columns else set()
    matched_samples = set(str(v) for v in matched["video_id"].dropna().astype(str).tolist()) if not matched.empty and "video_id" in matched.columns else set()
    audio_error_samples = set(str(v) for v in audio_errors["video_id"].dropna().astype(str).tolist()) if not audio_errors.empty and "video_id" in audio_errors.columns else set()

    decay_group = groupby_rows(selected_decay, "video_id") if not selected_decay.empty and "video_id" in selected_decay.columns else {}

    base = base.copy()
    if "time_causality_usable" in base.columns:
        base["time_causality_usable"] = base["time_causality_usable"].apply(safe_bool)
    if "chunk_id" in base.columns:
        base = base.sort_values(by=["chunk_id"], kind="stable").reset_index(drop=True)
    if limit is not None and limit > 0:
        base = base.head(limit).copy()

    rows = []
    for _, r in base.iterrows():
        chunk_id = str(r.get("chunk_id", "")).strip()
        video_id = str(r.get("video_id", "")).strip()
        video_path = str(r.get("video_chunk_path", "")).strip()
        audio_chunk_path = str(r.get("audio_chunk_path", "")).strip()
        video_exists = file_exists(video_path)
        audio_exists = file_exists(audio_chunk_path)
        event_rows = decay_group.get(video_id, pd.DataFrame()) if video_id in decay_group else pd.DataFrame()
        if event_rows.empty and chunk_id in decay_group:
            event_rows = decay_group.get(chunk_id, pd.DataFrame())
        matched_event_rows = matched[matched["video_id"].astype(str) == chunk_id] if not matched.empty and "video_id" in matched.columns else pd.DataFrame()
        visual_event_rows = visual[visual["video_id"].astype(str) == chunk_id] if not visual.empty and "video_id" in visual.columns else pd.DataFrame()
        audio_event_rows = audio[audio["video_id"].astype(str) == chunk_id] if not audio.empty and "video_id" in audio.columns else pd.DataFrame()

        has_visual = chunk_id in visual_samples
        has_audio = chunk_id in audio_samples
        has_matches = chunk_id in matched_samples
        has_audio_error = chunk_id in audio_error_samples
        if not bool(r.get("time_causality_usable", True)):
            status = "ineligible"
            skip_reason = "time_causality_usable_false"
        elif not video_exists:
            status = "missing_video"
            skip_reason = "video_file_missing"
        elif (not audio_exists) and (not has_audio_error):
            status = "missing_audio"
            skip_reason = "audio_file_missing_and_not_recoverable"
        elif has_audio_error:
            status = "audio_failed"
            skip_reason = "audio_extraction_failed"
        elif (not has_visual) or (not has_audio):
            status = "needs_detection"
            skip_reason = "missing_visual_or_audio_detection"
        elif not has_matches:
            status = "no_match"
            skip_reason = "no_matched_av_events"
        else:
            status = "ok"
            skip_reason = ""

        first_event = event_rows.iloc[0].to_dict() if not event_rows.empty else {}
        row = {
            "sample_id": chunk_id,
            "video_id": video_id,
            "chunk_id": chunk_id,
            "video_path": video_path,
            "event_clip_path": str(first_event.get("review_event_clip_path", "")) if first_event else "",
            "event_audio_path": str(first_event.get("review_event_audio_path", "")) if first_event else "",
            "event_clip_paths": ";".join(sorted(set(str(x) for x in event_rows.get("review_event_clip_path", pd.Series(dtype=str)).dropna().astype(str).tolist()))) if not event_rows.empty and "review_event_clip_path" in event_rows.columns else "",
            "event_audio_paths": ";".join(sorted(set(str(x) for x in event_rows.get("review_event_audio_path", pd.Series(dtype=str)).dropna().astype(str).tolist()))) if not event_rows.empty and "review_event_audio_path" in event_rows.columns else "",
            "seed_category": str(r.get("seed_category", "")),
            "source_type": str(r.get("primary_category", r.get("detected_event_type", ""))),
            "title": str(r.get("title", "")),
            "visual_event_file": str(project_root / "outputs_full" / "ov_avel" / "visual_events.csv") if has_visual else "",
            "audio_event_file": str(project_root / "outputs_full" / "flexsed" / "audio_events.csv") if has_audio else "",
            "existing_matched_event_file": str(project_root / "outputs_full" / "matched_events" / "matched_av_events.csv") if has_matches else "",
            "has_visual_events": bool(has_visual),
            "has_audio_events": bool(has_audio),
            "has_existing_matches": bool(has_matches),
            "status": status,
            "skip_reason": skip_reason,
            "review_event_clip_path": str(first_event.get("review_event_clip_path", "")) if first_event else "",
            "review_event_audio_path": str(first_event.get("review_event_audio_path", "")) if first_event else "",
            "review_full_video_path": str(first_event.get("review_full_video_path", "")) if first_event else "",
            "review_start_sec": first_event.get("review_start_sec", ""),
            "review_end_sec": first_event.get("review_end_sec", ""),
            "review_status": first_event.get("review_status", ""),
            "review_error_message": first_event.get("review_error_message", ""),
            "audio_error_message": "" if audio_errors.empty or not has_audio_error else str(audio_errors[audio_errors["video_id"].astype(str) == chunk_id].iloc[0].get("review_error_message", "")),
            "base_manifest_path": str(base_manifest_path),
            "selected_decay_manifest_path": str(selected_decay_path) if selected_decay_path.exists() else "",
        }
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest = sort_df(manifest, ["sample_id"])
    expected_cols = [
        "sample_id",
        "video_id",
        "chunk_id",
        "video_path",
        "event_clip_path",
        "event_audio_path",
        "event_clip_paths",
        "event_audio_paths",
        "seed_category",
        "source_type",
        "title",
        "visual_event_file",
        "audio_event_file",
        "existing_matched_event_file",
        "has_visual_events",
        "has_audio_events",
        "has_existing_matches",
        "status",
        "skip_reason",
        "review_event_clip_path",
        "review_event_audio_path",
        "review_full_video_path",
        "review_start_sec",
        "review_end_sec",
        "review_status",
        "review_error_message",
        "audio_error_message",
        "base_manifest_path",
        "selected_decay_manifest_path",
    ]
    manifest = manifest.reindex(columns=expected_cols)
    write_csv_df(out_csv, manifest)
    return manifest


def _standardize_events(
    df: pd.DataFrame,
    event_type: str,
    label_col: str,
    id_col: str,
    label_mapping: dict,
    time_col: str = "peak_sec",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = []
    for idx, r in df.iterrows():
        sample_id = str(r.get("video_id", "")).strip()
        label_raw = str(r.get(label_col, "")).strip()
        if event_type == "visual":
            start_sec = safe_float(r.get("start_sec"))
            end_sec = safe_float(r.get("end_sec"))
            peak_sec = safe_float(r.get("peak_sec"), (start_sec + end_sec) / 2.0 if not math.isnan(start_sec) and not math.isnan(end_sec) else math.nan)
            event_time_sec = peak_sec if not math.isnan(peak_sec) else start_sec
            event_time_source = "peak_sec" if not math.isnan(peak_sec) else "start_sec"
            onset_sec = math.nan
        else:
            start_sec = safe_float(r.get("start_sec"))
            end_sec = safe_float(r.get("end_sec"))
            peak_sec = safe_float(r.get("peak_sec"), start_sec)
            onset_sec = start_sec
            event_time_sec = onset_sec
            event_time_source = "start_sec"
        confidence = safe_float(r.get("confidence"))
        label_norm = normalize_label(label_raw)
        out.append(
            {
                "sample_id": sample_id,
                "video_id": sample_id,
                "event_id": str(r.get(id_col, f"{event_type[0]}{idx:04d}")),
                "event_type": event_type,
                "label_raw": label_raw,
                "label_norm": label_norm,
                "label_category": label_category(label_raw, label_mapping),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "event_time_sec": event_time_sec,
                "event_time_source": event_time_source,
                "onset_sec": onset_sec if event_type == "audio" else math.nan,
                "confidence": confidence,
                "bbox": None,
                "extra": {
                    "video_path": str(r.get("video_path", "")),
                    "raw_score": safe_float(r.get("raw_score")),
                    "peak_sec": peak_sec,
                    "source_row_index": int(idx),
                },
            }
        )
    return pd.DataFrame(out)


def parse_visual_audio_events(
    cfg: dict,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    skip_existing: bool = False,
) -> dict:
    project_root, out_root = _base_paths(cfg, project_root, output_root)
    parsed_dir = ensure_dir(out_root / "parsed_events")
    visual_out = parsed_dir / "visual_events.jsonl"
    audio_out = parsed_dir / "audio_events.jsonl"
    status_out = parsed_dir / "parse_status.csv"
    if skip_existing and visual_out.exists() and audio_out.exists() and status_out.exists():
        return {
            "visual_events": str(visual_out),
            "audio_events": str(audio_out),
            "parse_status": str(status_out),
        }

    visual_path = _pick_priority_file(
        [
            project_root / "outputs_full" / "ov_avel" / "visual_events.csv",
            project_root / "outputs" / "ov_avel" / "visual_events.csv",
            project_root / "tmp_small_out" / "ov_avel" / "visual_events.csv",
        ]
    )
    audio_path = _pick_priority_file(
        [
            project_root / "outputs_full" / "flexsed" / "audio_events.csv",
            project_root / "outputs" / "flexsed" / "audio_events.csv",
            project_root / "tmp_small_out" / "flexsed" / "audio_events.csv",
        ]
    )
    if visual_path is None or audio_path is None:
        raise FileNotFoundError("Could not find both visual_events.csv and audio_events.csv")

    visual_df = pd.read_csv(visual_path)
    audio_df = pd.read_csv(audio_path)
    label_mapping = load_label_mapping(project_root / "time_causality_eval" / "configs" / "event_label_mapping.yaml")
    v_std = _standardize_events(visual_df, "visual", "visual_label", "visual_event_id", label_mapping)
    a_std = _standardize_events(audio_df, "audio", "audio_label", "audio_event_id", label_mapping)
    expected_cols = [
        "sample_id",
        "video_id",
        "event_id",
        "event_type",
        "label_raw",
        "label_norm",
        "label_category",
        "start_sec",
        "end_sec",
        "event_time_sec",
        "event_time_source",
        "onset_sec",
        "confidence",
        "bbox",
        "extra",
    ]
    v_std = v_std.reindex(columns=expected_cols)
    a_std = a_std.reindex(columns=expected_cols)

    with open(visual_out, "w", encoding="utf-8") as f:
        for rec in v_std.to_dict(orient="records"):
            import json as _json

            f.write(_json.dumps(rec, ensure_ascii=False, default=str))
            f.write("\n")
    with open(audio_out, "w", encoding="utf-8") as f:
        for rec in a_std.to_dict(orient="records"):
            import json as _json

            f.write(_json.dumps(rec, ensure_ascii=False, default=str))
            f.write("\n")

    status = []
    for sample_id, g in v_std.groupby("sample_id", dropna=False):
        a_g = a_std[a_std["sample_id"] == sample_id]
        status.append(
            {
                "sample_id": sample_id,
                "video_id": sample_id,
                "visual_event_count": int(len(g)),
                "audio_event_count": int(len(a_g)),
                "visual_source_file": str(visual_path),
                "audio_source_file": str(audio_path),
                "parse_status": "ok" if len(g) > 0 and len(a_g) >= 0 else "partial",
            }
        )
    for sample_id, g in a_std.groupby("sample_id", dropna=False):
        if sample_id not in {x["sample_id"] for x in status}:
            status.append(
                {
                    "sample_id": sample_id,
                    "video_id": sample_id,
                    "visual_event_count": 0,
                    "audio_event_count": int(len(g)),
                    "visual_source_file": str(visual_path),
                    "audio_source_file": str(audio_path),
                    "parse_status": "ok",
                }
            )
    status_df = pd.DataFrame(status)
    if not status_df.empty:
        status_df = status_df.sort_values(by=["sample_id"], kind="stable")
    status_df = status_df.reindex(
        columns=[
            "sample_id",
            "video_id",
            "visual_event_count",
            "audio_event_count",
            "visual_source_file",
            "audio_source_file",
            "parse_status",
        ]
    )
    write_csv_df(status_out, status_df)
    return {
        "visual_events": str(visual_out),
        "audio_events": str(audio_out),
        "parse_status": str(status_out),
    }


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import json as _json

            rows.append(_json.loads(line))
    return pd.DataFrame(rows)


def write_jsonl_records(path: str | Path, rows: Sequence[dict]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False, default=str))
            f.write("\n")


def read_jsonl_records(path: str | Path) -> pd.DataFrame:
    return _read_jsonl(Path(path))


def _cluster_key_time(row: pd.Series, cluster_type: str, rep_mode: str) -> float:
    if cluster_type == "visual":
        if rep_mode == "start":
            return safe_float(row.get("start_sec"))
        return safe_float(row.get("event_time_sec"), safe_float(row.get("start_sec")))
    if rep_mode == "onset":
        return safe_float(row.get("onset_sec"), safe_float(row.get("start_sec")))
    return safe_float(row.get("event_time_sec"), safe_float(row.get("start_sec")))


def _cluster_events_for_type(
    df: pd.DataFrame,
    cluster_type: str,
    gap_sec: float,
    rep_mode: str,
    cfg: dict,
    label_mapping: dict,
) -> Tuple[pd.DataFrame, List[dict]]:
    if df.empty:
        return pd.DataFrame(), []
    min_dur = float(cfg["event_clustering"]["min_cluster_duration_sec"])
    max_dur = float(cfg["event_clustering"]["max_cluster_duration_sec"])
    merge_overlapping = bool(cfg["event_clustering"].get("merge_overlapping_events", True))
    merge_adjacent = bool(cfg["event_clustering"].get("merge_adjacent_events", True))

    grouped_rows = []
    records = []
    cluster_prefix = "vc" if cluster_type == "visual" else "ac"
    for sample_id, g in df.groupby("sample_id", dropna=False):
        g = g.sort_values(by=["start_sec", "end_sec", "confidence"], ascending=[True, True, False], kind="stable").reset_index(drop=True)
        current = []
        cluster_idx = 0

        def flush_cluster(rows: List[pd.Series]):
            nonlocal cluster_idx
            if not rows:
                return
            start_sec = min(safe_float(r["start_sec"]) for r in rows)
            end_sec = max(safe_float(r["end_sec"]) for r in rows)
            duration = max(0.0, end_sec - start_sec)
            raw_labels = [str(r.get("label_raw", "")) for r in rows]
            norm_labels = [normalize_label(r.get("label_raw", "")) for r in rows]
            confs = [safe_float(r.get("confidence")) for r in rows]
            label_counts = Counter(raw_labels)
            dominant_label = label_counts.most_common(1)[0][0] if label_counts else "unknown"
            rep_candidates = [(_cluster_key_time(r, cluster_type, rep_mode), i, r) for i, r in enumerate(rows)]
            if cluster_type == "visual" and rep_mode in {"confidence_peak", "peak", "confidence"}:
                rep_candidates.sort(key=lambda x: (-safe_float(x[2].get("confidence")), x[0], x[1]))
            else:
                rep_candidates.sort(key=lambda x: (x[0], -safe_float(x[2].get("confidence"))))
            representative_time_sec = rep_candidates[0][0] if rep_candidates else start_sec
            cluster_status = "ok"
            if duration < min_dur:
                cluster_status = "short_cluster"
            elif duration > max_dur:
                cluster_status = "long_cluster"
            cluster_id = f"{cluster_prefix}_{cluster_idx:04d}"
            cluster_idx += 1
            rec = {
                "sample_id": str(sample_id),
                "video_id": str(sample_id),
                "cluster_id": cluster_id,
                "cluster_type": cluster_type,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "representative_time_sec": float(representative_time_sec),
                "duration_sec": float(duration),
                "dominant_label": str(dominant_label),
                "labels_raw": sorted(set(raw_labels)),
                "labels_norm": sorted(set(norm_labels)),
                "max_confidence": float(np.nanmax(confs)) if confs else math.nan,
                "mean_confidence": float(np.nanmean(confs)) if confs else math.nan,
                "num_raw_events": int(len(rows)),
                "raw_event_ids": [str(r.get("event_id", "")) for r in rows],
                "raw_event_times_sec": [float(_cluster_key_time(r, cluster_type, rep_mode)) for r in rows],
                "cluster_status": cluster_status,
                "cluster_time_source": rep_mode,
                "extra": {
                    "raw_event_labels": raw_labels,
                    "raw_event_confidences": confs,
                    "raw_rows": [r.to_dict() for r in rows],
                },
            }
            records.append(rec)
            grouped_rows.append(rec)

        for _, row in g.iterrows():
            if not current:
                current = [row]
                continue
            prev_end = max(safe_float(r["end_sec"]) for r in current)
            cur_start = safe_float(row["start_sec"])
            gap = cur_start - prev_end
            should_merge = False
            if merge_overlapping and cur_start <= prev_end:
                should_merge = True
            if merge_adjacent and gap <= gap_sec:
                should_merge = True
            if should_merge:
                current.append(row)
            else:
                flush_cluster(current)
                current = [row]
        flush_cluster(current)

    cluster_df = pd.DataFrame(grouped_rows)
    if not cluster_df.empty:
        cluster_df = cluster_df.sort_values(by=["sample_id", "cluster_id"], kind="stable").reset_index(drop=True)
    return cluster_df, records


def cluster_events(
    cfg: dict,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    skip_existing: bool = False,
) -> dict:
    project_root, out_root = _base_paths(cfg, project_root, output_root)
    parsed_dir = ensure_dir(out_root / "parsed_events")
    v_out = parsed_dir / "visual_event_clusters.jsonl"
    a_out = parsed_dir / "audio_event_clusters.jsonl"
    summary_csv = parsed_dir / "event_cluster_summary.csv"
    if skip_existing and v_out.exists() and a_out.exists() and summary_csv.exists():
        return {"visual_clusters": str(v_out), "audio_clusters": str(a_out), "summary": str(summary_csv)}

    v_path = parsed_dir / "visual_events.jsonl"
    a_path = parsed_dir / "audio_events.jsonl"
    if not v_path.exists() or not a_path.exists():
        parse_visual_audio_events(cfg, project_root, out_root, skip_existing=skip_existing)
    v_df = read_jsonl_records(v_path)
    a_df = read_jsonl_records(a_path)
    if v_df.empty or a_df.empty:
        raise RuntimeError("Parsed events empty; cannot cluster.")

    label_mapping = load_label_mapping(project_root / "time_causality_eval" / "configs" / "event_label_mapping.yaml")
    ev_cfg = cfg["event_clustering"]
    v_clusters, v_records = _cluster_events_for_type(
        v_df,
        "visual",
        float(ev_cfg["visual_cluster_gap_sec"]),
        str(ev_cfg["visual_representative_time"]),
        cfg,
        label_mapping,
    )
    a_clusters, a_records = _cluster_events_for_type(
        a_df,
        "audio",
        float(ev_cfg["audio_cluster_gap_sec"]),
        str(ev_cfg["audio_representative_time"]),
        cfg,
        label_mapping,
    )
    write_jsonl_records(v_out, v_records)
    write_jsonl_records(a_out, a_records)

    summary_rows = []
    for sample_id in sorted(set(v_df["sample_id"].astype(str)) | set(a_df["sample_id"].astype(str))):
        vv = v_df[v_df["sample_id"].astype(str) == sample_id]
        aa = a_df[a_df["sample_id"].astype(str) == sample_id]
        vc = v_clusters[v_clusters["sample_id"].astype(str) == sample_id]
        ac = a_clusters[a_clusters["sample_id"].astype(str) == sample_id]
        summary_rows.append(
            {
                "sample_id": sample_id,
                "n_visual_events_raw": int(len(vv)),
                "n_audio_events_raw": int(len(aa)),
                "n_visual_clusters": int(len(vc)),
                "n_audio_clusters": int(len(ac)),
                "visual_compression_ratio": float(len(vc) / max(len(vv), 1)),
                "audio_compression_ratio": float(len(ac) / max(len(aa), 1)),
                "status": "ok",
            }
        )
    write_csv_df(summary_csv, pd.DataFrame(summary_rows))
    return {"visual_clusters": str(v_out), "audio_clusters": str(a_out), "summary": str(summary_csv)}


def _mode_cfg(cfg: dict, mode: str) -> dict:
    mode_cfg = cfg.get("matching_modes", {}).get(mode, {})
    merged = {
        "early_tolerance_sec": float(mode_cfg.get("early_tolerance_sec", cfg["timing"]["early_tolerance_sec"])),
        "sync_tolerance_sec": float(mode_cfg.get("sync_tolerance_sec", cfg["timing"]["sync_tolerance_sec"])),
        "max_match_delay_sec": float(mode_cfg.get("max_match_delay_sec", cfg["timing"]["max_match_delay_sec"])),
        "require_label_compatibility": bool(mode_cfg.get("require_label_compatibility", False)),
        "allow_many_to_one": bool(mode_cfg.get("allow_many_to_one", False)),
        "allow_one_to_many": bool(mode_cfg.get("allow_one_to_many", False)),
    }
    return merged


def _violation_threshold(cfg: dict) -> float:
    """Return the scoring threshold, distinct from event-match tolerance."""
    return float(cfg.get("timing", {}).get("violation_threshold_sec", -0.001))


def _match_clusters_one_to_one(v_df: pd.DataFrame, a_df: pd.DataFrame, cfg: dict, project_root: Path, mode: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_mapping = load_label_mapping(project_root / "time_causality_eval" / "configs" / "event_label_mapping.yaml")
    matching_cfg = cfg["matching"]
    mode_cfg = _mode_cfg(cfg, mode)
    early_tol = mode_cfg["early_tolerance_sec"]
    sync_tol = mode_cfg["sync_tolerance_sec"]
    max_delay = mode_cfg["max_match_delay_sec"]
    require_label = mode_cfg["require_label_compatibility"]
    w_time = float(matching_cfg["cost_weights"]["time_distance"])
    w_label = float(matching_cfg["cost_weights"]["label_compatibility"])
    w_conf = float(matching_cfg["cost_weights"]["confidence"])
    large_cost = 1e6
    matched = []
    unmatched_v = []
    unmatched_a = []
    for sample_id in sorted(set(v_df["sample_id"].astype(str).tolist()) | set(a_df["sample_id"].astype(str).tolist())):
        vv = v_df[v_df["sample_id"].astype(str) == sample_id].sort_values(by=["representative_time_sec", "mean_confidence", "cluster_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
        aa = a_df[a_df["sample_id"].astype(str) == sample_id].sort_values(by=["representative_time_sec", "mean_confidence", "cluster_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
        if vv.empty and aa.empty:
            continue
        if vv.empty:
            unmatched_a.extend([r.to_dict() | {"match_status": "potential_hallucinated_audio"} for _, r in aa.iterrows()])
            continue
        if aa.empty:
            unmatched_v.extend([r.to_dict() | {"match_status": "potential_missing_audio"} for _, r in vv.iterrows()])
            continue
        nv, na = len(vv), len(aa)
        cost = np.full((max(nv, na), max(nv, na)), large_cost, dtype=float)
        meta = {}
        for i, v in vv.iterrows():
            for j, a in aa.iterrows():
                delay = safe_float(a["representative_time_sec"]) - safe_float(v["representative_time_sec"])
                if delay < -early_tol or delay > max_delay:
                    continue
                compat_score, compat_reason = label_compatibility(v["dominant_label"], a["dominant_label"], label_mapping)
                if require_label and compat_score < 0.8:
                    continue
                conf_score = float(np.nanmean([safe_float(v["mean_confidence"]), safe_float(a["mean_confidence"])]))
                c = abs(delay) * w_time - compat_score * w_label - conf_score * w_conf
                cost[i, j] = c
                meta[(i, j)] = {"delay": delay, "compat_score": compat_score, "compat_reason": compat_reason, "conf_score": conf_score}
        try:
            from scipy.optimize import linear_sum_assignment

            rr, cc = linear_sum_assignment(cost)
            used_v = set()
            used_a = set()
            core_pairs = []
            for i, j in zip(rr, cc):
                if i >= nv or j >= na or cost[i, j] >= large_cost / 2:
                    continue
                used_v.add(i)
                used_a.add(j)
                core_pairs.append((i, j, "core"))
        except Exception:
            candidates = [(cost[i, j], i, j) for (i, j) in meta.keys()]
            candidates.sort()
            used_v, used_a, core_pairs = set(), set(), []
            for _, i, j in candidates:
                if i in used_v or j in used_a:
                    continue
                used_v.add(i)
                used_a.add(j)
                core_pairs.append((i, j, "core"))

        matched_v = set()
        matched_a = set()
        for i, j, role in core_pairs:
            v = vv.iloc[i].to_dict()
            a = aa.iloc[j].to_dict()
            m = meta[(i, j)]
            delay = float(m["delay"])
            compat_score = float(m["compat_score"])
            conf_score = float(m["conf_score"])
            matched_v.add(i)
            matched_a.add(j)
            matched.append(
                {
                    "sample_id": sample_id,
                    "video_id": sample_id,
                    "matching_mode": mode,
                    "link_role": role,
                    "visual_cluster_id": v["cluster_id"],
                    "audio_cluster_id": a["cluster_id"],
                    "visual_label": v["dominant_label"],
                    "audio_label": a["dominant_label"],
                    "visual_time_sec": safe_float(v["representative_time_sec"]),
                    "audio_onset_sec": safe_float(a["representative_time_sec"]),
                    "delay_sec": delay,
                    "label_compatible": compat_score >= 0.8,
                    "match_confidence": max(0.0, min(1.0, 0.5 * compat_score + 0.3 * conf_score + 0.2 * math.exp(-abs(delay) / max(max_delay, 1e-6)))),
                    "is_causality_violation": delay < _violation_threshold(cfg),
                    "is_sync_valid": (-early_tol) <= delay <= sync_tol,
                    "label_compatibility_score": compat_score,
                }
            )
        for i, v in vv.iterrows():
            if i not in matched_v:
                unmatched_v.append(v.to_dict() | {"match_status": "potential_missing_audio"})
        for j, a in aa.iterrows():
            if j not in matched_a:
                unmatched_a.append(a.to_dict() | {"match_status": "potential_hallucinated_audio"})
    matched_df = pd.DataFrame(matched)
    unmatched_v_df = pd.DataFrame(unmatched_v)
    unmatched_a_df = pd.DataFrame(unmatched_a)
    if not matched_df.empty:
        matched_df = matched_df.sort_values(by=["sample_id", "visual_time_sec", "audio_onset_sec"], kind="stable").reset_index(drop=True)
    return matched_df, unmatched_v_df, unmatched_a_df


def _match_relaxed(v_df: pd.DataFrame, a_df: pd.DataFrame, cfg: dict, project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_mapping = load_label_mapping(project_root / "time_causality_eval" / "configs" / "event_label_mapping.yaml")
    mode_cfg = _mode_cfg(cfg, "relaxed")
    early_tol = mode_cfg["early_tolerance_sec"]
    sync_tol = mode_cfg["sync_tolerance_sec"]
    max_delay = mode_cfg["max_match_delay_sec"]
    matching_cfg = cfg["matching"]
    w_time = float(matching_cfg["cost_weights"]["time_distance"])
    w_label = float(matching_cfg["cost_weights"]["label_compatibility"])
    w_conf = float(matching_cfg["cost_weights"]["confidence"])
    rows = []
    core_rows = []
    matched_v = set()
    matched_a = set()
    for sample_id in sorted(set(v_df["sample_id"].astype(str).tolist()) | set(a_df["sample_id"].astype(str).tolist())):
        vv = v_df[v_df["sample_id"].astype(str) == sample_id].sort_values(by=["representative_time_sec", "mean_confidence", "cluster_id"], kind="stable").reset_index(drop=True)
        aa = a_df[a_df["sample_id"].astype(str) == sample_id].sort_values(by=["representative_time_sec", "mean_confidence", "cluster_id"], kind="stable").reset_index(drop=True)
        if vv.empty and aa.empty:
            continue
        if vv.empty:
            continue
        if aa.empty:
            continue
        # core one-to-one
        core_df, uv, ua = _match_clusters_one_to_one(vv, aa, cfg, project_root, "relaxed")
        core_df = core_df.copy()
        core_df["link_role"] = "core"
        core_rows.append(core_df)
        if not core_df.empty and "visual_cluster_id" in core_df.columns:
            matched_v.update(core_df["visual_cluster_id"].astype(str).tolist())
        if not core_df.empty and "audio_cluster_id" in core_df.columns:
            matched_a.update(core_df["audio_cluster_id"].astype(str).tolist())
        # supplementary links: nearest neighbor on both sides within relaxed window
        for _, v in vv.iterrows():
            best = None
            for _, a in aa.iterrows():
                delay = safe_float(a["representative_time_sec"]) - safe_float(v["representative_time_sec"])
                if delay < -early_tol or delay > max_delay:
                    continue
                compat_score, _ = label_compatibility(v["dominant_label"], a["dominant_label"], label_mapping)
                conf_score = float(np.nanmean([safe_float(v["mean_confidence"]), safe_float(a["mean_confidence"])]))
                score = abs(delay) * w_time - compat_score * w_label - conf_score * w_conf
                if best is None or score < best[0]:
                    best = (score, a, delay, compat_score, conf_score)
            if best is not None:
                _, a, delay, compat_score, conf_score = best
                rows.append(
                    {
                        "sample_id": sample_id,
                        "video_id": sample_id,
                        "matching_mode": "relaxed",
                        "link_role": "supplemental",
                        "visual_cluster_id": v["cluster_id"],
                        "audio_cluster_id": a["cluster_id"],
                        "visual_label": v["dominant_label"],
                        "audio_label": a["dominant_label"],
                        "visual_time_sec": safe_float(v["representative_time_sec"]),
                        "audio_onset_sec": safe_float(a["representative_time_sec"]),
                        "delay_sec": float(delay),
                        "label_compatible": compat_score >= 0.8,
                        "match_confidence": max(0.0, min(1.0, 0.5 * compat_score + 0.3 * conf_score + 0.2 * math.exp(-abs(delay) / max(max_delay, 1e-6)))),
                        "is_causality_violation": delay < _violation_threshold(cfg),
                        "is_sync_valid": (-early_tol) <= delay <= sync_tol,
                        "label_compatibility_score": compat_score,
                    }
                )
    core_df = pd.concat(core_rows, ignore_index=True) if core_rows else pd.DataFrame()
    supplemental_df = pd.DataFrame(rows)
    matched_df = pd.concat([core_df, supplemental_df], ignore_index=True) if not supplemental_df.empty else core_df
    if not matched_df.empty:
        matched_df = matched_df.sort_values(by=["sample_id", "visual_time_sec", "audio_onset_sec", "link_role"], kind="stable").reset_index(drop=True)
    # unmatched based on core only
    matched_v_core = set(core_df["visual_cluster_id"].astype(str).tolist()) if not core_df.empty else set()
    matched_a_core = set(core_df["audio_cluster_id"].astype(str).tolist()) if not core_df.empty else set()
    unmatched_v = []
    unmatched_a = []
    for _, row in v_df.iterrows():
        if str(row["cluster_id"]) not in matched_v_core:
            unmatched_v.append(row.to_dict() | {"match_status": "potential_missing_audio"})
    for _, row in a_df.iterrows():
        if str(row["cluster_id"]) not in matched_a_core:
            unmatched_a.append(row.to_dict() | {"match_status": "potential_hallucinated_audio"})
    return matched_df, pd.DataFrame(unmatched_v), pd.DataFrame(unmatched_a)


def match_events_for_causality(
    cfg: dict,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    skip_existing: bool = False,
) -> dict:
    project_root, out_root = _base_paths(cfg, project_root, output_root)
    match_dir = ensure_dir(out_root / "matched_events")
    parsed_dir = ensure_dir(out_root / "parsed_events")
    cluster_v = parsed_dir / "visual_event_clusters.jsonl"
    cluster_a = parsed_dir / "audio_event_clusters.jsonl"
    if not cluster_v.exists() or not cluster_a.exists():
        cluster_events(cfg, project_root, out_root, skip_existing=skip_existing)
    v_df = read_jsonl_records(cluster_v)
    a_df = read_jsonl_records(cluster_a)
    if v_df.empty or a_df.empty:
        raise RuntimeError("Clustered events are empty; cannot match.")

    outputs = {}
    modes = ["strict", "normal", "relaxed"]
    for mode in modes:
        mode_dir = ensure_dir(match_dir / mode)
        out_csv = mode_dir / "causality_matched_events.csv"
        out_jsonl = mode_dir / "causality_matched_events.jsonl"
        out_unmatched_v = mode_dir / "unmatched_visual_clusters.csv"
        out_unmatched_a = mode_dir / "unmatched_audio_clusters.csv"
        out_summary = mode_dir / "match_summary.json"
        if skip_existing and out_csv.exists() and out_jsonl.exists() and out_unmatched_v.exists() and out_unmatched_a.exists() and out_summary.exists():
            outputs[mode] = {
                "matched_csv": str(out_csv),
                "matched_jsonl": str(out_jsonl),
                "unmatched_visual_csv": str(out_unmatched_v),
                "unmatched_audio_csv": str(out_unmatched_a),
            }
            continue
        if mode == "relaxed":
            matched_df, uv_df, ua_df = _match_relaxed(v_df, a_df, cfg, project_root)
        else:
            matched_df, uv_df, ua_df = _match_clusters_one_to_one(v_df, a_df, cfg, project_root, mode)
        if matched_df.empty:
            matched_df = pd.DataFrame(columns=["sample_id", "video_id", "matching_mode", "link_role", "visual_cluster_id", "audio_cluster_id", "visual_label", "audio_label", "visual_time_sec", "audio_onset_sec", "delay_sec", "label_compatible", "match_confidence", "is_causality_violation", "is_sync_valid", "label_compatibility_score"])
        matched_df["matching_mode"] = mode
        if "link_role" not in matched_df.columns:
            matched_df["link_role"] = "core"
        write_csv_df(out_csv, matched_df)
        write_jsonl_records(out_jsonl, matched_df.to_dict(orient="records"))
        write_csv_df(out_unmatched_v, uv_df)
        write_csv_df(out_unmatched_a, ua_df)
        summary = {
            "matching_mode": mode,
            "num_matched_rows": int(len(matched_df)),
            "num_core_matched_rows": int((matched_df.get("link_role", pd.Series(dtype=str)) == "core").sum()) if not matched_df.empty else 0,
            "num_unmatched_visual": int(len(uv_df)),
            "num_unmatched_audio": int(len(ua_df)),
            "early_tolerance_sec": _mode_cfg(cfg, mode)["early_tolerance_sec"],
            "sync_tolerance_sec": _mode_cfg(cfg, mode)["sync_tolerance_sec"],
            "max_match_delay_sec": _mode_cfg(cfg, mode)["max_match_delay_sec"],
        }
        write_json(out_summary, summary)
        outputs[mode] = {
            "matched_csv": str(out_csv),
            "matched_jsonl": str(out_jsonl),
            "unmatched_visual_csv": str(out_unmatched_v),
            "unmatched_audio_csv": str(out_unmatched_a),
        }
    return outputs


def _sample_metrics_from_groups(sample_id: str, manifest_row: pd.Series, matched_df: pd.DataFrame, uv_df: pd.DataFrame, ua_df: pd.DataFrame, cfg: dict) -> dict:
    timing = cfg["timing"]
    app = cfg["applicability"]
    early_tol = float(timing["early_tolerance_sec"])
    sync_tol = float(timing["sync_tolerance_sec"])
    mv = matched_df[matched_df["sample_id"].astype(str) == sample_id].copy()
    uv = uv_df[uv_df.get("sample_id", pd.Series(dtype=str)).astype(str) == sample_id].copy() if not uv_df.empty else pd.DataFrame()
    ua = ua_df[ua_df.get("sample_id", pd.Series(dtype=str)).astype(str) == sample_id].copy() if not ua_df.empty else pd.DataFrame()
    n_visual = len(mv) + len(uv)
    n_audio = len(mv) + len(ua)
    n_matched = len(mv)
    n_uv = len(uv)
    n_ua = len(ua)
    delays = pd.to_numeric(mv["delay_sec"], errors="coerce").dropna().astype(float).tolist() if not mv.empty else []
    early_flags = [d < 0 for d in delays]
    violation_flags = [d < _violation_threshold(cfg) for d in delays]
    sync_flags = [(-early_tol) <= d <= sync_tol for d in delays]
    low_v_count = int((pd.to_numeric(mv.get("visual_confidence", pd.Series(dtype=float)), errors="coerce") < float(cfg["event_filter"]["min_visual_confidence"])).fillna(False).sum()) if not mv.empty else 0
    low_a_count = int((pd.to_numeric(mv.get("audio_confidence", pd.Series(dtype=float)), errors="coerce") < float(cfg["event_filter"]["min_audio_confidence"])).fillna(False).sum()) if not mv.empty else 0

    sequence_applicable = (
        n_visual >= int(app["min_visual_events_for_sequence"])
        and n_audio >= int(app["min_audio_events_for_sequence"])
        and n_matched >= int(app["min_matched_events_for_sequence"])
    )
    if sequence_applicable and not mv.empty:
        vv = mv.sort_values(by=["visual_event_time_sec", "audio_onset_sec"], kind="stable").reset_index(drop=True)
        aa = mv.sort_values(by=["audio_onset_sec", "visual_event_time_sec"], kind="stable").reset_index(drop=True)
        visual_order = vv["visual_event_id"].astype(str).tolist()
        audio_order = aa["audio_event_id"].astype(str).tolist()
        # ranks of each matched pair in visual and audio order
        visual_rank = {eid: i for i, eid in enumerate(vv["visual_event_id"].astype(str).tolist())}
        audio_rank = {eid: i for i, eid in enumerate(aa["audio_event_id"].astype(str).tolist())}
        common_ids = [str(eid) for eid in mv["visual_event_id"].astype(str).tolist()]
        vis_ranks = [visual_rank[eid] for eid in common_ids]
        aud_ranks = [audio_rank[a] for a in mv["audio_event_id"].astype(str).tolist()]
        kt, sr = kendall_spearman_from_orders(vis_ranks, aud_ranks)
        order_inversions = 0
        total_pairs = 0
        for i in range(len(vis_ranks)):
            for j in range(i + 1, len(vis_ranks)):
                total_pairs += 1
                if (vis_ranks[i] - vis_ranks[j]) * (aud_ranks[i] - aud_ranks[j]) < 0:
                    order_inversions += 1
        order_inversion_rate = order_inversions / total_pairs if total_pairs else math.nan
        sequence_alignment_score = 1.0 - order_inversion_rate if not math.isnan(order_inversion_rate) else math.nan
        matched_sequence_ratio = n_matched / max(min(n_visual, n_audio), 1)
        normalized_edit_distance = sequence_edit_distance(visual_order, [str(x) for x in aa["visual_event_id"].astype(str).tolist()])
    else:
        visual_order = []
        audio_order = []
        vis_ranks = []
        aud_ranks = []
        order_inversions = math.nan
        total_pairs = math.nan
        order_inversion_rate = math.nan
        sequence_alignment_score = math.nan
        kt = math.nan
        sr = math.nan
        matched_sequence_ratio = math.nan
        normalized_edit_distance = math.nan

    missing_event_rate = n_uv / max(n_visual, 1)
    hallucinated_event_rate = n_ua / max(n_audio, 1)
    visual_event_recall_by_audio = n_matched / max(n_visual, 1)
    audio_event_precision_by_visual = n_matched / max(n_audio, 1)
    causality_violation_rate = sum(violation_flags) / max(n_matched, 1)
    early_audio_rate = sum(early_flags) / max(n_matched, 1)
    valid_sync_rate = sum(sync_flags) / max(n_matched, 1)

    manual_reasons = []
    if low_v_count > 0:
        manual_reasons.append("low_confidence_visual")
    if low_a_count > 0:
        manual_reasons.append("low_confidence_audio")
    if not mv.empty and bool((mv["label_compatible"] == False).any()):
        manual_reasons.append("label_incompatibility")
    if n_uv > 0 or n_ua > 0:
        manual_reasons.append("unmatched_events_present")
    if sequence_applicable and not math.isnan(sequence_alignment_score) and sequence_alignment_score < 0.8:
        manual_reasons.append("sequence_misalignment")
    if n_matched > 0 and causality_violation_rate > 0:
        manual_reasons.append("causality_violation")

    delay_stats = mean_median_std(delays)
    out = {
        "sample_id": sample_id,
        "video_id": str(manifest_row.get("video_id", sample_id)),
        "chunk_id": str(manifest_row.get("chunk_id", sample_id)),
        "seed_category": str(manifest_row.get("seed_category", "")),
        "source_type": str(manifest_row.get("source_type", "")),
        "title": str(manifest_row.get("title", "")),
        "video_path": str(manifest_row.get("video_path", "")),
        "event_clip_path": str(manifest_row.get("event_clip_path", "")),
        "event_audio_path": str(manifest_row.get("event_audio_path", "")),
        "status": str(manifest_row.get("status", "")),
        "skip_reason": str(manifest_row.get("skip_reason", "")),
        "n_visual_events": int(n_visual),
        "n_audio_events": int(n_audio),
        "n_matched_events": int(n_matched),
        "n_unmatched_visual_events": int(n_uv),
        "n_unmatched_audio_events": int(n_ua),
        "n_causality_violations": int(sum(violation_flags)),
        "causality_violation_rate": float(causality_violation_rate),
        "early_audio_rate": float(early_audio_rate),
        "valid_sync_rate": float(valid_sync_rate),
        "sync_valid_rate": float(valid_sync_rate),
        "delay_mean_sec": delay_stats["mean"],
        "delay_median_sec": delay_stats["median"],
        "delay_std_sec": delay_stats["std"],
        "delay_p10_sec": delay_stats["p10"],
        "delay_p90_sec": delay_stats["p90"],
        "missing_event_rate": float(missing_event_rate),
        "hallucinated_event_rate": float(hallucinated_event_rate),
        "visual_event_recall_by_audio": float(visual_event_recall_by_audio),
        "audio_event_precision_by_visual": float(audio_event_precision_by_visual),
        "low_confidence_visual_count": int(low_v_count),
        "low_confidence_audio_count": int(low_a_count),
        "sequence_applicable": bool(sequence_applicable),
        "visual_order": ";".join(visual_order) if visual_order else "",
        "audio_order": ";".join(audio_order) if audio_order else "",
        "matched_visual_order": ";".join([str(x) for x in mv["visual_event_id"].astype(str).tolist()]) if not mv.empty else "",
        "matched_audio_order": ";".join([str(x) for x in mv["audio_event_id"].astype(str).tolist()]) if not mv.empty else "",
        "order_inversion_count": int(order_inversions) if not math.isnan(order_inversions) else math.nan,
        "order_inversion_rate": float(order_inversion_rate),
        "sequence_alignment_score": float(sequence_alignment_score),
        "kendall_tau": float(kt),
        "spearman_rank_corr": float(sr),
        "normalized_edit_distance": float(normalized_edit_distance),
        "matched_sequence_ratio": float(matched_sequence_ratio),
        "manual_review_needed": bool(len(manual_reasons) > 0),
        "manual_review_reason": ";".join(manual_reasons),
    }
    return out


def compute_time_causality_metrics(
    cfg: dict,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    skip_existing: bool = False,
) -> dict:
    project_root, out_root = _base_paths(cfg, project_root, output_root)
    metrics_dir = ensure_dir(out_root / "metrics")
    event_csv = metrics_dir / "time_causality_event_metrics.csv"
    video_csv = metrics_dir / "time_causality_video_metrics.csv"
    summary_json = metrics_dir / "time_causality_summary.json"
    if skip_existing and event_csv.exists() and video_csv.exists() and summary_json.exists():
        return {
            "event_metrics": str(event_csv),
            "video_metrics": str(video_csv),
            "summary": str(summary_json),
        }

    manifest_path = out_root / "manifests" / "time_causality_manifest.csv"
    if not manifest_path.exists():
        build_time_causality_manifest(cfg, project_root, out_root, skip_existing=skip_existing)
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise RuntimeError("Time causality manifest is empty.")

    matched_path = out_root / "matched_events" / "causality_matched_events.csv"
    uv_path = out_root / "matched_events" / "unmatched_visual_events.csv"
    ua_path = out_root / "matched_events" / "unmatched_audio_events.csv"
    if not matched_path.exists() or not uv_path.exists() or not ua_path.exists():
        match_events_for_causality(cfg, project_root, out_root, skip_existing=skip_existing)
    matched = pd.read_csv(matched_path) if matched_path.exists() else pd.DataFrame()
    unmatched_v = pd.read_csv(uv_path) if uv_path.exists() else pd.DataFrame()
    unmatched_a = pd.read_csv(ua_path) if ua_path.exists() else pd.DataFrame()

    manifest = manifest.sort_values(by=["sample_id"], kind="stable").reset_index(drop=True)
    event_rows = []
    video_rows = []
    for _, m in manifest.iterrows():
        sample_id = str(m["sample_id"])
        if str(m.get("status", "")) in {"missing_video", "missing_audio", "ineligible"}:
            row = _sample_metrics_from_groups(sample_id, m, matched, unmatched_v, unmatched_a, cfg)
            video_rows.append(row)
            continue
        row = _sample_metrics_from_groups(sample_id, m, matched, unmatched_v, unmatched_a, cfg)
        video_rows.append(row)
        mv = matched[matched["sample_id"].astype(str) == sample_id].copy() if not matched.empty else pd.DataFrame()
        for _, e in mv.iterrows():
            event_rows.append(
                {
                    **row,
                    "visual_event_id": e.get("visual_event_id", ""),
                    "audio_event_id": e.get("audio_event_id", ""),
                    "visual_label_raw": e.get("visual_label_raw", ""),
                    "audio_label_raw": e.get("audio_label_raw", ""),
                    "visual_label_norm": e.get("visual_label_norm", ""),
                    "audio_label_norm": e.get("audio_label_norm", ""),
                    "visual_event_time_sec": e.get("visual_event_time_sec", ""),
                    "audio_onset_sec": e.get("audio_onset_sec", ""),
                    "delay_sec": e.get("delay_sec", ""),
                    "label_compatible": e.get("label_compatible", ""),
                    "label_compatibility_score": e.get("label_compatibility_score", ""),
                    "label_compatibility_reason": e.get("label_compatibility_reason", ""),
                    "match_confidence": e.get("match_confidence", ""),
                    "match_status": e.get("match_status", ""),
                    "is_causality_violation": e.get("is_causality_violation", ""),
                    "is_sync_valid": e.get("is_sync_valid", ""),
                    "early_by_sec": e.get("early_by_sec", ""),
                    "delay_abs_sec": e.get("delay_abs_sec", ""),
                }
            )

    event_df = pd.DataFrame(event_rows)
    video_df = pd.DataFrame(video_rows)
    event_df = event_df.reindex(
        columns=[
            "sample_id",
            "video_id",
            "visual_event_id",
            "audio_event_id",
            "event_clip_path",
            "event_audio_path",
            "video_path",
            "status",
            "skip_reason",
            "n_visual_events",
            "n_audio_events",
            "n_matched_events",
            "n_unmatched_visual_events",
            "n_unmatched_audio_events",
            "n_causality_violations",
            "delay_sec",
            "causality_violation_rate",
            "early_audio_rate",
            "valid_sync_rate",
            "sync_valid_rate",
            "delay_mean_sec",
            "delay_median_sec",
            "delay_std_sec",
            "delay_p10_sec",
            "delay_p90_sec",
            "missing_event_rate",
            "hallucinated_event_rate",
            "visual_event_recall_by_audio",
            "audio_event_precision_by_visual",
            "low_confidence_visual_count",
            "low_confidence_audio_count",
            "sequence_applicable",
            "visual_order",
            "audio_order",
            "matched_visual_order",
            "matched_audio_order",
            "order_inversion_count",
            "order_inversion_rate",
            "sequence_alignment_score",
            "kendall_tau",
            "spearman_rank_corr",
            "normalized_edit_distance",
            "matched_sequence_ratio",
            "match_confidence",
            "label_compatible",
            "match_status",
            "is_causality_violation",
            "is_sync_valid",
            "manual_review_needed",
            "manual_review_reason",
            "seed_category",
            "source_type",
            "title",
        ]
    )
    video_df = video_df.reindex(
        columns=[
            "sample_id",
            "video_id",
            "chunk_id",
            "seed_category",
            "source_type",
            "title",
            "video_path",
            "event_clip_path",
            "event_audio_path",
            "status",
            "skip_reason",
            "n_visual_events",
            "n_audio_events",
            "n_matched_events",
            "n_unmatched_visual_events",
            "n_unmatched_audio_events",
            "n_causality_violations",
            "causality_violation_rate",
            "early_audio_rate",
            "valid_sync_rate",
            "sync_valid_rate",
            "delay_mean_sec",
            "delay_median_sec",
            "delay_std_sec",
            "delay_p10_sec",
            "delay_p90_sec",
            "missing_event_rate",
            "hallucinated_event_rate",
            "visual_event_recall_by_audio",
            "audio_event_precision_by_visual",
            "low_confidence_visual_count",
            "low_confidence_audio_count",
            "sequence_applicable",
            "visual_order",
            "audio_order",
            "matched_visual_order",
            "matched_audio_order",
            "order_inversion_count",
            "order_inversion_rate",
            "sequence_alignment_score",
            "kendall_tau",
            "spearman_rank_corr",
            "normalized_edit_distance",
            "matched_sequence_ratio",
            "manual_review_needed",
            "manual_review_reason",
        ]
    )
    write_csv_df(event_csv, event_df)
    write_csv_df(video_csv, video_df)

    def _agg_series(s: pd.Series):
        vals = pd.to_numeric(s, errors="coerce").dropna().astype(float).tolist()
        return {
            "mean": float(np.mean(vals)) if vals else math.nan,
            "median": float(np.median(vals)) if vals else math.nan,
            "count": int(len(vals)),
        }

    summary = {
        "num_samples": int(len(video_df)),
        "num_samples_with_visual_events": int((video_df["n_visual_events"] > 0).sum()) if "n_visual_events" in video_df.columns else 0,
        "num_samples_with_audio_events": int((video_df["n_audio_events"] > 0).sum()) if "n_audio_events" in video_df.columns else 0,
        "num_samples_with_matches": int((video_df["n_matched_events"] > 0).sum()) if "n_matched_events" in video_df.columns else 0,
        "num_event_rows": int(len(event_df)),
        "num_matched_events": int(len(matched)),
        "num_unmatched_visual_events": int(len(unmatched_v)),
        "num_unmatched_audio_events": int(len(unmatched_a)),
        "causality_violation_rate": _agg_series(video_df["causality_violation_rate"]) if "causality_violation_rate" in video_df.columns else {},
        "sync_valid_rate": _agg_series(video_df["sync_valid_rate"]) if "sync_valid_rate" in video_df.columns else {},
        "sequence_alignment_score": _agg_series(video_df["sequence_alignment_score"]) if "sequence_alignment_score" in video_df.columns else {},
        "missing_event_rate": _agg_series(video_df["missing_event_rate"]) if "missing_event_rate" in video_df.columns else {},
        "hallucinated_event_rate": _agg_series(video_df["hallucinated_event_rate"]) if "hallucinated_event_rate" in video_df.columns else {},
        "delay_sec": mean_median_std(pd.to_numeric(matched["delay_sec"], errors="coerce").dropna().astype(float).tolist()) if not matched.empty and "delay_sec" in matched.columns else {},
        "manual_review_cases": int((video_df["manual_review_needed"] == True).sum()) if "manual_review_needed" in video_df.columns else 0,
        "low_confidence_visual_event_count": int(video_df.get("low_confidence_visual_count", pd.Series(dtype=float)).fillna(0).sum()) if "low_confidence_visual_count" in video_df.columns else 0,
        "low_confidence_audio_event_count": int(video_df.get("low_confidence_audio_count", pd.Series(dtype=float)).fillna(0).sum()) if "low_confidence_audio_count" in video_df.columns else 0,
    }

    # group stats
    group_stats = {}
    for group_key in ["seed_category", "source_type"]:
        if group_key in video_df.columns:
            group_stats[group_key] = {}
            for value, g in video_df.groupby(group_key, dropna=False):
                group_stats[group_key][str(value)] = {
                    "count": int(len(g)),
                    "causality_violation_rate_mean": float(pd.to_numeric(g["causality_violation_rate"], errors="coerce").mean()),
                    "sync_valid_rate_mean": float(pd.to_numeric(g["sync_valid_rate"], errors="coerce").mean()),
                    "sequence_alignment_score_mean": float(pd.to_numeric(g["sequence_alignment_score"], errors="coerce").mean()),
                    "missing_event_rate_mean": float(pd.to_numeric(g["missing_event_rate"], errors="coerce").mean()),
                    "hallucinated_event_rate_mean": float(pd.to_numeric(g["hallucinated_event_rate"], errors="coerce").mean()),
                }
    summary["group_stats"] = group_stats
    summary["status_counts"] = Counter(video_df["status"].astype(str).tolist()) if "status" in video_df.columns else {}
    summary["manual_review_reason_counts"] = Counter(
        [r for r in video_df.get("manual_review_reason", pd.Series(dtype=str)).fillna("").astype(str).tolist() if r]
    )
    write_json(summary_json, summary)
    return {
        "event_metrics": str(event_csv),
        "video_metrics": str(video_csv),
        "summary": str(summary_json),
    }


def analyze_time_causality_results(
    cfg: dict,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    skip_existing: bool = False,
) -> dict:
    project_root, out_root = _base_paths(cfg, project_root, output_root)
    summary_dir = ensure_dir(out_root / "summaries")
    fig_dir = ensure_dir(summary_dir / "figures")
    md_path = summary_dir / "time_causality_summary.md"
    json_path = summary_dir / "time_causality_summary.json"
    if skip_existing and md_path.exists() and json_path.exists():
        return {"summary_md": str(md_path), "summary_json": str(json_path), "figures": str(fig_dir)}

    video_csv = out_root / "metrics" / "time_causality_video_metrics.csv"
    event_csv = out_root / "metrics" / "time_causality_event_metrics.csv"
    if not video_csv.exists() or not event_csv.exists():
        compute_time_causality_metrics(cfg, project_root, out_root, skip_existing=skip_existing)
    video_df = pd.read_csv(video_csv)
    event_df = pd.read_csv(event_csv)

    overall = {
        "num_samples": int(len(video_df)),
        "num_visual_samples": int((video_df["n_visual_events"] > 0).sum()) if "n_visual_events" in video_df.columns else 0,
        "num_audio_samples": int((video_df["n_audio_events"] > 0).sum()) if "n_audio_events" in video_df.columns else 0,
        "num_matched_samples": int((video_df["n_matched_events"] > 0).sum()) if "n_matched_events" in video_df.columns else 0,
        "causality_violation_rate_mean": float(pd.to_numeric(video_df["causality_violation_rate"], errors="coerce").mean()) if "causality_violation_rate" in video_df.columns else math.nan,
        "causality_violation_rate_median": float(pd.to_numeric(video_df["causality_violation_rate"], errors="coerce").median()) if "causality_violation_rate" in video_df.columns else math.nan,
        "sync_valid_rate_mean": float(pd.to_numeric(video_df["sync_valid_rate"], errors="coerce").mean()) if "sync_valid_rate" in video_df.columns else math.nan,
        "sync_valid_rate_median": float(pd.to_numeric(video_df["sync_valid_rate"], errors="coerce").median()) if "sync_valid_rate" in video_df.columns else math.nan,
        "sequence_alignment_score_mean": float(pd.to_numeric(video_df["sequence_alignment_score"], errors="coerce").mean()) if "sequence_alignment_score" in video_df.columns else math.nan,
        "sequence_alignment_score_median": float(pd.to_numeric(video_df["sequence_alignment_score"], errors="coerce").median()) if "sequence_alignment_score" in video_df.columns else math.nan,
        "missing_event_rate_mean": float(pd.to_numeric(video_df["missing_event_rate"], errors="coerce").mean()) if "missing_event_rate" in video_df.columns else math.nan,
        "missing_event_rate_median": float(pd.to_numeric(video_df["missing_event_rate"], errors="coerce").median()) if "missing_event_rate" in video_df.columns else math.nan,
        "hallucinated_event_rate_mean": float(pd.to_numeric(video_df["hallucinated_event_rate"], errors="coerce").mean()) if "hallucinated_event_rate" in video_df.columns else math.nan,
        "hallucinated_event_rate_median": float(pd.to_numeric(video_df["hallucinated_event_rate"], errors="coerce").median()) if "hallucinated_event_rate" in video_df.columns else math.nan,
        "manual_review_cases": int((video_df["manual_review_needed"] == True).sum()) if "manual_review_needed" in video_df.columns else 0,
    }

    # select cases
    def _rank_good(g):
        return (
            pd.to_numeric(g["causality_violation_rate"], errors="coerce").fillna(1).mean()
            + pd.to_numeric(g["missing_event_rate"], errors="coerce").fillna(1).mean()
            + pd.to_numeric(g["hallucinated_event_rate"], errors="coerce").fillna(1).mean()
        )

    good = video_df.copy()
    if not good.empty:
        good["good_score"] = (
            pd.to_numeric(good["causality_violation_rate"], errors="coerce").fillna(1.0)
            + pd.to_numeric(good["missing_event_rate"], errors="coerce").fillna(1.0)
            + pd.to_numeric(good["hallucinated_event_rate"], errors="coerce").fillna(1.0)
            + pd.to_numeric(good["order_inversion_rate"], errors="coerce").fillna(1.0)
        )
        good_cases = good.sort_values(by=["good_score", "sample_id"], ascending=[True, True], kind="stable")
    else:
        good_cases = good
    causality_fail = video_df[pd.to_numeric(video_df.get("causality_violation_rate", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0]
    if not causality_fail.empty:
        causality_fail = causality_fail.sort_values(by=["delay_mean_sec", "causality_violation_rate"], ascending=[True, False], kind="stable")
    sequence_fail = video_df[pd.to_numeric(video_df.get("sequence_alignment_score", pd.Series(dtype=float)), errors="coerce").fillna(1) < 0.8]
    missing_fail = video_df[pd.to_numeric(video_df.get("missing_event_rate", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0]
    halluc_fail = video_df[pd.to_numeric(video_df.get("hallucinated_event_rate", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0]
    manual_review_mask = video_df.get("manual_review_needed", pd.Series(dtype=bool)).fillna(False) if "manual_review_needed" in video_df.columns else pd.Series(dtype=bool)
    manual_review = video_df[manual_review_mask == True] if "manual_review_needed" in video_df.columns else pd.DataFrame()

    def _top_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.head(n).copy() if not df.empty else df.copy()

    selections = {
        "good_cases": _top_rows(good_cases, int(cfg["reporting"]["max_good_cases"])),
        "causality_failure_cases": _top_rows(causality_fail, int(cfg["reporting"]["max_bad_cases"])),
        "manual_review_cases": _top_rows(manual_review, int(cfg["reporting"]["max_manual_review_cases"])),
    }

    for name, df in selections.items():
        if df.empty:
            continue
        if "sample_id" not in df.columns:
            continue
        if name == "good_cases":
            df.to_csv(fig_dir.parent / "good_cases_preview.csv", index=False)

    for stale in [
        "missing_rate_raw_vs_clustered.png",
        "raw_vs_cluster_event_counts.png",
        "hallucinated_rate_raw_vs_clustered.png",
        "delay_distribution_by_mode.png",
        "sequence_score_vs_coverage.png",
        "sequence_alignment_score_histogram.png",
        "missing_vs_hallucinated_scatter.png",
        "event_count_distribution.png",
    ]:
        old = fig_dir / stale
        if old.exists():
            old.unlink()
    save_histogram_png(pd.to_numeric(event_df["delay_sec"], errors="coerce"), fig_dir / "onset_delay_histogram.png", "Onset Delay Distribution", "delay_sec")
    save_histogram_png(pd.to_numeric(normal_video["causality_violation_rate"], errors="coerce"), fig_dir / "causality_violation_rate_histogram.png", "Causality Violation Rate", "rate")

    md_lines = [
        "# Time and Causality Summary",
        "",
        "This module is an expert-model-assisted causality validation layer built on OV-AVEL and FlexSED outputs.",
        "",
        "Important interpretation notes:",
        "- `early_tolerance_sec` controls event association only.",
        "- Matched delays below `violation_threshold_sec` are counted as violations.",
        "- The primary report keeps only Test point 1.",
        "- The main sync metric is the audio-conditioned sync rate: synced audio clusters / all audio clusters.",
        "- Missing / hallucinated / sequence outputs are retained as diagnostics, not main conclusions.",
        "- Low-confidence or label-conflict cases are routed to `manual_review` rather than auto-failure.",
        "",
        f"- Total samples: {overall['num_samples']}",
        f"- Samples with visual events: {overall['num_visual_samples']}",
        f"- Samples with audio events: {overall['num_audio_samples']}",
        f"- Samples with matched events: {overall['num_matched_samples']}",
        f"- Causality violation rate mean/median: {overall['causality_violation_rate_mean']:.4f} / {overall['causality_violation_rate_median']:.4f}",
        f"- Sync valid rate (matched-based) mean/median: {overall['sync_valid_rate_mean']:.4f} / {overall['sync_valid_rate_median']:.4f}",
        f"- Sync valid rate (audio-conditioned) mean/median: {overall['audio_sync_valid_rate_mean']:.4f} / {overall['audio_sync_valid_rate_median']:.4f}",
        f"- Manual review cases: {overall['manual_review_cases']}",
        "",
        "Recommended interpretation:",
        "1. This is not a manually annotated GT benchmark.",
        "2. Papers should describe it as expert-model-assisted TP1 validation.",
        "3. `manual_review` samples should be inspected before making strong claims.",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    write_json(json_path, {"overall": overall, "figures_dir": str(fig_dir), "selections": {k: int(len(v)) for k, v in selections.items()}})

    review_dir = ensure_dir(out_root / "review_cases")
    for stale_name in [
        "sequence_failure_cases.csv",
        "missing_event_cases.csv",
        "hallucinated_event_cases.csv",
        "good_cases.csv",
        "causality_failure_cases.csv",
        "manual_review_cases.csv",
    ]:
        stale_path = review_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    for name, df in selections.items():
        if df.empty:
            continue
        out = df.copy()
        cols = [c for c in [
            "sample_id",
            "video_id",
            "event_clip_path",
            "event_audio_path",
            "video_path",
            "visual_order",
            "audio_order",
            "delay_sec",
            "delay_mean_sec",
            "causality_violation_rate",
            "sync_valid_rate",
            "sequence_alignment_score",
            "missing_event_rate",
            "hallucinated_event_rate",
            "failure_type",
            "manual_review_reason",
        ] if c in out.columns]
        out = out[cols].copy()
        out["failure_type"] = name
        out["notes"] = name.replace("_", " ")
        out.to_csv(review_dir / f"{name}.csv", index=False)

    return {"summary_md": str(md_path), "summary_json": str(json_path), "figures": str(fig_dir), "review_cases": str(review_dir)}


def export_time_causality_review_cases(
    cfg: dict,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    skip_existing: bool = False,
) -> dict:
    project_root, out_root = _base_paths(cfg, project_root, output_root)
    review_dir = ensure_dir(out_root / "review_cases")
    for stale_name in [
        "good_cases.csv",
        "causality_failure_cases.csv",
        "manual_review_cases.csv",
        "sequence_failure_cases.csv",
        "missing_event_cases.csv",
        "hallucinated_event_cases.csv",
    ]:
        stale_path = review_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    video_csv = out_root / "metrics" / "time_causality_video_metrics.csv"
    if not video_csv.exists():
        analyze_time_causality_results(cfg, project_root, out_root, skip_existing=skip_existing)
    video_df = pd.read_csv(video_csv)

    def _prepare_case_df(df: pd.DataFrame, failure_type: str, n: int) -> pd.DataFrame:
        if df.empty:
            return df
        cols = [c for c in [
            "sample_id",
            "video_id",
            "event_clip_path",
            "event_audio_path",
            "video_path",
            "delay_mean_sec",
            "delay_median_sec",
            "delay_std_sec",
            "causality_violation_rate",
            "manual_review_reason",
        ] if c in df.columns]
        out = df[cols].copy()
        out["failure_type"] = failure_type
        out["notes"] = failure_type.replace("_", " ")
        return out.head(n)

    good = video_df.copy()
    if not good.empty:
        good["good_score"] = (
            pd.to_numeric(good["causality_violation_rate"], errors="coerce").fillna(1.0)
            + pd.to_numeric(good["delay_mean_sec"], errors="coerce").fillna(1.0).clip(lower=0)
        )
        good = good.sort_values(by=["good_score", "sample_id"], kind="stable")
    selections = {
        "good_cases": _prepare_case_df(good, "good_cases", int(cfg["reporting"]["max_good_cases"])),
        "causality_failure_cases": _prepare_case_df(video_df[pd.to_numeric(video_df.get("causality_violation_rate", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0].sort_values(by=["delay_mean_sec"], kind="stable"), "causality_failure_cases", int(cfg["reporting"]["max_bad_cases"])),
        "manual_review_cases": _prepare_case_df(
            video_df[video_df.get("manual_review_needed", pd.Series(dtype=bool)).fillna(False) == True]
            if "manual_review_needed" in video_df.columns
            else pd.DataFrame(),
            "manual_review_cases",
            int(cfg["reporting"]["max_manual_review_cases"]),
        ),
    }
    for name, df in selections.items():
        df.to_csv(review_dir / f"{name}.csv", index=False)
    return {"review_dir": str(review_dir), "selection_counts": {k: int(len(v)) for k, v in selections.items()}}


def _load_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _sequence_stats_from_core_matches(
    core_df: pd.DataFrame,
    n_visual_clusters: int | None = None,
    n_audio_clusters: int | None = None,
) -> dict:
    if core_df is None or core_df.empty or len(core_df) < 2:
        if core_df is None or core_df.empty:
            coverage = 0.0
        else:
            denom = max(1, min(int(n_visual_clusters or len(core_df)), int(n_audio_clusters or len(core_df))))
            coverage = float(len(core_df) / denom)
        return {
            "sequence_applicable": False,
            "sequence_alignment_score": math.nan,
            "matched_sequence_coverage": coverage,
            "order_inversion_count": math.nan,
            "order_inversion_rate": math.nan,
            "kendall_tau": math.nan,
            "spearman_rank_corr": math.nan,
            "matched_visual_order": "",
            "matched_audio_order": "",
            "visual_order": "",
            "audio_order": "",
            "sequence_weak_evidence": True,
            "sequence_review_reason": "insufficient_matched_events",
        }

    df = core_df.sort_values(by=["visual_time_sec", "audio_onset_sec", "visual_cluster_id", "audio_cluster_id"], kind="stable").reset_index(drop=True)
    visual_order = df["visual_cluster_id"].astype(str).tolist()
    audio_order = df["audio_cluster_id"].astype(str).tolist()
    vis_ranks = list(range(len(df)))
    audio_sorted = df.sort_values(by=["audio_onset_sec", "visual_time_sec", "audio_cluster_id"], kind="stable").reset_index(drop=True)
    audio_rank_map = {str(v): i for i, v in enumerate(audio_sorted["audio_cluster_id"].astype(str).tolist())}
    aud_ranks = [audio_rank_map[str(x)] for x in df["audio_cluster_id"].astype(str).tolist()]
    inversions = 0
    total_pairs = 0
    for i in range(len(vis_ranks)):
        for j in range(i + 1, len(vis_ranks)):
            total_pairs += 1
            if (vis_ranks[i] - vis_ranks[j]) * (aud_ranks[i] - aud_ranks[j]) < 0:
                inversions += 1
    order_inversion_rate = inversions / total_pairs if total_pairs else math.nan
    sequence_alignment_score = 1.0 - order_inversion_rate if not math.isnan(order_inversion_rate) else math.nan
    kt, sr = kendall_spearman_from_orders(vis_ranks, aud_ranks)
    denom = max(1, min(int(n_visual_clusters or len(df)), int(n_audio_clusters or len(df))))
    coverage = len(df) / denom
    return {
        "sequence_applicable": True,
        "sequence_alignment_score": float(sequence_alignment_score),
        "matched_sequence_coverage": float(coverage),
        "order_inversion_count": int(inversions),
        "order_inversion_rate": float(order_inversion_rate),
        "kendall_tau": float(kt),
        "spearman_rank_corr": float(sr),
        "matched_visual_order": ";".join(visual_order),
        "matched_audio_order": ";".join(audio_order),
        "visual_order": ";".join(visual_order),
        "audio_order": ";".join(audio_order),
        "sequence_weak_evidence": bool(coverage < 0.5),
        "sequence_review_reason": "weak_sequence_coverage" if coverage < 0.5 else "",
    }


def _build_raw_detector_metrics(
    cfg: dict,
    project_root: Path,
    manifest: pd.DataFrame,
    raw_visual: pd.DataFrame,
    raw_audio: pd.DataFrame,
    raw_matched: pd.DataFrame,
    raw_unmatched_visual: pd.DataFrame,
    raw_unmatched_audio: pd.DataFrame,
) -> pd.DataFrame:
    early_tol = float(cfg["timing"]["early_tolerance_sec"])
    rows = []
    for _, m in manifest.iterrows():
        sid = str(m["sample_id"])
        vv = raw_visual[raw_visual["video_id"].astype(str) == sid] if not raw_visual.empty else pd.DataFrame()
        aa = raw_audio[raw_audio["video_id"].astype(str) == sid] if not raw_audio.empty else pd.DataFrame()
        mm = raw_matched[raw_matched["video_id"].astype(str) == sid] if not raw_matched.empty else pd.DataFrame()
        uv = raw_unmatched_visual[raw_unmatched_visual["video_id"].astype(str) == sid] if not raw_unmatched_visual.empty else pd.DataFrame()
        ua = raw_unmatched_audio[raw_unmatched_audio["video_id"].astype(str) == sid] if not raw_unmatched_audio.empty else pd.DataFrame()
        delays = pd.to_numeric(mm.get("av_offset_sec", pd.Series(dtype=float)), errors="coerce").dropna().astype(float).tolist() if not mm.empty else []
        seq_stats = {
            "sequence_applicable": False,
            "sequence_alignment_score": math.nan,
            "matched_sequence_coverage": math.nan,
            "order_inversion_count": math.nan,
            "order_inversion_rate": math.nan,
            "kendall_tau": math.nan,
            "spearman_rank_corr": math.nan,
            "matched_visual_order": "",
            "matched_audio_order": "",
            "visual_order": "",
            "audio_order": "",
            "sequence_weak_evidence": True,
            "sequence_review_reason": "raw_detector_diagnostic_only",
        }
        rows.append(
            {
                "sample_id": sid,
                "video_id": sid,
                "matching_mode": "raw",
                "n_visual_events_raw": int(len(vv)),
                "n_audio_events_raw": int(len(aa)),
                "n_raw_matched_events": int(len(mm)),
                "n_raw_unmatched_visual_events": int(len(uv)),
                "n_raw_unmatched_audio_events": int(len(ua)),
                "raw_missing_event_rate": float(len(uv) / max(len(vv), 1)),
                "raw_hallucinated_event_rate": float(len(ua) / max(len(aa), 1)),
                "raw_causality_violation_rate": float(sum(d < _violation_threshold(cfg) for d in delays) / max(len(delays), 1)),
                "raw_early_audio_rate": float(sum(d < 0 for d in delays) / max(len(delays), 1)),
                "raw_sync_valid_rate": float(sum((-early_tol) <= d <= float(cfg["timing"]["sync_tolerance_sec"]) for d in delays) / max(len(delays), 1)),
                "raw_delay_mean_sec": mean_median_std(delays)["mean"],
                "raw_delay_median_sec": mean_median_std(delays)["median"],
                "raw_delay_std_sec": mean_median_std(delays)["std"],
                "raw_delay_p10_sec": mean_median_std(delays)["p10"],
                "raw_delay_p90_sec": mean_median_std(delays)["p90"],
                "manual_review_needed": bool(len(uv) > 0 or len(ua) > 0),
                "review_reason": "raw_detector_disagreement" if (len(uv) > 0 or len(ua) > 0) else "",
                "video_path": str(m.get("video_path", "")),
                "event_clip_path": str(m.get("event_clip_path", "")),
                "event_audio_path": str(m.get("event_audio_path", "")),
                "seed_category": str(m.get("seed_category", "")),
                "source_type": str(m.get("source_type", "")),
                "title": str(m.get("title", "")),
                **seq_stats,
            }
        )
    return pd.DataFrame(rows)


def _build_clustered_mode_metrics(
    cfg: dict,
    mode: str,
    manifest: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    matched_df: pd.DataFrame,
    unmatched_v: pd.DataFrame,
    unmatched_a: pd.DataFrame,
    raw_visual: pd.DataFrame,
    raw_audio: pd.DataFrame,
    raw_unmatched_visual: pd.DataFrame,
    raw_unmatched_audio: pd.DataFrame,
) -> pd.DataFrame:
    mode_cfg = _mode_cfg(cfg, mode)
    rows = []
    if matched_df is None:
        matched_df = pd.DataFrame()
    if unmatched_v is None:
        unmatched_v = pd.DataFrame()
    if unmatched_a is None:
        unmatched_a = pd.DataFrame()
    for _, m in manifest.iterrows():
        sid = str(m["sample_id"])
        cs = cluster_summary[cluster_summary["sample_id"].astype(str) == sid] if not cluster_summary.empty else pd.DataFrame()
        n_visual_raw = int(raw_visual[raw_visual["video_id"].astype(str) == sid].shape[0]) if not raw_visual.empty else 0
        n_audio_raw = int(raw_audio[raw_audio["video_id"].astype(str) == sid].shape[0]) if not raw_audio.empty else 0
        n_visual_clusters = int(cs["n_visual_clusters"].iloc[0]) if not cs.empty else 0
        n_audio_clusters = int(cs["n_audio_clusters"].iloc[0]) if not cs.empty else 0
        mm = matched_df[matched_df["sample_id"].astype(str) == sid].copy() if not matched_df.empty else pd.DataFrame()
        if not mm.empty and "link_role" in mm.columns:
            core = mm[mm["link_role"].astype(str) == "core"].copy()
        elif not mm.empty:
            core = mm.copy()
            core["link_role"] = "core"
        else:
            core = pd.DataFrame()
        uv = unmatched_v[unmatched_v["sample_id"].astype(str) == sid].copy() if not unmatched_v.empty else pd.DataFrame()
        ua = unmatched_a[unmatched_a["sample_id"].astype(str) == sid].copy() if not unmatched_a.empty else pd.DataFrame()
        delays = pd.to_numeric(core.get("delay_sec", pd.Series(dtype=float)), errors="coerce").dropna().astype(float).tolist() if not core.empty else []
        n_matched = int(len(core))
        n_unmatched_v = int(len(uv))
        n_unmatched_a = int(len(ua))
        coverage = n_matched / max(min(n_visual_clusters, n_audio_clusters), 1)
        sync_valid_visual_cluster_ids = {
            str(row["visual_cluster_id"])
            for _, row in core.iterrows()
            if bool(row.get("is_sync_valid", False))
        }
        sync_valid_audio_cluster_ids = {
            str(row["audio_cluster_id"])
            for _, row in core.iterrows()
            if bool(row.get("is_sync_valid", False))
        }
        visual_sync_valid_rate = float(len(sync_valid_visual_cluster_ids) / max(n_visual_clusters, 1))
        audio_sync_valid_rate = float(len(sync_valid_audio_cluster_ids) / max(n_audio_clusters, 1))
        seq_stats = _sequence_stats_from_core_matches(core, n_visual_clusters=n_visual_clusters, n_audio_clusters=n_audio_clusters)
        review_reasons = []
        if n_matched < 2:
            review_reasons.append("insufficient_matched_events")
        if seq_stats["sequence_weak_evidence"]:
            review_reasons.append("weak_sequence_evidence")
        if pd.to_numeric(core.get("match_confidence", pd.Series(dtype=float)), errors="coerce").fillna(0).lt(0.35).any():
            review_reasons.append("low_match_confidence")
        if not core.empty and (core["label_compatible"] == False).any():
            review_reasons.append("label_incompatibility")
        if n_unmatched_v > 0 or n_unmatched_a > 0:
            review_reasons.append("unmatched_clusters_present")
        rows.append(
            {
                "sample_id": sid,
                "video_id": sid,
                "matching_mode": mode,
                "n_visual_events_raw": n_visual_raw,
                "n_audio_events_raw": n_audio_raw,
                "n_visual_clusters": n_visual_clusters,
                "n_audio_clusters": n_audio_clusters,
                "n_matched_clusters": n_matched,
                "n_unmatched_visual_clusters": n_unmatched_v,
                "n_unmatched_audio_clusters": n_unmatched_a,
                "causality_violation_rate": float(sum(d < _violation_threshold(cfg) for d in delays) / max(len(delays), 1)),
                "early_audio_rate": float(sum(d < 0 for d in delays) / max(len(delays), 1)),
                "sync_valid_rate": float(sum((-mode_cfg["early_tolerance_sec"]) <= d <= mode_cfg["sync_tolerance_sec"] for d in delays) / max(len(delays), 1)),
                "visual_sync_valid_rate": visual_sync_valid_rate,
                "audio_sync_valid_rate": audio_sync_valid_rate,
                "delay_mean_sec": mean_median_std(delays)["mean"],
                "delay_median_sec": mean_median_std(delays)["median"],
                "delay_std_sec": mean_median_std(delays)["std"],
                "missing_cluster_rate": float(n_unmatched_v / max(n_visual_clusters, 1)),
                "hallucinated_cluster_rate": float(n_unmatched_a / max(n_audio_clusters, 1)),
                "visual_cluster_recall_by_audio": float(n_matched / max(n_visual_clusters, 1)),
                "audio_cluster_precision_by_visual": float(n_matched / max(n_audio_clusters, 1)),
                "sequence_alignment_score": seq_stats["sequence_alignment_score"],
                "matched_sequence_coverage": float(coverage),
                "order_inversion_count": seq_stats["order_inversion_count"],
                "order_inversion_rate": seq_stats["order_inversion_rate"],
                "kendall_tau": seq_stats["kendall_tau"],
                "spearman_rank_corr": seq_stats["spearman_rank_corr"],
                "sequence_applicable": seq_stats["sequence_applicable"],
                "sequence_weak_evidence": seq_stats["sequence_weak_evidence"],
                "manual_review_needed": bool(len(review_reasons) > 0),
                "review_reason": ";".join(sorted(set(review_reasons))),
                "seed_category": str(m.get("seed_category", "")),
                "source_type": str(m.get("source_type", "")),
                "title": str(m.get("title", "")),
                "video_path": str(m.get("video_path", "")),
                "event_clip_path": str(m.get("event_clip_path", "")),
                "event_audio_path": str(m.get("event_audio_path", "")),
                "raw_missing_event_rate": float(len(raw_unmatched_visual[raw_unmatched_visual["video_id"].astype(str) == sid]) / max(n_visual_raw, 1)) if not raw_unmatched_visual.empty else 0.0,
                "raw_hallucinated_event_rate": float(len(raw_unmatched_audio[raw_unmatched_audio["video_id"].astype(str) == sid]) / max(n_audio_raw, 1)) if not raw_unmatched_audio.empty else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    return df


def compute_time_causality_metrics(
    cfg: dict,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    skip_existing: bool = False,
) -> dict:
    project_root, out_root = _base_paths(cfg, project_root, output_root)
    metrics_dir = ensure_dir(out_root / "metrics")
    raw_video_csv = metrics_dir / "raw_detector_video_metrics.csv"
    strict_csv = metrics_dir / "clustered_strict_video_metrics.csv"
    normal_csv = metrics_dir / "clustered_normal_video_metrics.csv"
    relaxed_csv = metrics_dir / "clustered_relaxed_video_metrics.csv"
    event_csv = metrics_dir / "time_causality_event_metrics.csv"
    summary_json = metrics_dir / "time_causality_summary.json"
    if skip_existing and raw_video_csv.exists() and strict_csv.exists() and normal_csv.exists() and relaxed_csv.exists() and event_csv.exists() and summary_json.exists():
        return {
            "raw_detector_video_metrics": str(raw_video_csv),
            "clustered_strict_video_metrics": str(strict_csv),
            "clustered_normal_video_metrics": str(normal_csv),
            "clustered_relaxed_video_metrics": str(relaxed_csv),
            "event_metrics": str(event_csv),
            "summary": str(summary_json),
        }

    manifest = _load_csv_or_empty(out_root / "manifests" / "time_causality_manifest.csv")
    if manifest.empty:
        manifest = build_time_causality_manifest(cfg, project_root, out_root, skip_existing=skip_existing)
    cluster_summary = _load_csv_or_empty(out_root / "parsed_events" / "event_cluster_summary.csv")
    if cluster_summary.empty:
        cluster_events(cfg, project_root, out_root, skip_existing=skip_existing)
        cluster_summary = _load_csv_or_empty(out_root / "parsed_events" / "event_cluster_summary.csv")

    raw_visual = _load_csv_or_empty(project_root / "outputs_full" / "ov_avel" / "visual_events.csv")
    raw_audio = _load_csv_or_empty(project_root / "outputs_full" / "flexsed" / "audio_events.csv")
    raw_matched = _load_csv_or_empty(project_root / "outputs_full" / "matched_events" / "matched_av_events.csv")
    raw_unmatched_v = _load_csv_or_empty(project_root / "outputs_full" / "matched_events" / "unmatched_visual_events.csv")
    raw_unmatched_a = _load_csv_or_empty(project_root / "outputs_full" / "matched_events" / "unmatched_audio_events.csv")

    raw_video_df = _build_raw_detector_metrics(cfg, project_root, manifest, raw_visual, raw_audio, raw_matched, raw_unmatched_v, raw_unmatched_a)
    write_csv_df(raw_video_csv, raw_video_df)

    mode_frames = {}
    event_rows = []
    for mode in ["strict", "normal", "relaxed"]:
        mode_dir = out_root / "matched_events" / mode
        matched_df = _load_csv_or_empty(mode_dir / "causality_matched_events.csv")
        uv_df = _load_csv_or_empty(mode_dir / "unmatched_visual_clusters.csv")
        ua_df = _load_csv_or_empty(mode_dir / "unmatched_audio_clusters.csv")
        video_df = _build_clustered_mode_metrics(
            cfg,
            mode,
            manifest,
            cluster_summary,
            matched_df,
            uv_df,
            ua_df,
            raw_visual,
            raw_audio,
            raw_unmatched_v,
            raw_unmatched_a,
        )
        mode_frames[mode] = video_df
        write_csv_df(metrics_dir / f"clustered_{mode}_video_metrics.csv", video_df)
        if mode == cfg["reporting"]["primary_matching_mode"]:
            if not matched_df.empty:
                if "link_role" in matched_df.columns:
                    core = matched_df[matched_df["link_role"].astype(str) == "core"].copy()
                else:
                    core = matched_df.copy()
                for _, e in core.iterrows():
                    event_rows.append(
                        {
                            "sample_id": e.get("sample_id", ""),
                            "video_id": e.get("video_id", ""),
                            "matching_mode": mode,
                            "visual_cluster_id": e.get("visual_cluster_id", ""),
                            "audio_cluster_id": e.get("audio_cluster_id", ""),
                            "visual_label": e.get("visual_label", ""),
                            "audio_label": e.get("audio_label", ""),
                            "visual_time_sec": e.get("visual_time_sec", ""),
                            "audio_onset_sec": e.get("audio_onset_sec", ""),
                            "delay_sec": e.get("delay_sec", ""),
                            "label_compatible": e.get("label_compatible", ""),
                            "match_confidence": e.get("match_confidence", ""),
                            "is_causality_violation": e.get("is_causality_violation", ""),
                            "is_sync_valid": e.get("is_sync_valid", ""),
                        }
                    )

    event_df = pd.DataFrame(event_rows)
    if event_df.empty:
        event_df = pd.DataFrame(columns=["sample_id", "video_id", "matching_mode", "visual_cluster_id", "audio_cluster_id", "visual_label", "audio_label", "visual_time_sec", "audio_onset_sec", "delay_sec", "label_compatible", "match_confidence", "is_causality_violation", "is_sync_valid"])
    write_csv_df(event_csv, event_df)

    def _mean_median(df: pd.DataFrame, col: str):
        vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).tolist() if col in df.columns else []
        return {"mean": float(np.mean(vals)) if vals else math.nan, "median": float(np.median(vals)) if vals else math.nan}

    def _full_stats(df: pd.DataFrame, col: str):
        vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).tolist() if col in df.columns else []
        return mean_median_std(vals) if vals else {"mean": math.nan, "median": math.nan, "std": math.nan, "p10": math.nan, "p90": math.nan}

    summary = {
        "num_samples": int(len(manifest)),
        "raw_visual_events": int(raw_video_df["n_visual_events_raw"].sum()) if "n_visual_events_raw" in raw_video_df.columns else 0,
        "raw_audio_events": int(raw_video_df["n_audio_events_raw"].sum()) if "n_audio_events_raw" in raw_video_df.columns else 0,
        "visual_clusters": int(mode_frames["normal"]["n_visual_clusters"].sum()) if "n_visual_clusters" in mode_frames["normal"].columns else 0,
        "audio_clusters": int(mode_frames["normal"]["n_audio_clusters"].sum()) if "n_audio_clusters" in mode_frames["normal"].columns else 0,
        "raw_matched_events": int(raw_video_df["n_raw_matched_events"].sum()) if "n_raw_matched_events" in raw_video_df.columns else 0,
        "clustered_strict_matched": int((mode_frames["strict"].get("n_matched_clusters", pd.Series(dtype=float)).sum()) if "strict" in mode_frames else 0),
        "clustered_normal_matched": int((mode_frames["normal"].get("n_matched_clusters", pd.Series(dtype=float)).sum()) if "normal" in mode_frames else 0),
        "clustered_relaxed_matched": int((mode_frames["relaxed"].get("n_matched_clusters", pd.Series(dtype=float)).sum()) if "relaxed" in mode_frames else 0),
        "raw_missing_event_rate": _mean_median(raw_video_df, "raw_missing_event_rate"),
        "raw_hallucinated_event_rate": _mean_median(raw_video_df, "raw_hallucinated_event_rate"),
        "raw_causality_violation_rate": _mean_median(raw_video_df, "raw_causality_violation_rate"),
        "clustered_strict_missing_cluster_rate": _mean_median(mode_frames["strict"], "missing_cluster_rate"),
        "clustered_normal_missing_cluster_rate": _mean_median(mode_frames["normal"], "missing_cluster_rate"),
        "clustered_relaxed_missing_cluster_rate": _mean_median(mode_frames["relaxed"], "missing_cluster_rate"),
        "clustered_strict_hallucinated_cluster_rate": _mean_median(mode_frames["strict"], "hallucinated_cluster_rate"),
        "clustered_normal_hallucinated_cluster_rate": _mean_median(mode_frames["normal"], "hallucinated_cluster_rate"),
        "clustered_relaxed_hallucinated_cluster_rate": _mean_median(mode_frames["relaxed"], "hallucinated_cluster_rate"),
        "clustered_strict_causality_violation_rate": _mean_median(mode_frames["strict"], "causality_violation_rate"),
        "clustered_normal_causality_violation_rate": _mean_median(mode_frames["normal"], "causality_violation_rate"),
        "clustered_relaxed_causality_violation_rate": _mean_median(mode_frames["relaxed"], "causality_violation_rate"),
        "clustered_normal_onset_delay_sec": _full_stats(event_df, "delay_sec"),
        "clustered_strict_delay_mean_sec": _mean_median(mode_frames["strict"], "delay_mean_sec"),
        "clustered_normal_delay_mean_sec": _mean_median(mode_frames["normal"], "delay_mean_sec"),
        "clustered_relaxed_delay_mean_sec": _mean_median(mode_frames["relaxed"], "delay_mean_sec"),
        "clustered_strict_delay_median_sec": _mean_median(mode_frames["strict"], "delay_median_sec"),
        "clustered_normal_delay_median_sec": _mean_median(mode_frames["normal"], "delay_median_sec"),
        "clustered_relaxed_delay_median_sec": _mean_median(mode_frames["relaxed"], "delay_median_sec"),
        "clustered_strict_delay_std_sec": _mean_median(mode_frames["strict"], "delay_std_sec"),
        "clustered_normal_delay_std_sec": _mean_median(mode_frames["normal"], "delay_std_sec"),
        "clustered_relaxed_delay_std_sec": _mean_median(mode_frames["relaxed"], "delay_std_sec"),
        "clustered_strict_sync_valid_rate": _mean_median(mode_frames["strict"], "sync_valid_rate"),
        "clustered_normal_sync_valid_rate": _mean_median(mode_frames["normal"], "sync_valid_rate"),
        "clustered_relaxed_sync_valid_rate": _mean_median(mode_frames["relaxed"], "sync_valid_rate"),
        "clustered_strict_visual_sync_valid_rate": _mean_median(mode_frames["strict"], "visual_sync_valid_rate"),
        "clustered_normal_visual_sync_valid_rate": _mean_median(mode_frames["normal"], "visual_sync_valid_rate"),
        "clustered_relaxed_visual_sync_valid_rate": _mean_median(mode_frames["relaxed"], "visual_sync_valid_rate"),
        "clustered_strict_audio_sync_valid_rate": _mean_median(mode_frames["strict"], "audio_sync_valid_rate"),
        "clustered_normal_audio_sync_valid_rate": _mean_median(mode_frames["normal"], "audio_sync_valid_rate"),
        "clustered_relaxed_audio_sync_valid_rate": _mean_median(mode_frames["relaxed"], "audio_sync_valid_rate"),
        "clustered_strict_sequence_alignment_score": _mean_median(mode_frames["strict"], "sequence_alignment_score"),
        "clustered_normal_sequence_alignment_score": _mean_median(mode_frames["normal"], "sequence_alignment_score"),
        "clustered_relaxed_sequence_alignment_score": _mean_median(mode_frames["relaxed"], "sequence_alignment_score"),
        "clustered_strict_matched_sequence_coverage": _mean_median(mode_frames["strict"], "matched_sequence_coverage"),
        "clustered_normal_matched_sequence_coverage": _mean_median(mode_frames["normal"], "matched_sequence_coverage"),
        "clustered_relaxed_matched_sequence_coverage": _mean_median(mode_frames["relaxed"], "matched_sequence_coverage"),
        "sequence_not_applicable_samples": int((mode_frames["normal"]["sequence_applicable"] == False).sum()) if "sequence_applicable" in mode_frames["normal"].columns else 0,
        "manual_review_cases": int((mode_frames["normal"]["manual_review_needed"] == True).sum()) if "manual_review_needed" in mode_frames["normal"].columns else 0,
        "mode_counts": {mode: int(len(df)) for mode, df in mode_frames.items()},
    }
    write_json(summary_json, summary)
    return {
        "raw_detector_video_metrics": str(raw_video_csv),
        "clustered_strict_video_metrics": str(strict_csv),
        "clustered_normal_video_metrics": str(normal_csv),
        "clustered_relaxed_video_metrics": str(relaxed_csv),
        "event_metrics": str(event_csv),
        "summary": str(summary_json),
    }


def analyze_time_causality_results(
    cfg: dict,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    skip_existing: bool = False,
) -> dict:
    project_root, out_root = _base_paths(cfg, project_root, output_root)
    summary_dir = ensure_dir(out_root / "summaries")
    fig_dir = ensure_dir(summary_dir / "figures")
    md_path = summary_dir / "time_causality_summary.md"
    json_path = summary_dir / "time_causality_summary.json"
    if skip_existing and md_path.exists() and json_path.exists():
        return {"summary_md": str(md_path), "summary_json": str(json_path), "figures": str(fig_dir)}

    metrics_dir = out_root / "metrics"
    raw_video = _load_csv_or_empty(metrics_dir / "raw_detector_video_metrics.csv")
    strict_video = _load_csv_or_empty(metrics_dir / "clustered_strict_video_metrics.csv")
    normal_video = _load_csv_or_empty(metrics_dir / "clustered_normal_video_metrics.csv")
    relaxed_video = _load_csv_or_empty(metrics_dir / "clustered_relaxed_video_metrics.csv")
    event_df = _load_csv_or_empty(metrics_dir / "time_causality_event_metrics.csv")
    if normal_video.empty or event_df.empty or raw_video.empty:
        compute_time_causality_metrics(cfg, project_root, out_root, skip_existing=skip_existing)
        raw_video = _load_csv_or_empty(metrics_dir / "raw_detector_video_metrics.csv")
        strict_video = _load_csv_or_empty(metrics_dir / "clustered_strict_video_metrics.csv")
        normal_video = _load_csv_or_empty(metrics_dir / "clustered_normal_video_metrics.csv")
        relaxed_video = _load_csv_or_empty(metrics_dir / "clustered_relaxed_video_metrics.csv")
        event_df = _load_csv_or_empty(metrics_dir / "time_causality_event_metrics.csv")

    def _mm(df: pd.DataFrame, col: str):
        return mean_median_std(pd.to_numeric(df[col], errors="coerce").dropna().astype(float).tolist()) if not df.empty and col in df.columns else {"mean": math.nan, "median": math.nan}

    summary = {
        "num_samples": int(len(normal_video)),
        "raw_visual_events": int(raw_video["n_visual_events_raw"].sum()) if "n_visual_events_raw" in raw_video.columns else 0,
        "raw_audio_events": int(raw_video["n_audio_events_raw"].sum()) if "n_audio_events_raw" in raw_video.columns else 0,
        "visual_clusters": int(normal_video["n_visual_clusters"].sum()) if "n_visual_clusters" in normal_video.columns else 0,
        "audio_clusters": int(normal_video["n_audio_clusters"].sum()) if "n_audio_clusters" in normal_video.columns else 0,
        "raw_matched_events": int(raw_video["n_raw_matched_events"].sum()) if "n_raw_matched_events" in raw_video.columns else 0,
        "clustered_strict_matched": int(strict_video["n_matched_clusters"].sum()) if "n_matched_clusters" in strict_video.columns else 0,
        "clustered_normal_matched": int(normal_video["n_matched_clusters"].sum()) if "n_matched_clusters" in normal_video.columns else 0,
        "clustered_relaxed_matched": int(relaxed_video["n_matched_clusters"].sum()) if "n_matched_clusters" in relaxed_video.columns else 0,
        "raw_missing_event_rate": _mm(raw_video, "raw_missing_event_rate"),
        "raw_hallucinated_event_rate": _mm(raw_video, "raw_hallucinated_event_rate"),
        "clustered_strict_missing_cluster_rate": _mm(strict_video, "missing_cluster_rate"),
        "clustered_normal_missing_cluster_rate": _mm(normal_video, "missing_cluster_rate"),
        "clustered_relaxed_missing_cluster_rate": _mm(relaxed_video, "missing_cluster_rate"),
        "clustered_strict_hallucinated_cluster_rate": _mm(strict_video, "hallucinated_cluster_rate"),
        "clustered_normal_hallucinated_cluster_rate": _mm(normal_video, "hallucinated_cluster_rate"),
        "clustered_relaxed_hallucinated_cluster_rate": _mm(relaxed_video, "hallucinated_cluster_rate"),
        "clustered_strict_causality_violation_rate": _mm(strict_video, "causality_violation_rate"),
        "clustered_normal_causality_violation_rate": _mm(normal_video, "causality_violation_rate"),
        "clustered_relaxed_causality_violation_rate": _mm(relaxed_video, "causality_violation_rate"),
        "clustered_normal_onset_delay_sec": _mm(event_df, "delay_sec"),
        "clustered_strict_delay_mean_sec": _mm(strict_video, "delay_mean_sec"),
        "clustered_normal_delay_mean_sec": _mm(normal_video, "delay_mean_sec"),
        "clustered_relaxed_delay_mean_sec": _mm(relaxed_video, "delay_mean_sec"),
        "clustered_strict_delay_median_sec": _mm(strict_video, "delay_median_sec"),
        "clustered_normal_delay_median_sec": _mm(normal_video, "delay_median_sec"),
        "clustered_relaxed_delay_median_sec": _mm(relaxed_video, "delay_median_sec"),
        "clustered_strict_delay_std_sec": _mm(strict_video, "delay_std_sec"),
        "clustered_normal_delay_std_sec": _mm(normal_video, "delay_std_sec"),
        "clustered_relaxed_delay_std_sec": _mm(relaxed_video, "delay_std_sec"),
        "clustered_strict_sync_valid_rate": _mm(strict_video, "sync_valid_rate"),
        "clustered_normal_sync_valid_rate": _mm(normal_video, "sync_valid_rate"),
        "clustered_relaxed_sync_valid_rate": _mm(relaxed_video, "sync_valid_rate"),
        "clustered_strict_visual_sync_valid_rate": _mm(strict_video, "visual_sync_valid_rate"),
        "clustered_normal_visual_sync_valid_rate": _mm(normal_video, "visual_sync_valid_rate"),
        "clustered_relaxed_visual_sync_valid_rate": _mm(relaxed_video, "visual_sync_valid_rate"),
        "clustered_strict_audio_sync_valid_rate": _mm(strict_video, "audio_sync_valid_rate"),
        "clustered_normal_audio_sync_valid_rate": _mm(normal_video, "audio_sync_valid_rate"),
        "clustered_relaxed_audio_sync_valid_rate": _mm(relaxed_video, "audio_sync_valid_rate"),
        "clustered_strict_sequence_alignment_score": _mm(strict_video, "sequence_alignment_score"),
        "clustered_normal_sequence_alignment_score": _mm(normal_video, "sequence_alignment_score"),
        "clustered_relaxed_sequence_alignment_score": _mm(relaxed_video, "sequence_alignment_score"),
        "clustered_strict_matched_sequence_coverage": _mm(strict_video, "matched_sequence_coverage"),
        "clustered_normal_matched_sequence_coverage": _mm(normal_video, "matched_sequence_coverage"),
        "clustered_relaxed_matched_sequence_coverage": _mm(relaxed_video, "matched_sequence_coverage"),
        "sequence_not_applicable_samples": int((normal_video["sequence_applicable"] == False).sum()) if "sequence_applicable" in normal_video.columns else 0,
        "manual_review_cases": int((normal_video["manual_review_needed"] == True).sum()) if "manual_review_needed" in normal_video.columns else 0,
        "mode_counts": {
            "strict": int(len(strict_video)),
            "normal": int(len(normal_video)),
            "relaxed": int(len(relaxed_video)),
        },
    }

    save_histogram_png(pd.to_numeric(event_df["delay_sec"], errors="coerce"), fig_dir / "delay_distribution_by_mode.png", "Delay Distribution", "delay_sec")
    save_histogram_png(pd.to_numeric(normal_video["causality_violation_rate"], errors="coerce"), fig_dir / "causality_violation_rate_histogram.png", "Causality Violation Rate", "rate")

    md_lines = [
        "# Time and Causality Summary",
        "",
        "Current Time and Causality evaluation is built on OV-AVEL and FlexSED expert-model outputs.",
        "The main report keeps only onset delay and causality violation rate.",
        "Other outputs are retained only for internal diagnostics.",
        "",
        f"- Total samples: {summary['num_samples']}",
        f"- Normal onset delay mean/median/std: {summary['clustered_normal_onset_delay_sec']['mean']:.4f} / {summary['clustered_normal_onset_delay_sec']['median']:.4f} / {summary['clustered_normal_onset_delay_sec']['std']:.4f}",
        f"- Normal causality violation mean/median: {summary['clustered_normal_causality_violation_rate']['mean']:.4f} / {summary['clustered_normal_causality_violation_rate']['median']:.4f}",
        "",
        "1. Matching tolerance and the 1 ms scoring threshold are distinct.",
        "2. This is not a manually annotated GT benchmark.",
        "3. Papers should describe it as expert-model-assisted TP1 validation.",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    write_json(json_path, summary)
    return {"summary_md": str(md_path), "summary_json": str(json_path), "figures": str(fig_dir)}


def export_time_causality_review_cases(
    cfg: dict,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
    skip_existing: bool = False,
) -> dict:
    project_root, out_root = _base_paths(cfg, project_root, output_root)
    review_dir = ensure_dir(out_root / "review_cases")
    for stale_name in [
        "good_cases.csv",
        "causality_failure_cases.csv",
        "manual_review_cases.csv",
        "sequence_failure_cases.csv",
        "missing_event_cases.csv",
        "hallucinated_event_cases.csv",
    ]:
        stale_path = review_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    video_csv = out_root / "metrics" / "clustered_normal_video_metrics.csv"
    if not video_csv.exists():
        analyze_time_causality_results(cfg, project_root, out_root, skip_existing=skip_existing)
    video_df = _load_csv_or_empty(video_csv)
    if video_df.empty:
        return {"review_dir": str(review_dir), "selection_counts": {}}

    def _prepare_case_df(df: pd.DataFrame, failure_type: str, n: int) -> pd.DataFrame:
        if df.empty:
            return df
        cols = [c for c in [
            "sample_id",
            "video_id",
            "video_path",
            "event_clip_path",
            "event_audio_path",
            "delay_mean_sec",
            "delay_median_sec",
            "delay_std_sec",
            "causality_violation_rate",
            "review_reason",
        ] if c in df.columns]
        out = df[cols].copy()
        out["failure_type"] = failure_type
        out["notes"] = failure_type.replace("_", " ")
        return out.head(n)

    good = video_df.copy()
    good["good_score"] = (
        pd.to_numeric(good["causality_violation_rate"], errors="coerce").fillna(1.0)
        + pd.to_numeric(good["delay_mean_sec"], errors="coerce").fillna(1.0).clip(lower=0)
    )
    good = good.sort_values(by=["good_score", "sample_id"], kind="stable")
    selections = {
        "good_cases": _prepare_case_df(good, "good_cases", int(cfg["reporting"]["max_good_cases"])),
        "causality_failure_cases": _prepare_case_df(video_df[pd.to_numeric(video_df["causality_violation_rate"], errors="coerce").fillna(0) > 0].sort_values(by=["delay_mean_sec"], kind="stable"), "causality_failure_cases", int(cfg["reporting"]["max_bad_cases"])),
        "manual_review_cases": _prepare_case_df(video_df[video_df.get("manual_review_needed", pd.Series(dtype=bool)).fillna(False) == True], "manual_review_cases", int(cfg["reporting"]["max_manual_review_cases"])),
    }
    for name, df in selections.items():
        df.to_csv(review_dir / f"{name}.csv", index=False)
    return {"review_dir": str(review_dir), "selection_counts": {k: int(len(v)) for k, v in selections.items()}}


def load_config_and_run(args: argparse.Namespace):
    cfg = load_yaml(args.config)
    project_root = args.project_root or cfg["input"]["project_root"]
    output_root = args.output_root or cfg["output"]["output_root"]
    return cfg, project_root, output_root
