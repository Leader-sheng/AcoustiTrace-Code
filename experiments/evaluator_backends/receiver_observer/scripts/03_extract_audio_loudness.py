from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import clean_text, ensure_dir, extract_audio, load_yaml, read_csv_dicts, safe_float, write_csv_dicts


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "receiver_observer_eval_config.yaml"


def load_audio(audio_path: Path, sr: int) -> tuple[np.ndarray, int]:
    import librosa

    y, sr_loaded = librosa.load(str(audio_path), sr=sr, mono=True)
    return y.astype(np.float32), sr_loaded


def compute_curves(y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> dict:
    import librosa

    if len(y) < frame_length:
        return {"status": "too_short"}
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    loudness = db.copy()
    dynamic_range = float(np.nanpercentile(db, 95) - np.nanpercentile(db, 5))
    silence_ratio = float(np.mean(db < (np.nanmedian(db) - 20.0)))
    return {
        "status": "success",
        "frame_times_sec": times,
        "rms_curve": rms,
        "spl_curve_db": db,
        "loudness_curve_db": loudness,
        "dynamic_range_db": dynamic_range,
        "silence_ratio": silence_ratio,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--output-root", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    manifest = Path(args.manifest or (Path(cfg["output"]["root"]) / "manifests" / "receiver_observer_eval_manifest.csv"))
    output_root = Path(args.output_root or (Path(cfg["output"]["root"]) / cfg["audio"]["output_subdir"]))
    ensure_dir(output_root)

    rows = read_csv_dicts(manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    status_rows = []
    errors = []
    for row in rows:
        sample_id = clean_text(row.get("sample_id"), "")
        if not sample_id or clean_text(row.get("status"), "") != "ok":
            status_rows.append({"sample_id": sample_id, "status": "skipped", "error_message": clean_text(row.get("skip_reason"), "manifest_skip")})
            continue

        event_audio_raw = clean_text(row.get("event_audio_path"), "")
        event_audio_path = Path(event_audio_raw) if event_audio_raw else None
        event_clip_path = Path(clean_text(row.get("event_clip_path"), clean_text(row.get("video_path"), "")))
        sample_out = output_root / sample_id
        ensure_dir(sample_out)
        npz_path = sample_out / "audio_features.npz"
        json_path = sample_out / "audio_features.json"
        if args.skip_existing and npz_path.exists() and json_path.exists():
            status_rows.append({"sample_id": sample_id, "status": "skipped_existing", "audio_src": str(audio_src) if 'audio_src' in locals() else "", "npz_path": str(npz_path), "json_path": str(json_path), "dynamic_range_db": "", "silence_ratio": ""})
            continue

        audio_src = event_audio_path if event_audio_path is not None and event_audio_path.is_file() else None
        if audio_src is None and event_clip_path.exists():
            extracted = sample_out / "extracted_audio.wav"
            if extract_audio(event_clip_path, extracted, sr=cfg["audio"]["sample_rate"]):
                audio_src = extracted
        if audio_src is None or not audio_src.exists():
            status_rows.append({"sample_id": sample_id, "status": "failed", "error_message": "missing_audio"})
            errors.append({"sample_id": sample_id, "error": "missing_audio"})
            continue

        try:
            y, sr = load_audio(audio_src, cfg["audio"]["sample_rate"])
            curves = compute_curves(y, sr, cfg["audio"]["frame_length"], cfg["audio"]["hop_length"])
            if curves.get("status") != "success":
                status_rows.append({"sample_id": sample_id, "status": curves["status"], "error_message": curves["status"]})
                errors.append({"sample_id": sample_id, "error": curves["status"]})
                continue
            np.savez_compressed(
                npz_path,
                frame_times_sec=curves["frame_times_sec"],
                rms_curve=curves["rms_curve"],
                spl_curve_db=curves["spl_curve_db"],
                loudness_curve_db=curves["loudness_curve_db"],
            )
            json_path.write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "audio_src": str(audio_src),
                        "frame_times_sec": curves["frame_times_sec"].tolist(),
                        "rms_curve": curves["rms_curve"].tolist(),
                        "spl_curve_db": curves["spl_curve_db"].tolist(),
                        "loudness_curve_db": curves["loudness_curve_db"].tolist(),
                        "dynamic_range_db": curves["dynamic_range_db"],
                        "silence_ratio": curves["silence_ratio"],
                        "audio_valid_ratio": float(np.mean(np.isfinite(curves["spl_curve_db"]))),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            status_rows.append(
                {
                    "sample_id": sample_id,
                    "status": "success",
                    "audio_src": str(audio_src),
                    "npz_path": str(npz_path),
                    "json_path": str(json_path),
                    "dynamic_range_db": f"{curves['dynamic_range_db']:.6f}",
                    "silence_ratio": f"{curves['silence_ratio']:.6f}",
                }
            )
        except Exception as e:
            status_rows.append({"sample_id": sample_id, "status": "failed", "error_message": str(e)})
            errors.append({"sample_id": sample_id, "error": str(e)})

    write_csv_dicts(output_root / "audio_feature_status.csv", status_rows, ["sample_id", "status", "audio_src", "npz_path", "json_path", "dynamic_range_db", "silence_ratio", "error_message"])
    write_csv_dicts(output_root / "audio_feature_errors.csv", errors, ["sample_id", "error"])
    print(json.dumps({"status_csv": str(output_root / "audio_feature_status.csv"), "error_csv": str(output_root / "audio_feature_errors.csv"), "rows": len(status_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
