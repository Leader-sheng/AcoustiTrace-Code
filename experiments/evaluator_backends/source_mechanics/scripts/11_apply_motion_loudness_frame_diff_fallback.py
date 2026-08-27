from __future__ import annotations

"""Apply the 8 fps frame-difference fallback to valid Qwen Motion judgments."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


OUTPUT_FIELD = "frame_diff_fallback_postprocess_max_rms_db_no_margin"


def number(value) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def truth(value) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def frame_difference_strength(video_path: Path, sample_fps: float = 8.0) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"valid": False, "reason": "video_open_failed"}
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or sample_fps)
    stride = max(1, int(round(source_fps / sample_fps)))
    previous = None
    mean_differences: list[float] = []
    area_ratios: list[float] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride:
            frame_index += 1
            continue
        frame_index += 1
        # Match the archived evaluator: sample the original frame grid at 8 fps
        # and measure raw grayscale change.  Resizing or blurring changes the
        # visual-strength ordering and therefore is not allowed here.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None:
            difference = np.abs(gray - previous)
            mean_differences.append(float(np.mean(difference)))
            area_ratios.append(float(np.mean(difference >= 12)))
        previous = gray
    capture.release()
    if not mean_differences:
        return {"valid": False, "reason": "insufficient_frames"}
    return {
        "valid": True,
        "reason": "",
        "frame_count": len(mean_differences) + 1,
        "visual_motion_mean": float(np.mean(mean_differences)),
        "visual_motion_max": float(np.max(mean_differences)),
        "visual_motion_area_ratio": float(np.mean(area_ratios)),
        "visual_action_strength": float(np.max(mean_differences)),
    }


def compare_audio(row: pd.Series, stronger: str, metric: str) -> dict:
    first = number(row.get(f"cluster_1_{metric}"))
    second = number(row.get(f"cluster_2_{metric}"))
    prefix = f"frame_diff_dynamic_pairwise_{metric}"
    if stronger not in {"cluster_1", "cluster_2"} or not math.isfinite(first) or not math.isfinite(second):
        return {
            f"frame_diff_stronger_minus_weaker_{metric}": float("nan"),
            f"{prefix}_valid": False,
            f"{prefix}_no_margin": "",
            f"{prefix}_margin_1db": "",
            f"{prefix}_margin_3db": "",
        }
    margin = first - second if stronger == "cluster_1" else second - first
    return {
        f"frame_diff_stronger_minus_weaker_{metric}": margin,
        f"{prefix}_valid": True,
        f"{prefix}_no_margin": bool(margin > 0.0),
        f"{prefix}_margin_1db": bool(margin >= 1.0),
        f"{prefix}_margin_3db": bool(margin >= 3.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.qwen_metrics_csv)
    output_rows = []
    for _, row in source.iterrows():
        output = row.to_dict()
        first = frame_difference_strength(Path(str(row.get("cluster_1_clip_path", ""))))
        second = frame_difference_strength(Path(str(row.get("cluster_2_clip_path", ""))))
        for prefix, metrics in (("cluster_1", first), ("cluster_2", second)):
            output[f"{prefix}_frame_diff_motion_failed"] = not metrics.get("valid", False)
            output[f"{prefix}_frame_diff_motion_error"] = metrics.get("reason", "")
            for name in (
                "frame_count", "visual_motion_mean", "visual_motion_max",
                "visual_motion_area_ratio", "visual_action_strength",
            ):
                output[f"{prefix}_frame_diff_{name}"] = metrics.get(name, float("nan"))

        frame_stronger = ""
        if first.get("valid") and second.get("valid"):
            first_strength = number(first.get("visual_action_strength"))
            second_strength = number(second.get("visual_action_strength"))
            frame_stronger = "cluster_1" if first_strength >= second_strength else "cluster_2"
        output["frame_diff_judgment_status"] = "valid" if frame_stronger else "invalid"
        output["frame_diff_visual_stronger_cluster"] = frame_stronger
        output["frame_diff_c2_minus_c1_visual_action_strength"] = (
            number(second.get("visual_action_strength")) - number(first.get("visual_action_strength"))
            if first.get("valid") and second.get("valid")
            else float("nan")
        )
        output["frame_diff_stronger_minus_weaker_visual_action_strength"] = (
            abs(output["frame_diff_c2_minus_c1_visual_action_strength"])
            if frame_stronger
            else float("nan")
        )
        for metric in ("max_rms_db", "energy_sum_db", "mean_rms_db", "top2_mean_rms_db"):
            output.update(compare_audio(row, frame_stronger, metric))
        frame_audio_pass = truth(output.get("frame_diff_dynamic_pairwise_max_rms_db_no_margin"))

        qwen_status = str(row.get("qwen_visual_judgment_status", ""))
        qwen_pass = truth(row.get("qwen_dynamic_pairwise_max_rms_db_no_margin"))
        if qwen_status != "valid" or qwen_pass is None:
            final_score = float("nan")
            decision = "invalid_qwen_judgment"
            rescue = False
        elif qwen_pass:
            final_score = True
            decision = "qwen_audio_pass"
            rescue = False
        elif frame_audio_pass is True:
            final_score = True
            decision = "frame_diff_fallback_rescue"
            rescue = True
        else:
            final_score = False
            decision = "fail"
            rescue = False
        output["frame_diff_fallback_eligible"] = (
            bool(qwen_pass is False) if qwen_status == "valid" else ""
        )
        output["frame_diff_fallback_rescue"] = rescue if qwen_status == "valid" else ""
        output[OUTPUT_FIELD] = final_score
        output["frame_diff_fallback_postprocess_decision"] = decision
        output["frame_diff_fallback_postprocess_physical_score"] = (
            100.0 * float(final_score) if isinstance(final_score, bool) else float("nan")
        )
        output["frame_diff_fallback_postprocess_rule"] = (
            "if_qwen_valid_fail_then_use_frame_diff_visual_stronger_cluster_"
            "and_pass_when_that_cluster_has_higher_audio_max_rms"
        )
        output_rows.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(output_rows)
    metrics_path = args.output_dir / "motion_loudness_metrics.csv"
    result.to_csv(metrics_path, index=False)
    results_path = args.output_dir / "motion_loudness_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    scores = pd.to_numeric(result.get(OUTPUT_FIELD), errors="coerce")
    summary = {
        "input_rows": int(len(result)),
        "valid_count": int(scores.notna().sum()),
        "invalid_count": int(scores.isna().sum()),
        "fallback_rescue_count": int(result.get("frame_diff_fallback_rescue", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "physical_score": float(100.0 * scores.mean()) if scores.notna().any() else None,
        "metrics_csv": str(metrics_path),
    }
    (args.output_dir / "motion_loudness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
