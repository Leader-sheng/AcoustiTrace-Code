from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import ensure_dir, load_config, parse_detection_lines, read_csv_dicts, write_csv_dicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["ov_avel"]["output_dir"])
    raw_root = out_dir / "raw"
    parsed = []
    per_video_stats = []

    for sample_dir in sorted(raw_root.glob("*")):
        if not sample_dir.is_dir():
            continue
        result_csv = sample_dir / "events.csv"
        result_txt = sample_dir / "detection_results.txt"
        rows = []
        if result_csv.exists():
            rows = read_csv_dicts(result_csv)
        if not rows and result_txt.exists():
            dets = parse_detection_lines(result_txt.read_text(encoding="utf-8", errors="ignore"))
            vid = sample_dir.name
            rows = [
                {
                    "video_id": vid,
                    "video_path": "",
                    "visual_event_id": f"v{i:04d}",
                    "visual_label": "foreground_event",
                    "start_sec": det["start_sec"],
                    "end_sec": det["end_sec"],
                    "peak_sec": (det["start_sec"] + det["end_sec"]) / 2.0,
                    "confidence": det["score"],
                    "raw_score": det["score"],
                }
                for i, det in enumerate(dets)
            ]
        if not rows:
            continue
        vid = rows[0]["video_id"]
        for row in rows:
            row["video_id"] = vid
            row["video_path"] = row.get("video_path", "")
            parsed.append(row)
        per_video_stats.append({"video_id": vid, "num_visual_events": len(rows)})

    write_csv_dicts(
        out_dir / "visual_events.csv",
        parsed,
        ["video_id", "video_path", "visual_event_id", "visual_label", "start_sec", "end_sec", "peak_sec", "confidence", "raw_score"],
    )
    write_csv_dicts(out_dir / "visual_event_stats.csv", per_video_stats, ["video_id", "num_visual_events"])

    summary = {
        "num_videos": len(per_video_stats),
        "num_visual_events": len(parsed),
        "output_csv": str(out_dir / "visual_events.csv"),
    }
    with open(out_dir / "visual_parse_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
