from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ensure_dir, load_config, read_csv_dicts, write_csv_dicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["flexsed"]["output_dir"])
    raw_root = out_dir / "raw"
    parsed = []
    stats = []

    for sample_dir in sorted(raw_root.glob("*")):
        if not sample_dir.is_dir():
            continue
        result_csv = sample_dir / "events.csv"
        if not result_csv.exists():
            continue
        rows = read_csv_dicts(result_csv)
        if not rows:
            continue
        vid = rows[0]["video_id"]
        parsed.extend(rows)
        stats.append({"video_id": vid, "num_audio_events": len(rows)})

    write_csv_dicts(
        out_dir / "audio_events.csv",
        parsed,
        ["video_id", "video_path", "audio_event_id", "audio_label", "start_sec", "end_sec", "peak_sec", "confidence", "raw_score"],
    )
    write_csv_dicts(out_dir / "audio_event_stats.csv", stats, ["video_id", "num_audio_events"])
    summary = {
        "num_videos": len(stats),
        "num_audio_events": len(parsed),
        "output_csv": str(out_dir / "audio_events.csv"),
    }
    with open(out_dir / "audio_parse_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
