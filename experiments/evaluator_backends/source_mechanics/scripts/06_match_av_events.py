from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import load_config, read_csv_dicts, write_csv_dicts, safe_float


def greedy_match(v_events, a_events, max_offset):
    pairs = []
    candidates = []
    for vi, v in enumerate(v_events):
        for ai, a in enumerate(a_events):
            off = safe_float(a["peak_sec"]) - safe_float(v["peak_sec"])
            dist = abs(off)
            if dist <= max_offset:
                conf = safe_float(v["confidence"]) * safe_float(a["confidence"]) * np.exp(-dist / 0.2)
                candidates.append((dist, -conf, vi, ai, off, conf))
    candidates.sort()
    used_v = set()
    used_a = set()
    for _, _, vi, ai, off, conf in candidates:
        if vi in used_v or ai in used_a:
            continue
        used_v.add(vi)
        used_a.add(ai)
        pairs.append((vi, ai, off, conf))
    return pairs, used_v, used_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg["output"]["root"])
    max_offset = float(cfg["event_matching"]["max_av_offset_sec"])

    visual = read_csv_dicts(out_root / "ov_avel" / "visual_events.csv")
    audio = read_csv_dicts(out_root / "flexsed" / "audio_events.csv")
    videos = read_csv_dicts(out_root / "index" / "videos.csv")
    video_path_map = {r["video_id"]: r["video_path"] for r in videos}

    by_video_v = defaultdict(list)
    by_video_a = defaultdict(list)
    for r in visual:
        by_video_v[r["video_id"]].append(r)
    for r in audio:
        by_video_a[r["video_id"]].append(r)

    matched = []
    unmatched_v = []
    unmatched_a = []
    match_id = 0
    for vid in sorted(set(by_video_v) | set(by_video_a)):
        v_events = sorted(by_video_v.get(vid, []), key=lambda x: safe_float(x["peak_sec"]))
        a_events = sorted(by_video_a.get(vid, []), key=lambda x: safe_float(x["peak_sec"]))
        pairs, used_v, used_a = greedy_match(v_events, a_events, max_offset)
        for vi, ai, off, conf in pairs:
            v = v_events[vi]
            a = a_events[ai]
            matched.append(
                {
                    "video_id": vid,
                    "video_path": video_path_map.get(vid, v.get("video_path", "")),
                    "match_id": f"m{match_id:06d}",
                    "visual_event_id": v["visual_event_id"],
                    "audio_event_id": a["audio_event_id"],
                    "visual_label": v["visual_label"],
                    "audio_label": a["audio_label"],
                    "visual_start_sec": v["start_sec"],
                    "visual_end_sec": v["end_sec"],
                    "visual_peak_sec": v["peak_sec"],
                    "audio_start_sec": a["start_sec"],
                    "audio_end_sec": a["end_sec"],
                    "audio_peak_sec": a["peak_sec"],
                    "av_offset_sec": off,
                    "match_confidence": conf,
                }
            )
            match_id += 1
        for i, v in enumerate(v_events):
            if i not in used_v:
                unmatched_v.append(v)
        for i, a in enumerate(a_events):
            if i not in used_a:
                unmatched_a.append(a)

    matched_dir = out_root / "matched_events"
    write_csv_dicts(
        matched_dir / "matched_av_events.csv",
        matched,
        [
            "video_id",
            "video_path",
            "match_id",
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
        ],
    )
    write_csv_dicts(
        matched_dir / "unmatched_visual_events.csv",
        unmatched_v,
        ["video_id", "video_path", "visual_event_id", "visual_label", "start_sec", "end_sec", "peak_sec", "confidence", "raw_score"],
    )
    write_csv_dicts(
        matched_dir / "unmatched_audio_events.csv",
        unmatched_a,
        ["video_id", "video_path", "audio_event_id", "audio_label", "start_sec", "end_sec", "peak_sec", "confidence", "raw_score"],
    )

    summary = {
        "num_matched": len(matched),
        "num_unmatched_visual": len(unmatched_v),
        "num_unmatched_audio": len(unmatched_a),
        "max_av_offset_sec": max_offset,
    }
    with open(matched_dir / "match_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
