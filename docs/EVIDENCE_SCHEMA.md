# Structured Evidence Schema

Batch input is JSON Lines. Every row contains `sample_id`, `model`, `task`,
`evaluator`, and an evaluator-specific `evidence` object. Run:

```powershell
acoustitrace score-batch examples/evidence.jsonl output/scored_evidence.csv
```

## Evidence Fields

| Evaluator | Required evidence |
|---|---|
| `motion_loudness` | `level_pairs_db`: visually stronger/visually weaker event RMS pairs |
| `log_attack_time` | `generated_attack_seconds`, `reference_attack_seconds` |
| `impact_decay` | `fit_r2`, `tail_residual_mae_db`, `peak_to_floor_db` |
| `rt60_consistency` | `audio_rt60`, `visual_rt60` in seconds |
| `causality_violation` | `onset_delays_seconds`: audio onset minus visual contact |
| `range_attenuation` | `window_r2_values`: native valid local-window R2 values |
| `approach_gain` | `native_score` on [0, 1] from the receiver backend |
| `lateral_stability` | `native_score` on [0, 1] from the receiver backend |

Applicability and extraction gates belong to the evidence backend. When a gate
fails, emit an invalid score row with an empty score; never substitute zero.
