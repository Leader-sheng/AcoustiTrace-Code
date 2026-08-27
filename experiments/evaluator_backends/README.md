# Evaluator backends

These scripts convert raw media into the structured evidence consumed by the
`acoustitrace` score API. Their thresholds and score definitions follow the
paper and supplementary document.

| Directory | Included stages |
|---|---|
| `receiver_observer` | VDA depth/point-cloud export, Grounded-SAM source tracking, audio-level extraction, visual applicability, Range, Approach, and Lateral readouts |
| `source_mechanics` | OV-AVEL/FlexSED parsing, event matching, two-cluster Qwen motion judgment, frame-difference fallback, Motion--Loudness, and Impact Decay readouts |
| `time_causality_eval` | event parsing, confidence filtering, temporal clustering, association, and the final causality readout |
| `log_attack_time` | paired generated/reference impact attack-time extraction and scoring |

Third-party repositories and weights are not vendored. Paths in the example
YAML files are repository-relative placeholders and may be overridden from the
command line. Invalid outputs remain in coverage accounting and do not receive
zero scores.
