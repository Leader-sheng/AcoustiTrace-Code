# Implementation Map

English | [简体中文](IMPLEMENTATION_MAP_zh-CN.md)

| Evaluator | Clean score entry point | Included evidence backend |
|---|---|---|
| Motion--Loudness | `motion_loudness` | OV-AVEL/FlexSED matching, two time-contiguous clusters, Qwen motion-only judgment, 8 fps frame-difference fallback |
| Log Attack Time | `log_attack_time` | paired impact localization input, envelope extraction, T10/T90 measurement |
| Impact Decay | `impact_decay` | event-window extraction, exponential-plus-floor fit, residual diagnostics |
| RT60 Consistency | `rt60_consistency` | strict audio proxy plus released Qwen3-VL Sabine-guided inference runtime |
| Causality Violation | `causality_violation` | confidence filtering, clustering, event association, 1 ms violation scoring |
| Range Attenuation | `range_attenuation` | Grounded-SAM tracking, VDA relative range, sign-aware local-window exponent search |
| Approach Gain | `native_receiver_score` | relative-depth approach screening and native monotonic/trend readout |
| Lateral Stability | `native_receiver_score` | Unified v2 constant-range local windows followed by lateral visual screening |

The raw-media scripts, including the RT60 inference runtime, live under
`experiments/evaluator_backends/`. External detector repositories and model
weights are intentionally not vendored. RT60 fine-tuning code is planned as a
later release.
