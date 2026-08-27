# RT60 evaluator runtime

This directory contains only the code needed to run the released RT60
Consistency evaluator:

- `audio_rt60_proxy.py` estimates the strict 500 Hz audio decay proxy;
- `visual_physics_runtime.py` loads the frozen Qwen3-VL backbone, released LoRA
  adapter, and continuous Sabine-guided physics head;
- `rt60_runtime.yaml` records the inference architecture and preprocessing
  configuration.

Training code and controlled validation experiments are outside the first
public release. The released runtime architecture is recorded in
`models/rt60_evaluator_contract.json`.
