# Model artifacts

English | [简体中文](README_zh-CN.md)

AcoustiTrace's learned evaluator is the visual 500 Hz RT60 estimator. It is
not a full fine-tuned copy of Qwen3-VL. The release checkpoint contains a LoRA
adapter and a Sabine-guided continuous physics head, while the frozen base
model is downloaded separately from `Qwen/Qwen3-VL-8B-Instruct`.

Expected checkpoint layout (the top-level evaluator uses this exact path):

```text
checkpoints/acoustitrace-rt60/
  physics_head.pt
  vlm_adapter/
    adapter_config.json
    adapter_model.safetensors
  processor/                 # optional; base processor is the fallback
```

The inference architecture and checkpoint contract are stored in
`models/rt60_evaluator_contract.json`.

The released checkpoint is hosted at
[`Missouter/AcoustiTrace-RT60`](https://huggingface.co/Missouter/AcoustiTrace-RT60).
Download and verify it with:

```bash
python scripts/download_weights.py \
  --repo-id Missouter/AcoustiTrace-RT60 \
  --local-dir checkpoints/acoustitrace-rt60

python scripts/verify_checkpoint.py \
  checkpoints/acoustitrace-rt60
```

The model repository also provides processor files, evaluator configuration,
and `SHA256SUMS`. For an immutable experiment, pass a commit hash through
`--revision` and record it with the evaluation output. The frozen
`Qwen/Qwen3-VL-8B-Instruct` base model is not redistributed and must be
downloaded separately.

The AcoustiTrace RT60 artifacts are available for non-commercial academic and
research use under the
[model repository license](https://huggingface.co/Missouter/AcoustiTrace-RT60/blob/main/LICENSE).
Use also remains subject to applicable upstream terms, including the
Matterport3D Academic Use Terms. No upstream training data is included in the
model repository.

Other evaluator dependencies (Video-Depth-Anything, Grounded-SAM, OV-AVEL,
FlexSED, and their checkpoints) remain third-party downloads governed by their
own licenses.
