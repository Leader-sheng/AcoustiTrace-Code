# AcoustiTrace

English | [简体中文](README_zh-CN.md)

AcoustiTrace is a diagnostic benchmark for evaluating whether generated audio-video content follows acoustic principles. It evaluates seven dimensions for text-to-audio-video (T2AV) generation and adds Log Attack Time for image-to-audio-video (I2AV) generation.

[Paper](https://arxiv.org/abs/2608.02035) · [Project page](https://leader-sheng.github.io/AcoustiTrace/) · [RT60 weights](https://huggingface.co/Missouter/AcoustiTrace-RT60)

## Benchmark

| Task | Prompts | Evaluation dimensions |
|---|---:|---|
| T2AV | 605 | Range Attenuation, Approach Gain, Lateral Stability, Motion--Loudness, Impact Decay, Causality Violation, RT60 Consistency |
| I2AV | 143 | Log Attack Time |

## Installation

### Requirements

- Linux
- Python 3.10--3.12 (Python 3.10 recommended)
- NVIDIA GPU and a CUDA-compatible PyTorch build
- FFmpeg, Git, and Git LFS

Create the environment and install AcoustiTrace:

```bash
conda create -n acoustitrace python=3.10 -y
conda activate acoustitrace

python -m pip install --upgrade pip
pip install -r requirements-eval.txt
pip install -e .
```

On Linux, `requirements-eval.txt` pins the tested source-evaluator stack:
PyTorch 2.8.0, TorchAudio 2.8.0, TorchVision 0.23.0, vLLM 0.11.0,
Transformers 4.57.6, and `qwen-vl-utils==0.0.14`. Install it in a clean
environment; vLLM and GroundingDINO include compiled CUDA components.

Most users should use the single environment above. On vendor images that must
retain a different PyTorch build for Receiver Observer, create a second clean
environment such as `.venv-source` with `requirements-eval.txt`, then point the
`source_mechanics`, `causality`, and `rt60` backend commands to
`{repo_root}/.venv-source/bin/python`. The top-level command still runs from the
main environment. The exact split-environment configuration is documented in
[the installation guide](docs/DEVELOPMENT_MACHINE.md#1-system-requirements).

### External evaluators

Clone the following repositories under `third_party/`:

```text
third_party/
├── Video-Depth-Anything/
├── Grounded-Segment-Anything/
├── OV-AVEL/
└── FlexSED/
```

The exact clone commands and tested revisions are listed in [the installation guide](docs/DEVELOPMENT_MACHINE.md#2-external-evaluator-repositories).

### Checkpoints

Place the evaluator checkpoints in the following structure:

```text
checkpoints/
├── video_depth_anything/
├── grounded_sam/
├── ov_avel/
├── flexsed/
│   ├── flexsed_as.pt
│   └── laion-clap-htsat-unfused/
├── qwen3-vl/
└── acoustitrace-rt60/
```

Checkpoint names and third-party download links are documented in [the checkpoint guide](models/README.md) and [the installation guide](docs/DEVELOPMENT_MACHINE.md#3-checkpoint-layout).

Download the released AcoustiTrace RT60 evaluator directly from Hugging Face,
then verify its expected file contract:

```bash
python scripts/download_weights.py \
  --repo-id Missouter/AcoustiTrace-RT60 \
  --local-dir checkpoints/acoustitrace-rt60

python scripts/verify_checkpoint.py checkpoints/acoustitrace-rt60
```

The repository contains the LoRA adapter, Sabine-guided physics head, processor
files, configuration, and SHA-256 checksums. It does not include the frozen
`Qwen/Qwen3-VL-8B-Instruct` base model. The RT60 artifacts are released for
non-commercial academic and research use under the terms in the
[model repository license](https://huggingface.co/Missouter/AcoustiTrace-RT60/blob/main/LICENSE).

## Prepare the benchmark

The repository already includes the release manifests and I2AV-LAT assets:

```text
data/
├── prompts/
│   ├── t2av_605.jsonl
│   └── i2av_lat_143.jsonl
├── i2av_conditioning_assets/   # 143 PNGs
└── references/i2av_lat/        # 143 five-second WAVs
```

Use the corresponding `prompt_id` verbatim as each generated filename:

```text
submissions/MyModel/
├── t2av/
│   ├── <prompt-id>.mp4
│   └── ...
└── i2av_lat/
    ├── <prompt-id>.mp4
    └── ...
```

The canonical filename is exactly `<prompt_id>.mp4`, with matching case. Do not
add a model prefix, numeric index, or custom suffix; the model name belongs only
in the `submissions/<MODEL>/` directory. The 143 rows whose manifest task is
`i2av` belong in `i2av_lat/`, not `i2av/`.

Export model-neutral generation requests before generation. Every request
contains the exact `output_filename` and repository-relative `output_relpath`:

```bash
acoustitrace export-generation-requests \
  data/prompts/t2av_605.jsonl generation_requests/t2av_605.jsonl
acoustitrace export-generation-requests \
  data/prompts/i2av_lat_143.jsonl generation_requests/i2av_lat_143.jsonl
```

With `--videos-dir`, the evaluator enforces this MP4 layout and rejects unknown,
extra, or non-canonical video files. Use an explicit `--outputs` manifest only
when another container format or storage layout is unavoidable.

### Bundled I2AV-LAT assets

The release includes all 143 conditioning PNGs in
`data/i2av_conditioning_assets/` and all 143 five-second reference WAVs in
`data/references/i2av_lat/`. No Greatest Hits download or preprocessing is
required before evaluation. Verify the bundled assets with:

```bash
python scripts/check_i2av_lat_assets.py
```

The derived images and audio are distributed under CC BY 4.0. Attribution,
source mapping, and modification details are recorded in
[`data/ASSET_NOTICE.md`](data/ASSET_NOTICE.md).

For provenance reproduction or asset recovery only, the repository retains the
range-based official-archive downloader, the resumable Zenodo fallback, and the
preparation script. See [the installation guide](docs/DEVELOPMENT_MACHINE.md#4-release-manifests-and-bundled-i2av-lat-assets).

## Run evaluation

Create a local backend configuration:

```bash
cp configs/evaluator_backends.development.json \
  configs/evaluator_backends.local.json
```

Check the environment, repositories, checkpoints, and prompt manifests:

```bash
python scripts/check_evaluator_setup.py
```

For a split environment, also validate the source evaluator interpreter:

```bash
python scripts/check_evaluator_setup.py \
  --source-python .venv-source/bin/python
```

Before a full run, verify all eight native dimensions with a deterministic
ten-assignment smoke test:

```bash
python evaluate.py \
  --prompts data/prompts/t2av_605.jsonl data/prompts/i2av_lat_143.jsonl \
  --videos-dir submissions/MyModel \
  --model MyModel \
  --backend-config configs/evaluator_backends.local.json \
  --output-dir results/MyModel-smoke \
  --smoke-limit-per-dimension 10
```

Smoke outputs are marked `smoke_test=true` and `official_scores=false` in
`run.json`. A backend may complete successfully while marking an individual
sample invalid when the generated video contains no reliable track, matched
audio-visual event, or decay fit. Inspect `failures.jsonl` and the per-dimension
valid rate in `summary.csv`; invalid samples are excluded from the mean but
remain in coverage accounting.

Remove the smoke flag to run the complete suite. The orchestrator executes the
GPU backends sequentially, so one command is sufficient and multiple evaluator
processes should not be launched on the same GPU:

```bash
python evaluate.py \
  --prompts data/prompts/t2av_605.jsonl data/prompts/i2av_lat_143.jsonl \
  --videos-dir submissions/MyModel \
  --model MyModel \
  --backend-config configs/evaluator_backends.local.json \
  --output-dir results/MyModel \
  --resume
```

## Outputs

```text
results/MyModel/
├── summary.csv
├── scores.csv
├── evidence.jsonl
├── generated_outputs.csv
├── failures.jsonl
├── run.json
├── logs/
└── work/
```

- `summary.csv`: aggregated scores for the eight dimensions
- `scores.csv`: prompt-level scores and validity states
- `evidence.jsonl`: structured evidence extracted by the evaluators
- `failures.jsonl`: missing, invalid, or failed samples
- `run.json`: configuration and metadata for the evaluation run

## Repository structure

```text
AcoustiTrace/
├── evaluate.py                       # end-to-end evaluation entry point
├── src/acoustitrace/                 # scoring, aggregation, and data contracts
├── experiments/evaluator_adapters/  # evaluator backend adapters
├── experiments/evaluator_backends/  # raw-media evidence extractors
├── data/prompts/                     # benchmark prompt manifests
├── configs/                          # evaluator configuration
├── models/                           # checkpoint instructions
├── docs/                             # detailed documentation
└── tests/                            # tests
```

For the evaluator input/output protocol and individual implementation entry points, see [End-to-end evaluation](docs/END_TO_END.md) and [Implementation map](docs/IMPLEMENTATION_MAP.md).

## Roadmap

- [ ] Organize and release the conditioning images and prompt manifests for the remaining I2AV dimensions.
- [ ] Complete and release the LoRA fine-tuning code and training configuration for the RT60 evaluator.

## License

The AcoustiTrace source code is released under the
[Apache License 2.0](LICENSE). The RT60 evaluator weights are distributed under
the separate license in the
[model repository](https://huggingface.co/Missouter/AcoustiTrace-RT60/blob/main/LICENSE).
Bundled benchmark assets and external dependencies retain the licenses and
attribution terms documented in [data/ASSET_NOTICE.md](data/ASSET_NOTICE.md)
and their respective upstream repositories.

## Citation

```bibtex
@article{li2026acoustitrace,
  title   = {AcoustiTrace: When Plausible Sound Violates Physics},
  author  = {Li, Shiyang and Cao, Yuewen and Liu, Yihao and Pu, Yuandong and
             Zhang, Baochang and Li, Xiaofei and Zou, Changqing},
  journal = {arXiv preprint arXiv:2608.02035},
  year    = {2026}
}
```
