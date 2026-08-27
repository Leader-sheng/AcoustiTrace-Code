# Development-machine setup

English | [简体中文](DEVELOPMENT_MACHINE_zh-CN.md)

This document describes the intended public user experience: install the
evaluator once, place weights and generated videos in fixed directories, then
run one command. All adapters are wired into that command.

## 1. System requirements

- Linux is the primary supported evaluator platform.
- Python 3.10--3.12 (Python 3.10 is recommended).
- NVIDIA GPU with a CUDA-compatible PyTorch installation.
- FFmpeg available on `PATH`.
- Git and Git LFS.

Create the main environment:

```bash
conda create -n acoustitrace python=3.10 -y
conda activate acoustitrace

pip install -r requirements-eval.txt
pip install -e .
```

The Linux requirements pin the mutually compatible vLLM 0.11.0, PyTorch 2.8.0,
TorchAudio 2.8.0, and TorchVision 0.23.0 stack. Qwen3-VL additionally uses
`qwen-vl-utils==0.0.14`. Create a clean environment instead of installing this
stack over a vendor-specific or nightly PyTorch build. The vLLM wheel contains
compiled CUDA code; use the wheel variant appropriate for the target driver if
the default PyPI wheel is unsuitable.

If a machine must retain a vendor PyTorch build for the receiver or RT60
backend, install the pinned full stack in a second environment and point the
source-mechanics, causality, and Motion--Loudness commands at that interpreter
in `configs/evaluator_backends.local.json`. The top-level `evaluate.py` command
does not change. Validate this layout with:

```bash
python scripts/check_evaluator_setup.py \
  --source-python /path/to/source-environment/bin/python
```

## 2. External evaluator repositories

Clone the upstream repositories into the following exact locations:

```text
third_party/
  Video-Depth-Anything/
  Grounded-Segment-Anything/
  OV-AVEL/
  FlexSED/
```

```bash
mkdir -p third_party
git clone https://github.com/DepthAnything/Video-Depth-Anything \
  third_party/Video-Depth-Anything
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git \
  third_party/Grounded-Segment-Anything
git clone https://github.com/jasongief/OV-AVEL.git third_party/OV-AVEL
git clone https://github.com/JHU-LCAP/FlexSED.git third_party/FlexSED

# Revisions recorded from the evaluator environment snapshot.
git -C third_party/Video-Depth-Anything checkout 4f5ae23172ba60fd7bc11ef671cca678842c7072
git -C third_party/Grounded-Segment-Anything checkout 126abe633ffe333e16e4a0a4e946bc1003caf757
git -C third_party/OV-AVEL checkout b5fe1d685d0c6d0d6fd80312b5ccde79f9b73ea6
git -C third_party/FlexSED checkout 89aac3351e807de57c4fe3148d982922a3318e46

# Patch current PyTorch C++ dispatch compatibility, then build the
# GroundingDINO CUDA extension.
python scripts/patch_groundingdino_torch_compat.py
CUDA_HOME=/path/to/matching/cuda \
  pip install --no-build-isolation --no-deps -e \
  third_party/Grounded-Segment-Anything/GroundingDINO
```

Follow each upstream repository's installation instructions inside the active
environment. AcoustiTrace does not redistribute or relicense these projects.
The public source adapter calls the unmodified ImageBind modules vendored by
OV-AVEL; it does not require the unpublished single-video CLI patch that was
present in the original server checkout.
The CUDA toolkit (including `nvcc`) used for GroundingDINO must match the CUDA
version of PyTorch in the active environment; an NVIDIA driver alone is not
enough to compile the extension. `check_evaluator_setup.py` imports
`groundingdino._C` directly so a missing or ABI-incompatible extension is
reported before an evaluation starts.

## 3. Checkpoint layout

Place checkpoints as follows:

```text
checkpoints/
  video_depth_anything/
    metric_video_depth_anything_vitl.pth
  grounded_sam/
    groundingdino_swint_ogc.pth
    sam_vit_b_01ec64.pth
  ov_avel/
    imagebind_huge.pth
  flexsed/
    flexsed_as.pt
    laion-clap-htsat-unfused/
      config.json
      pytorch_model.bin
      tokenizer.json
      ...
  qwen3-vl/
    config.json
    model-00001-of-00004.safetensors
    ...
  acoustitrace-rt60/
    physics_head.pt
    vlm_adapter/
      adapter_config.json
      adapter_model.safetensors
    processor/                 # optional
```

Upstream downloads:

- Video-Depth-Anything Large:
  `https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth`
- GroundingDINO Swin-T:
  `https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`
- SAM ViT-B:
  `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`
- ImageBind Huge:
  `https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth`
- FlexSED:
  `https://huggingface.co/Higobeatz/FlexSED/resolve/main/ckpts/flexsed_as.pt`
- CLAP text encoder used by FlexSED:
  `https://huggingface.co/laion/clap-htsat-unfused`
- Qwen3-VL-8B-Instruct:
  `https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct`

For example, after accepting any upstream model terms:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir checkpoints/qwen3-vl

hf download laion/clap-htsat-unfused \
  --local-dir checkpoints/flexsed/laion-clap-htsat-unfused
```

Download the released AcoustiTrace RT60 adapter and physics head from Hugging
Face, then verify the local checkpoint contract:

```bash
python scripts/download_weights.py \
  --repo-id Missouter/AcoustiTrace-RT60 \
  --local-dir checkpoints/acoustitrace-rt60
python scripts/verify_checkpoint.py checkpoints/acoustitrace-rt60
```

The model repository is
[`Missouter/AcoustiTrace-RT60`](https://huggingface.co/Missouter/AcoustiTrace-RT60).
It includes the LoRA adapter, physics head, processor files, configuration, and
SHA-256 checksums, but not the Qwen3-VL base weights. The RT60 artifacts are
provided for non-commercial academic and research use under the model
repository's license and applicable upstream terms.

## 4. Release manifests and bundled I2AV-LAT assets

The first release expects exactly:

```text
data/prompts/
  t2av_605.jsonl
  i2av_lat_143.jsonl
```

The T2AV allocation is the rebalanced public suite:

| prompt pool | unique prompts | evaluator assignments |
|---|---:|---|
| Receiver Distance | 90 | Range Attenuation |
| Receiver Observer | 90 | Approach Gain and Lateral Stability |
| Motion--Loudness | 90 | Motion--Loudness |
| Impact/Source Generation | 90 | Impact Decay |
| Causality | 79 | Causality Violation |
| RT60 | 166 | RT60 Consistency |
| I2AV Log Attack Time | 143 | Log Attack Time |

Receiver rows carry evaluator-only target names under:

```json
{
  "evaluator_inputs": {
    "receiver_observer": {
      "detection_targets": ["car", "vehicle"]
    }
  }
}
```

I2AV-LAT rows carry evaluator-only reference audio under:

```json
{
  "evaluator_inputs": {
    "log_attack_time": {
      "reference_audio_path": "data/references/i2av_lat/i2av_source_log_attack_time_000001.wav"
    }
  }
}
```

`evaluator_inputs` is never exported to the model generation request.

The release already includes the 143 conditioning PNGs and 143 five-second
reference WAVs required by the I2AV Log Attack Time evaluator. No dataset
download is required for normal use. Validate the bundled files before a run:

```bash
python scripts/check_i2av_lat_assets.py
```

The bundled derived assets are covered by CC BY 4.0; see
[`data/ASSET_NOTICE.md`](../data/ASSET_NOTICE.md) for attribution and the exact
release transformations.

The public source map identifies the corresponding Greatest Hits recordings
without retaining private filesystem paths. To reproduce or repair the bundled
assets, download only those archive members and rerun the preparation step:

```bash
python scripts/download_greatest_hits_subset.py
python scripts/prepare_i2av_lat_references.py --overwrite
```

The downloader uses HTTP range requests against the official low-resolution
archive. If the legacy host is unreachable, use the resumable SyncFusion Zenodo
fallback instead:

```bash
python scripts/download_greatest_hits_zenodo_subset.py
```

These recovery commands are optional and are not part of the standard setup.
They write `outputs/i2av_lat_reference_report.csv`; the bundled-asset validator
remains the authoritative pre-run check.

## 5. Generated-video directory

Use every stable `prompt_id` verbatim as its generated filename:

```text
submissions/MyModel/
  t2av/
    <t2av-prompt-id>.mp4
    ... 605 files
  i2av_lat/
    <i2av-lat-prompt-id>.mp4
    ... 143 files
```

The canonical layout is `submissions/<MODEL>/<TASK>/<prompt_id>.mp4`. Filenames
are case-sensitive and must not gain model prefixes, numeric indices, or custom
suffixes. Public rows with `task=i2av` use the directory name `i2av_lat/`.
Directory discovery accepts only canonical MP4 files and rejects unknown,
extra, or non-canonical video files. Use an explicit `--outputs` manifest when
another container or storage layout is unavoidable. A prompt must have exactly
one matching video. The evaluator preserves missing and invalid generations in
coverage accounting rather than silently assigning zero.

## 6. One-command evaluation

Copy the backend configuration once:

```bash
cp configs/evaluator_backends.development.json \
  configs/evaluator_backends.local.json
```

Check the environment, third-party repositories, checkpoints, and prompt
manifests before launching GPU work:

```bash
python scripts/check_evaluator_setup.py
```

First append `--smoke-limit-per-dimension 10` to the command below. This runs a
deterministic ten assignments per selected dimension and marks the output as a
non-official smoke test. Once all backends pass, remove the flag and run:

```bash
python evaluate.py \
  --prompts data/prompts/t2av_605.jsonl data/prompts/i2av_lat_143.jsonl \
  --videos-dir submissions/MyModel \
  --model MyModel \
  --backend-config configs/evaluator_backends.local.json \
  --output-dir results/MyModel \
  --resume
```

Outputs:

```text
results/MyModel/
  generated_outputs.csv
  evidence.jsonl
  scores.csv
  summary.csv
  failures.jsonl
  run.json
  logs/
  work/
```

`--resume` reuses a backend only when its input fingerprint and previous
successful state match.

## 7. Evaluator backends

The default configuration includes the following native adapters:

- Receiver Observer: Range Attenuation, Approach Gain, Lateral Stability.
- Source Mechanics: Motion--Loudness and Impact Decay.
- Causality: Causality Violation.
- RT60: RT60 Consistency.
- Log Attack Time: I2AV Log Attack Time.

The file `configs/evaluator_backends.mock.json` is only for control-plane
tests. Mock runs are marked `official_scores=false` and require the explicit
`--allow-mock` flag.
