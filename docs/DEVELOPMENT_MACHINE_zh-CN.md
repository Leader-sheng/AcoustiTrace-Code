# 开发机环境配置

[English](DEVELOPMENT_MACHINE.md) | 简体中文

本说明介绍 AcoustiTrace 完整 evaluator 的安装与运行方法：配置一次环境，将权重和生成视频放入指定目录，然后使用一条命令完成八个维度的评测。

## 1. 系统要求

- Linux
- Python 3.10--3.12（推荐 Python 3.10）
- NVIDIA GPU 及与 CUDA 兼容的 PyTorch
- 已加入 `PATH` 的 FFmpeg
- Git 和 Git LFS

创建主环境：

```bash
conda create -n acoustitrace python=3.10 -y
conda activate acoustitrace

pip install -r requirements-eval.txt
pip install -e .
```

Linux 依赖文件固定了一组相互兼容的版本：vLLM 0.11.0、PyTorch 2.8.0、TorchAudio 2.8.0、TorchVision 0.23.0；Qwen3-VL 另外固定使用 `qwen-vl-utils==0.0.14`。请在干净环境中安装，不要直接覆盖服务器厂商提供的 PyTorch 或 nightly 版本。vLLM wheel 包含已编译的 CUDA 代码；若默认 PyPI wheel 与目标驱动不匹配，应选择对应 CUDA 版本的官方 wheel。

如果开发机必须保留厂商 PyTorch 来运行 receiver 或 RT60 后端，可在第二个环境中安装上述固定版本，并在 `configs/evaluator_backends.local.json` 中把 source mechanics、causality 与 Motion--Loudness 的命令指向该解释器；顶层 `evaluate.py` 命令无需改变。使用下面的命令同时检查两个环境：

```bash
python scripts/check_evaluator_setup.py \
  --source-python /path/to/source-environment/bin/python
```

## 2. 外部 evaluator 仓库

将依赖仓库克隆到以下固定位置：

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

# 项目评测环境中记录的版本。
git -C third_party/Video-Depth-Anything checkout 4f5ae23172ba60fd7bc11ef671cca678842c7072
git -C third_party/Grounded-Segment-Anything checkout 126abe633ffe333e16e4a0a4e946bc1003caf757
git -C third_party/OV-AVEL checkout b5fe1d685d0c6d0d6fd80312b5ccde79f9b73ea6
git -C third_party/FlexSED checkout 89aac3351e807de57c4fe3148d982922a3318e46

# 兼容当前 PyTorch C++ dispatch API，然后编译 GroundingDINO CUDA 扩展。
python scripts/patch_groundingdino_torch_compat.py
CUDA_HOME=/path/to/matching/cuda \
  pip install --no-build-isolation --no-deps -e \
  third_party/Grounded-Segment-Anything/GroundingDINO
```

请在当前环境中按照各上游仓库的说明完成安装。AcoustiTrace 的声源相关 adapter 直接调用 OV-AVEL 中原始的 ImageBind 模块，不依赖未公开的单视频 CLI 修改。
GroundingDINO 使用的 CUDA toolkit（包括 `nvcc`）必须与当前环境中的 PyTorch CUDA 版本匹配；仅安装 NVIDIA 驱动不足以编译该扩展。`check_evaluator_setup.py` 会直接导入 `groundingdino._C`，避免到正式评测时才发现扩展缺失或 ABI 不兼容。

## 3. 权重目录

按照以下结构放置权重：

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
    processor/                 # 可选
```

第三方权重下载地址：

- Video-Depth-Anything Large：
  `https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth`
- GroundingDINO Swin-T：
  `https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`
- SAM ViT-B：
  `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`
- ImageBind Huge：
  `https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth`
- FlexSED：
  `https://huggingface.co/Higobeatz/FlexSED/resolve/main/ckpts/flexsed_as.pt`
- FlexSED 使用的 CLAP 文本编码器：
  `https://huggingface.co/laion/clap-htsat-unfused`
- Qwen3-VL-8B-Instruct：
  `https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct`

例如，在接受相应模型条款后，可使用以下命令下载 Qwen3-VL：

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir checkpoints/qwen3-vl

hf download laion/clap-htsat-unfused \
  --local-dir checkpoints/flexsed/laion-clap-htsat-unfused
```

从 Hugging Face 下载已经公开的 AcoustiTrace RT60 LoRA adapter 与 physics head，
并校验本地权重结构：

```bash
python scripts/download_weights.py \
  --repo-id Missouter/AcoustiTrace-RT60 \
  --local-dir checkpoints/acoustitrace-rt60
python scripts/verify_checkpoint.py checkpoints/acoustitrace-rt60
```

模型仓库为
[`Missouter/AcoustiTrace-RT60`](https://huggingface.co/Missouter/AcoustiTrace-RT60)，
包含 LoRA adapter、physics head、processor 文件、配置与 SHA-256 校验值，但不包含
Qwen3-VL 基础模型权重。RT60 权重依据模型仓库许可证开放用于非商业学术与研究
用途，并仍受适用上游条款约束。详细结构见[权重说明](../models/README_zh-CN.md)。

## 4. 发布 manifest 与随仓库提供的 I2AV-LAT 资产

公开评测使用以下文件：

```text
data/prompts/
  t2av_605.jsonl
  i2av_lat_143.jsonl
```

各 prompt pool 的数量与 evaluator 分配如下：

| Prompt pool | 唯一 prompt 数 | Evaluator |
|---|---:|---|
| Receiver Distance | 90 | Range Attenuation |
| Receiver Observer | 90 | Approach Gain、Lateral Stability |
| Motion--Loudness | 90 | Motion--Loudness |
| Impact/Source Generation | 90 | Impact Decay |
| Causality | 79 | Causality Violation |
| RT60 | 166 | RT60 Consistency |
| I2AV Log Attack Time | 143 | Log Attack Time |

接收端相关记录在 `evaluator_inputs` 中提供仅供 evaluator 使用的检测目标：

```json
{
  "evaluator_inputs": {
    "receiver_observer": {
      "detection_targets": ["car", "vehicle"]
    }
  }
}
```

I2AV-LAT 记录在 `evaluator_inputs` 中提供仅供 evaluator 使用的参考音频：

```json
{
  "evaluator_inputs": {
    "log_attack_time": {
      "reference_audio_path": "data/references/i2av_lat/i2av_source_log_attack_time_000001.wav"
    }
  }
}
```

`evaluator_inputs` 不会导出给生成模型。

发布包已经包含 I2AV Log Attack Time evaluator 所需的 143 张条件 PNG 和
143 份五秒参考 WAV，常规使用无需下载 Greatest Hits。运行前可直接校验：

```bash
python scripts/check_i2av_lat_assets.py
```

这些随仓库发布的衍生素材遵循 CC BY 4.0；署名和发布处理说明见
[`data/ASSET_NOTICE_zh-CN.md`](../data/ASSET_NOTICE_zh-CN.md)。

公开来源映射仍列出对应的 143 条 Greatest Hits 记录，且不包含私有文件路径。
如需复现来源或修复资产，可按需下载官方低分辨率压缩包中的对应成员并重建：

```bash
python scripts/download_greatest_hits_subset.py
python scripts/prepare_i2av_lat_references.py --overwrite
```

如果旧版 Greatest Hits 主机无法访问，可从 SyncFusion 发布的 Zenodo 分片中
以可续跑方式恢复同一批资产：

```bash
python scripts/download_greatest_hits_zenodo_subset.py
```

上述恢复命令均为可选工具，不属于标准安装流程。恢复脚本会写入
`outputs/i2av_lat_reference_report.csv`；正式运行前仍以随仓库资产校验器的结果
为准。

## 5. 生成视频目录

使用稳定的 `prompt_id` 原样命名对应视频：

```text
submissions/MyModel/
  t2av/
    <t2av-prompt-id>.mp4
    ... 605 files
  i2av_lat/
    <i2av-lat-prompt-id>.mp4
    ... 143 files
```

规范布局固定为 `submissions/<MODEL>/<TASK>/<prompt_id>.mp4`。文件名区分大小写，不得添加模型名前缀、序号或自定义后缀；`task=i2av` 的公开 LAT 子集固定使用目录名 `i2av_lat/`。目录自动发现模式只接受规范 MP4，并会拒绝未知、额外或非规范扩展名的视频。需要使用其他容器或路径时，必须通过显式的 `--outputs` manifest 提交。每个 prompt 必须且只能匹配一个视频。

## 6. 一站式评测

复制默认后端配置：

```bash
cp configs/evaluator_backends.development.json \
  configs/evaluator_backends.local.json
```

正式运行 GPU evaluator 前，先检查环境、外部仓库、权重和 prompt manifest：

```bash
python scripts/check_evaluator_setup.py
```

请先在下面的命令中加入 `--smoke-limit-per-dimension 10`，固定抽取每个维度十条 assignment 进行非正式原生后端测试；全部通过后移除该参数，再运行全部八个维度：

```bash
python evaluate.py \
  --prompts data/prompts/t2av_605.jsonl data/prompts/i2av_lat_143.jsonl \
  --videos-dir submissions/MyModel \
  --model MyModel \
  --backend-config configs/evaluator_backends.local.json \
  --output-dir results/MyModel \
  --resume
```

评测结果保存在：

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

`--resume` 仅在输入指纹一致且对应后端上一次成功完成时复用已有结果。

## 7. Evaluator 后端

默认配置包含以下后端：

- Receiver Observer：Range Attenuation、Approach Gain、Lateral Stability
- Source Mechanics：Motion--Loudness、Impact Decay
- Causality：Causality Violation
- RT60：RT60 Consistency
- Log Attack Time：I2AV Log Attack Time

`configs/evaluator_backends.mock.json` 仅用于控制流程测试；mock 结果会标记为 `official_scores=false`，且必须显式添加 `--allow-mock` 才能运行。
