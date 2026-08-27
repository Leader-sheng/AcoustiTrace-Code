# AcoustiTrace

[English](README.md) | 简体中文

AcoustiTrace 是一个面向生成式音视频的诊断评测基准，用于检验生成结果是否遵循声学规律。基准包含文本生成音视频（T2AV）的七个评测维度，并为图像生成音视频（I2AV）增加 Log Attack Time 评测。

[论文](https://arxiv.org/abs/2608.02035) · [项目主页](https://leader-sheng.github.io/AcoustiTrace/) · [RT60 权重](https://huggingface.co/Missouter/AcoustiTrace-RT60)

## 基准概览

| 任务 | Prompt 数量 | 评测维度 |
|---|---:|---|
| T2AV | 605 | Range Attenuation、Approach Gain、Lateral Stability、Motion--Loudness、Impact Decay、Causality Violation、RT60 Consistency |
| I2AV | 143 | Log Attack Time |

## 环境安装

### 系统要求

- Linux
- Python 3.10--3.12（推荐 Python 3.10）
- NVIDIA GPU 及与 CUDA 兼容的 PyTorch
- FFmpeg、Git 和 Git LFS

创建环境并安装 AcoustiTrace：

```bash
conda create -n acoustitrace python=3.10 -y
conda activate acoustitrace

python -m pip install --upgrade pip
pip install -r requirements-eval.txt
pip install -e .
```

Linux 下的 `requirements-eval.txt` 固定了经过实测的声源 evaluator 依赖：
PyTorch 2.8.0、TorchAudio 2.8.0、TorchVision 0.23.0、vLLM 0.11.0、
Transformers 4.57.6 与 `qwen-vl-utils==0.0.14`。请在干净环境中安装；
vLLM 与 GroundingDINO 均包含需要匹配 CUDA 的编译组件。

多数用户直接使用上述单环境即可。如果服务器镜像必须保留另一套 PyTorch 运行
Receiver Observer，可另外创建 `.venv-source`，在其中安装
`requirements-eval.txt`，并把 `source_mechanics`、`causality` 与 `rt60`
后端命令的解释器设为 `{repo_root}/.venv-source/bin/python`。顶层命令仍从主环境
运行。双环境的完整配置见[安装说明](docs/DEVELOPMENT_MACHINE_zh-CN.md#1-系统要求)。

### 外部 evaluator

将以下仓库克隆到 `third_party/`：

```text
third_party/
├── Video-Depth-Anything/
├── Grounded-Segment-Anything/
├── OV-AVEL/
└── FlexSED/
```

具体的克隆命令和经过测试的版本见[安装说明](docs/DEVELOPMENT_MACHINE_zh-CN.md)。

### 权重文件

按照以下结构放置 evaluator 权重：

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

具体文件名和第三方权重下载链接见[权重说明](models/README_zh-CN.md)与[安装说明](docs/DEVELOPMENT_MACHINE_zh-CN.md)。使用以下命令下载并校验已经公开的 AcoustiTrace RT60 evaluator：

```bash
python scripts/download_weights.py \
  --repo-id Missouter/AcoustiTrace-RT60 \
  --local-dir checkpoints/acoustitrace-rt60

python scripts/verify_checkpoint.py checkpoints/acoustitrace-rt60
```

该 Hugging Face 仓库包含 LoRA adapter、由 Sabine 关系引导的 physics head、
processor 文件、配置文件和 SHA-256 校验值，但不包含冻结的
`Qwen/Qwen3-VL-8B-Instruct` 基础模型。RT60 权重依据
[模型仓库许可证](https://huggingface.co/Missouter/AcoustiTrace-RT60/blob/main/LICENSE)
开放用于非商业学术与研究用途。

## 准备评测数据

仓库已经包含发布 manifest 与 I2AV-LAT 资产：

```text
data/
├── prompts/
│   ├── t2av_605.jsonl
│   └── i2av_lat_143.jsonl
├── i2av_conditioning_assets/   # 143 张 PNG
└── references/i2av_lat/        # 143 份五秒 WAV
```

使用对应的 `prompt_id` 原样命名每个生成视频：

```text
submissions/MyModel/
├── t2av/
│   ├── <prompt-id>.mp4
│   └── ...
└── i2av_lat/
    ├── <prompt-id>.mp4
    └── ...
```

规范文件名固定为 `<prompt_id>.mp4`，大小写必须与 manifest 完全一致。不要添加模型名前缀、序号或自定义后缀；模型名只放在 `submissions/<MODEL>/` 这一层。`task=i2av` 的 143 条 LAT 样本固定放入 `i2av_lat/`，而不是 `i2av/`。

可先导出模型无关的生成请求；每条请求都明确包含 `output_filename` 和相对路径 `output_relpath`：

```bash
acoustitrace export-generation-requests \
  data/prompts/t2av_605.jsonl generation_requests/t2av_605.jsonl
acoustitrace export-generation-requests \
  data/prompts/i2av_lat_143.jsonl generation_requests/i2av_lat_143.jsonl
```

使用 `--videos-dir` 时，一站式脚本严格检查上述 MP4 布局，并拒绝未知、额外或非规范扩展名的视频。只有使用显式的 `--outputs` manifest 时，才允许记录其他容器格式或自定义存放路径。

### 随仓库提供的 I2AV-LAT 资产

发布包直接包含 `data/i2av_conditioning_assets/` 下的 143 张条件 PNG，以及
`data/references/i2av_lat/` 下的 143 份五秒参考 WAV。正式评测前无需再下载或
预处理 Greatest Hits。可运行下面的命令校验随仓库提供的资产：

```bash
python scripts/check_i2av_lat_assets.py
```

这些衍生图像和音频以 CC BY 4.0 发布；署名、来源映射和修改说明见
[`data/ASSET_NOTICE_zh-CN.md`](data/ASSET_NOTICE_zh-CN.md)。

仓库仍保留官方压缩包的 Range 下载器、可续传的 Zenodo 备用入口和准备脚本，
仅用于来源复现或资产恢复。详见[开发机安装说明](docs/DEVELOPMENT_MACHINE_zh-CN.md#4-发布-manifest-与随仓库提供的-i2av-lat-资产)。

## 运行评测

创建本地后端配置：

```bash
cp configs/evaluator_backends.development.json \
  configs/evaluator_backends.local.json
```

检查运行环境、外部仓库、权重和 prompt manifest：

```bash
python scripts/check_evaluator_setup.py
```

双环境配置还应显式检查声源 evaluator 的解释器：

```bash
python scripts/check_evaluator_setup.py \
  --source-python .venv-source/bin/python
```

正式全量评测前，先用同一入口对八个维度各运行十条 assignment：

```bash
python evaluate.py \
  --prompts data/prompts/t2av_605.jsonl data/prompts/i2av_lat_143.jsonl \
  --videos-dir submissions/MyModel \
  --model MyModel \
  --backend-config configs/evaluator_backends.local.json \
  --output-dir results/MyModel-smoke \
  --smoke-limit-per-dimension 10
```

smoke 结果会在 `run.json` 中标记为 `smoke_test=true`、`official_scores=false`。
后端成功执行并不意味着每条生成视频都可评：没有可靠轨迹、声画匹配事件或衰减
拟合的样本会被标记为 invalid。请结合 `failures.jsonl` 与 `summary.csv` 中的
valid rate 判断；无效样本不进入均值，但会保留在覆盖率统计中。

移除 smoke 参数即可运行完整套件。一站式脚本会顺序执行各 GPU 后端，一张 GPU
只需启动这一条命令，不要并行启动多个 evaluator：

```bash
python evaluate.py \
  --prompts data/prompts/t2av_605.jsonl data/prompts/i2av_lat_143.jsonl \
  --videos-dir submissions/MyModel \
  --model MyModel \
  --backend-config configs/evaluator_backends.local.json \
  --output-dir results/MyModel \
  --resume
```

## 输出结果

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

- `summary.csv`：八个维度的聚合分数
- `scores.csv`：每个 prompt 的分数与有效性状态
- `evidence.jsonl`：各 evaluator 提取的结构化证据
- `failures.jsonl`：缺失、无效或评测失败的样本
- `run.json`：本次评测的配置与运行信息

## 项目结构

```text
AcoustiTrace/
├── evaluate.py                       # 一站式评测入口
├── src/acoustitrace/                 # 计分、聚合与数据契约
├── experiments/evaluator_adapters/  # evaluator 后端适配器
├── experiments/evaluator_backends/  # 原始媒体证据提取后端
├── data/prompts/                     # benchmark prompt manifest
├── configs/                          # evaluator 配置
├── models/                           # 权重说明
├── docs/                             # 详细文档
└── tests/                            # 测试
```

Evaluator 的输入输出协议和各维度的代码入口见[端到端评测说明](docs/END_TO_END_zh-CN.md)与[实现索引](docs/IMPLEMENTATION_MAP_zh-CN.md)。

## 后续计划

- [ ] 整理并公开 I2AV 其余维度的条件图像与 prompt manifest。
- [ ] 完善并公开 RT60 evaluator 的 LoRA 微调代码与训练配置。

## 许可证

AcoustiTrace 源代码采用 [Apache License 2.0](LICENSE) 开源。RT60 evaluator
权重遵循[模型仓库](https://huggingface.co/Missouter/AcoustiTrace-RT60/blob/main/LICENSE)
中的独立许可证。仓库内附带的 benchmark 资产与外部依赖继续遵循
[data/ASSET_NOTICE.md](data/ASSET_NOTICE.md) 及相应上游仓库注明的许可证和署名条款。

## 引用

```bibtex
@article{li2026acoustitrace,
  title   = {AcoustiTrace: When Plausible Sound Violates Physics},
  author  = {Li, Shiyang and Cao, Yuewen and Liu, Yihao and Pu, Yuandong and
             Zhang, Baochang and Li, Xiaofei and Zou, Changqing},
  journal = {arXiv preprint arXiv:2608.02035},
  year    = {2026}
}
```
