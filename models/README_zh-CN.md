# 模型权重

[English](README.md) | 简体中文

AcoustiTrace 中需要额外训练权重的 evaluator 是视觉 500 Hz RT60 估计器。该模型并不是 Qwen3-VL 的完整微调副本：公开权重包含一个 LoRA adapter 和一个由 Sabine 关系引导的连续值 physics head，冻结的基础模型 `Qwen/Qwen3-VL-8B-Instruct` 需单独下载。

顶层 evaluator 使用以下固定目录结构：

```text
checkpoints/acoustitrace-rt60/
  physics_head.pt
  vlm_adapter/
    adapter_config.json
    adapter_model.safetensors
  processor/                 # 可选；缺省时使用基础模型的 processor
```

推理结构与权重契约记录在 `models/rt60_evaluator_contract.json`。

公开权重托管于
[`Missouter/AcoustiTrace-RT60`](https://huggingface.co/Missouter/AcoustiTrace-RT60)。
使用以下命令下载并校验：

```bash
python scripts/download_weights.py \
  --repo-id Missouter/AcoustiTrace-RT60 \
  --local-dir checkpoints/acoustitrace-rt60

python scripts/verify_checkpoint.py \
  checkpoints/acoustitrace-rt60
```

模型仓库同时提供 processor 文件、evaluator 配置与 `SHA256SUMS`。需要冻结实验
版本时，可通过 `--revision` 指定 commit hash，并将其与评测结果一并记录。冻结的
`Qwen/Qwen3-VL-8B-Instruct` 基础模型不在该仓库中，需要单独下载。

AcoustiTrace RT60 权重依据
[模型仓库许可证](https://huggingface.co/Missouter/AcoustiTrace-RT60/blob/main/LICENSE)
开放用于非商业学术与研究用途，同时仍受 Matterport3D Academic Use Terms 等适用
上游条款约束。模型仓库不包含任何上游训练数据。

Video-Depth-Anything、Grounded-SAM、OV-AVEL、FlexSED 及其权重属于第三方依赖，请按照各项目的许可证和下载说明获取。
