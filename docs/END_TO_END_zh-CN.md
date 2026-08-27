# 端到端评测流程

[English](END_TO_END.md) | 简体中文

## 1. 固定并校验 prompt

公开版本读取 `data/prompts/t2av_605.jsonl` 和 `data/prompts/i2av_lat_143.jsonl`。`validate-release-suite` 会校验 prompt 总数、evaluator 成员关系、类别分配和 I2AV 条件图像。

## 2. 导出生成请求

`acoustitrace export-generation-requests` 为每个 prompt 输出一条与具体模型无关的 JSON 记录。生成模型 adapter 读取以下字段：

```text
prompt_id, task, prompt, evaluator_membership,
conditioning_asset_id and/or conditioning_asset_path,
output_subdir, output_filename, output_relpath
```

T2AV adapter 只使用文本；I2AV adapter 使用 prompt 及其引用的条件图像。模型特有的随机种子、时长、分辨率、API 任务 ID 和服务版本应记录在 adapter 的 provenance 中，而不应写入冻结的 prompt 文本。

生成器必须把结果写到请求给出的 `output_relpath`。规范命名为 `t2av/<prompt_id>.mp4` 或 `i2av_lat/<prompt_id>.mp4`；不得改写、重新编号或添加模型名前缀。

## 3. 返回生成结果

每个尝试生成的 prompt 都必须在输出 manifest 中占一行，包括生成失败的样本：

```text
prompt_id,task,model,video_path,status,error,seed,generator_revision
```

当 `status=success` 时必须提供视频路径。失败或无法读取的结果仍需保留在 manifest 中，以便统计有效样本覆盖率。使用以下命令进行校验：

```bash
acoustitrace validate-outputs outputs.csv --prompts prompts.jsonl --check-files
```

如果直接使用一站式命令的 `--videos-dir`，则无需另写 output manifest，但目录必须严格遵循上述 MP4 命名。自定义路径或其他容器格式仅通过显式 `--outputs` manifest 支持。

## 4. 统一媒体格式

运行 evaluator 前，请使用无损或高质量的中间格式，并记录源视频编码。必须保留原始音轨；不要用静音替代缺失音频，否则会把无效生成错误地变成一个可计分的声学失败样本。

## 5. 提取结构化证据

根据每个 prompt 的 evaluator 成员关系，将样本路由到相应后端：

| Evaluator | 证据提取后端 |
|---|---|
| Range Attenuation | 接收端深度/掩码轨迹与音频声级曲线 |
| Approach Gain | 接收端深度/掩码轨迹与原生接近增益输出 |
| Lateral Stability | 等距横向轨迹与声级稳定性 |
| Motion--Loudness | 音视频事件定位、视觉顺序与事件 RMS |
| Impact Decay | 匹配的冲击窗口与衰减包络拟合 |
| Causality Violation | 聚类后的音视频事件与起始时间差 |
| Log Attack Time | 生成/参考冲击声的起音时长 |
| RT60 Consistency | 音频表观 RT60 与视觉 RT60 估计值 |

后端脚本位于 `experiments/evaluator_backends/`，输出字段定义见 `docs/EVIDENCE_SCHEMA.md`。

## 6. 计分与聚合

`acoustitrace score-batch` 将 evidence JSONL 映射为以模型、prompt 和 evaluator 为单位的分数记录；随后，`acoustitrace aggregate` 输出条件均值、有效/尝试数量、覆盖率及 percentile bootstrap 置信区间。

`python evaluate.py` 会统一调度上述步骤，但不会改变各后端的计分定义。每个计算量较大的后端会先写出结构化证据，再执行分数映射和有效性感知聚合。公开目录结构和一站式命令见[开发机环境配置](DEVELOPMENT_MACHINE_zh-CN.md)。
