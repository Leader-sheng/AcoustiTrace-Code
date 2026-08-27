# Evaluator 实现索引

[English](IMPLEMENTATION_MAP.md) | 简体中文

| Evaluator | 分数计算入口 | 已包含的证据提取后端 |
|---|---|---|
| Motion--Loudness | `motion_loudness` | OV-AVEL/FlexSED 匹配、时间连续双 cluster、Qwen 仅动作判断和 8 fps 帧差回退 |
| Log Attack Time | `log_attack_time` | 成对冲击事件定位输入、包络提取与 T10/T90 测量 |
| Impact Decay | `impact_decay` | 事件窗口提取、指数项与噪声底拟合、残差诊断 |
| RT60 Consistency | `rt60_consistency` | 严格的音频 proxy，以及已发布的 Qwen3-VL、Sabine 引导推理运行时 |
| Causality Violation | `causality_violation` | 置信度过滤、聚类、事件关联和 1 ms 违例计分 |
| Range Attenuation | `range_attenuation` | Grounded-SAM 跟踪、VDA 相对距离、符号自适应局部窗指数搜索 |
| Approach Gain | `native_receiver_score` | 相对深度接近筛选，以及原生单调性/趋势输出 |
| Lateral Stability | `native_receiver_score` | Unified v2 等距局部窗计算，再执行横向视觉筛选 |

原始媒体处理脚本和 RT60 推理运行时均位于 `experiments/evaluator_backends/`。
外部检测器仓库与模型权重不直接复制到本项目中；RT60 微调代码计划后续发布。
