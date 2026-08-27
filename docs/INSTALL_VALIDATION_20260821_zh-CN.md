# 开发机安装验证记录（2026-08-21，2026-08-24 更新）

本文记录在 PJLab WebIDE 开发机上按照公开 README 配置 AcoustiTrace evaluator 时，实测流程与文档之间的差异。完成本轮验证后，应将其中具有普遍性的内容合并回正式安装文档；机器特有事项不应写成所有用户都必须执行的步骤。

## 开发机环境

- Ubuntu 24.04.4 LTS
- NVIDIA A800-SXM4-80GB
- 系统 Python 3.12.3
- 服务器预装 PyTorch 2.12.0a0（CUDA 13.2 构建），`torch.cuda.is_available()` 为 `True`

## 与 README 不同或 README 未提及的事项

### 1. Python 版本

README 以 Python 3.10 为目标环境，但这台开发机只提供 Python 3.12。为了保留服务器预装且可用的 CUDA/PyTorch 组合，本次使用：

```bash
python -m venv --system-site-packages .venv
```

这属于开发机兼容处理，不应替代公开 README 中的 Python 3.10 基准环境。需要在最终验证结果中明确 Python 3.12 是否可完整运行所有 evaluator。

### 2. 系统工具

基础镜像最初没有 `ffmpeg`、`ffprobe` 和 Git LFS，需要额外安装：

```bash
apt-get update
apt-get install -y ffmpeg git-lfs python3-venv
```

README 已列出 FFmpeg 和 Git LFS 为要求，但没有给出 Ubuntu 安装命令。该命令可考虑加入开发机文档。

### 3. Git 可执行文件冲突

开发机默认 `PATH` 首先解析到 `/shared/bin/git`。该 Git 缺少 `git-remote-https` helper，访问 HTTPS 仓库会报：

```text
git: 'remote-https' is not a git command
```

系统自带 `/usr/bin/git` 及 `/usr/lib/git-core/git-remote-https` 是完整的。本次下载固定版本第三方仓库时显式使用 `/usr/bin/git`，并设置：

```bash
export GIT_EXEC_PATH=/usr/lib/git-core
```

这是这台开发机的路径问题，不是 AcoustiTrace 的通用安装要求。

### 4. PyTorch 与 setuptools 约束

服务器预装的 PyTorch 2.12.0a0 要求 `setuptools<82`。常规执行 `pip install --upgrade pip setuptools wheel` 会安装 setuptools 84，并产生版本冲突。完成依赖安装后需要将 setuptools 回退到兼容范围，并重新运行依赖检查。

公开文档不应无条件要求升级 setuptools；更稳妥的做法是只升级 pip/wheel，或按目标 PyTorch 环境约束 setuptools。

### 5. Hugging Face 下载路径

该开发机不能直接访问 Hugging Face，需要使用站内代理。`hf download`/HTTP2/Xet 路径出现过连接中断；最终使用 `wget -c` 的 HTTP/1.1 断点续传完成 FlexSED 权重，并按仓库 API 清单补齐 Qwen3-VL 与 BERT 的配置、tokenizer 和模型索引文件。

代理地址和凭据属于开发机私密配置，不应写入公开 README、日志或仓库。

### 6. vLLM/PyTorch 版本组合（已修正）

直接执行 `pip install -r requirements-eval.txt` 时，pip 会解析当前最新的 vLLM 及其大规模依赖树，并试图安装另一套 PyTorch/CUDA 依赖。实测解析和下载持续约 35 分钟，仍未形成稳定环境，因此中止；继续安装可能覆盖开发机已经验证可用的 NVIDIA PyTorch 2.12/CUDA 13.2 组合。

这不是单纯的下载慢问题。现已固定为 vLLM 0.11.0、PyTorch/TorchAudio 2.8.0、TorchVision 0.23.0 与 `qwen-vl-utils==0.0.14`；公开安装文档要求使用干净环境。对于已预装厂商 PyTorch 的开发机，预检支持通过 `--source-python` 检查第二个声源 evaluator 环境，避免覆盖已验证的主环境。

### 7. 大型 wheel 经 pip 下载不稳定

通过开发机代理下载 OpenCV 和 PyAV 的大型 wheel 时，pip 多次停滞；使用 `wget -c` 断点续传后再执行本地 wheel 安装成功。这是代理/网络路径的机器特有问题，不应写进通用安装命令，但文档可以提醒用户在大型 wheel 下载中断时使用 pip 缓存或离线 wheel。

### 8. Python 3.12 下的 Decord 元数据告警

`pip check` 报告 `decord 0.6.0 is not supported on this platform`，但实际执行 `import decord` 成功，且加载的是当前虚拟环境中的包。该告警来自 Decord 0.6.0 的旧 wheel/平台元数据；仍需在真实视频读取 smoke test 中确认二进制接口是否完整。Python 3.10 基准环境通常可避免这一不确定性。

### 9. 测试发现的发布包问题

- 按 README 的 `python -m unittest discover -s tests -v` 运行 31 项测试：30 项通过，1 项失败。
- 失败项是 `test_private_paths_and_user_identifiers_are_not_shipped`。测试会扫描 README 要求用户克隆到仓库内的 `third_party/`，从 Grounded-Segment-Anything 上游源码/Dockerfile 中匹配到路径文本。这是 hygiene 测试未排除第三方仓库产生的误报，不是 AcoustiTrace 自身泄漏私有路径。
- 单独使用 `python -m unittest tests.test_end2end -v` 会因 `tests/` 不是 package、测试内部使用顶层导入而失败；按 README 的 discover 形式运行同一 `test_end2end.py` 则通过。这说明 README 给出的测试命令是正确的，但测试模块不支持点名模块运行。
- `examples/prompts_subset.jsonl` 目前不能直接用于 README 风格的一站式 smoke run：T2AV 示例没有 `receiver.detection_targets`，I2AV 示例没有 `log_attack_time.reference_audio_path`，会在 release-suite validation 阶段退出。单元测试内部动态构造的完整 mock manifest 可以通过，但仓库里现成的 example 文件尚未满足当前 evaluator contract。

### 10. 整包安装中止后容易遗漏后续依赖

早期未固定版本的 vLLM 使 `pip install -r requirements-eval.txt` 中途停止，排在依赖解析链中的包不一定已经真正安装。逐项导入核验时发现：

- `imageio-ffmpeg` 最初缺失，已单独安装并验证；
- README 固定 commit 的 `pytorchvideo` 最初缺失；通过 pip 的 VCS 安装会再次依赖开发机 Git HTTPS 路径，首次 clone 在网络阶段长时间无进展，仍需完成或改用预下载源码包。

预检脚本现已覆盖原生后端的直接运行依赖，并在隔离子进程中执行真实 import；这能发现仅检查 `find_spec` 无法识别的 Torch/TorchAudio/vLLM ABI 冲突。Grounded-SAM 漏列的 `addict` 也已加入依赖文件。

### 11. LTX-2.3 全量素材试运行新增发现

- `LTX2.3.zip` 含 605 个 T2AV 视频；`LTX2.3_i2av.zip` 含 748 个 I2AV 视频。依据旧文件名中携带的公开 `prompt_id`，已无歧义地整理出 605 个 T2AV 视频和 143 个 I2AV-LAT 视频，未发现缺失或重复。
- 605+143 一站式 dry-run 通过；使用 mock 后端完整跑通 748 个 prompt、8 个维度和 838 条 evaluator assignment，838 条均有效。同步代码后重新运行 43 项单元测试，全部通过。
- RT60 原生后端已在 166 个 T2AV 样本上完成正式运行：145 个样本有效，有效率 87.35%，LTX-2.3 的公开套件均分为 39.69。该结果验证了最终 LoRA adapter、physics head 和基础模型目录能够被当前封装实际加载。
- Receiver 后端首次运行暴露出 `01_run_vda_depth.py` 将 YAML 路径字符串直接与 `Path` 运算的错误；入口现已统一把 `repo_root` 解析为 `Path`。修复后 VDA 已完成缓存。另一个空 `event_audio_path` 被错误解释成当前目录的问题也已修复：后端会回退到自身提取的 `extracted_audio.wav`。正式重跑后 Range Attenuation 有 57/90 个有效样本，Approach Gain 有 61/90 个有效样本。
- Lateral Stability 的首轮 0/90 不是评分公式问题，而是 1,980 个动态 Grounded-SAM 关键帧均因缺少 `addict` 失败，随后被旧逻辑当作可用缓存永久跳过。依赖和缓存判断均已修正；轨迹阶段也会在动态检测恢复后自动替换旧的静态回退轨迹。
- Source Mechanics 与 Causality 后端首次运行发现当前 `.venv` 缺少 `torchaudio`。README 的 PyTorch 安装命令已经包含 TorchAudio，但预检此前没有检查；预检现已加入 `torchaudio`，并明确标注其为 OV-AVEL 依赖。
- OV-AVEL 从任意工作目录启动时，ImageBind 会把相对的 BPE 文件路径误解为当前工作目录；同时其文本张量默认留在 CPU，导致与 GPU 模型混用。公开 runner 现将 BPE 路径锚定到 ImageBind checkout，并显式把文本输入移动到模型设备。十条扩大验证已越过该阶段。
- FlexSED 上游 API 将 `laion/clap-htsat-unfused` 写成 Hugging Face 模型名，即使 `flexsed_as.pt` 已在本地也会尝试联网。公开 runner 现把这一依赖重定向到 `checkpoints/flexsed/laion-clap-htsat-unfused/`，预检、中英文 README 与开发机文档均已列出该目录。开发机已下载完整 CLAP 配置、tokenizer 和 586 MiB 模型文件，并进入真实样本推理。
- 独立 `.venv-source` 已完成 vLLM 0.11.0、PyTorch 2.8.0+cu128 和 GPU 实测。开发机全局 `TRITON_PTXAS_PATH` 指向 CUDA 13.2，超出 Triton 3.4 的版本识别范围；Motion--Loudness 入口现会优先选用当前 Triton wheel 自带的 CUDA 12.8 `ptxas`，也允许通过 `ACOUSTITRACE_TRITON_PTXAS_PATH` 显式覆盖。修复后 Qwen3-VL/vLLM 已完成真实视频排序。
- Source adapter 现将 Qwen 排序阶段与已完成的事件定位证据隔离：Motion--Loudness 阶段报错不再连带丢弃 Impact Decay 结果。三条小样本回归中，Impact Decay 3/3 有效；Motion--Loudness 1/3 有效，另外两条因没有形成可配对的定位事件而按 evaluator contract 判为无效，不是后端故障。Causality Violation 独立回归为 3/3 有效。
- Grounded-SAM 的 CUDA 扩展已针对当前 PyTorch C++ API 重编译；Receiver 三维真实小样本回归中，Range Attenuation 2/3、Approach Gain 2/3、Lateral Stability 3/3 有效。其余样本分别因没有可靠远离时段或接近段不占主导而自然失效。
- Zenodo TAR 经开发机代理读取时会随机中断。恢复脚本现把分片写入持久化 `.tar.part`，同时使用 HTTP Range 在单进程重连及进程重启后从精确字节位置继续；直连文件端点在该开发机上也明显快于旧的 `?download=1` 重定向。
- I2AV Log Attack Time 已使用真实生成视频与五秒参考音频完成两轮十条样本回归，20 条均有效；三条联合回归的有效率同样为 3/3。
- 八维一站式联合回归已从同一个 `evaluate.py` 入口顺序执行全部五个原生后端组。该轮覆盖每个维度三条 assignment、共 21 个唯一 prompt；五个后端均以返回码 0 完成，`run.json` 状态为 `success`，八个维度均至少产生一条有效分数。

## 已完成的准备工作

- 从对象存储下载并校验 AcoustiTrace 权重快照（19 个对象，约 24.4 GiB）。
- 补齐 FlexSED `flexsed_as.pt`（430,911,508 字节）。
- 补齐 Qwen3-VL-8B-Instruct 的配置、tokenizer、processor 和 safetensors index。
- 补齐 Grounded-SAM 所需 BERT 配置和 tokenizer 文件。
- 按安装文档固定 commit 下载四个第三方 evaluator 仓库。
- 安装 FFmpeg、Git LFS 和 Python venv 支持。
- 创建保留服务器 CUDA PyTorch 的 `.venv`，安装核心 evaluator 依赖；OpenCV、Transformers、PEFT、Qwen-VL utils、Librosa、PyAV、Decord、imageio-ffmpeg 等均可导入，CUDA 可见。
- 补装服务器预置 `triton-kernels` 声明依赖的 pytest；此后 `pip check` 只剩 Decord 的 Python 3.12 平台元数据告警。
- 五个 evaluator adapter（receiver、source mechanics、causality、log attack time、RT60）的 `--help` 入口均能启动。
- 按 discover 方式运行一站式 mock 回归测试，通过。
- 已从冻结的 current-605 T2AV assignment 表转换出 605 条唯一 prompt：Range 90、Receiver Observer 90（同时分配 Approach Gain 与 Lateral Stability）、Motion--Loudness 90、Impact Decay 90、Causality 79、RT60 166。
- 已将官方 I2AV Log Attack Time manifest 的 143 条 prompt 与发布条件图包逐 ID 对齐，生成 `i2av_lat_143.jsonl`。
- 两份发布 manifest 均通过 `validate-release-suite`；本地最终发布候选已运行 46 项测试并全部通过，且发布清单不含本机或服务器私有绝对路径。

## 当前验证状态

固定的 vLLM/PyTorch/TorchAudio 环境、Grounded-SAM 动态检测、Receiver 三维、Motion--Loudness、Impact Decay、Causality Violation、RT60 与 Log Attack Time 均已通过真实媒体验证。八维三条 assignment 联合回归的五个后端组全部成功，确认一站式调度、双环境解释器切换、GPU 串行释放、证据归一化、计分与聚合链路可以共同运行。

I2AV-LAT 的 143 个目标均已生成五秒参考音频，并以各来源视频解码后的第 2 帧（索引 1）生成和校验条件图；143/143 份资产均通过准备脚本检查。

在三条联合回归通过后，已额外提交每个维度十条 assignment 的扩大验证任务。该任务使用 70 个唯一 prompt 覆盖 80 个 evaluator assignment，并只选择已具备参考音频的十条 I2AV-LAT 样本；`dry-run` 路由校验通过，任务按五个后端组串行运行。它是扩大规模的发布前验证，不用于替换论文 leaderboard。

## 后续验证

- 等待十条 assignment/维度的扩大验证任务完成，并归档其 `run.json`、`summary.csv` 与 `failures.jsonl`。
