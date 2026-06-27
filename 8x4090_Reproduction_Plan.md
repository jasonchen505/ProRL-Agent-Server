# Polar 项目 8x4090 复现计划

> 硬件环境: 8 x NVIDIA RTX 4090 (24GB VRAM, 共 192GB)
> 目标: 在有限算力下完整走通 Polar 的核心流程，深入理解项目

---

## 目录

1. [资源评估与可行性分析](#1-资源评估与可行性分析)
2. [分阶段复现计划](#2-分阶段复现计划)
3. [Phase 1: 环境搭建与 Smoke Test](#3-phase-1-环境搭建与-smoke-test)
4. [Phase 2: 多 Harness 推理验证](#4-phase-2-多-harness-推理验证)
5. [Phase 3: GRPO 训练全流程](#5-phase-3-grpo-训练全流程)
6. [关键配置修改清单](#6-关键配置修改清单)
7. [预期问题与解决方案](#7-预期问题与解决方案)
8. [学习检查点](#8-学习检查点)

---

## 1. 资源评估与可行性分析

### 1.1 硬件对比

| 参数 | 原始环境 (H100) | 你的环境 (4090) | 影响 |
|------|----------------|----------------|------|
| 每卡 VRAM | 80GB | 24GB | 模型/context 受限 |
| 总 VRAM (8卡) | 640GB | 192GB | 只能用小模型 |
| FP16 算力 | 989 TFLOPS | 165 TFLOPS | 训练速度 ~1/6 |
| 内存带宽 | 3.35 TB/s | 1.0 TB/s | 推理速度 ~1/3 |
| 卡间互联 | NVLink 900GB/s | PCIe ~32GB/s | TP 通信瓶颈 |
| 典型用途 | 生产训练 | 桌面/研究 | 小规模验证 |

### 1.2 模型选择策略

**原始配置 vs 适配配置**:

| 组件 | 原始模型 | 适配模型 | 显存需求 |
|------|---------|---------|---------|
| 推理演示 | Qwen3.6-27B (TP=4) | **Qwen3.5-4B** (TP=1) | ~12-16GB/卡 |
| GRPO 训练 | Qwen3.5-4B (TP=2) | **Qwen3.5-4B** (TP=2) | ~16-20GB/卡 |

**为什么选择 Qwen3.5-4B**:
- 论文中训练实验就是用这个模型，结果可对比
- 4B 参数在 4090 24GB 下 TP=1 可推理，TP=2 可训练
- 支持 GatedDeltaNet 混合注意力架构 (论文创新点之一)

### 1.3 GPU 分配方案 (8 卡全用于训练)

```
GPU 0-3: Slime 训练 (Actor, TP=2, 4 卡)
GPU 4-7: SGLang 推理 (Rollout, TP=1, 每卡一个引擎)
```

### 1.4 可行性总结

| 场景 | 可行性 | 最少 GPU | 预估时间 |
|------|--------|---------|---------|
| Calculator smoke test | **可行** | 1 卡 | 30 min |
| 单 harness 推理验证 | **可行** | 1 卡 | 1-2 h |
| 多 harness 对比 | **可行** | 2-4 卡 | 2-4 h |
| SWE-bench 评估 | **可行 (小规模)** | 2-4 卡 | 数小时 |
| GRPO 完整训练 | **可行 (极慢)** | 8 卡 | 数天/epoch |

---

## 2. 分阶段复现计划

```
Phase 1 (Day 1-2): 环境搭建 + Calculator Smoke Test
    |-- 安装 Polar + SGLang
    |-- 构建 calculator Docker 镜像
    |-- 用 Qwen3.5-4B 单卡跑通 calculator
    |-- 理解 Proxy, Gateway, Rollout Server 的交互

Phase 2 (Day 3-5): 多 Harness 推理验证
    |-- 测试 mini_swe_agent, shell 等轻量 harness
    |-- 观察 trajectory 构建 (prefix_merging vs per_request)
    |-- 理解 evaluator 工作机制
    |-- 尝试 SWE-bench 小规模评估 (10-20 instances)

Phase 3 (Day 5-10): GRPO 训练全流程
    |-- 安装 Slime + Megatron-LM
    |-- 转换 Qwen3.5-4B 权重到 Megatron 格式
    |-- 跑通 swegym_slime_grpo 训练 (小 batch)
    |-- 观察训练曲线和 reward 变化
```

---

## 3. Phase 1: 环境搭建与 Smoke Test

### 3.1 安装 Polar

```bash
cd /data/home/yizhou/ProRL-Agent-Server
uv venv --python 3.12
uv pip install -e .
source .venv/bin/activate
```

### 3.2 安装 SGLang 推理服务器

```bash
uv pip install "sglang==0.5.13"
```

### 3.3 下载模型

```bash
# Qwen3.5-4B (训练+推理)
huggingface-cli download Qwen/Qwen3.5-4B

# 或者用 modelscope
modelscope download Qwen/Qwen3.5-4B
```

### 3.4 修改 Calculator Topology

创建适配 4090 的 topology 文件:

```yaml
# examples/calculator/topology.4090.yaml
rollout:
  host: 127.0.0.1
  port: 8080
  public_url: http://127.0.0.1:8080
  save_dir: ./rollout_results

gateway:
  heartbeat_interval_seconds: 30
  nodes:
    - id: localhost-node-01
      host: 127.0.0.1
      port: 8100
      public_url: http://127.0.0.1:8100
      max_init_workers: 4
      max_run_workers: 4
      max_postrun_workers: 4
      model_served: Qwen/Qwen3.5-4B
      inference:
        engine: sglang
        base_url: http://127.0.0.1:8000
```

### 3.5 启动推理服务器

```bash
# 单卡启动 SGLang
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B \
  --port 8000 \
  --tp 1 \
  --context-length 32768 \
  --mem-fraction-static 0.85
```

### 3.6 启动 Polar 服务

```bash
# 终端 1: Rollout Server
polar serve_rollout -c examples/calculator/topology.4090.yaml

# 终端 2: Gateway Node
polar serve_gateway -c examples/calculator/topology.4090.yaml --node-id localhost-node-01
```

### 3.7 构建并运行 Calculator

```bash
# 构建 Docker 镜像
uv run python examples/calculator/build_image.py

# 运行单个 harness (先用最简单的 mini_swe_agent)
uv run python examples/calculator/run.py \
  --harness mini_swe_agent \
  --backend docker
```

### 3.8 验证产出

检查 `./rollout_results/` 目录:
- `task_calculator-mini_swe_agent-*/` 包含 session 结果
- 每个 session 有 trajectory 数据 (prompt_ids, response_ids, loss_mask)
- Dashboard 可视化: `polar dashboard -c examples/calculator/topology.4090.yaml`

---

## 4. Phase 2: 多 Harness 推理验证

### 4.1 测试轻量 Harness

```bash
# mini_swe_agent (Python, 无需 npm)
uv run python examples/calculator/run.py --harness mini_swe_agent

# hermes (Python, 无需 npm)
uv run python examples/calculator/run.py --harness hermes

# shell (最简单的 harness)
# 需要自己创建一个 shell harness 的 task payload
```

### 4.2 测试 npm-based Harness (需要网络)

```bash
# codex (如果能访问 OpenAI API)
uv run python examples/calculator/run.py --harness codex

# pi
uv run python examples/calculator/run.py --harness pi
```

### 4.3 观察 Trajectory 构建

```python
# 读取 session 结果，分析 trajectory
import json
from pathlib import Path

results_dir = Path("./rollout_results")
for task_dir in results_dir.iterdir():
    if not task_dir.is_dir():
        continue
    for ses_file in task_dir.glob("ses_*.json"):
        data = json.loads(ses_file.read_text())
        traj = data.get("trajectory", {})
        print(f"Session: {ses_file.stem}")
        print(f"  Status: {traj.get('status')}")
        print(f"  Traces: {len(traj.get('traces', []))}")
        for i, trace in enumerate(traj.get('traces', [])):
            print(f"  Trace {i}: prompt={len(trace['prompt_ids'])} tokens, "
                  f"response={len(trace['response_ids'])} tokens, "
                  f"loss_mask sum={sum(trace['loss_mask'])}")
```

### 4.4 SWE-bench 小规模评估

```bash
# 修改 swebench_verified 的 topology 为 4090 版本
# 只跑 10-20 个 instance 验证流程
```

---

## 5. Phase 3: GRPO 训练全流程

### 5.1 安装训练依赖

```bash
# 安装 Slime
git clone https://github.com/THUDM/slime.git
cd slime && git checkout v0.3.0
uv pip install -e .

# 安装 Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && git checkout 26.04-alpha.rc1

# 安装其他依赖
uv pip install flash-linear-attention==0.5.0
uv pip install transformer-engine==2.5.0
```

### 5.2 转换模型权重

```bash
# 下载 Qwen3.5-4B
huggingface-cli download Qwen/Qwen3.5-4B --local-dir tmp/Qwen3.5-4B

# 转换为 Megatron torch_dist 格式
bash examples/swegym_slime_grpo/convert_weights.sh
```

### 5.3 准备训练数据

```bash
# 下载 SWE-Gym 数据
python examples/swegym_slime_grpo/prepare_data.py
```

### 5.4 修改训练配置 (4090 适配)

关键修改:

```bash
# 原始配置 (H100)
ACTOR_NUM_GPUS_PER_NODE=4
ROLLOUT_NUM_GPUS=4
ROLLOUT_NUM_GPUS_PER_ENGINE=1
ROLLOUT_BATCH_SIZE=4
N_SAMPLES_PER_PROMPT=16
MAX_TOKENS_PER_GPU=30000
SGLANG_CONTEXT_LENGTH=50000

# 4090 适配配置
ACTOR_NUM_GPUS_PER_NODE=4
ROLLOUT_NUM_GPUS=4
ROLLOUT_NUM_GPUS_PER_ENGINE=1
ROLLOUT_BATCH_SIZE=2          # 减半，降低显存峰值
N_SAMPLES_PER_PROMPT=4         # 从 16 降到 4，大幅减少推理量
MAX_TOKENS_PER_GPU=15000       # 减半，避免 OOM
SGLANG_CONTEXT_LENGTH=32768    # 从 50K 降到 32K，减少 KV Cache
```

### 5.5 启动训练

```bash
# 使用修改后的配置启动
export ACTOR_NUM_GPUS_PER_NODE=4
export ROLLOUT_NUM_GPUS=4
export ROLLOUT_BATCH_SIZE=2
export N_SAMPLES_PER_PROMPT=4
export MAX_TOKENS_PER_GPU=15000
export SGLANG_CONTEXT_LENGTH=32768

bash examples/swegym_slime_grpo/run.sh
```

### 5.6 监控训练

```bash
# W&B 监控 (如果配置了)
wandb login

# GPU 监控
nvidia-smi -l 1

# Polar Dashboard
polar dashboard -c tmp/swegym_slime_grpo/topology.yaml
```

---

## 6. 关键配置修改清单

### 6.1 Calculator Topology (topology.4090.yaml)

| 配置项 | 原始值 | 修改值 | 原因 |
|--------|--------|--------|------|
| model_served | Qwen3.6-27B | Qwen3.5-4B | 27B 放不进 4090 24GB |
| gateway nodes | 2 个 | 1 个 | 减少资源占用 |
| max_run_workers | 8 | 4 | 匹配并发能力 |
| engine base_url | :8000 + :8001 | :8000 | 只用 1 个推理实例 |

### 6.2 SGLang 启动参数

| 配置项 | 原始值 | 修改值 | 原因 |
|--------|--------|--------|------|
| tp | 4 | 1 | 4B 模型单卡可放下 |
| context-length | 262144 | 32768 | 减少 KV Cache 显存 |
| mem-fraction-static | 0.85 | 0.85 | 保持不变 |
| GPU | 0-3 或 4-7 | 0 | 单卡即可 |

### 6.3 GRPO 训练参数

| 配置项 | 原始值 | 修改值 | 原因 |
|--------|--------|--------|------|
| rollout_batch_size | 4 | 2 | 降低显存峰值 |
| n_samples_per_prompt | 16 | 4 | 减少推理量 |
| max_tokens_per_gpu | 30000 | 15000 | 避免 OOM |
| sglang_context_length | 50000 | 32768 | 减少 KV Cache |
| tensor_model_parallel_size | 2 | 2 | 保持不变 |
| save_interval | 10 | 5 | 更频繁保存 checkpoint |

---

## 7. 预期问题与解决方案

### 7.1 推理阶段

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| OOM on 4090 | KV Cache 太大 | 减小 context-length 到 16K-32K |
| 推理速度极慢 | 4090 带宽低 | 减少 concurrent sessions |
| Harness 安装失败 | 网络问题 | 优先用 mini_swe_agent/hermes (pip) |
| Docker 镜像构建失败 | 磁盘空间 | 检查磁盘，清理旧镜像 |

### 7.2 训练阶段

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 训练 OOM | 激活值太大 | 启用 full recompute + 减小 batch |
| TP=2 通信慢 | 4090 无 NVLink | 这是硬件限制，只能接受 |
| Slime 安装失败 | 依赖冲突 | 用 venv 隔离，严格按照 launch_e2e.sh |
| 权重转换失败 | 格式不兼容 | 检查 Megatron-LM 版本 |
| Rollout 超时 | 推理太慢 | 减小 n_samples_per_prompt |

### 7.3 常见踩坑

1. **SGLang VLM patch**: Qwen3.5-4B 是 VLM checkpoint，text-only 推理需要 SGLang VLM input_ids patch
2. **Apptainer vs Docker**: HPC 用 Apptainer，本地开发用 Docker
3. **端口冲突**: 确保 8000, 8080, 8100, 9000 端口未被占用
4. **CUDA 版本**: 4090 需要 CUDA 11.8+，推荐 12.x
5. **Python 版本**: 项目要求 Python >= 3.11

---

## 8. 学习检查点

### Phase 1 完成后应理解

- [ ] Polar 的三层架构 (Rollout Server / Gateway / Inference)
- [ ] Proxy 如何透明代理 LLM API 调用
- [ ] Session 生命周期 (INIT -> READY -> RUNNING -> POST-RUN)
- [ ] Trajectory 的数据结构 (prompt_ids, response_ids, loss_mask, logprobs)
- [ ] Evaluator 如何打分

### Phase 2 完成后应理解

- [ ] 不同 harness 的区别 (tool calling format, context management)
- [ ] Prefix Merging vs Per-Request 的实际效果差异
- [ ] Token-faithful 的含义和重要性
- [ ] Reward signal 如何从 evaluator 传递到 trajectory

### Phase 3 完成后应理解

- [ ] GRPO 的完整训练循环 (rollout -> reward -> advantage -> update)
- [ ] Slime 如何与 Polar 集成 (slime_bridge)
- [ ] 权重更新的 Pause-Update-Resume 协议
- [ ] DAPO 的 zero-variance filtering
- [ ] 异步 RL 的优势 (为什么比同步 batch-by-batch 好)
