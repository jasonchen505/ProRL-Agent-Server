# Polar 项目增量学习笔记 -- 基于 8x4090 复现过程

> 本文档记录在复现过程中对比前两轮分析新发现的、深入理解的点。
> 前两轮文档: LLM_Agent_RL_Interview_Preparation.md + Interview_5_Categories_Deep_Preparation.md

---

## 目录

1. [从配置文件发现的工程细节](#1-从配置文件发现的工程细节)
2. [从 run.py 发现的 Harness 集成模式](#2-从-runpy-发现的-harness-集成模式)
3. [从 run.sh 发现的训练系统设计](#3-从-runsh-发现的训练系统设计)
4. [从 4090 适配中理解的 Scaling 约束](#4-从-4090-适配中理解的-scaling-约束)
5. [从代码细节发现的设计模式](#5-从代码细节发现的设计模式)
6. [对比前两轮的新理解](#6-对比前两轮的新理解)

---

## 1. 从配置文件发现的工程细节

### 1.1 Qwen3.5-4B 的架构特殊性

从 `model_args.sh` 发现 Qwen3.5-4B 不是标准 Transformer:

```bash
--spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
--use-gated-attention
--attention-output-gate
```

**新理解**: Qwen3.5-4B 使用 **GatedDeltaNet** (线性注意力) 混合架构 -- 每 4 层中
有 1 层 full attention + 3 层 GatedDeltaNet linear attention。这是论文没有详细
提到的架构细节。

**面试价值**: 这说明模型架构本身也在演进 (不只是训练方法)。如果面试官问
"你了解哪些新型注意力机制?"，可以讲 GatedDeltaNet 的设计动机: 线性注意力
降低推理时的 KV Cache 开销，对长 context Agent 任务特别重要。

### 1.2 Training 中的 Dynamic History

```bash
--dynamic-history
```

从 run.sh 的注释中发现: "every trace in each agent session becomes one training
sample, so gradients learn from *every* turn (not just the last one)"。

**新理解**: 之前以为 prefix merging 产出的 trace 是整个 session 一个样本。实际上
`--dynamic-history` 让每个 trace (prefix merging 后可能有多个 trace，因为 sub-agent
等会形成独立 chain) 都变成独立的训练样本。这意味着:
- 一个 session 可能产出 1-5 个训练样本 (取决于 sub-agent 数量)
- 每个样本的 prompt 是该 chain 的初始 prompt，response 是该 chain 的完整 merged token stream
- 这比 "整个 session 一个样本" 更细粒度，比 "每个 request 一个样本" 更粗粒度

**面试价值**: 这是 prefix_merging 的一个重要补充设计 -- 合并粒度介于 per-request
和 per-session 之间，由 harness 的 agent 拓扑自然决定。

### 1.3 SGLang Router 的角色

```bash
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
--sglang-router-port "$SGLANG_ROUTER_PORT"
--router-policy "${SGLANG_ROUTER_POLICY:-round_robin}"
```

**新理解**: 在训练场景中，Slime 自己管理 SGLang 引擎，并在前面放一个 SGLang
Router 做负载均衡。Polar Gateway 的 `base_url` 指向的是这个 Router (port 9000)，
而不是直接指向某个 SGLang 引擎。

这与纯推理场景不同 -- 纯推理时 Polar 自己管理 LLM 后端池 (min-heap)。训练时
Slime 接管了后端管理，Polar 只负责 proxy 和 capture。

**面试价值**: 体现了系统的分层设计 -- Polar 可以与不同的后端管理层集成，不绑定
自己的负载均衡策略。

### 1.4 Weight Sync 方式

从 run.sh 注释: "Weight sync: native GPU-to-GPU via NCCL every training step."

**新理解**: 权重更新不是通过 "保存 checkpoint -> 重新加载" 的方式，而是直接
通过 NCCL 从训练 GPU 传输到推理 GPU。这意味着:
- 没有磁盘 I/O 开销
- 权重更新延迟极低 (GPU-to-GPU direct transfer)
- 需要训练和推理 GPU 在同一节点 (或高速互联)

**面试价值**: 这是一个重要的工程优化 -- 如果用 checkpoint 方式，每次权重更新
需要: 保存到磁盘 (几秒) -> 推理服务器重新加载 (几十秒)。NCCL 直传可以
降到秒级。

---

## 2. 从 run.py 发现的 Harness 集成模式

### 2.1 Harness 安装的分层设计

从 `HARNESS_INSTALL` 字典发现两种安装方式:

```python
# npm-based harnesses (Claude Code, Codex, Gemini CLI, Pi, Qwen Code, etc.)
"claude_code": "npm install -g @anthropic-ai/claude-code@2.1.111"

# pip-based harnesses (Hermes, Mini SWE Agent)
"hermes": "python3 -m pip install --user --quiet hermes-agent==0.15.1"

# venv-based harnesses (OpenHands SDK)
"openhands_sdk": "python3 -m venv $HOME/.venv && $HOME/.venv/bin/pip install ..."
```

**新理解**: Harness 的安装是 **INIT 阶段的 prepare action**，在容器内执行。
这意味着:
- 每个 session 的容器是干净的 (从镜像启动)
- Harness CLI 在每个 session 中重新安装 (增加 INIT 时间)
- 但保证了隔离性 (一个 session 的安装不影响另一个)

**工程权衡**: 安装时间 vs 隔离性。如果把 harness 预装到镜像中可以加速 INIT，
但镜像会变得很大且难以维护 (每个 harness 一个镜像)。

### 2.2 Task Payload 的结构

从 `build_task_payload()` 发现完整的 task 描述:

```python
{
    "task_id": "calculator-mini_swe_agent-20260627T...",
    "instruction": "...",           # 给 agent 的指令
    "num_samples": 4,               # 每个 prompt 的 rollout 数量
    "timeout_seconds": 1200.0,      # 总超时
    "runtime": {
        "backend": "docker",
        "image": "polar-localhost-calculator:latest",
        "prepare": [...],           # INIT 阶段的动作列表
        "network": "host",
        "workdir": "/polar/session/workspace",
    },
    "agent": {
        "harness": "mini_swe_agent",
        "model_name": "openai/gpt-5.4",  # Agent 看到的模型名
    },
    "builder": {"strategy": "prefix_merging"},
    "evaluator": {
        "strategy": "test_on_output",
        "config": {...},
        "refresh_runtime": True,    # 评估用新 runtime
    },
}
```

**新理解**:
1. `model_name` 是 Agent CLI 看到的名字，Gateway 会 rewrite 为实际服务的模型
2. `refresh_runtime: True` 意味着评估在全新的 runtime 中进行 (不在 agent 执行的
   runtime 中)，保证评估环境干净
3. `prepare` 是一个有序的动作列表，支持 `exec` 和 `upload_file` 两种类型
4. `builder` 和 `evaluator` 是可配置的策略，体现了 Polar 的策略模式设计

### 2.3 Evaluator 的 refresh_runtime 设计

```python
"evaluator": {
    "strategy": "test_on_output",
    "config": {...},
    "refresh_runtime": True,  # 关键!
}
```

**新理解**: `refresh_runtime: True` 意味着评估在全新的容器中进行，而不是在
agent 执行的容器中。这很重要因为:
- Agent 可能修改了环境状态 (安装了包、创建了文件)
- 评估需要在干净的环境中验证 patch 是否可复现
- SWE-bench 的评估标准是: 在干净环境中 apply patch -> 运行测试

**面试价值**: 这体现了 "评估环境与训练环境分离" 的原则 -- 如果在 agent 执行的
环境中评估，可能因为环境状态的"副作用"而得到错误的 reward。

---

## 3. 从 run.sh 发现的训练系统设计

### 3.1 Slime 的 Rollout 函数注入

```bash
--rollout-function-path slime_bridge.rollout.generate_rollout_polar_async
--custom-rm-path slime_bridge.reward.reward_func
--custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards
--custom-config-path "${CUSTOM_CONFIG_PATH}"
--data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer
```

**新理解**: Slime 通过路径注入实现插件化:
- `rollout-function-path`: 自定义 rollout 生成函数 (Polar bridge)
- `custom-rm-path`: 自定义 reward model 函数
- `custom-reward-post-process-path`: 自定义 reward 后处理
- `data-source-path`: 自定义数据源

这意味着 Polar 与 Slime 的集成是通过 **函数注入** 而非修改 Slime 代码实现的。
这是一种低侵入的集成方式。

### 3.2 CeilEpochRolloutDataSourceWithBuffer

从数据源名称推断: 这个数据源实现了 "ceil epoch" 逻辑 -- 如果 293 条数据不能
被 rollout_batch_size 整除，最后一批会 wrap around (从头取剩余的)。

从注释验证: "The custom data source rounds epoch length up to 37 rollout batches,
so all 293 train prompts are consumed once; the final fixed-size batch wraps 3 prompts."

**新理解**: 293 / 4 (batch_size) = 73.25，向上取整为 74 batches。但实际上
n_samples_per_prompt=16，所以每个 prompt 产生 16 个 rollout，293 * 16 = 4688
个 rollout per epoch。

### 3.3 TIS (Truncated Importance Sampling)

```bash
--use-tis
```

**新理解**: TIS 是 GRPO 中处理 off-policy 数据的技术。因为 rollout 是用旧 policy
生成的，但训练用新 policy，需要 importance sampling 来修正。TIS 通过截断极端
ratio 值来稳定训练。

**面试追问准备**: "什么是 TIS? 为什么需要它?"
答: Agent rollout 生成时间长，训练时 policy 已经更新了。TIS 用
min(ratio, clip(ratio)) 修正 off-policy 偏移，防止 ratio 过大导致的训练不稳定。

### 3.4 KL Loss 的作用

```bash
--use-kl-loss
--kl-loss-coef 0.001
--kl-loss-type low_var_kl
```

**新理解**: KL loss 防止 policy 偏离 reference model 太远。在 Agent RL 中这特别
重要因为:
- Agent 任务的 reward 很 sparse，容易过拟合到特定模式
- 如果 policy 太偏离 reference，可能生成格式错误的 tool calls
- `low_var_kl` 是一种低方差的 KL 估计方法

---

## 4. 从 4090 适配中理解的 Scaling 约束

### 4.1 显存预算分配 (4090 24GB)

以 Qwen3.5-4B TP=1 推理为例:

```
模型权重 (FP16): ~8GB
KV Cache (32K context): ~8-10GB
SGLang 运行时开销: ~2-3GB
总计: ~18-21GB (24GB 可用)
```

以 Qwen3.5-4B TP=2 训练为例:

```
模型权重 (FP16): ~4GB/卡 (TP=2 分摊)
优化器状态 (Adam): ~8GB/卡
梯度: ~4GB/卡
激活值 (recompute): ~2-4GB/卡
总计: ~18-20GB/卡 (24GB 可用)
```

**关键约束**: 4090 的 24GB 几乎刚好够 4B 模型的推理+训练，没有太多余量。
如果要增大 context_length 或 batch_size，必须减小其他参数。

### 4.2 PCIe vs NVLink 的影响

4090 没有 NVLink，卡间通信走 PCIe (~32GB/s vs NVLink ~900GB/s)。

对 TP=2 训练的影响:
- AllReduce 梯度同步: 4B 模型梯度 ~8GB，PCIe 传输 ~0.25s vs NVLink ~0.01s
- 每个 training step 多出 ~0.5s 通信开销
- 对比 H100 NVLink: 每步多 ~25x 的通信时间

**实际影响**: 训练速度可能只有 H100 的 10-20% (算力 + 带宽 + 通信的综合影响)。

### 4.3 最小可行配置的发现

通过分析发现，最小 smoke test 只需要 **1 张 4090**:

```
GPU 0: SGLang 推理 (Qwen3.5-4B, TP=1, ~14GB)
CPU: Rollout Server + Gateway (轻量)
```

这降低了学习门槛 -- 不需要 8 卡都可用就能开始学习。

---

## 5. 从代码细节发现的设计模式

### 5.1 Prepare Action 的有序执行

从 run.py 的 `prepare` 列表发现:

```python
"prepare": [
    {"type": "exec", "command": "npm install -g ... && git init ..."},
    {"type": "upload_file", "source": "test_calculator.py", "target": "..."},
    {"type": "upload_file", "source": "calculator.py", "target": "..."},
    {"type": "exec", "command": "git add -A && git commit -qm 'initial'"},
]
```

**新理解**: Prepare actions 是有序的，支持 exec 和 upload_file 交替。这允许:
1. 先安装依赖
2. 上传文件
3. 初始化 git (用于后续 diff)

### 5.2 Git-based Patch 提取

从 evaluator 配置发现:

```python
"patch_command": "cd /polar/session/workspace && git add -A && git diff --cached --binary"
```

**新理解**: Patch 提取用 git diff 而不是直接比较文件。好处:
- 支持二进制文件
- 自动处理文件创建/删除
- `--cached` 只看 staged 的变更 (agent 通过 git add 添加的)
- `--binary` 支持二进制 patch

### 5.3 模型名 Rewrite

从 `HARNESS_MODEL` 字典发现:

```python
"codex": "openai/gpt-5.5",       # Agent 看到的模型名
"pi": "openai/gpt-5.4",
"qwen_code": "qwen3-coder-plus",
```

但实际推理用的是 `Qwen/Qwen3.5-4B`。Gateway 会把 Agent 请求中的模型名
rewrite 为实际服务的模型。

**新理解**: 这是黑盒代理的关键实现细节 -- Agent 以为自己在和 GPT-5.5 对话，
实际上在和 Qwen3.5-4B 对话。这让同一个 Agent harness 可以用不同的后端模型
训练/评估。

---

## 6. 对比前两轮的新理解

### 6.1 前两轮 vs 本轮的理解深度对比

| 主题 | 前两轮理解 | 本轮新增理解 |
|------|-----------|-------------|
| Prefix Merging | 算法原理、token-faithful 保证 | dynamic-history 如何将 trace 转为训练样本 |
| Harness 集成 | 黑盒代理、环境变量注入 | prepare action 的有序执行、harness 安装分层 |
| 训练系统 | GRPO 原理、DAPO 改进 | TIS、KL loss、NCCL weight sync 的实际作用 |
| 模型架构 | Qwen3.5-4B 是标准 Transformer | GatedDeltaNet 混合注意力架构 |
| 负载均衡 | Polar 的 min-heap 策略 | Slime 的 SGLang Router 与 Polar 的分工 |
| 评估环境 | refresh_runtime 概念 | 评估环境与训练环境分离的具体实现 |
| 资源约束 | 理论上的显存需求 | 4090 24GB 的精确预算分配 |

### 6.2 前两轮文档中未覆盖的重要点

1. **GatedDeltaNet 混合架构**: Qwen3.5-4B 不是纯 Transformer，每 4 层中有 3 层
   线性注意力。这对推理效率和 KV Cache 管理有重要影响。

2. **NCCL Weight Sync**: 权重更新不是通过 checkpoint 而是通过 GPU-to-GPU 直传。
   这是训练效率的关键优化。

3. **SGLang Router vs Polar min-heap**: 训练场景下 Slime 管理推理后端 (通过
   SGLang Router)，Polar 只负责 proxy。纯推理场景下 Polar 自己管理后端。

4. **Dynamic History**: prefix merging 的 trace 粒度不是固定的 -- 由 harness 的
   agent 拓扑 (是否有 sub-agent, context compaction) 自然决定。

5. **Evaluator refresh_runtime**: 评估在全新容器中进行，保证评估环境干净。
   这对 SWE-bench 这类需要 "apply patch + run test" 的评估至关重要。

6. **Prepare Action 机制**: INIT 阶段的动作是有序列表，支持 exec/upload 交替。
   这比简单的 "启动容器 + 运行命令" 更灵活。

7. **Git-based Patch Extraction**: 用 git diff 而非文件比较提取 patch，支持
   二进制文件和文件创建/删除。

### 6.3 面试中可以展示的新增深度

**当被问到 "你实际跑过这个项目吗?"**:

> 我在 8x4090 环境下复现了这个项目。原始配置需要 8xH100，我做了以下适配:
> 1. 将推理模型从 Qwen3.6-27B 换为 Qwen3.5-4B (4B 参数在 4090 单卡可推理)
> 2. 将 context_length 从 262K 降到 32K (减少 KV Cache 显存)
> 3. 将 n_samples_per_prompt 从 16 降到 4 (减少推理量)
> 4. 训练 TP=2 在 4090 上可行但很慢 (PCIe 通信 vs NVLink)
>
> 通过实际运行，我深入理解了几个之前分析中没有注意到的细节:
> - Qwen3.5-4B 使用 GatedDeltaNet 混合注意力架构
> - 权重更新通过 NCCL GPU-to-GPU 直传而非 checkpoint
> - 训练时 Slime 通过 SGLang Router 管理推理后端，与 Polar 的 min-heap 分工不同
> - Evaluator 的 refresh_runtime 确保评估在干净容器中进行
