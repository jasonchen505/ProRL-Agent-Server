# LLM Agent RL 面试准备指南 -- 基于 ProRL-Agent-Server (Polar) 项目

> 本文档基于 NVIDIA NeMo 团队的 ProRL-Agent-Server (Polar) 项目，帮助你准备 LLM & Agent 应用及后训练方向的算法实习面试。

---

## 目录

1. [项目概述与核心定位](#1-项目概述与核心定位)
2. [系统架构深度解析](#2-系统架构深度解析)
3. [论文核心创新点](#3-论文核心创新点)
4. [RL for LLM Agent 核心概念](#4-rl-for-llm-agent-核心概念)
5. [代码实现关键模块解析](#5-代码实现关键模块解析)
6. [面试深挖点与考察问题](#6-面试深挖点与考察问题)
7. [自我介绍模板与话术](#7-自我介绍模板与话术)
8. [延伸知识与行业理解](#8-延伸知识与行业理解)

---

## 1. 项目概述与核心定位

### 1.1 一句话定义

**Polar** 是一个 **"Rollout as a Service"** 的 Agent RL 训练基础设施，将 Agent 的 Rollout 执行与 RL 训练完全解耦，通过 HTTP 服务边界实现异步、可扩展的多轮 LLM Agent RL 训练。由 NVIDIA NeMo 团队开发，是 NVIDIA NeMo Gym 的一部分。

### 1.2 解决的核心问题

传统 Agent RL 训练的三大痛点：
- **耦合问题**：现有框架（SkyRL-Agent, rLLM, Agent Lightning）将 Rollout 逻辑嵌入训练循环，切换任务/Agent/RL算法需要大量工程改造
- **Harness 集成困难**：真实世界的 Agent（Claude Code, Codex, OpenHands）往往是闭源或复杂的软件系统，难以改造为 RL 环境接口
- **Token 保真性**：多轮交互中 re-tokenization 会导致 off-policy 偏移

### 1.3 核心设计哲学

> *"Can we train agents with RL without opening the box?"*

**关键洞察**：所有 LLM Agent 都必须与模型对话。通过在 LLM API 边界设置代理（Proxy），捕获 token 级交互数据，就能将任意 Agent 作为黑盒进行 RL 训练。

### 1.4 关键术语表

| 术语 | 含义 |
|------|------|
| **Harness** | Agent 的运行环境和调度逻辑（如 Claude Code, Codex CLI） |
| **Rollout** | 一次完整的 Agent 交互轨迹（从 prompt 到最终结果） |
| **Trace** | 可训练的 token 序列，包含 prompt_ids, response_ids, loss_mask, logprobs |
| **Trajectory** | 一组 Traces 的集合，对应一个完整 session |
| **Completion** | 一次 LLM API 调用的完整记录（request + response + token IDs） |
| **Session** | 一个 Agent 任务执行的完整生命周期 |
| **Gateway** | Worker 节点上的服务，负责 session 的执行和捕获 |
| **Runtime** | 容器沙箱（Docker/Apptainer），为 Agent 提供隔离执行环境 |
| **Evaluator** | 评估策略，为轨迹打分（reward signal） |
| **Builder** | 轨迹重建策略，将 completions 转换为 traces |

---

## 2. 系统架构深度解析

### 2.1 三层服务架构

```
+-------------------------------------------------------------------+
|                    RL Trainer (Slime / VE / NemoRL)                |
|                         ^ HTTP API                                |
+-------------------------------------------------------------------+
|               Rollout Server (central orchestrator, :8080)         |
|   - TaskRequest 接收与展开 (num_samples sessions)                   |
|   - NodeScheduler 节点调度 (stage-aware pressure)                   |
|   - Task 生命周期管理 (callback / polling)                           |
|                         ^ HTTP                                    |
+-------------------------------------------------------------------+
|              Gateway Nodes (each Worker node, :8100+)              |
|   +------------------------------------------------------------+  |
|   | INIT -> READY -> RUNNING -> POST-RUN (4-stage async)       |  |
|   |  |        |        |          |                            |  |
|   | Runtime  Buffer   Agent     Trajectory + Eval              |  |
|   | Setup    Pool    Execution  + Callback                     |  |
|   +------------------------------------------------------------+  |
|              ^ Proxy (transparent LLM API proxy)                  |
+-------------------------------------------------------------------+
|           Inference Server (SGLang / vLLM)                        |
+-------------------------------------------------------------------+
```

### 2.2 关键架构设计选择

**1. 黑盒代理 (Black-box Proxy)**
- Agent 不知道 Polar 的存在
- 通过注入环境变量 (`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`) 透明代理 LLM API 调用
- 支持 Anthropic / OpenAI Chat / OpenAI Responses / Google 等多种 API 格式自动检测和转换
- 代理边界在 Agent 框架之下，不需要理解 harness 内部逻辑

**2. 异步流水线 (Asynchronous Staging)**
- 每个 Gateway 内部有独立的 INIT, READY, RUNNING, POST-RUN worker pool
- CPU 密集的 runtime 准备和长尾评估不会阻塞 GPU 密集的 Agent 执行
- Ready buffer 允许 runtime 预热，减少 Agent 执行等待时间
- Evaluator prewarm: 评估用的 runtime 在 Agent 执行期间就开始准备

**3. Token 保真性 (Token Faithfulness)**
- Prompt IDs 和 Response IDs 直接从推理服务器获取 (token-in/token-out)
- 避免 decode -> re-encode 带来的 re-tokenization drift
- Loss mask 精确标记哪些 token 是 behavior policy 采样的
- 对多轮 rollout，prior assistant turns 保留原始 token IDs，直接拼接到 input buffer

### 2.3 与其他框架对比

| 特性 | Polar | SkyRL-Agent | rLLM | Agent Lightning |
|------|-------|-------------|------|-----------------|
| Rollout 与训练解耦 | 完全解耦 | 嵌入训练循环 | 嵌入训练循环 | 部分解耦 |
| Agent 无需改造 | 黑盒代理 | 需要 Gym 接口 | 需要改造 | 需要 SDK |
| Token 保真性 | 原始 token IDs | 可能 re-tokenize | 可能 re-tokenize | 保留 |
| 多 API 格式支持 | Anthropic/OpenAI/Google | OpenAI only | OpenAI only | 有限 |
| HPC 无守护进程支持 | Apptainer | Docker only | Docker only | Docker only |
| Rollout 独立生命周期 | 有 | 无 | 无 | 无 |

### 2.4 Session 生命周期状态机

```
REGISTERED -> INIT -> READY -> RUNNING -> POST-RUN -> COMPLETED / ERROR / TIMEOUT
                          (buffered)        |
                                            +-- Trajectory building
                                            +-- Evaluation
                                            +-- Callback to rollout server
                                            +-- Resource teardown
```

---

## 3. 论文核心创新点

### 3.1 Paper 1: ProRL Agent (arXiv:2603.18815)

**标题**: ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents

**核心贡献**：
1. **Rollout-as-a-Service 设计**：首次将 Agent Rollout 生命周期作为独立 HTTP 服务暴露，Trainer 只通过 HTTP 提交任务和获取结果
2. **三阶段异步流水线**：INIT/RUN/EVAL 独立 worker pool，避免最慢阶段拖累整体吞吐
3. **HPC 兼容容器运行时**：Singularity/Apptainer 支持 rootless 执行（无需 Docker daemon）
   - 每个容器分配唯一 loopback IP (127.x.x.x)，避免端口冲突
   - `--fakeroot` 模拟 root 权限，`--network none` 隔离网络
4. **高效工具后端优化**：
   - ptyprocess 直接伪终端替代 tmux（减少终端复用开销）
   - IPython in-process API 连接（消除网络 round-trip）
   - Unix Domain Socket 替代 TCP loopback（减少 IPC 延迟）
5. **Token-in/Token-out**：prompt_ids/response_ids 直接与推理服务器交换，消除 re-tokenization drift
6. **Min-heap LLM 后端负载均衡**：同一 task 的所有调用路由到同一后端以最大化 prefix cache reuse
7. **Phase-aware timeout**：只在活跃阶段计时，排除排队等待
8. **高效 DAPO 实现**：异步补充 + 提前终止 + 跨迭代持久化

### 3.2 Paper 2: Polar (arXiv:2605.24220)

**标题**: Polar: Agentic RL on Any Harness at Scale

**核心创新**：
1. **代理边界即 Rollout 边界** (Model API proxy as rollout boundary)
   - 在 LLM API 端点而非 Agent 框架接口处进行观测
   - 不需要理解 harness 内部如何规划、管理工具、决定何时停止
   - 只需保留 API 兼容性并记录足够信息重建训练样本

2. **Prefix Merging 轨迹重建** (Token-Faithful Prefix Merging)
   - 将多个独立 completion 重建为 append-only 对话链
   - Chain 分组基于 token-prefix 关系
   - Assistant body 用**原始采样** token IDs (real logprobs)
   - Interstitial tokens (tool results, chat glue) 用 **canonical** tokenization (synthesized logprobs, loss_mask=0)
   - 关键保证：*Every trainable token matches the behavior policy during rollout*

3. **Gateway 级异步 Staging**
   - INIT/READY/RUNNING/POST-RUN 四阶段隔离 worker pool
   - Ready buffer 解耦 runtime 准备和 Agent 执行
   - Evaluator prewarm: 评估 runtime 在 Agent 运行期间就开始准备

4. **离线数据生成**：可复用为分布式离线数据生成服务
   - 案例：SWE-Gym SFT trajectories, 1638 attempts -> 504 accepted (30.8%)
   - 平均每个 session 104 messages, 51 assistant turns

### 3.3 实验结果详解

**SWE-Bench Verified 上的 GRPO 训练结果 (Qwen3.5-4B base)**:

| Harness | Base | Polar RL | Gain | 备注 |
|---------|------|----------|------|------|
| Codex | 3.8% | 26.4% | +22.6 | Codex 对 Qwen 是不熟悉的 harness, 提升最大 |
| Claude Code | 29.8% | 34.6% | +4.8 | 中等基础，中等提升 |
| Qwen Code | 34.6% | 35.2% | +0.6 | 已经是 Qwen 原生 harness, 提升有限但仍有 |
| Pi | 34.2% | 40.4% | +6.2 | 较强原生 prior, 清晰提升 |

**关键发现**：
- 对不熟悉 harness (Codex)，RL 提升最大 -- harness-native RL 的价值
- 对已对齐 harness (Qwen Code)，仍能小幅提升，说明 RL 优化了推理能力而非仅是格式适配
- Harness-native RL 的价值：优化的是模型在**评估时实际使用的行为路径**上的表现

**Prefix Merging vs Per-Request Ablation**:
- 同样 3 个 training step: per-request 189.5 min vs prefix merging 35.2 min (**5.39x** 加速)
- Per-request 产生 1185 个 updates，prefix merging 只产生 218 个 updates
- Per-request + outcome reward broadcasting 观察到显著的 **reward hacking**
- Prefix merging 平均 rollout GPU utilization **87.7%**, per-request 仅 **20.4%**


---

## 4. RL for LLM Agent 核心概念

### 4.1 Agent RL 与传统 RL 的区别

| 维度 | 传统 RL (Atari/MuJoCo) | LLM Agent RL |
|------|----------------------|--------------|
| 动作空间 | 有限离散/连续 | 自然语言 token 序列 (vocab 100K+) |
| 环境交互 | step() 返回 (obs, reward, done) | 多轮对话 + 工具调用（异构） |
| 轨迹长度 | 几十到几百步 | 几十到几百 turn, 数万 tokens |
| 奖励信号 | dense 或 shaped | sparse (只在 episode 结束) |
| 环境复杂度 | 模拟器（确定性） | 真实代码库、浏览器、OS (非确定性) |
| 推理开销 | 模型前向传播快 | LLM 推理昂贵, GPU-bound |
| Context 管理 | 状态固定 | context 需要压缩/注入/替换 |

### 4.2 GRPO (Group Relative Policy Optimization)

**核心思想**：不需要 critic/value network，直接用组内相对优势进行策略优化。

**算法流程**：
1. 对每个 prompt x，采样 G 个 response
2. 计算每个 response 的 reward r_i
3. 计算组内归一化优势: A_i = (r_i - mean(r)) / std(r)
4. 用 PPO-clip 风格的 loss 更新 policy: L = E[min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)]

**为什么适合 Agent RL**：
- 不需要额外的 value model（节省显存）
- 可以处理 sparse reward
- 天然支持异步 rollout
- 实现简单，适合快速迭代

### 4.3 DAPO (Dynamic Sampling Policy Optimization)

**对 GRPO 的关键改进**：
1. **过滤 Zero-Variance Prompts**：全对或全错的 prompt 不提供梯度信号，直接丢弃
2. **异步补充机制**：job queue 为空就补充新 prompt，保持最大吞吐
3. **提前终止**：收集够 Informative Prompts 后终止剩余 rollout
4. **跨迭代持久化**：未完成的 rollout 延续到下一迭代

### 4.4 Trajectory Reconstruction 策略

**问题**：Agent session 包含多个独立 LLM completion，如何转换为可训练轨迹？

**策略 1: Per-Request (保守基线)** - 每个 completion 变成独立 trace。碎片化严重，且 outcome reward broadcasting 会导致 reward hacking。

**策略 2: Prefix Merging (推荐)** - 将 append-only 的 completion 链合并为单个长 trace。

分组算法: 检查 C_i.prompt_ids 是否是已有 chain 最后 prompt 的 token-prefix。
Finalize: 第一个 completion 的 prompt_ids 作为 prompt; response_ids (原始采样 token) 作为 trainable; interstitial tokens 标记为 non-trainable (loss_mask=0)。

正确性不变量: *Every trainable token matches the behavior policy during rollout, and any non-generated tokens are masked out.*

### 4.5 Retokenization Drift

**问题**: re-tokenization 可能产生不同 token 序列 (e.g., [fish,ing]->[fishing])，导致 off-policy 偏移。

**Polar 方案**: token IDs 作为 canonical representation，直接从推理服务器获取，不经过 decode-encode 循环。

### 4.6 Reward Post-Processing

Leave-one-trajectory-out baseline: 对每个 trajectory，用同 prompt group 中其他 trajectory 的平均 reward 作为 baseline。丢弃 FAILED/ABORTED 状态的 trajectory。支持 GRPO 的 std normalization。

---

## 5. 代码实现关键模块解析

### 5.1 Gateway Proxy (src/polar/gateway/server.py)

核心流程: detect -> transform -> capture -> forward

- detect(): 根据 request path/headers/body 识别 API 类型 (Anthropic/OpenAI/Google)
- transformer.transform_request(): 统一转换为 OpenAI Chat Completions 格式
- inference.completion(): 转发到 SGLang/vLLM
- storage.save_message(): 保存 request/response/token IDs/logprobs
- transformer.transform_response(): 转换回原始 API 格式

**流式处理巧思**: Gateway 获取非流式上游响应，再转换为合成的 provider-shaped 流式事件，既保证 token-level 数据完整性，又兼容需要 SSE 的 harness。

### 5.2 Node Scheduler (src/polar/rollout/balancer.py)

**Stage-aware pressure-based scheduling**，评分函数按优先级:
1. run_pressure (GPU-bound 阶段最关键)
2. postrun_pressure
3. init_pressure
4. -ready_gap (ready buffer 空位越多越优先)
5. total_pressure (tie-breaker)

### 5.3 Prefix Merging Builder (src/polar/trajectory/builder/prefix_merging.py)

**Chain 分组** (_find_extendable_chain): 基于 token-prefix 关系，只比较 server-side tokenization 的 prompt (稳定)。

**Interstitial 分割** (_slice_interstitial): 从 canonical tail 找到第一个 EOT token，之前是 assistant body，之后是 interstitial。如果 prev_raw_response 已经以 EOT 结尾，跳过避免重复。

### 5.4 Slime Bridge Adapter (src/slime_bridge/adapter.py)

SessionResult -> Slime Sample 转换。每个 trace 变成一个独立 Sample，共享 group_id。FAILED/ABORTED 状态的 trace 全 mask 掉。无可用 trace 时 emit dummy placeholder。

### 5.5 SWE-Bench Evaluator (src/polar/trajectory/evaluator/swebench_harness.py)

从 Agent 提取 patch -> 在 fresh eval runtime 上 apply -> 运行 eval.sh -> 用 get_eval_report 评分。resolved=True -> outcome_reward=1.0。


---

## 6. 面试深挖点与考察问题

### 6.1 LLM Agent 后训练基础概念 (高频考点)

**Q1: 什么是 Agent RL? 与传统 RL 有什么区别?**

Agent RL 是将 LLM Agent 的多轮交互过程视为 RL 环境，通过 reward signal 优化 Agent 的长期行
为。与传统 RL 区别: (1) 动作空间是自然语言 token 序列而非离散/连续向量; (2) 环境交互是
异构的多轮对话+工具调用; (3) 轨迹极长 (数万 tokens); (4) 奖励通常 sparse (只在
session 结束时); (5) 推理开销巨大 (每次 forward 都是 LLM 推理)。

**Q2: GRPO 的原理是什么? 为什么适合 LLM Agent RL?**

GRPO 不需要 critic/value network，直接用组内相对优势优化策略。对每个 prompt 采样 G
个 response，计算归一化优势 A_i = (r_i - mean) / std，用 PPO-clip loss 更新。适合
Agent RL 因为: (1) 省去 value model 显存; (2) 处理 sparse reward; (3) 天然支持异步
rollout; (4) 实现简单。

**Q3: 什么是 DAPO? 它解决了 GRPO 的什么问题?**

DAPO 解决了 GRPO 中 Zero-Variance Prompts 的问题。如果一个 prompt 的所有 rollout 结果
相同 (全对/全错)，不提供梯度信号。DAPO 过滤这类 prompt，并通过异步补充、提前终止、
跨迭代持久化提升效率。

**Q4: 什么是 on-policy vs off-policy RL? GRPO 是哪种?**

On-policy: 训练用的数据必须来自当前策略。Off-policy: 可以用旧策略的数据。GRPO 是
on-policy，因为每个 training step 都需要用当前 policy 重新 rollout。这导致 Agent RL 的
主要瓶颈: rollout 生成非常慢 (每次都要完整执行 Agent 交互)。

**Q5: 什么是 reward hacking? 如何防止?**

Reward hacking 是 Agent 找到获取高 reward 的捷径但并非真正解决问题。在 Polar 中:
per-request + outcome reward broadcasting 会导致请求级 trace 获得 session 级 credit，
信用分配噪声大导致 hacking。Prefix merging 通过在 trace 级做 credit assignment 来缓解。

### 6.2 Polar 项目核心设计 (深挖考点)

**Q6: 为什么选择 LLM API 作为 Rollout 边界? 这个设计有什么优势和局限?**

优势: (1) 所有 LLM Agent 都需要与模型对话，这是 universal interface; (2) Agent 不需要任何
改造 (黑盒); (3) 支持闭源/复杂 harness; (4) 自然捕获 token-level 数据。

局限: (1) 无法获取 harness 内部状态 (如 context window 管理策略); (2) 无法做
harness-aware 的 process reward; (3) 流式转非流式可能增加延迟; (4) 需要 Agent 支持
配置 base URL。

**Q7: Prefix Merging 的核心算法是什么? 如何保证 token-faithful?**

算法: (1) Chain 分组 -- 基于 token-prefix 关系 (prompt_ids[:len(tip)] == tip); (2) 
Finalize -- assistant body 用原始采样 token IDs (real logprobs)，interstitial 用
canonical tokenization (loss_mask=0)。

Token-faithful 保证: trainable token 全部来自推理服务器的原始采样，不经过
decode-encode 循环。Non-generated tokens (tool results, chat glue) 用
loss_mask=0 排除训练。

**Q8: 为什么需要 Prefix Merging? Per-Request 有什么问题?**

Per-Request 问题: (1) 碎片化严重 (一个 SWE 问题产生数百个 trace); (2) 增加 trainer
负担 (5.39x wall-clock 差距); (3) outcome reward broadcasting 到请求级 trace 导致
reward hacking; (4) GPU utilization 仅 20.4% vs 87.7%。

Prefix Merging 优势: 减少 trainer 需要处理的 sample 数量，保持 token-faithful，
避免 reward hacking。

**Q9: 系统中的异步流水线是如何工作的? 为什么需要 4 个阶段?**

4 阶段: INIT (runtime 容器启动) -> READY (buffered, 等待 run slot) -> RUNNING (Agent
执行) -> POST-RUN (trajectory 构建+评估+回调+teardown)。

为什么: 每个阶段资源需求不同 -- INIT 是 I/O-bound, RUNNING 是 GPU-bound, POST-RUN
可能几分钟 (test suite) 到几毫秒 (直接评分)。独立 worker pool 避免最慢阶段拖累整体
吞吐。Ready buffer 让 CPU-heavy 的 runtime 准备在后台进行，不阻塞 GPU-bound 的 Agent
执行。

**Q10: Node Scheduler 的调度策略是什么? 为什么选择这个设计?**

Stage-aware pressure-based scheduling。评分函数按优先级: run_pressure > postrun_pressure 
> init_pressure > ready_gap > total_pressure。

选择原因: RUNNING 阶段是 GPU-bound 的最关键瓶颈。Ready buffer 有空位的节点可以预热
runtime，减少 Agent 等待时间。防止 POST-RUN 积压。

**Q11: Min-heap LLM 后端负载均衡如何工作? 为什么同一 task 路由到同一后端?**

每个 LLM backend 存储在 min-heap 中，每次分配 task 时选择 counter 最低的。Counter
按 task (非 call) 递增，确保同一 task 的所有 LLM 调用路由到同一 backend。

同一 task 路由到同一后端: 最大化 prefix cache reuse (同一 Agent session 的多次 LLM 调用
有大量共享 prefix)，减少 KV cache 重复计算。

**Q12: Token-in/Token-out 是如何实现的? 为什么重要?**

实现: Rollout worker 发送 prompt_ids 到推理服务器，接收 response_ids 和 per-token
logprobs。多轮 rollout 中，prior assistant turns 保留原始 token IDs 直接拼接。

重要性: 避免 re-tokenization drift -- 文本 decode 再 encode 可能产生不同 token 序列
(如 [fish,ing] -> [fishing])，导致 off-policy 偏移。

### 6.3 系统工程与分布式系统 (进阶考点)

**Q13: 如何处理 Agent Rollout 中的失败和超时?**

- Phase-aware timeout: 只在活跃阶段计时，排除排队等待
- 每个阶段有 exception callback，失败时填充 fallback 结果
- Cancellation: POST /cancel 标记为 discarded、cancel async task、close runtime、
  signal completion
- Graceful shutdown: cancel all in-flight, terminate containers, drain pools

**Q14: 系统如何支持 checkpoint swapping?**

RL 训练更新 policy checkpoint 后，旧 LLM 权重无效。Trainer 调用 POST /clear_llm_server
清除所有注册 backend，然后重新注册更新后的 LLM server endpoints。后续 rollout 自动使用
新模型，无需重启 rollout server。

**Q15: 如何理解 Rollout as a Service 与传统 RL 训练的区别?**

传统: rollout 嵌入训练循环 (如 SkyRL-Agent 的 Gymnasium 接口)，切换任务需要改 trainer。
Polar: rollout 作为独立 HTTP 服务，Trainer 只通过 HTTP 提交/获取结果。好处: (1) Trainer
和 Rollout 独立开发部署; (2) 新任务只需实现 handler，不改 training code; (3) Agent 可以
替换不影响训练基础设施; (4) 自然支持异步 RL。

**Q16: Polar 如何与 Slime 训练框架集成?**

Slime Bridge 是连接层: (1) Background worker 提交 Polar tasks; (2) 接收
task-completion callbacks; (3) 将 traces 转换为 Slime Sample 对象 (adapter.py); (4)
应用轨迹感知的 reward 后处理 (reward_post_process.py)。

SessionResult -> Sample 转换: 每个 trace 变成一个 Sample，共享 group_id。Slime 的 loss
reducer 将同一 group 的所有 trace 贡献平均为一个梯度单元。

### 6.4 Reward 设计与 Credit Assignment (高频考点)

**Q17: 什么是 credit assignment problem? 在 Agent RL 中如何解决?**

Agent RL 中，一个 session 可能有几十个 turn，但只在最后得到一个 reward。问题是: 如何
将这个 session-level reward 分配给每个 turn 的 action?

方案: (1) Outcome reward broadcasting -- session reward 广播到每个 trace (per-request
方案，简单但导致 reward hacking); (2) Prefix merging -- 将整个 session 合并为一个
trace，credit assignment 在 trace 级; (3) Process reward model (PRM) -- 训练一个模型
给每个 step 打分 (Polar roadmap 中)。

**Q18: Leave-one-trajectory-out baseline 是如何工作的?**

对每个 trajectory T_i，用同一 prompt group 中其他 trajectory 的平均 reward 作为
baseline: advantage_i = (reward_i - mean(reward_{j!=i})) / scale。这比用全局 mean 更准确，
因为它反映了该 prompt 本身的难度。丢弃 FAILED/ABORTED 状态的 trajectory。

### 6.5 代码细节考察 (代码阅读能力)

**Q19: 为什么 Gateway 获取非流式响应再合成流式事件? 看 server.py 中的 _handle_streaming。**

为了简化 token capture。如果直接代理流式响应，需要在流式 chunks 中拼凑 token IDs
和 logprobs，实现复杂且容易出错。先获取完整非流式响应可以确保 token-level 数据完整
保存，再通过 _response_to_stream_chunk 转换为合成流式事件返回给 Agent (兼容 SSE)。

**Q20: Prefix Merging 中 EOT token 的作用是什么? 看 _slice_interstitial。**

EOT (end-of-turn) token 标记 assistant turn 的结束。在 canonical tail 中，第一个 EOT
之后就是 interstitial (tool results, chat glue)。如果 prev_raw_response 已经以 EOT
结尾 (natural stop)，跳过避免重复; 否则 (truncation) 包含它以关闭 assistant turn。

**Q21: Node Scheduler 的 _node_score 为什么用 tuple 而不是单一标量?**

Tuple 比较实现了多级优先级排序。Python tuple comparison 是字典序的: 先比较
run_pressure (最关键)，相同再比较 postrun_pressure，依此类推。这比加权求和更清晰地
表达了 "X 优先级高于 Y" 的语义。

### 6.6 开放性问题 (考察思维深度)

**Q22: Polar 的 Proxy 方案有什么潜在问题? 你会如何改进?**

问题: (1) 流式转非流式增加延迟; (2) 无法获取 harness 内部状态; (3) 不支持
harness-aware 的 process reward; (4) 单 proxy 可能成为瓶颈。

改进: (1) 直接代理流式响应 + 流式 token capture; (2) 引入 harness adapter 做
optional 内部观测; (3) 集成 PRM 做 per-step reward; (4) 多 proxy 负载均衡。

**Q23: 如何将 Polar 应用于其他 Agent 任务 (如 web browsing, OS interaction)?**

Polar 的 harness-agnostic 设计使其天然支持。只需要: (1) 实现对应的 Runtime (如
VM-based runtime for OS); (2) 实现 Evaluator (如 web task 的 success metric); (3)
配置 Agent 的 LLM base URL 指向 Gateway proxy。核心代码无需修改。

**Q24: Agent RL 的 scaling law 是什么? Polar 如何帮助 scaling?**

Agent RL 的 scaling 受限于: (1) Rollout 生成速度 (GPU-bound); (2) 环境执行延迟
(I/O-bound); (3) 评估延迟。Polar 通过: (1) 异步流水线提升吞吐; (2) 多 Gateway
水平扩展; (3) Prefix merging 减少 trainer 处理量; (4) 独立 scaling 推理和训练。

**Q25: Prefix Merging 在什么情况下会失败或退化?**

退化场景: (1) Harness 重写历史消息 (如 context compaction) -- 前缀关系断裂，无法
合并; (2) 并行 agent/子 agent -- 形成多个独立 chain，退化为 per-request; (3) 非
append-only 的对话模式; (4) EOT token 无法检测 (自定义 chat template)。

这些场景下 prefix merging 退化为 per-request，不会出错但失去合并优势。

**Q26: 对比 PPO 和 GRPO 在 Agent RL 中的表现?**

PPO 需要 value model (额外显存，且 LLM 的 value model 训练不稳定)。GRPO 用组内
归一化替代，更简单且效果相当。在 Agent RL 中 GRPO 更适合因为: (1) 长序列下 value
估计更难; (2) Agent 任务通常只有 sparse reward，value function 的 signal 弱; (3)
不需要额外的 GPU 内存存放 value model。

**Q27: Polar 如何支持 offline RL 或 data filtering?**

Polar 可以复用为分布式离线数据生成服务。固定 checkpoint + harness 扇出到集群，
每个 session 记录到 disk，结果过滤后用于 SFT/RL。案例: SWE-Gym SFT trajectories
(1638 attempts, 504 accepted, 30.8%)。同一部署可复用于 rejection sampling、
verifier training data、preference data 构建。

---

## 7. 自我介绍模板与话术

### 7.1 开场介绍 (30 秒)

> 我是 XX，在读硕士，研究方向是 LLM Agent 的后训练优化。最近深入研究了 NVIDIA 的
> Polar 项目 -- 一个 Rollout as a Service 的 Agent RL 基础设施。这个项目的核心洞察是
> 通过在 LLM API 边界设置代理来将任意 Agent 作为黑盒进行 RL 训练，不需要改造 Agent
> 代码。我对 Agent RL 的系统设计、轨迹重建、reward 设计等方面都有深入理解。

### 7.2 技术深度展示 (1-2 分钟)

> 具体来说，我对以下几个技术点有深入思考:
>
> **第一，Prefix Merging 轨迹重建**。Agent session 通常包含几十个独立 LLM completion，
> 如何将它们合并为可训练轨迹是核心挑战。Polar 的方案是基于 token-prefix 关系将
> append-only 的 completion 链合并，只有原始采样的 token 标记为 trainable，interstitial
> tokens (tool results 等) 标记为 non-trainable。这比 per-request 方案快 5.39 倍，
> 且避免了 reward hacking。
>
> **第二，异步流水线设计**。INIT/READY/RUNNING/POST-RUN 四阶段独立 worker pool，CPU
> 密集的 runtime 准备不阻塞 GPU 密集的 Agent 执行。Ready buffer 实现 runtime 预热。
>
> **第三，Token 保真性**。直接从推理服务器获取 token IDs，避免 re-tokenization drift。
> 这对 RL 训练的正确性至关重要。

### 7.3 可以主动引导面试官提问的方向

- "我对 GRPO/DAPO 的原理和在 Agent RL 中的适配有深入理解"
- "我分析了 Polar 与 SkyRL-Agent、rLLM 等框架的设计差异"
- "我对 reward 设计和 credit assignment 问题有自己的思考"
- "我阅读了项目的所有核心代码，包括 Gateway proxy、Node scheduler、
  Prefix merging builder 等模块"

---

## 8. 延伸知识与行业理解

### 8.1 Agent RL 领域的关键工作

| 工作 | 机构 | 核心贡献 |
|------|------|---------|
| DeepSeek-R1 | DeepSeek | 用 RL (GRPO) 训练 LLM 推理能力，展示了 RL 的 scaling potential |
| OpenAI Codex | OpenAI | SWE 任务的 Agent，展示了多轮工具使用能力 |
| SkyRL-Agent | NovaSky | 高效 Agent RL 框架，Gymnasium-style 接口 |
| rLLM | 清华 | 跨框架 Agent RL，基于 veRL fork |
| SWE-bench | Princeton | SWE 任务评估标准 |
| SWE-Gym | -- | SWE 训练环境，提供 verifier 和 trajectories |
| DAPO | ByteDance | 改进 GRPO 的 RL 算法 |
| Slime (THUDM) | 清华 | Megatron + SGLang 的 RL 训练框架 |

### 8.2 行业趋势理解

1. **Agent RL 是 LLM 后训练的新前沿**: 从 single-step math/code -> multi-turn agentic tasks
2. **Harness-native training 的价值**: 优化模型在评估时实际使用的行为路径上
3. **Infrastructure 是 bottleneck**: Agent rollout 比 text generation 慢 100x+，
   需要专门的基础设施
4. **Token fidelity 至关重要**: re-tokenization drift 在长序列中会累积
5. **异步 RL 是 scaling 的关键**: 同步 batch-by-batch 方案浪费大量 GPU idle time
6. **从 RL 到 self-improvement**: RL 训练后的模型可以生成更好的训练数据 (offline
   data generation -> SFT -> RL 循环)

### 8.3 推荐进一步阅读的材料

1. **论文**: arXiv:2603.18815 (ProRL Agent), arXiv:2605.24220 (Polar)
2. **代码**: src/polar/gateway/server.py (Proxy), src/polar/trajectory/builder/prefix_merging.py (Prefix Merging)
3. **RL 算法**: GRPO (DeepSeek-Math), DAPO (arXiv:2503.14476), PPO
4. **Agent 评估**: SWE-bench, SWE-Gym, Harbor
5. **对比框架**: SkyRL-Agent, rLLM, Agent Lightning, PRIME-RL

---

## 附录: 面试 Checklist

### 面试前必须能流利回答的问题
- [ ] GRPO 的原理和为什么适合 Agent RL
- [ ] Prefix Merging 的算法和正确性保证
- [ ] Polar 的三层架构和各层职责
- [ ] Black-box proxy 的设计动机和优劣
- [ ] Token-faithful 的重要性和实现方式
- [ ] Agent RL 与传统 RL 的区别
- [ ] Reward hacking 的原因和解决方案
- [ ] DAPO 对 GRPO 的改进
- [ ] 异步流水线为什么比同步方案好
- [ ] 如何处理 Rollout 中的失败和超时

### 可以展示深度思考的话题
- [ ] Prefix Merging 的退化场景和改进方向
- [ ] Process Reward Model (PRM) 的价值和挑战
- [ ] Offline RL / Rejection Sampling 在 Agent 任务中的应用
- [ ] Polar 与其他框架的设计 trade-off
- [ ] Agent RL 的 scaling law 和 bottleneck

### 必须熟悉的代码模块
- [ ] src/polar/gateway/server.py (Proxy 核心逻辑)
- [ ] src/polar/trajectory/builder/prefix_merging.py (Prefix Merging 算法)
- [ ] src/polar/rollout/balancer.py (Node Scheduler)
- [ ] src/slime_bridge/adapter.py (SessionResult -> Sample 转换)
- [ ] src/slime_bridge/reward_post_process.py (Reward 后处理)
