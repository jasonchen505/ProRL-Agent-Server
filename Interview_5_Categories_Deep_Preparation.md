# Polar 项目技术面试五类能力深度应对

> 基于 ProRL-Agent-Server (Polar) 项目的深度技术分析，针对五类面试考察能力准备。

---

## 目录

1. [底层原理深度理解](#1-底层原理深度理解)
2. [实验和方案验证能力](#2-实验和方案验证能力)
3. [问题定位能力](#3-问题定位能力)
4. [工程落地能力](#4-工程落地能力)
5. [业务与实际场景的理解](#5-业务与实际场景的理解)

---

## 1. 底层原理深度理解

> 重点不是回答清楚概念，而是讲清楚方法解决什么问题、存在哪些局限性、有哪些改进方法。

### 1.1 Prefix Merging: 为什么这么设计?

**解决的问题**:
Agent session 会产生几十个独立 LLM completion (每个 turn 一次调用)。朴素方案 (per-request)
把每个 completion 变成独立 trace，导致: (1) 碎片化严重 (一个 SWE 问题数百个 trace);
(2) outcome reward 广播到请求级 trace 导致 reward hacking -- 一个 session 中间某步恰好
"对了"但整体失败，也会被标记为正 reward，污染梯度信号。

**为什么用 token-prefix 而不是 message-level 对比来分组 chain**:
Message-level 对比依赖 harness 的消息格式一致性，但不同 turn 的 tool result 格式可能
变化、harness 可能对历史消息做 reformat。Token-prefix 对比的是 server-side tokenization
的结果 -- 两次 tokenize 相同文本一定得到相同 token 序列。这使得分组对 BPE re-tokenization
鲁棒。关键: 我们只比较 prompt 的 token (server-side 生成)，从不比较 sampled response
的 token (那些可能在下一轮 prompt 中被 re-tokenize)。

**局限性**:
1. **Context compaction 断裂**: 很多 harness 会压缩历史 (如 Claude Code 的 auto-compact)，
   压缩后的 prompt 不再是前一个 prompt 的前缀，chain 断裂，退化为 per-request
2. **并行 agent/子 agent**: 形成独立 chain，无法合并
3. **非 append-only 对话模式**: 如果 harness 重写历史消息 (如删除中间步骤)，
   前缀关系不成立
4. **EOT token 检测**: 依赖 auto-detection，自定义 chat template 可能失败

**改进方向**:
1. **Harness-aware merging**: 如果能获取 harness 的 context 管理策略，可以更智能地
   识别 compaction 边界
2. **Hierarchical merging**: 先按 sub-agent 分组，再在每组内做 prefix merging
3. **Process Reward Model (PRM)**: 为每个 turn 提供独立 reward signal，避免
   credit assignment 问题

### 1.2 Black-box Proxy: 为什么选择 LLM API 作为观测边界?

**解决的问题**:
真实 Agent harness (Claude Code, Codex, OpenHands) 往往是闭源二进制或复杂软件系统，
无法改造为 RL 环境接口。传统方案 (如 SkyRL-Agent) 需要 harness 实现 Gymnasium-style
接口，每加一个新 harness 就需要一套集成代码。

**设计洞察**:
所有 LLM Agent 都必须与模型对话 -- LLM API 是 universal interface。在 API 边界设置
proxy，不需要理解 harness 内部逻辑就能捕获训练所需的一切: prompt, response,
token IDs, logprobs。

**局限性**:
1. **无法获取 harness 内部状态**: 不知道 harness 如何管理 context window、何时
   压缩历史、如何选择工具 -- 这些信息对 process reward 有价值
2. **流式转非流式延迟**: 为了保证 token capture 完整性，gateway 先获取完整非流式
   再合成流式事件，首次 token 延迟增加
3. **单 proxy 瓶颈**: 高并发下 proxy 可能成为瓶颈 (虽然实际中 LLM 推理本身是
   更大的瓶颈)
4. **Harness 需要支持配置 base URL**: 闭源 harness 如果不暴露配置就无法接入

**改进方向**:
1. **直接代理流式响应 + 流式 capture**: 避免非流式延迟增加，但实现复杂
2. **Optional harness adapter**: 可选的内部观测层，用于获取 context 管理信息
3. **多 proxy 负载均衡**: 大规模部署时分散 proxy 压力

### 1.3 异步流水线: 为什么需要 4 个阶段?

**解决的问题**:
一个 rollout 包含三个资源需求完全不同的阶段: INIT (I/O-bound, 容器启动),
RUNNING (GPU-bound, LLM 推理), POST-RUN (CPU-bound + I/O, 轨迹构建 + 测试套件
执行)。如果用单一 worker 顺序执行，大部分时间在等待最慢的阶段。

**设计选择**:
独立 worker pool 让三个阶段可以并行处理不同 job。Ready buffer 是关键 -- 它
解耦了 INIT 和 RUNNING: INIT 完成的 session 进入 buffer 等待 run slot，这样
CPU-heavy 的 runtime 准备可以在 Agent 执行的同时在后台进行。Evaluator prewarm
进一步优化: 在 Agent 运行期间就开始准备评估用的 runtime。

**局限性**:
1. **内存开销**: Ready buffer 中的 runtime 占用内存，buffer 过大可能 OOM
2. **阶段粒度固定**: 不能根据任务特性动态调整阶段数量
3. **POST-RUN 内部仍有串行**: trajectory 构建和评估在同一个 worker 内串行执行

**改进方向**:
1. **动态 worker pool sizing**: 根据当前负载自动调整各阶段 worker 数量
2. **Evaluator 独立 pool**: 将评估从 POST-RUN 中独立出来，评估可以并行
3. **Runtime pool 复用**: 预启动的 runtime 可以跨 session 复用 (类似连接池)

### 1.4 Token-in/Token-out: 为什么重要?

**解决的问题**:
RL 训练需要精确知道 behavior policy 在每个 token 位置的概率分布。如果用文本
传输轨迹，decode->re-encode 可能产生不同的 token 序列 (re-tokenization drift)，
例如 [fish, ing] -> [fishing]。这导致训练时的 policy ratio 计算错误。

**实现细节**:
Gateway proxy 在每次 LLM 调用时记录 prompt_token_ids, response_token_ids,
per-token logprobs。多轮 rollout 中，prior assistant turns 保留原始 token IDs
直接拼接到 input buffer，只有新消息 (如环境观测) 才做 fresh tokenization。

**局限性**:
1. **Token ID 与特定 tokenizer 绑定**: 如果训练和推理用不同 tokenizer (理论上
   不应该发生，但实际中模型更新时可能出现)，token IDs 无意义
2. **增加数据传输量**: token IDs 比文本更长，增加网络传输和存储开销

### 1.5 GRPO vs PPO: 为什么选 GRPO?

**GRPO 解决的问题**:
LLM RL 训练中，PPO 需要一个与 policy 同等规模的 value model。对于 7B+ 的 LLM，
这意味着双倍显存开销。而且 LLM 的 value function 训练本身就不稳定 (长序列、
sparse reward)。

**GRPO 的巧妙之处**:
用同一 prompt group 内多个 rollout 的 reward 统计量 (mean, std) 替代 value function。
这避免了 value model 的显存和训练开销，同时保留了相对排序的梯度信号。

**局限性**:
1. **样本效率低**: 每个 prompt 需要多个 rollout (通常 4-16 个)，计算开销大
2. **Baseline 估计粗糙**: 组内 mean 是无偏但高方差的 baseline 估计
3. **Sparse reward 下退化**: 如果所有 rollout 都对或都错 (zero-variance)，
   不产生梯度信号 -- DAPO 通过过滤解决了部分问题
4. **不适合需要 dense reward 的任务**: 代码生成等任务可能需要 per-step reward

**改进方向**:
1. **GAE (Generalized Advantage Estimation)**: 结合 value function 做更精确的
   advantage 估计 (但需要 value model)
2. **Process Reward Model**: 为每个推理步骤提供 reward signal
3. **Rejection Sampling + SFT**: 先用 rejection sampling 收集正样本，再 SFT
   (Polar 论文中的 offline data generation 案例)

---

## 2. 实验和方案验证能力

> 面试官不仅关注做了什么，更关注怎么证明它有效。追问实验细节能看出是否有真正深入理解。

### 2.1 Prefix Merging vs Per-Request 的消融实验

**实验设置**:
- 同一模型 (Qwen3.5-4B)、同一硬件、同一拓扑、同一任务 (SWE-Gym GRPO)
- 只改变轨迹重建策略: per_request vs prefix_merging
- 测量 3 个 training step 的 wall-clock time 和 GPU utilization

**核心结果**:

| 指标 | Per-Request | Prefix Merging | 倍数 |
|------|-------------|----------------|------|
| Trainer updates | 1,185 | 218 | 5.43x |
| Wall-clock time | 189.5 min | 35.2 min | 5.39x |
| Rollout GPU util | 20.4% | 87.7% | 4.30x |

**为什么 per-request GPU 利用率低**:
Per-request 产生大量 trainer updates，每个 update 都需要 trainer 做 forward/backward。
Trainer 处理这些 update 的时间里，rollout 端在等待 trainer 准备好下一个 batch。
这造成了 rollout GPU 的大量 idle time。Prefix merging 减少了 5.43x 的 trainer
updates，rollout 几乎不需要等待。

**追问: per-request + outcome reward broadcasting 为什么会导致 reward hacking?**

一个 SWE session 可能有 50+ 个 request-level traces。如果 session 最终成功
(outcome_reward=1)，所有 50+ 个 traces 都被标记 reward=1，包括那些中间步骤的
错误探索。这给模型错误的 credit assignment 信号 -- "即使中间步骤做了错误的事，
只要最后成功了就是好的"。

**追问: 你如何验证 prefix merging 的 token-faithfulness?**

`tests/trajectory/test_engine_trajectory_equivalence.py` 测试了 SGLang 和 vLLM
两个引擎在 normalize 后产生 byte-identical 的 Trace 对象。这验证了:
(1) 不同引擎的 token IDs 一致; (2) prefix merging 的 interstitial 处理一致;
(3) loss_mask 对齐正确。

### 2.2 Harness-Native RL 的验证

**实验设计**:
同一 base model (Qwen3.5-4B) 分别在 4 个不同 harness (Codex, Claude Code,
Qwen Code, Pi) 上做 GRPO 训练，在 SWE-Bench Verified 上评估。

**关键发现与解释**:

| Harness | Base | RL | Gain | 解释 |
|---------|------|----|------|------|
| Codex | 3.8% | 26.4% | +22.6 | Codex 的 action protocol/patch style 对 Qwen 完全陌生 |
| Claude Code | 29.8% | 34.6% | +4.8 | 中等熟悉度 |
| Qwen Code | 34.6% | 35.2% | +0.6 | 已经是 Qwen 原生 harness |
| Pi | 34.2% | 40.4% | +6.2 | 较强原生 prior |

**追问: 为什么 Codex 提升最大?**

Codex 对 Qwen 是完全不熟悉的工具格式 -- 它使用特定的 function calling schema、
特定的 patch 生成方式、特定的 context 管理策略。Qwen 没有在 Codex 格式上做过
训练，所以 base model 几乎无法正确使用 Codex harness (3.8%)。Polar 的 RL 训练
直接优化了模型在 Codex 执行路径上的行为 -- 不是教模型 "如何写代码"，而是教它
"如何使用 Codex 这个工具来写代码"。

**追问: Qwen Code 只提升了 0.6%，是否说明 RL 没用?**

不是。Qwen Code 已经是 Qwen 的原生 harness，base model 已经很好地适配了
(34.6%)。0.6% 的提升说明 RL 优化了推理能力本身 (更善于解决问题) 而不仅是
格式适配。如果只优化格式，base model 已经够好了，不会有提升。

### 2.3 离线数据生成的验证

**实验设置**:
- 模型: Qwen3.5-122B-A10B (TP=8, max_model_len=32768)
- Harness: pi-coding-agent v0.67.68
- 数据: SWE-Gym 1,638 instances across 7 repos
- 配置: max_concurrent=5-8, timeout=3600s

**结果分析**:

| Repo | Attempts | Accepted | Rate |
|------|----------|----------|------|
| getmoto/moto | 343 | 184 | 53.6% |
| python/mypy | 257 | 101 | 39.3% |
| pandas-dev/pandas | 477 | 98 | 19.7% |
| Total | 1,638 | 504 | 30.8% |

**追问: 为什么不同 repo 的 acceptance rate 差异这么大?**

Bug-fix heavy 的 repo (如 moto) 问题通常更简单、测试更明确，agent 更容易生成
正确的 patch。而 data-frame/dataflow 类的 repo (如 pandas, dask) 测试套件更长、
问题更复杂，需要更深的理解和更长的执行链。

**追问: 如何提高 acceptance rate?**

1. **Rejection sampling**: 每个 prompt 生成多个 rollout，只保留成功的
2. **更强的 teacher model**: 用更大的模型 (如 122B vs 4B) 生成数据
3. **更长的 timeout**: 当前 3600s 对复杂问题可能不够
4. **更好的 harness**: 优化 agent 的工具使用策略

---

## 3. 问题定位能力

> 模型上线后能力突然下降、系统突然缓慢、实验结果与预期不一致，怎么排查?

### 3.1 场景: Agent Rollout 突然大面积超时

**排查思路**:

**Step 1: 区分是 Gateway 还是 Inference 的问题**

查看 `/health` endpoint 的 metrics:
```json
{
  "metrics": {"init_inflight": 0, "run_inflight": 8, "postrun_inflight": 0},
  "inference": {"status": "ok"},
  "available_run": 0
}
```

如果 `run_inflight` = `max_run_workers` 且 `available_run` = 0，说明 RUN 阶段
积压。进一步查看 inference server 的状态 -- 如果 inference 响应慢 (如模型加载
问题、GPU OOM)，每个 session 的 RUN 时间变长，导致后续 session 排队。

**Step 2: 检查是否是权重更新导致的暂停**

Polar 的 `/admin/inference/pause` 会在权重更新期间暂停所有推理。如果 pause
后没有正确 resume，所有新的推理请求会一直等待。检查日志中是否有 "Paused
inference generation proxy" 但没有对应的 "Resumed"。

**Step 3: 检查是否是 Runtime 容器问题**

INIT 阶段的超时可能是: (1) 容器镜像损坏; (2) 磁盘空间不足导致容器启动失败;
(3) 网络问题导致镜像拉取超时。检查 INIT 阶段的 inflight 和 queue_depth。

**Step 4: 检查是否是具体 Harness 问题**

如果只有特定 harness 超时 (如 Claude Code 但 Codex 正常)，可能是:
(1) Harness 本身 bug 导致死循环; (2) Harness 的 LLM 调用格式变化导致
proxy 解析失败; (3) Harness 的工具调用卡住 (如 bash 命令不返回)。

### 3.2 场景: RL 训练 reward 不上升

**排查思路**:

**Step 1: 检查 rollout 数据质量**

查看 `save_dir/task_*/` 下的 session 结果:
- 如果大部分 session 是 ERROR/TIMEOUT，说明 rollout 本身有问题
- 如果 session 是 COMPLETED 但 reward 全是 0，说明 evaluator 有问题
- 如果 reward 有正有负但均值不升，可能是 lr 太小或 prompt 太难

**Step 2: 检查 Zero-Variance Prompts**

DAPO 会过滤 zero-variance prompts (全对/全错)。如果过滤率太高 (>80%)，
说明大部分 prompt 对训练没有贡献。可能原因: (1) prompt 太简单 (全对);
(2) prompt 太难 (全错); (3) rollout 数量太少 (n=1 时没有 variance)。

**Step 3: 检查 Token-Faithfulness**

验证 trajectory 中的 token IDs 是否与推理服务器返回的一致。如果存在
re-tokenization drift，policy ratio 计算会出错，梯度信号是错误的。

**Step 4: 检查 Reward Hacking**

如果 reward 快速上升但 SWE-bench 评估分数不升，可能是 reward hacking:
(1) 模型学会了获取 reward 的捷径而非真正解决问题;
(2) per-request + outcome broadcasting 导致错误的 credit assignment。

### 3.3 场景: Gateway Proxy 响应缓慢

**排查思路**:

**Step 1: 确认瓶颈在哪**

Proxy 的链路: Agent -> Gateway -> Inference Server -> Gateway -> Agent。
在 Gateway 日志中看每步的耗时:
- `detect` + `transform_request`: 通常 <1ms
- `inference.completion`: 取决于模型大小和并发量
- `storage.save_message`: 通常 <1ms (内存操作)
- `transform_response`: 通常 <1ms

如果 `inference.completion` 慢，瓶颈在推理服务器。如果其他步骤慢，
检查 Gateway 自身的资源使用。

**Step 2: 检查内存使用**

`SessionStore` 在内存中保存所有 completion records。高并发下可能内存
增长过快。检查 `save_message` 的 `deepcopy` 是否有内存泄漏。

**Step 3: 检查 CompletionWriter 的队列**

如果磁盘写入跟不上推理速度，CompletionWriter 的队列可能满。
队满时会丢弃写入 (drop-tail)，日志中会看到 "Dropped N completions"。
这不影响训练 (内存中数据是权威的)，但会影响离线分析。

---

## 4. 工程落地能力

> 理论可行的方案实际工程落地中不可行，关键在理论结合实际。

### 4.1 算法部署: 如何保证系统稳定?

**问题**: RL 训练中需要频繁更新模型权重。权重更新期间如何保证正在进行的
rollout 不受影响?

**Polar 的方案**:

1. **Pause-Update-Resume 协议**:
   ```
   Trainer -> POST /admin/inference/pause
     Gateway 设置 _generation_paused = True
     等待所有 inflight 推理请求完成 (最多 300s)
   Trainer -> 更新模型权重
   Trainer -> POST /admin/inference/resume
     Gateway 设置 _generation_paused = False
     通知所有等待的请求
   ```

2. **Checkpoint Swapping**:
   ```
   Trainer -> POST /clear_llm_server  (清除所有注册的 backend)
   Trainer -> 重新加载权重到推理服务器
   Trainer -> POST /add_llm_server    (注册新的 backend endpoint)
   ```
   后续 rollout 自动使用新模型，无需重启 rollout server。

**工程权衡**:
- Pause 期间新请求会阻塞 (在 `_acquire_generation_slot` 中等待)
- 如果推理服务器卡死，pause 可能超时 (300s)，此时需要人工干预
- 权重更新期间的 GPU 利用率为 0 -- 这是 on-policy RL 的固有开销

### 4.2 数据回滚与监控

**Polar 的数据持久化策略**:

1. **内存为主 + best-effort 磁盘**:
   - `SessionStore` 在内存中保存所有 completion records (权威数据源)
   - `CompletionWriter` 异步写入磁盘 (队满丢弃，不阻塞推理)
   - 回调成功后清除 registry 中的 trajectory payload 以释放内存

2. **FsIndex 恢复**:
   - 如果 rollout server 重启，`FsIndex` 扫描 `save_dir` 下的 JSON 文件恢复
     task 级别的摘要信息
   - 但 session 级别的 completion records 丢失 (内存数据)

3. **Dashboard 监控**:
   - 实时查看 task/session 状态
   - SSE 事件流推送状态变化
   - GPU 利用率监控 (`scripts/monitor_wandb_gpu.py`)

**工程权衡**:
- 磁盘持久化是 best-effort: 推理热路径不阻塞，但极端情况下可能丢失数据
- Session 重试: 失败的 session 不自动重试 (RL 训练可以容忍部分失败)
- Callback 失败: 不阻塞，trainer 有 poll 作为 safety net

### 4.3 配置验证与 Fail-Fast

**Polar 的 Pydantic strict 模式**:
```python
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
```

- `extra="forbid"`: 未知字段立即报错 (fail-fast)
- `frozen=True`: 配置不可变 (防止运行时修改)
- URL 格式验证、端口范围检查、worker 数量正数检查

**工程价值**: 配置错误在启动时就暴露，而不是运行时出现奇怪的行为。

### 4.4 容器化与 HPC 兼容

**问题**: HPC 集群禁止 Docker daemon，所有进程必须无 root 权限运行。

**Polar 的方案**: Apptainer (原 Singularity) 支持:
- `--fakeroot`: 模拟 root 权限安装软件包
- `--network none`: 网络隔离
- 每个容器分配唯一 loopback IP (127.x.x.x)，避免端口冲突
- .sif 单文件格式，适合 Slurm 共享文件系统

**工程挑战**:
- Apptainer 不如 Docker 生态丰富
- 镜像构建需要 Jinja2 模板 + 三种缓存策略 (Scratch/Versioned/Lock)
- 端口管理需要线程安全的分配器

### 4.5 流式处理的工程决策

**问题**: Agent harness 期望 SSE 流式响应，但 token capture 需要完整 response。

**Polar 的方案**: 合成 streaming
1. Gateway 发送非流式请求到推理服务器
2. 获取完整 response 并保存 token-level 数据
3. 将 response 转换为 synthetic delta chunk
4. 以 SSE 格式返回给 Agent

**工程权衡**:
- 优势: Token capture 简单可靠，数据完整
- 劣势: 首次 token 延迟增加 (用户看到的 "流式" 实际要等完整推理)
- 这是**训练优先**的设计决策 -- 训练数据的完整性比用户体验更重要

---

## 5. 业务与实际场景的理解

> 一个项目真正需要产生的是有用的场景价值和业务价值。

### 5.1 Polar 适合什么场景?

**最适合的场景**:
1. **SWE (软件工程) Agent 训练**: 代码修复、代码生成 -- 有明确的 verifiable
   reward (测试通过/失败)，rollout 时间长 (分钟级)，需要真实代码库环境
2. **多轮工具使用训练**: 计算器、搜索引擎、代码执行 -- 需要多轮交互的
   agent 任务
3. **Harness-specific RL**: 优化模型在特定 harness (如 Claude Code, Codex)
   上的表现 -- 模型需要学会使用特定工具格式

**不太适合的场景**:
1. **单轮生成任务**: 如单轮问答、翻译 -- 不需要多轮 rollout，用传统
   RL (如 PPO on single-turn) 更高效
2. **需要 dense reward 的任务**: 如数学推理步骤评分 -- 需要 PRM，Polar
   目前只支持 session/trace 级 reward
3. **实时交互场景**: Polar 是 batch 训练系统，不适合在线学习

### 5.2 用户更关心什么?

**Agent RL 研究者关心**:
1. **能否快速迭代**: 添加新 harness 的成本有多低?
2. **Token fidelity**: 训练数据是否正确?
3. **GPU 利用率**: 是否浪费了推理 GPU?
4. **调试便利性**: 能否方便地查看 rollout 轨迹和 reward?

**工程团队关心**:
1. **部署复杂度**: 需要多少组件? 配置是否简单?
2. **稳定性**: 单点故障? 容错机制?
3. **可扩展性**: 能否水平扩展到数百个 GPU?
4. **与现有基础设施的集成**: 是否支持 Slurm/K8s?

**业务方关心**:
1. **训练效果**: RL 后模型能力提升多少?
2. **成本**: 需要多少 GPU hours?
3. **可复用性**: 同一基础设施能否支持不同任务?

### 5.3 上线成本有多高?

**部署组件**:
1. Rollout Server: 1 个 CPU 节点 (轻量)
2. Gateway Nodes: 每个推理节点 1 个 (CPU 密集但不需 GPU)
3. Inference Server: 每个 GPU 节点 1 个 (GPU 密集)
4. Trainer: 1 个 GPU 节点 (训练)
5. Dashboard: 可选，1 个 CPU 节点

**最小部署**: 1 台 8xGPU 机器即可 (rollout server + gateway + inference + trainer
共存)

**GPU Hours 估算** (以 SWE-Gym 为例):
- 293 tasks x 4 samples x ~5min/rollout = ~97 GPU-hours per epoch
- 训练通常 50-100 epochs = ~5000-10000 GPU-hours
- 离线数据生成: 1638 tasks x ~3min/task = ~82 GPU-hours

### 5.4 如果资源有限，首先优化什么?

**优先级排序**:

**P0 (必须有)**:
1. **推理服务器优化**: LLM 推理是最大的 GPU 瓶颈。优化 KV cache、
   prefix cache、batch scheduling
2. **超时和失败处理**: Agent rollout 失败率高，必须有完善的容错机制
3. **Token fidelity**: 训练数据的正确性是底线

**P1 (应该有)**:
1. **Prefix Merging**: 5.39x 的训练加速，ROI 极高
2. **异步流水线**: 避免 GPU idle time
3. **Node Scheduler**: 多节点部署时的负载均衡

**P2 (锦上添花)**:
1. **Dashboard**: 方便调试但非必需
2. **离线数据生成**: 可复用同一基础设施但不是核心功能
3. **更多 harness 预设**: 用户可以用 shell harness 通用方案

### 5.5 与现有方案的对比优势

**vs SkyRL-Agent**:
- SkyRL 需要 harness 实现 Gymnasium 接口，Polar 不需要
- SkyRL 的 rollout 嵌入训练循环，切换任务/Agent 需要改 trainer
- Polar 的 rollout 作为独立服务，新任务只需在 rollout 端实现

**vs rLLM**:
- rLLM 基于 veRL fork，与 veRL 强绑定
- Polar 是 trainer-agnostic 的，可以接入任何训练框架

**vs 纯 SFT**:
- RL 可以优化 long-horizon 行为，SFT 只能模仿 demonstrations
- RL 可以从失败中学习 (negative reward)，SFT 只能从成功中学习
- 但 RL 的 rollout 成本远高于 SFT 的数据收集

---

## 附录: 面试回答模板

### 当被问到 "你做了什么?"

> 我深入研究了 NVIDIA NeMo 的 Polar 项目 -- 一个 Rollout as a Service 的 Agent RL
> 基础设施。这个项目的核心创新是在 LLM API 边界设置代理来将任意 Agent 作为黑盒进行
> RL 训练。我对 Prefix Merging 轨迹重建算法、异步流水线设计、token-faithful 保证等
> 技术细节有深入理解，并分析了项目中 10+ 个核心模块的实现。

### 当被问到 "遇到了什么问题?"

> 项目本身是开源框架，但我通过代码分析发现了几个工程挑战:
> 1. 流式处理的 token capture 完整性问题 -- 通过合成 streaming 解决
> 2. 多轮 rollout 中的 re-tokenization drift -- 通过 token-in/token-out 解决
> 3. 请求级 reward broadcasting 导致的 reward hacking -- 通过 prefix merging 解决
> 4. HPC 环境下无 Docker daemon 的容器化需求 -- 通过 Apptainer 解决

### 当被问到 "有什么改进方向?"

> 我认为有三个改进方向:
> 1. **Process Reward Model**: 当前只支持 session/trace 级 reward，可以引入 PRM
>    做 per-step reward，改善 credit assignment
> 2. **直接代理流式响应**: 当前的合成 streaming 增加首次 token 延迟，可以实现
>    流式 token capture 来优化
> 3. **Runtime Pool 复用**: 当前每个 session 启动新容器，可以预启动 runtime 池
>    来减少 INIT 阶段延迟
