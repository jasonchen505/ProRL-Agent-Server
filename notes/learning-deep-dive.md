# ProRL-Agent-Server 深入学习文档

> 上游：`NVIDIA-NeMo/ProRL-Agent-Server`（"Agentic RL on Any Harness at Scale"，Apache-2.0）
> Fork：`jasonchen505/ProRL-Agent-Server`
> 学习日期：2026-10-14（以当天 GitHub API 读取为准）
> 阶段：只学习，不提 PR、不认领 issue、不改 `stable`。本文档是唯一的产出，位于 fork 的 `notes/learning` 分支。
> 方法与诚实边界：全部结论来自 GitHub MCP 读取的 README、模块 README、核心源码与 issue/PR 列表；**没有**拉起服务、没有跑测试、没有 GPU 验证。凡是未实际运行验证的点，下文明确标注"未验证"。

## 0. 一句话

Polar 是一个 **Rollout-as-a-Service**：把任意 agent harness（Claude Code、Codex、OpenHands……）当作 RL 环境，用一个透明代理截获 harness 与推理引擎之间的全部 LLM 调用，从线上的请求/响应重建可训练的 token 级轨迹（trajectory），附上 evaluator 打出的 reward，再经 HTTP 交给任意训练框架（目前只有 Slime bridge 落地）做 GRPO 之类的 RL 训练。核心设计口号是 **harness 零侵入**：harness 不用改代码、不解析 transcript，Polar 在中间"偷听"并重建轨迹。

配套论文（README 引用）：Polar（arXiv:2605.24220）、ProRL Agent（arXiv:2603.18815）。

## 1. Fork 状态：先看清分支，再谈贡献

| | 分支 | HEAD | 说明 |
|---|---|---|---|
| 上游默认分支 | `stable` | `45feeffe`（2026-10-07，`fabric integration (#56)`） | **真正的开发线**，全部 4 个 open PR 的 base 都是它 |
| 上游 `main` | `main` | `e7508cb`（2025-10-02，`ci: Add initial repo template and CI`） | **陷阱**：只是初始模板分支，受保护，内容与 `stable` 完全不同；不要以它为基准 |
| 本 fork | `stable`（仅有分支） | `6a1b2e2c`（2026-06-27，`Add paper and markdown files`） | 默认分支是 `stable`，fork 下**没有 `main` 分支** |

- **Fork 与上游 `stable` 的关系**：落后 2 个 commit（`#48` Harbor + Rubric 评估器、`#56` NeMo Fabric 集成），超前 1 个 commit（`6a1b2e2c` 个人笔记提交）。
- **`6a1b2e2c` 的内容（+7473 行，全部是个人笔记，绝不能进上游 PR diff）**：根目录 4 个 md（`8x4090_Reproduction_Plan.md`、`Incremental_Learning_Notes.md`、`Interview_5_Categories_Deep_Preparation.md`、`LLM_Agent_RL_Interview_Preparation.md`，共约 2000 行）+ `paper/` 下两份 arXiv 论文源码树（`arXiv-2603.18815v1`、`arXiv-2605.24220v1`，含 tex、字体、图片）。
- **贡献分支规则**：今后任何上游 PR 分支必须从**上游 `stable`**（`45feeffe`）建，不能从 fork `stable` 建（否则 4 个 md + `paper/` 会混进 diff）。本文档所在的 `notes/learning` 分支是例外：它只放学习笔记，永不向 upstream 发 PR。
- 本次任务前 fork 下**不存在** `notes/learning` 分支；本文档是该分支的第一个提交，fork `stable` 未被触碰。

## 2. 上游活跃度与治理

- 规模（搜索 API 当天）：857⭐、96 forks，创建于 2025-09-24，`updated_at` 2026-10-14T02:21:58Z，语言 Python，License Apache-2.0。
- `stable` 提交节奏：2026-10-07 `#56` → 2026-08-13 `#48` → 2026-06-26 `#45` → 2026-06-16 `#43` → 2026-06-01 `#37` → 2026-05-26 `#33` → 2026-05-22 `#32` → 2026-05-20 `#31` → 2026-05-14 `#28`。**近 3 个月只有 2 个 landing（#48、#56），且都出自 billxbf（Binfeng Xu）**；外部 PR 在 open 列表里挂着数周到数月未合并（见 §9）。维护模式是"单维护者 + 慢合并"，不是"没人维护"，也不是"快速响应"。
- Open issue 只有 **2 个**（`open_issues_count = 6 = 2 issue + 4 PR`，不是 6 个 issue）：
  - `#39`（bug，assignee billxbf）：opencode harness 零 completion，见 §11.1。
  - `#46`（enhancement，无 assignee）：问是否支持 SWE-bench 之外的更多 coding benchmark（如 Tau-bench），2026-07-01 提出后无后续。
- Open PR **4 个**（全部 base `stable`）：`#57`（JSON 非 ASCII 保留，EgorBu，2026-10-09）、`#53`（slime `rollout_id`/`group_id` 兼容，YolandaLyj，2026-08-20）、`#51`（anthropic `output_tokens` 被零 usage chunk 重置，andrewwhitecdw，2026-08-17）、`#49`（opt-in ThunderAgent 调度器，HaoKang-Timmy，2026-08-13）。
- **治理缺失（已验证）**：`stable` 上**没有** `.github/`（无 CI workflow）、**没有** `CONTRIBUTING.md`、**没有** PR 模板、**没有** DCO/CLA 机器人痕迹。`main` 分支的模板 CI 与 `stable` 无关，不能当作 `stable` 有 CI。提 PR 时没有可遵守的格式，只能参考已开 PR 的体例（`#53` 的 Summary/Background/Tests 结构是本仓库最好的范本）。DCO 是否强制**未验证**，建议首个 PR 正文里写明 Signed-off-by 并观察。
- 另外上游还有约 30 个陈旧分支（`arp-integration`、`osworld_*`、`docker_osworld`、`fix_*` 等），大多是已放弃的实验线，学习时忽略。

## 3. 架构：五段流水线 + 两条外挂

```
训练框架 (Slime, 经 slime_bridge) / polar submit / 示例脚本
        │  POST /rollout/task/submit (task → N sessions)
        ▼
┌──────────────── Rollout Server ────────────────┐  polar serve_rollout (默认 :8080)
│ manager.py 任务生命周期 / pipeline.py 调度+等待  │
│ balancer.py 节点注册、心跳健康、压力调度        │
└──────────────────┬──────────────────────────────┘
                   │  POST /sessions (SessionDispatchRequest)
                   ▼
┌──────────────── Gateway Node × N ───────────────┐  polar serve_gateway --node-id <id> (:8100+)
│ INIT: 启动 runtime + prepare recipe              │
│ READY: 等 run 槽位                                │
│ RUNNING: harness 跑 agent, 代理截获全部 LLM 调用   │
│ POST-RUN: 建 trajectory → evaluator 打分 → 回调   │
└────────┬───────────────────────┬────────────────┘
         │ 代理 (catch-all)         │ exec / 文件
         ▼                        ▼
  推理引擎 (SGLang / vLLM)     Runtime 沙箱 (Docker / Apptainer, 一 session 一容器)
         ▲
         │  训练桥需要同步权重时可 pause/resume 生成
  Dashboard (polar dashboard :8090, 只读 React SPA + FastAPI, SSE fan-out)
```

关键设计决策（README + 代码双重确认）：

1. **服务化边界**：Rollout Server 与 Gateway Node 之间、Polar 与训练框架之间全部走 HTTP + 回调，不共享进程、不共享 Python 环境。这就是"trainer agnostic"的来源：Slime、NeMoRL、VERL 理论上都能接，但**目前只有 Slime bridge 真实落地**（roadmap 里 `More trainer bridges (NemoRL, VERL, etc.)` 还是未勾选）。
2. **异步优先**：任务提交立即返回 `{task_id, status: "running"}`；gateway 回调 + 轮询双保险（`pipeline.py`，回调丢了也能靠轮询兜底）；训练侧的权重同步用 pause/resume 门控生成（`POST /admin/inference/pause|resume`，只暂停出方向生成，不停 gateway 进程）。
3. **合成流式（synthetic streaming）**：即使 agent 要 SSE 流，gateway 也只向后端发一次非流式请求，再把完整回答 replay 成格式正确的 SSE（`gateway/README.md`）。这极大简化了 capture，但意味着 agent 侧的"流式体验"是假的——对依赖逐 token 到达时序的 harness 可能是行为差异（未验证有无 harness 受影响）。
4. **超时口径**：`timeout_seconds` 从 session 进入 INIT 才开始计，排队时间（REGISTERED）不计入 agent 的 wall-clock（`rollout/README.md`）。

## 4. 核心数据契约（读代码的落点）

| 契约 | 文件 | 内容 |
|---|---|---|
| `TaskRequest` | `src/polar/rollout/models.py` | 训练方提交的任务：`task_id`、`instruction`、`num_samples`（≥1）、`timeout_seconds`（默认 600）、`runtime`、`agent`、`builder`（默认 `per_request`）、`evaluator`（可空）、`callback_url`、`metadata`（**无校验的透传 dict**，常用来带 `group_id`/`policy_version`/`rollout_step`） |
| `SessionDispatchRequest` | 同上 | rollout → gateway 的派发：`remaining_timeout_seconds`（gateway 从 INIT 起计）、其余同 TaskRequest |
| `SessionStatus` | 同上 | `REGISTERED → INITIALIZING → READY → RUNNING → POST_RUN → BUILDING → EVALUATING → COMPLETED/ERROR/TIMEOUT`；`terminal()` / `active()` 两个集合方法 |
| `SessionResult` / `TaskResult` | 同上 | 终态结果：`session_id`、`task_id`、`status`、`trajectory`、`timing`（`register_to_init_queue_ms` / `init_ms` / `run_ms` / `postrun_ms`）、`error`、`metadata` |
| `CompletionRecord` | `src/polar/trajectory/models.py` | 一次被截获的模型调用：`completion_id`、`request`（发给 served 模型的请求）、`original_request`（agent 的原始请求）、`response`、`metadata`；`extra="allow"` |
| `CompletionSession` | 同上 | 一次 run 的全部 record；validator 按 `(timestamp, completion_id)` 排序，保证 builder 看到有序序列 |
| `Trace` | 同上 | 可训练单元：平行数组 `prompt_ids` / `response_ids` / `loss_mask` / `response_logprobs` + `prompt_messages` / `response_messages` / `tools` / `finish_reason` / `reward` / `metadata`；validator 强制 `loss_mask` 只含 0/1 且长度与 `response_ids` 一致，`response_logprobs` 长度也一致 |
| `Trajectory` | 同上 | `status ∈ {COMPLETED, TIMEOUT, ERROR}` + `traces` + `metadata` + `error` |
| `EvalResult` | 同上 | `outcome_reward`（可空）+ `trace_rewards`（可空）+ `metadata`；**builder 不设 reward**，reward 由 gateway 在 `_merge_eval_result` 里合并 |
| `AgentSpec` | `src/polar/agent/models.py` | `harness` 与 `import_path` **恰好二选一**（validator 强制）；`custom_shell` 仅当 `harness == "shell"` 时合法，且与 `mcp_servers` / `skills_path` 互斥；`MCPServerSpec` 按 transport 校验 `command` / `url` |
| `AgentRunResult` | 同上 | `status ∈ {completed, failed, timeout}` + `return_code` + `error` + `metadata` |
| `RuntimeSpec` / `PrepareAction` / `ExecInput` | `src/polar/runtime/models.py` | 沙箱规格；prepare 是 `upload_file / upload_dir / exec` 的有序列表；`eval_prepare` 省略时 eval 复用 `prepare` |
| `TopologyConfig` | `src/polar/config/topology.py` | 单一 `topology.yaml`：`rollout:` + `gateway.nodes[]`；**严格不可变**：`extra="forbid"`（typo 直接失败）、frozen；`public_url` 为空时从 `host:port` 派生（`0.0.0.0`/`::` → `127.0.0.1`）；`gateway.rollout_server_url` 为空时 fallback 到 `rollout.public_url`；节点 id 唯一 |
| `StrategySpec` / `EvaluatorSpec` | `src/polar/trajectory/models.py` | 策略名 + `config`；`EvaluatorSpec` 多 `env` + `refresh_runtime` |

**Reward 合并规则**（`gateway/node.py:701-741` `_merge_eval_result`，已读代码）：`trace_rewards` 长度必须**恰好等于** trace 数，否则整个 session 标 `ERROR`（reward 不部分应用）；只有一个 `outcome_reward` 时广播到全部 trace；两者都为空时 trace 的 `reward` 保持 `None`。这个"长度不匹配即整单失败"是严格但合理的 Fail-fast 设计，自定义 evaluator 最容易踩。

## 5. 模块详解

### 5.1 Rollout（中央编排）`src/polar/rollout/`

- `server.py`：FastAPI，`POST /rollout/task/submit`（立即返回）、`GET /rollout/task/{id}`、`GET /rollout/status`、节点注册/心跳/列表/下线、`/callbacks/session_result`、`/tasks` + `/events`（给 dashboard）、`/health`。
- `manager.py`：任务生命周期、按 `num_samples` 展开 session、终态回调给训练方。
- `pipeline.py`：核心调度循环。`_dispatch_session`（约 165-215 行）：`acquire_node()` 拿预留 → 向 gateway `POST /sessions`（超时取 `min(30s, 剩余)`）→ 失败则 `release_reservation` + `mark_unhealthy` 后重试；有个 `_accepted_duplicate_dispatch` 兜底：gateway 已接受但回包丢了时，用 GET 确认 session 归属，避免 409 死循环。结果持久化在 `pipeline.py:414-415`：`json.dumps(..., separators=(",", ":"), default=str)`（**未设 `ensure_ascii=False`**，与 open PR `#57` 直接相关，见 §11.4）。
- `balancer.py`（`NodeScheduler`，248 行）：调度是**压力优先**。`_node_score` 返回 `(run_pressure, postrun_pressure, init_pressure, -ready_gap, total_pressure, node_id)` 的字典序 tuple；`_can_accept_session` 要求 `total_sessions < admission_limit`（= init + ready_buffer + run + postrun 之和）且 postrun backlog `< max_postrun_workers × 2`；健康定义为心跳在 `heartbeat_interval × 2.5` 内（`stale_factor = 2.5`）。`ready_buffer_target = max_run_workers`（READY 缓冲与 run 容量 1:1）。注意 `heartbeat()`（`balancer.py:103-111`）会把 `dispatch_reservations` 无条件清零，见 §11.2 的竞态观察。
- `timer.py`：分阶段计时（INIT / RUN / POST-RUN），产出 `SessionTiming`。

### 5.2 Gateway（每 worker 节点）`src/polar/gateway/`

- `server.py`（840 行）：FastAPI。`/{path:path}` catch-all POST（`server.py:609`）是代理入口；`_handle_non_streaming` / `_handle_streaming`（671/704 行）；`GET /v1/models`、`GET /health`、`/admin/inference/pause|resume|status`、`/sessions` 系列、`/events`（SSE）。
- `node.py`（973 行，`GatewayNodeManager`）：阶段处理器。`_handle_init`（225 行）→ `_handle_run`（305 行）→ `_handle_postrun`（496 行）；`_build_session_result`（559 行）串起"建 trajectory → evaluator → 合并 reward → 回调"；`_build_trajectory`（626 行）经 `asyncio.to_thread` 调 builder（注意：里面对协程式 builder 用了 `asyncio.run`，之所以安全是因为它跑在 `to_thread` 的独立线程里，不是事件循环线程——读代码确认，不是 bug）；`_run_eval`（636 行）支持 `refresh_runtime` 时拉一个预热好的全新 runtime；`_merge_eval_result`（701 行）见 §4；`_runtime_env`（744 行）注入代理 env（`OPENAI_BASE_URL` 等 + API key = session id）。
- `session.py`（278 行）：内存 session 注册表、session id 校验、从入站代理请求解析 session id（API key 即 session id）。
- `detection.py`（50 行）：API 家族检测，优先级 **path → header → body**：`/v1/messages` → anthropic；`/v1/chat/completions` → openai_chat；`/v1/responses` → openai_responses；`generateContent` → google；header 有 `anthropic-version` → anthropic；body 有 `contents` → google；body 同时有 `input` + `instructions` → openai_responses；**其余一律默认 openai_chat**（见 §11.6 的静默 fallback 观察）。
- `transform/`：四种 API 到"OpenAI Chat Completions"内部 canonical 格式的双向转换。`base.py` 做共享归一化（`developer` → `system` 合并、Qwen3.5 强制 `enable_thinking=False`）；`openai_chat.py` 近透传（`max_completion_tokens` → `max_tokens`）；`openai_responses.py`（1059 行，Codex 用）含 reasoning / shell / function tool 的转换；`anthropic.py`（681 行）含 Claude-Code header 处理；`google.py` 含 functionDeclarations/Call/Response；`images.py` 跨 API 图片/文档归一化；`reasoning.py` 推理内容的跨 API round-trip。**模型名**由代理（`server.py`）换成 served 模型，transformer 在响应里把请求的模型名恢复，客户端无感知。
- `engine.py`（209 行）：推理后端策略（SGLang / vLLM），注入 token id / logprob 参数并规范化响应。
- `proxy.py`：`InferenceClient`，到推理引擎的 HTTP 客户端，带 pause/resume 生成门控。
- `storage.py`：内存 completion-record 存储（权威副本）；`completion_writer.py`（220 行）：后台任务把 completion 异步落盘，路径 `<save_dir>/task_<id>/sessions/<sid>/completions/<NNNN>-<id>.json`，受 `gateway.completion_persistence` 控制（`enabled` 默认 true、`max_field_bytes` 默认 1MB、`queue_size` 默认 1024）；落盘在 `completion_writer.py:216` 用 `json.dumps(payload, default=str)`（**同样未设 `ensure_ascii=False`**，与 `#57` 相关）。
- `dispatcher.py`：INIT / READY / RUNNING / POST-RUN 四个阶段隔离的 worker 池与状态机。

### 5.3 Harness（agent 启动器）`src/polar/agent/`

契约极小（`base.py`，56 行）：

```python
class BaseHarness(ABC):
    async def setup(self, runtime) -> None: ...          # 写配置，勿做重活
    def run_steps(self, instruction) -> list[ExecInput]: ...  # 返回要跑的命令
    def postrun_steps(self) -> list[ExecInput]: ...      # post-run 拆卸前 best-effort 执行
    async def postprocess(self, runtime, result) -> None: ...  # 产物收集
```

三种接入（`agent/README.md`）：**preset**（`agent.harness: "<name>"`）、**`shell`**（`agent.harness: "shell"` + `agent.custom_shell`，任意命令）、**`import_path`**（`your.module:YourHarness`，自己的类留在自己的 repo）。Preset 只是"薄启动器"，每个几十行，唯一价值是"写 agent 配置 + 发运行命令"的胶水——如果 agent 已读 `OPENAI_BASE_URL` / `OPENAI_API_KEY`，preset 几乎可以只有 `run_steps`。

Preset 表（`factory.py` `_builtin_harness_map`，12 个）：`claude_code`（anthropic）、`codex`（openai_responses）、`gemini_cli`（google）、`opencode`（openai_chat）、`openclaw`（openai_chat）、`openhands_sdk`（openai_chat，非流式）、`hermes`（openai_chat）、`pi`（openai_chat，非流式）、`qwen_code`（openai_chat）、`mini_swe_agent`（openai_chat）、`nemo_fabric`（按 `settings.adapter` 定）、`shell`。每个 preset 的 verified version 写在 `agent/README.md`（如 `claude_code@2.1.111`、`codex@0.125.0`、`opencode-ai@1.4.6`、`openhands-sdk==1.17.0`——注意 1.18+ 需要 Python 3.13，3.12 上 pin 死 1.17.0）。

**关键坑**（`agent/README.md` 明确写出）：`setup()` 在代理 env 可用**之前**运行，所以任何需要 `$OPENAI_BASE_URL` 的配置必须写在 `run_steps()` 的命令里（`openclaw` / `hermes` / `pi` 用 placeholder + sed 替换；`codex` 在 `run_steps` 里写 `auth.json` + `config.toml` 追加 `openai_base_url`）。**这正是 issue `#39` 的根因模式**：opencode preset 没做这一步（§11.1）。

**NeMo Fabric**（`#56` 新增，`nemo_fabric.py` 257 行 + `nemo_fabric_runner.py` 69 行 + 测试 223 行）：一个 preset 覆盖 Fabric 整合的全部 harness。`settings.adapter` 选 adapter（`claude` / `codex` / `deepagents` / `hermes` / `mini_swe_agent` / `nooa` / `nooa_bench` / `openclaw` / `openhands` / `cline` / `kilo` / `opencode` / `pi` / `qwen` 共 14 个）；Polar 生成 `FabricConfig`，`models.default` 指向对应协议的 gateway 端点（`${ANTHROPIC_BASE_URL}` 或 `${OPENAI_BASE_URL}`），`settings.config` deep-merge 覆盖（`null` 删 key）；runner 被上传进 runtime 调 `Fabric().run()`，结果写 `logs/agent/fabric/result.json`，失败则 step 失败且 Fabric 错误码追加到 session error。细节：`instructions.system.mode: append` 才能保留 Claude Code 原生 system prompt，否则 Fabric 的 Claude adapter 会用 Agent SDK 的一句话 system prompt 替换掉。

### 5.4 Trajectory：builder + evaluator `src/polar/trajectory/`

- 注册表（`registry.py`，`StrategyRegistry`，通用泛型）：存**类**不存实例，每次 `create(spec)` 用 `spec.config` 新建；策略名未注册但含 `:` 时按 `"module:ClassName"` import_path 解析并缓存。**策略名是稳定契约**（任务文件和 Slime 配置都按字符串引用），改名是 breaking change。
- Builder（2 个）：
  - `per_request`（默认）：一个 completion = 一个 trace，保真但不合并。
  - `prefix_merging`（`builder/prefix_merging.py`，369 行）：多轮 agent 每次重发递增的会话，连续请求共享公共前缀。合并判定是**严格 token 前缀**：新请求的 `prompt_ids` 必须以链上前一个 completion 的 prompt 开头（append-only）；消息级 key 先做候选过滤，且刻意忽略 tool-result 与空 assistant 消息（普通 tool 循环仍能合并）；前缀关系断裂（如上下文压缩重写了早期轮次）时开新 trace，部分合并的链可在断裂处截断。`loss_mask = 1` 给应训练的采样 token，`0` 给 interstitial/拷贝 token；采样 token 保留真实 logprob，interstitial 填 `0.0` 占位以保持数组对齐（训练侧用 TIS 纠 off-policy 比率）；轮次边界靠 end-of-turn token（自动检测或 `end_of_turn_token_id` 配置，Qwen/ChatML 即 `<|im_end|>`）。空 session → `Trajectory(status="ERROR", error="no completions")`。
  - 另有 `record_utils.py`：从 record 里抽 messages / token id / logprob / metadata 的 helper。
- Evaluator（5 个，`evaluator/`）：
  - `session_completed`：终态 `COMPLETED` 得 1.0，否则 0.0；不需要 runtime，smoke-test 用。
  - `test_on_output`：抽 agent 的 git diff、（可选）在全新 runtime 上 apply、跑测试命令，**按解析测试输出判分，不按 exit code**：读 `PASSED` / `FAILED` / `ERROR` / `SKIPPED <node>` 行，要求解析结果与 `expected_output_json` **精确相等**才给 1.0。空 diff 得 0.0。
  - `swebench_harness`：用 SWE-bench（或 SWE-Gym）harness 给真实 patch 打分，输入 `instance` dict + 同 test_on_output 的 patch 配置。
  - `harbor`：把 Harbor 任务的 `tests/` 注入 agent 的**活 runtime**，跑 `bash /tests/test.sh`，按 Harbor verifier 契约读回 reward；要求 `refresh_runtime: false`。
  - `harbor_rubric`（`#48` 新增，`harbor_rubric.py`，17KB + 测试 10KB）：harbor outcome + rubric 行为校准。任务带 `tests/rubric.md` 时，一次 judge 调用（OpenAI-compatible 端点）看任务 `instruction.md`、统一 meta rubric（Must-do/Best-practice、Must-avoid、效率、诚实、reward hacking 重罚）、任务 rubric、verifier 原始打分（有 `reward.json` 时）、以及每条 trace 的 `response_messages`（编号 `trace_0`…）。每条 trace 被当作不透明单元（一个 completion、一条合并多轮链、整个 rollout、或并行分支都可），judge 只评"该单元内有证据的行为"，不假设 builder 策略。返回 `{trace_id: [-5, 5] 整数}`；除灾难性的 `-5` 强制 reward 归 0 外，每条 trace 得 `clip((1-c)·outcome + c·(score/5), 0, 1)`，`c = rubric_coefficient` 默认 0.2。**Fails open**：无 rubric、judge 失败、缺 trace id 时回退到未缩放的 outcome reward。Judge 配置：`judge_base_url` / `judge_model` 必填，`judge_api_key_env` 默认 `JUDGE_API_KEY`（先查 evaluator `env` 再查进程 env）。
  - `_patch_utils.py`（12KB，`BasePatchEvaluator`）：`test_on_output` 与 `swebench_harness` 共享的 extract → filter → apply → test 流程。

### 5.5 Slime Bridge `src/slime_bridge/`

- 定位：Slime（THUDM）RL 训练循环与 Polar rollout server 之间的 HTTP 桥。**故意放在 `polar` 包之外**，因为 Slime / Ray / Megatron / torch 是另外安装的——Polar 本身不依赖它们。
- 入口：Slime 经 `--rollout-function-path` 调 `generate_rollout_polar_async`（`rollout.py`，53KB，最大文件）。桥做：向 `polar_rollout_url` 提交 async 任务 batch，用本地回调监听器 + 轮询安全网收结果；跟踪 rollout id / policy version，把 `group_id` / `policy_version` / `rollout_step` 盖到每个任务上；把 Polar `Trajectory` 转回 Slime `Sample`（每 trace 一个 Sample，同一 trajectory 的 Sample 共享 `group_id`，Slime loss reducer 按组平均一次）；过滤空/超大 trace；算 dynamic-trace leave-one-trajectory-out advantage；失败/中断 trajectory 的 advantage 置零。
- `adapter.py`（326 行）：`session_result_to_samples`；**注意**：当前代码仍用 Slime 0.3.0 tag 的 `group_id` 字段（`adapter.py:170` / `:216`），这正是 open PR `#53` 要兼容的点（新版 Slime 已恢复 `rollout_id`，旧桥会 `TypeError: unexpected keyword argument 'group_id'`）。
- `config.py`：`PolarSlimeConfig` + 渲染任务 payload / instruction / topology（让 gateway 指向 Slime 的 SGLang router）。
- `reward.py`：reward hook，直接读 Polar 已嵌入的 reward；`reward_post_process.py`：trajectory-aware、group-normalized reward shaping；`data_source.py`：`CeilEpochRolloutDataSourceWithBuffer`，把 epoch 长度向上取整避免跳过数据集尾部；`_messages.py`：prompt/message 扁平化（rollout + adapter 共享，README 注明是有损的）。
- 安装：Slime 必须从 THUDM git checkout 装（不是 PyPI 上同名的 `slime` 包）；`scripts/patch/patch_slime_router_tokens.sh` 保留 SGLang 原生 prompt/output token id 与 token 级 logprob，避免 Polar 本地重 tokenize；`scripts/patch/patch_sglang_0513_token_metadata.sh` 是 SGLang 0.5.13 的 token-metadata 补丁。README pin SGLang `==0.5.13`（与 calculator 示例的 vLLM 拓扑并存，**不要在同一环境装两个推理后端**）。

### 5.6 Runtime `src/polar/runtime/` + Dashboard `src/polar/platform/`

- Runtime：一个 `RuntimeSpec` → 一个容器，贯穿 session 的 init → run → eval 全程。宿主 session 目录 bind-mount 到容器内固定路径 `/polar/session`（`RUNTIME_SESSION_DIR`），该路径下的上传/下载是宿主侧纯文件拷贝（快），路径外 fallback 到 `docker cp` / tar 流；命令以 `bash -lc` 跑，工作目录 `cwd or spec.workdir or /polar/session`。`docker.py`（默认，支持 `--cpus` / `--memory`）、`apptainer.py`（无 daemon，集群友好，`--nv` 暴露 GPU）；`factory.py` 在构建前校验后端是否真支持 spec 要的东西（GPU、CPU/内存上限、断网）。容器内约定路径：`/polar/session` 下 `artifacts/`、`logs/`、`logs/agent/`、`logs/eval/`、`eval_artifacts/`。
- Dashboard：只读可观测面板。React SPA + 单个 FastAPI 服务，**双源合并**：live HTTP（rollout + 各 gateway）+ 磁盘（`save_dir` 下 `task_*/`，`FsIndex` 每几秒扫一次）；`SseFanout` 对每个上游开一个 `/events` 流、打 source 标签、重发布到进程内 `EventBus`，浏览器消费 `GET /api/events`。唯一会改变状态的操作是 cancel 代理。Wheel 只有在 `web/dist/` 构建后才带真实 UI（`cd web && npm install && npm run build`），否则只服务一个 JSON 占位。

## 6. RL 循环是怎么跑起来的

1. 训练侧（Slime）产出 prompt batch → bridge 渲染成 Polar `TaskRequest`（`num_samples` = 每 prompt 采样数，`metadata` 带 `group_id` / `policy_version` / `rollout_step`）。
2. Rollout server 把 task 展开成 N 个 session，`NodeScheduler` 按压力选 gateway，`pipeline` 派发并等待（回调 + 交错轮询双保险）。
3. Gateway：INIT（起 runtime + prepare）→ RUNNING（harness 在容器里跑 agent，全部 LLM 调用被代理截获并记成 `CompletionRecord`）→ POST-RUN（builder 重建 trajectory → evaluator 打分 → 合并 reward → 回调给 rollout）。
4. Bridge 把 `Trajectory` 转成 Slime `Sample`，算 advantage（dynamic-trace LOTO），Slime 做 GRPO 更新；需要同步权重时经 pause/resume 门控生成（in-flight 调用 drain 后再恢复）。
5. Eval 路径：Slime bridge 也有独立的 evaluation path，走 `eval_datasets`，发 W&B 指标。

训练侧关键数值关系（理解即可，未实测）：`loss_mask` 决定哪些 token 进 loss；interstitial 位置 logprob 为 0.0 占位 + TIS 纠 off-policy 比率；`prefix_merging` 把多轮 append-only 链压成一条 trace，避免同一 prompt 前缀被重复计 loss。

## 7. 例子与启动

| 例子 | 用途 | 入门成本 |
|---|---|---|
| `examples/calculator/` | 最小端到端 smoke test：每个 harness 改一个带 parser stub 的 `calculator.py`，evaluator 跑 `python3 test_calculator.py`；`run.py` 一次性提交全部 harness 并打印 reward 对比；支持 `--harness nemo_fabric`（全部 14 个 adapter）/ `--fabric-adapter pi` | 1 节点 8×H100（README 原话，需按自己硬件调整）；2 个推理服务器（`topology.vllm.yaml` / `topology.sgl.yaml`）+ rollout + 2 gateway |
| `examples/count_stars/` | VLM 最小测试 | 同上 |
| `examples/swebench_verified/` | benchmark 风格评测：`submit_swebench_tasks.py --harness <name> --max-tasks N --num-samples 1 --runtime-backend docker --model-name <model>`（正是 `#39` 的复现命令） | 同上 + Docker |
| `examples/swegym_slime_grpo/` | 真实训练：Qwen3.5-4B + async GRPO on SWE-Gym，单节点 8×{H100,H200,B200}（4 卡训练 + 4 卡推理）；`launch_e2e.sh` 一键：克隆 Slime + Megatron-LM、打两个补丁、建 293-task JSONL（`swegym_train_293.jsonl`，8.6MB）、拉 Apptainer 镜像、转权重、起 `run.sh`（Polar + Ray + Slime） | 重；SGLang 专用（Slime 拥有推理引擎，每步 NCCL 同步新权重） |
| `examples/tmax-15k/` | TMAX 评测（`#45` 引入） | 同 swebench_verified |

CLI（`pyproject.toml` 入口 `polar = "polar.cli:main"`，`cli.py` 12KB）：

```bash
polar serve_rollout -c topology.yaml                 # 中央编排 :8080
polar serve_gateway -c topology.yaml --node-id <id>  # 每 gateway 节点一个 :8100+
polar dashboard -c topology.yaml [--port 8090]       # 只读面板
polar submit <task.json|yaml> -c topology.yaml       # 提交任务并 tail
polar status -c topology.yaml                        # 一次性健康/拓扑检查
```

## 8. 测试与工程质量

- 测试：25 个 test 文件——`tests/gateway/` 9 个（transform 占 5 个：anthropic / google / openai_chat / openai_responses / reasoning，26KB + 27KB 最大两个）、`tests/slime_bridge/` 6 个、`tests/trajectory/` 4 个、`tests/platform/` 3 个、`tests/agent/` 1 个（`test_nemo_fabric.py`）、`tests/rollout/` 1 个（仅 balancer）、`tests/config/` 1 个（topology）。
- `pyproject.toml`：包名 `polar` 0.1.0，`requires-python >= 3.11`，运行时依赖只有 5 个（fastapi / uvicorn / httpx / pydantic / pyyaml）+ 可选 `swebench`；dev 组 pytest / pytest-asyncio / ruff / build / twine；ruff `line-length = 99`，`target-version = py311`；pytest markers `unit` / `integration` / `gpu`，`testpaths = ["tests"]`，`examples/` 等被排除在 pytest 收集之外。
- **覆盖缺口（读目录得出，未跑覆盖率）**：`tests/rollout/` 只有 balancer 测试，pipeline / manager / server 无测试；`tests/agent/` 只有 nemo_fabric，11 个其余 preset 无单测；runtime（docker/apptainer）无测试；`gateway/node.py`（973 行核心）无直接测试。测试重心在 transform + trajectory + bridge，与"代理截获正确性"和"轨迹重建正确性"这两个最难的点对齐，但调度与运行时是薄弱带。
- **无 CI**：见 §2。本地能跑的只有 `pytest` + `ruff`（PR `#53` 自称 "29 passed + Ruff + git diff --check" 是可信的范式）。

## 9. 与 uni-agent / verl / deepseek-harness 的对比

| | Polar（本仓库） | uni-agent（`verl-project/uni-agent`，此前已学） | verl / verl-recipe |
|---|---|---|---|
| 一句话 | Rollout-as-a-Service：harness 零侵入，代理截获重建轨迹 | Long-horizon agent 训练框架，一个 interaction stack 接推理/评测/轨迹/SFT/RL | RL 训练核心库 / 社区 recipe 集合 |
| 与训练框架的关系 | **服务边界**：HTTP + 回调，训练框架是客户端；目前只有 Slime bridge 落地 | **库/框架边界**：`AgentFrameworkRolloutAdapter → TransferQueue → verl trainer`，训练循环在同一进程体系内 | verl 是训练核心；recipe 是接线 |
| Harness 接入 | 代理截获：harness 不改代码，`shell` / `import_path` / preset 三路 | 注册式 Agent（5 个：react / claude_code / mini_swe_agent / openclaw / mem_agent）+ 8 Task + 6 Sandbox provider | recipe 级接线 |
| 轨迹表示 | `Trace`：`prompt_ids` / `response_ids` / `loss_mask` / `response_logprobs` 平行数组 + `reward` | `Trajectory`：token 级 + `response_mask` 为 loss mask | verl 的 batch / advantage 体系 |
| 推理后端 | SGLang / vLLM（gateway 内 `engine.py` 策略） | 通过 interaction stack 接推理 | verl 原生 |
| Eval | 内置 evaluator 策略（test_on_output / swebench_harness / harbor / harbor_rubric）+ reward 合并 | Task + Sandbox 层负责 | recipe 各自实现 |
| 成熟度信号 | 857⭐，单维护者慢合并，2 issue + 4 open PR，无 CI | 666⭐，112 commits 落后 fork，57 open issues+PR，26Q4 roadmap | verl-recipe 338⭐，57 open issues+PR |

**对 deepseek-harness 一线的启示**（用户原话：与 deepseek-harness 思路直接相关；以下为类比判断，未对照 deepseek-harness 代码验证）：Polar 的"代理截获 + prefix_merging 重建 + registry 策略"三件套是可直接借鉴的模式——harness 侧零侵入、轨迹重建与打分分离、策略名即契约。反过来，Polar 的代价是整套服务化运维（rollout + 多 gateway + 容器 runtime + 权重同步门控），如果目标只是单机跑通 RL 闭环，uni-agent 式的库内集成可能更轻；如果目标是"一个 rollout 服务供多个训练框架/多节点共享"，Polar 的边界设计是更对的答案。选择前先确认 deepseek-harness 的实际规模：单机实验 vs 多节点服务，两条路的取舍完全不同。

## 10. 值得注意的设计判断（学习所得，非贡献建议）

1. **约束即设计**：`TopologyConfig` 的 `extra="forbid"` + frozen、`AgentSpec` 的恰好二选一、`Trace` 的 loss_mask 长度校验、`_merge_eval_result` 的 trace_rewards 长度精确匹配——四处都是"不可能的状态让它 loud 地炸"，与 Molt 维护者 final pass 的风格一致。
2. **Canonical 格式收敛**：四种 API 全收敛到 OpenAI Chat Completions 再进后端，transform 只做"形状转换"，训练信号（logprobs / token id）留在 `engine.py`——分层干净，避免 transform 里塞后端特判。
3. **Fails open 只用在评估侧**：`harbor_rubric` judge 失败回退 outcome reward（评估失败不应毁掉一次 rollout）；但 reward 合并时长度不匹配则整单 ERROR（数据正确性不容妥协）。两个方向的取舍分得清。
4. **README 即契约**：每个模块 README 都写了"mental model + 主文件 + 关键坑"（如 setup() 在 env 之前跑），且与代码一致（抽查的 balancer / prefix_merging / registry 均吻合）。这种文档密度是该仓库最值得抄的工程习惯。
5. **单维护者 + 无 CI 的隐性成本**：`#51`、`#53` 这种带测试的小 fix 挂 2 个月未合并，说明"写了 PR"不等于"能落地"；选贡献角度时要优先"维护者不得不合"（回归/阻塞类）而非"锦上添花"。

## 11. 候选观察（仅观察，未实现、未认领、未发声）

> 状态说明：`#39` 已有 assignee（billxbf）；`#51` / `#53` / `#57` / `#49` 已有 open PR 覆盖对应问题。**不要**重复开 PR 或抢认领。下面每条都标注了"已验证到代码"与"还需验证"。

### 11.1 最强候选：`#39` opencode preset 缺 gateway baseURL（已验证仍存在）

- Issue：`#39`（2026-06-03，assignee billxbf，含 1 条评论未展开读），SWE-bench Verified 下 opencode 10 个 task 全部 0/10，gateway 日志里该 session 没有任何 `POST /v1/chat/completions`，4 秒退出，`error: "no completions"`；对照组同设置下 qwen_code / claude_code 均为 5/10。
- 根因（**已在当前 `stable` 代码上验证仍存在**）：`src/polar/agent/presets/opencode.py` 的 `setup()` 写出的 `$HOME/.config/opencode/opencode.json` 只有 `provider` + `permission` + 可选 `mcp`，**没有 `options.baseURL` / `options.apiKey`**；opencode CLI 不读 `OPENAI_BASE_URL` env var，默认走自己的内置 OpenAI 地址，于是根本没流量进 gateway。对照：`codex.py` 在 `run_steps` 里写 `auth.json` + `config.toml` 追加 `openai_base_url = "${OPENAI_BASE_URL}"`；`openclaw.py` 用 placeholder + sed 替换；opencode 是唯一没配 baseURL 的 preset。
- Issue 自带建议 fix（placeholder + sed，与 openclaw 同款模式），改动面预计 1 个文件 + 1 个测试。
- 落点：阻塞类（opencode 在该 benchmark 上 0/10 全灭），且是维护者已确认 assignee 的 bug；若要动手，**先在 issue 下问 assignee 是否欢迎外部 fix**，不要直接开 PR 抢活。
- 还需验证：opencode 1.4.6 的 config schema 是否确为 `provider.<name>.options.baseURL / apiKey`（issue 断言如此，但未对照 opencode 官方文档验证）；sed 替换时 `$OPENAI_BASE_URL` 在容器内的实际值（是否带 `/v1` 后缀，opencode 期望的是否一致）；`opencode run --format=json` 的退出码语义（失败时是否非零，便于测试断言）。

### 11.2 调度器竞态：heartbeat 清零进行中的 dispatch reservation（已验证代码路径，需维护者确认是否为有意设计）

- 代码：`src/polar/rollout/balancer.py:103-111`，`heartbeat()` 无条件执行 `node.dispatch_reservations = 0`；而 `pipeline.py` 的 `_dispatch_session` 在 `acquire_node()`（预留 +1）之后、向 gateway `POST /sessions`（超时可达 `min(30s, 剩余)`）之前，有一个 reservation 持有窗口。窗口期内到达的心跳会把该预留清零；同时成功派发后代码**不**显式 `release_reservation`（失败路径才释放，`pipeline.py:204`），说明 heartbeat 清零同时承担了"释放"的职责。
- 后果（推理，未实测）：清零后 `total_sessions`（含 reservation）与 `init_pressure`（含 reservation）都被低估，并发派发时可能超过 `_can_accept_session` 的 admission limit 而过度准入。心跳默认 30 秒间隔，窗口通常很短，实际触发概率取决于 gateway 响应延迟与并发度。
- 落点：纯 CPU 可单测（直接测 `NodeScheduler`，无需 GPU / 服务），改动面小（例如派发成功时显式释放 + heartbeat 只清零"已确认"的预留，或给 reservation 加时间戳）。但**必须先确认**：heartbeat 清零可能是维护者有意为之的简化设计（用心跳作最终一致的释放），贸然"修复"可能与设计意图冲突。适合以 issue 提问 + 附复现单测的方式探路，不适合直接甩 PR。
- 还需验证：多并发派发下是否真能观测到超 admission；`_maybe_remove_drained_locked` 与清零的交互（draining 节点在 heartbeat 时被移除的边界）。

### 11.3 Review 观察：`#51` 的测试路径与仓库布局不符（已验证，适合在 PR 下评论而非另开 PR）

- `#51`（anthropic `output_tokens` 被零 usage chunk 重置）的 fix 本身与当前代码吻合：`src/polar/gateway/transform/anthropic.py:90` 仍是 `self.output_tokens = usage.get("completion_tokens", self.output_tokens)`，零 chunk 确实会把它重置为 0；PR 提出的 `max()` 方案合理。
- 但 PR 正文称测试在 `tests/unit/gateway/transform/test_anthropic_output_tokens.py`，而仓库实际布局是 `tests/gateway/test_transform_anthropic.py`，**不存在 `tests/unit/` 目录**，且 `pyproject.toml` 的 `testpaths = ["tests"]`。若按 PR 正文落盘，测试会进一个新目录，与现有命名惯例断裂。
- 落点：零代码成本的 review 评论（指出路径 + 建议放到 `tests/gateway/test_transform_anthropic.py` 或按现有文件合并），比另开 PR 更合适。**不要**就此另开重复 PR。

### 11.4 已覆盖，勿重复：`#57` JSON 非 ASCII（已验证落点）

- `#57`（EgorBu，2026-10-09）要 `ensure_ascii=False`，动机是非英文可读。当前代码有两个落点：`src/polar/gateway/completion_writer.py:216`（`json.dumps(payload, default=str)`）与 `src/polar/rollout/pipeline.py:414-415`（`json.dumps(..., separators=(",", ":"), default=str)`），均未设 `ensure_ascii`，默认 `True` 会转义非 ASCII。
- 落点：已有 open PR，**勿重复**。如需参与，只能做 review（例如提醒：`pipeline.py:414-415` 与 `completion_writer.py:216` 是否都覆盖到；`ensure_ascii=False` 后文件仍需 UTF-8 写盘，`write_text` 默认编码在部分环境不是 UTF-8，值得在 PR 下提一句）。

### 11.5 已覆盖，勿重复：`#53` slime `rollout_id` / `group_id`（已验证旧代码）

- `src/slime_bridge/adapter.py:170` / `:216` 仍用 `group_id`；`#53`（YolandaLyj，2026-08-20，自称 29 测试通过 + Ruff）做按 `Sample` dataclass 字段探测的兼容（新 Slime 恢复 `rollout_id`，旧 tag 用 `group_id`），方案看起来是对的。
- 落点：已有 open PR，**勿重复**。值得注意的是它挂了近 2 个月未合并，再次印证 §2 的"慢合并"判断。

### 11.6 小候选：`detect()` 对未知 API 静默默认 openai_chat（已验证代码，需确认是否为有意）

- 代码：`src/polar/gateway/detection.py:20-49`，`detect()` 在 path / header / body 都匹配不上时**无日志、无告警**地返回 `APIType.OPENAI_CHAT`。
- 后果（推理）：一个新 harness 发了 Polar 不认识的协议形状时，不会得到"未知 API"的明确错误，而是被当作 OpenAI Chat 走下去，在更下游以难懂的方式失败（恰好是 `#39` 类"4 秒退出、零 completion、无明确报错"的排障噩梦的放大器）。
- 落点：改动面极小（fallback 处记一条 warning，带 path + 关键 header），纯 CPU 可测（`test_detection.py` 已有 1294 字节的测试文件可扩展）。但需先确认维护者是否刻意保持静默（例如某些 harness 的 body 形状确实会落到默认分支且工作正常），贸然加日志可能制造噪音。适合小 issue 提问。
- 还需验证：现有 preset 里是否有依赖默认分支的 harness（grep 各 preset 的实际请求 path 未做）。

### 11.7 明确不建议：`#46` 更多 benchmark、`#49` ThunderAgent、harbor_rubric 调参

- `#46`（Tau-bench 等）：无 assignee、无后续讨论，范围大（新 benchmark 需要新 evaluator + 新 example + 数据集接入），且 roadmap 只写了"更多内置 evaluator"，没有承诺具体 benchmark。单独提 PR 极易因"维护者未表态要不要"而石沉大海。先等维护者表态。
- `#49`（ThunderAgent）：已有 open PR，勿重复；且它引入的是外部调度器依赖，维护者是否愿意接外部代理进拓扑是设计决策，不是代码问题。
- harbor_rubric 的 `rubric_coefficient` 默认 0.2、`-5` 归零等数值：README 已写清且有测试覆盖，改数值需要实验数据支撑，没有实验就不要动。

## 12. 未验证清单（诚实边界）

- 未运行任何服务 / 测试 / 示例；未验证 calculator / swebench_verified 在真实 GPU 上能跑通。
- `#39` 的 1 条评论未展开读（MCP `list_issues` 未返回评论正文）；`#39` 正文后半部分还贴了第二个 codex 调查（4 轮退出、空 patch、怀疑 `openai_responses.py` 的 `transform_response` 丢 `finish_reason` / refusal / incomplete 信号）——已 grep 确认 `openai_responses.py`（1059 行）内无 `refusal` / `incomplete` 处理，但**未验证**这是否真导致 codex 提前退出（也可能是模型行为），列为 §11.7 式观察，不单独成候选。
- 未验证 opencode 官方 config schema、未验证 heartbeat 竞态的实际触发、未验证 DCO/CLA 是否强制、未验证 closed PR 的 merged 状态（MCP 返回的 `merged` 字段对全部 closed PR 均显示 False，与 `stable` 历史矛盾，不可信，故本文不引用它）。
- 未对照 deepseek-harness 代码做对比（§9 末段为类比判断）。
- `node.py` 的 eval prewarm（`_start_eval_prewarm` 等）与 `#31` 提交信息里提到的 `max_eval_prewarm_workers` / `eval_prepare` 配置项的关系未完全理清：当前 `topology.py` 里已无 `max_eval_prewarm_workers` 字段（全仓搜索 0 命中），`eval_prepare` 只在 `RuntimeSpec` 侧存在；可能是后续重构删了配置项，也可能是本学习漏了入口。写贡献代码前需再核对。

## 13. 给下一步的建议（给人看的，不是承诺）

1. 若想快速建立存在感：先在 `#39` 下问 assignee 是否欢迎外部 fix（附上"已确认当前 stable 仍缺 baseURL"的验证），再动手；同时在 `#51` 下留一条测试路径的 review 评论（零成本、高信号）。
2. 若想做独立小 PR：`§11.2` 的 heartbeat 竞态与 `§11.6` 的 detect fallback 都是纯 CPU 可测的小改动，但两者都先以 issue 提问确认设计意图，再写代码——这个仓库的慢合并节奏下，"问对问题"比"写对代码"更重要。
3. 任何 PR 分支都从上游 `stable`（`45feeffe`）建；提交前用 `git diff --stat` 确认 diff 里没有 `paper/` 与 4 个个人 md；PR 正文学 `#53` 的 Summary/Background/Tests 三段式。
