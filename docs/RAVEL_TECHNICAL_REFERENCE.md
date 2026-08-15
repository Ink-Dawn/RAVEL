# RAVEL 技术参考

版本：2026-08-12
正式策略：`RAVEL-Unified` / `ravel_unified`
实现入口：`src/dualmap/scheduler/global_scheduler/ravel_unified_global_scheduler.py`

本文档只描述当前可发布实现。旧的 Credit、Externality Twin、Exact-State、
Sidecar-Abort 和 MobileTriggered 版本仍可用于消融，但不属于正式 RAVEL
契约。

## 1. 目标与结论边界

RAVEL 的主要目标是最大化相同请求集合上的 request-level SLO attainment：

```text
SLO attainment = number of requests meeting their own SLO / all requests
```

请求类型的成功条件与 JITServe evaluator 一致：

- `LATENCY`：TTFT 不超过 `slo[0]`，且每个观测 TBT 不超过
  `slo[1]`；
- `THROUGHPUT`：E2E/TTLT 不超过 `slo[2]`；
- `COLLECTIVE`：每个 branch 的 E2E/TTLT 不超过 `slo[2]`；
- workflow-level Task SLO 是独立诊断项，不可与 request-level SLO
  混写。

SLO attainment 与平均/尾部延迟不是同一个目标。尤其是
completion-only workload，保护更多按时完成请求可能增加等待和 TTFT；
因此正式报告必须同时给出 TTFT/E2E tail，不能用一个指标替代另一个。

## 2. 不可违反的实验与信息契约

### 2.1 Router 可见

每个请求到达时，Router 可读取：

- prompt token 数和 token/block hash；
- request type；
- `stage_id`、`stage_num` 和 branch 元数据；
- client region；
- TTFT/TBT/TTLT SLO；
- 声明的 routing output hint；
- 当前 wall-clock time；
- topology、RTT 和已校准 service profile；
- 已经发生的 dispatch、first-token、completion 和聚合 engine 状态。

### 2.2 Router 不可见

Router 禁止读取：

- 测试 trace 的真实 `output_len`；
- 未来 arrival；
- 未来 cache eviction；
- 数据集名称作为策略分支；
- hindsight assignment；
- 运行结束后才知道的指标。

实验客户端为复现 trace，会将 trace `output_len` 发送给 vLLM，并设置
`ignore_eos=True`。该字段保存在 Request 中供数据面与评估使用，但正式
Router 只使用独立的 routing hint 或训练 profile。所有新增策略必须通过
输出防火墙测试。

### 2.3 公平后端

所有策略必须使用相同：

- 模型、GPU 服务类、vLLM 版本和 dtype；
- `max_model_len`、`max_num_seqs`、`max_num_batched_tokens`；
- block size；
- APC=on；
- Chunked Prefill=on；
- 相同 cache reset、trace 和 endpoint；
- open-loop 绝对时间回放。

真实跨域 endpoint 使用 `network_delay_mode=physical`：topology RTT 进入
Router quote，但数据面不再人工 sleep。所有 endpoint 位于同机、需要模拟
跨域传播时才使用 `synthetic`，在请求和响应路径各注入 RTT/2。该模式必须
进入 cell fingerprint，禁止在同一比较中混用。

RAVEL 不通过关闭 baseline 的 APC、降低 baseline 到达率或拒绝困难请求
获得优势。

## 3. 系统架构

```text
Client / trace replay
        |
        v
Central Router
  - topology and service profiles
  - Router-owned reversible queues
  - online engine/accounting state
  - RAVEL-Unified planner
        |
        +------------+-------------+
        |            |             |
        v            v             v
 Cluster A       Cluster B      Cluster C
 Sidecar/vLLM    Sidecar/vLLM   Sidecar/vLLM
```

请求在 Router/Sidecar 的 `WAITING_MOVABLE` 状态可以更改 metadata
placement。创建最终 HTTP task 后即进入不可逆边界；正式实现不迁移已经
物化的 KV，也不需要修改 vLLM 源码。

## 4. Service Profile

### 4.1 为什么必须校准

GPU、模型、dtype、vLLM、APC、Chunked Prefill 和并发配置都会改变
Prefill/Decode 服务曲线。代码内固定 token/s 无法跨机器成立。

每个服务类生成 schema-v2 profile：

- 多个 prompt 长度；
- 多个 active-sequence 档位；
- 每档重复样本；
- Prefix cache reset 与唯一 nonce；
- Prefill affine fit；
- Decode TPOT 分布；
- 完整 engine fingerprint。

### 4.2 Prefill 曲线

对并发档位 n，校准器拟合：

```text
P_n(p) = b_n + a_n * p
```

其中：

- `p`：API tokenizer 计算的 prompt token 数；
- `a_n`：`seconds_per_token`；
- `b_n`：`intercept_s`；
- 负截距投影为零后重新过原点拟合；
- profile 保存样本数、R-squared、MAE 和 p95 absolute residual。

运行时在相邻并发档位线性插值；低于/高于已测范围时夹到端点，不做
无界外推。

### 4.3 Decode 曲线

对 active sequence 数 n，profile 保存：

```text
D_n = measured post-first-token time / (completion_tokens - 1)
```

并保存 mean、median、p95、stdev 与逐 token 最大间隔统计。Router 使用
插值后的 `D_n` 估计 completion workload；adapter 的 Chunked Prefill
保护使用 Prefill 保守包络。

### 4.4 Adapter 保守包络

`scripts/start_vllm_ravel.sh` 将 profile 路径传给 adapter。adapter 取：

```text
a_guard = max_n a_n
b_guard = max_n b_n
```

因此对 profile 中任意 n：

```text
b_guard + a_guard * p >= b_n + a_n * p, for p >= 0
```

这是从校准曲线确定性推导，不是每台机器重新搜索策略参数。原始环境变量
`RAVEL_PREFILL_SECONDS_PER_TOKEN` 和 `RAVEL_PREFILL_INTERCEPT_S`
只保留为显式运维覆盖。

## 5. 基础状态与工作量

### 5.1 Prompt work

没有 engine-certified cached-token telemetry 时：

```text
prompt_work(i, r) = full_prompt_tokens(i)
```

Router shadow Prefix 只计算：

```text
estimated_hit = prompt_tokens - estimated_recompute_tokens
```

该值仅作确定性 tie-break，不从可行性工作量中扣除。这样避免把 shadow
locality 误当成真实 APC residency。发布实验的 vLLM APC 始终开启。

### 5.2 Engine 与 Router debt

对 replica r：

- `engine_work_r`：当前 engine 尚未完成的实际 pending prompt token；
- `queued_work_r(i)`：Router 队列中 deadline 不晚于请求 i 的 prompt
  work；
- `queued_count_r(i)`：对应请求数；
- `running_r`、`pending_r`：已观测 engine sequence 数；
- 当前决策中刚分配的 work 会立即进入 planning ledger，不等待 heartbeat。

一个请求 ID 在 Router queue、engine pending 和 engine running 集合中去重，
避免 prospective decode occupancy 重复计数。

## 6. Replica Quote

对请求 i 和 replica r，定义：

```text
p_i       = full prompt tokens
a_r(n)    = interpolated Prefill seconds/token
b_r(n)    = interpolated Prefill intercept
d_r(m)    = interpolated Decode TPOT
RTT_ir    = client-region to target-cluster RTT
elapsed_i = now - arrival_i
```

Prefill/等待估计：

```text
service_ir =
    (engine_work_r + queued_work_r(i) + p_i) * a_r(running_r)
  + (engine_pending_r + queued_count_r(i) + 1) * b_r(running_r)

TTFT_point_ir = elapsed_i + RTT_ir + service_ir
```

profile 的 Prefill 曲线已经在相应 Decode 并发下测量，因此不再额外加一个
任意的 Decode interference tax，避免双重计数。

对 `LATENCY`：

```text
objective_point_ir = TTFT_point_ir
```

对 completion 类型：

```text
objective_point_ir =
    TTFT_point_ir + max(0, E[L_i] - 1) * d_r(m_ir)

objective_risk_ir =
    TTFT_point_ir + max(0, U[L_i] - 1) * d_r(m_ir)
```

其中 `m_ir` 是接纳后 prospective sequence count；`E[L_i]` 和
`U[L_i]` 分别是请求可见的期望与保守输出 workload。

可行性：

```text
feasible(i, r) = objective_risk_ir <= request_SLO_i
```

### 6.1 Residual calibration

完成请求产生端到端 residual：

```text
e = observed_objective - predicted_objective
```

每个 replica/request type 保存有限窗口。风险分位数采用有限样本 rank：

```text
rank = ceil((n + 1) * (1 - epsilon))
```

rank 不存在时明确标记 uncalibrated。当前正式配置记录 residual，但
`residual_decision_enabled=False`，即不在冷启动阶段用不足样本构造虚假
保证。该项属于在线诊断与后续风险实验，不是当前主结果的隐藏加分项。

## 7. RAVEL-Unified 的两条语义路径

策略选择只依赖请求可见 SLO 语义：

```text
pure multi-stage COLLECTIVE cohort -> completion path
any non-workflow / LATENCY observed -> mixed TTFT/TBT path
```

不读取 `lmsys`、`burst` 或 `deepresearch` 名称。

### 7.1 Mixed TTFT/TBT path

候选按 absolute deadline 排序。该路径不使用加权总分，而是先比较硬风险
层，再比较两个具有明确物理意义的虚拟 admission ledger。

对 LATENCY 请求，定义一个 token 周期的归一化 Decode 需求：

```text
u_ir = d_r(1) / TBT_SLO_i
A_r  = sum of u_jr for LATENCY requests admitted to r
```

`d_r(1)` 是 profile 中 replica `r` 在单 sequence 下生成一个 token 的时间，
单位为秒；`TBT_SLO_i` 也是秒，所以 `u_ir` 无量纲，表示该请求消耗的一个
token-period 服务份额。`A_r + u_ir` 是 LATENCY admission 的 virtual finish。

对 mixed traffic 中的所有请求，定义：

```text
s_ir = profiled_prefill_seconds_ir + profiled_decode_seconds_ir
L_r  = sum of s_jr for requests admitted to r
```

`s_ir` 和 `L_r` 的单位都是秒。`L_r + s_ir` 是经典 greedy list scheduling
中的 virtual completion load，用于避免生命周期回调尚未返回时连续把请求
倾倒到同一 replica。它不是实时 Queue 预测，完成时不扣减；rebind 时旧
owner 的 charge 被精确移除，再加入新 owner，因此每个请求在 ledger 中恰好
计数一次。该累计量只比较 replica 间差值，系统重启或 scheduler 重建时归零。

风险项定义为：

```text
objective_lateness_ir = max(0, objective_ir - SLO_i) / SLO_i
ttft_lateness_ir      = max(0, TTFT_ir - TTFT_SLO_i) / TTFT_SLO_i
```

cohort planner 对每个 replica 使用以下 lexicographic key：

```text
(
  infeasible,
  pending-overflow cost,
  objective_lateness,
  ttft_lateness,
  A_r + u_ir,
  L_r + s_ir,
  predicted objective / SLO,
  convex marginal service work,
  moved,
  replica_id
)
```

凸边际项为：

```text
M_ir = ((W_r + own_service_ir)^2 - W_r^2) / SLO_i^2
```

它来自凸负载势函数的离散增量，不需要人为加权系数。初始单请求路径在
相同风险和 virtual finish 后使用当前 pressure、Prefix locality 与稳定
replica ID 打破平局；cohort 路径使用上式的凸边际项。Prefix 不越过
feasibility、lateness 或 virtual-load 层。

因此顺序具有明确含义：先避免可规避的 Full-SLO/TTFT 违约和 engine
overflow，再平衡 TBT 周期份额与 profile 服务秒数，最后才优化平均目标和
移动/locality。这里没有按 workload 名称选择权重，也没有通过搜索得到的
线性系数。

本地 vLLM priority 只对 LATENCY 请求编码 absolute deadline 与 TBT；同一
mixed cohort 中的 completion/flexible 请求保持 priority 0 和稳定 FCFS。
这是因为把所有 TTLT deadline 编成 EDF 会在持续 arrival 下长期推迟宽
deadline 请求，恶化 TTFT，却不一定提高其 completion SLO。

每个请求最多在 KV 物化前 rebind 一次。没有合法 target 属于实现错误，
不会静默丢弃请求。

### 7.2 Completion workflow path

completion-only 流量使用近似 maximum on-time set：

1. 按 absolute deadline 扫描；
2. 只将 quote feasible 且 Decode/TBT set feasible 的请求放入保护集；
3. 若保护集已满，只有当 newcomer 可行且替换当前最大 profiled service
   victim 能释放正服务量时才替换；
4. 被替换或当前不可行请求标记 deferred，不拒绝；
5. 第二遍尝试把 deferred 请求填入其他可行 replica。

这对应单机 Moore-Hodgson 思想在异构多 replica 上的确定性近似，不声明
求得 NP-hard 多机带网络时延问题的精确最优解。

### 7.3 Deadline-bounded soft admission

只有纯 multi-stage completion 流量且系统接近 sequence saturation 时启用：

```text
total_capacity = replicas * pending_request_limit
reserve         = one replica-equivalent capacity by default
activation      = total_capacity - reserve
```

`reserve` 来自部署 `max_num_seqs`，不是 GPU 型号常数，也不按数据集
配置。

保护集请求优先进入 engine。预测 miss 的 deferred 请求在以下条件同时满足
时留在可逆队列：

- engine 或 Router 队列仍有 protected 请求；
- 当前时间早于 deferred 请求自己的 objective deadline。

释放时间严格定义为：

```text
release_at_i = arrival_i + request_SLO_i
```

到达 `release_at_i` 后必须提交，不能无限饥饿。所有请求最终执行，因此
SLO 提升不是 rejection gain。该规则表示“已预测为迟到的工作让位于仍可
挽救的工作，但最多让到自己的 deadline”，不含经验性的 1.4/1.8 倍常数。

## 8. Workflow 输出语义 profile

真实输出长度在到达时未知。对多阶段 workflow，可从独立 training trace
生成 schema-v2 profile：

```text
context = (stage_id, stage_num)
expected = training mean
upper    = finite-sample nearest-rank q95
fallback = stage_id aggregate, then declared routing hint
```

`stage_num` 在请求到达时已知，能够区分同一 stage 在四阶段与六阶段
workflow 中不同的语义角色。profile 文件绑定 training source SHA256，
并标记 `training_only=true`。

在普通负载和 soft-admission 激活前，upper 用于 completion 可行性风险需求。
在 soft-admission 饱和区，maximum-cardinality 保护集使用 expected workload，
q95 仍记录为诊断；否则过度保守的 q95 会把大量可挽救长阶段请求提前判死。

该 profile 不是必须项。未知 context 回退到 stage aggregate，再回退到
声明 hint，不允许回退到真实测试输出。

## 9. vLLM Adapter 与动态 Chunked Prefill

Router 不能把本机 `perf_counter` absolute deadline 直接发送到另一台主机，
因为不同主机的 monotonic clock epoch 不一致。当前 phase-aware 协议编码的是
dispatch 时剩余 deadline budget、可选 TBT，以及 completion 请求的
profile-derived Decode service demand。engine 以本地 arrival clock 重建
deadline：

```text
local_deadline = engine_arrival + remaining_budget
```

普通 vLLM priority 不在该 signed-int64 保留负值区，adapter 对其保持 inert。
`RAVEL-Unified-Balanced` 使用 phase-aware Prefill EDF/SPT，但不提前下放
Router 已标记为 predicted miss 的请求。

另保留显式实验策略 `RAVEL-Unified-Standby`。它允许 predicted miss 在
Router deadline 前进入 engine waiting queue，并用 deferred bit、Prefill
service 与 held-out output q95 推导的 Decode occupancy 约束机会执行。该策略
是 latency-oriented 实验变体，不是发布默认值。

endpoint 重启后的首个请求会受到 CUDA/kernel/allocator 冷状态影响。正式
比较必须先执行所有策略共享、且不计入结果的统一 warm-up，随后重置 APC，
再开始交错测量；只调用 `/v1/models` 不能替代执行路径 warm-up。

若 active Decode 中存在 latency request，取最紧 TBT：

```text
T = min active latency TBT
a = max profile prefill seconds_per_token over calibrated concurrency
b = max profile prefill intercept over calibrated concurrency
raw = floor((T - b) / a)
chunk = floor(raw / block_size) * block_size
chunk = clamp(chunk, block_size, max_num_batched_tokens)
batch_budget = ceil((active_decode_sequences + chunk) / block_size)
               * block_size
```

`a,b` 取整个校准并发曲线的保守上界，不是代码常数。adapter 只在一次
`_schedule_chunked_prefill` 调用期间将 `max_num_batched_tokens` 临时收紧为
`min(original_budget, batch_budget)`，finally 恢复原值。没有 latency TBT
时完全调用 vLLM 原调度逻辑。

若 `T <= b + a * block_size`，一个最小 block 本身已经超过 profile TBT
预算；adapter 仍必须给 vLLM 至少一个合法 block，但这属于物理不可行状态，
不能宣称 TBT 可保证。Router 的 TBT feasibility/virtual admission 应在进入
该状态前分流；实验仍把发生的 violation 计入分母。

adapter 采用 feature detection：若目标 vLLM 没有
`Scheduler._schedule_chunked_prefill`，启动立即失败，不静默运行一个
未经验证的兼容模式。这样可以兼容多个具有该接口的版本，同时避免假装
任意版本都安全。

## 10. 状态机与不变量

请求主要状态：

```text
ARRIVED
  -> WAITING_MOVABLE
  -> HTTP_SUBMITTED / PREFILL_LOCKED
  -> FIRST_TOKEN
  -> COMPLETE | FAILED
```

必须满足：

1. 一个请求最多创建一个有效最终数据面 submission；
2. rebind 只能发生在 `WAITING_MOVABLE`；
3. request ID 在 planning state 中恰好计数一次；
4. first-token 释放 Prefill reservation，但请求继续贡献 Decode pressure；
5. completion/failed 清理 remaining state；
6. failed、timeout 和 no-feasible 请求仍进入结果，不从分母删除；
7. soft-deferred 请求最晚在自己的 objective deadline 释放；
8. Router 选择不得访问测试 `output_len` 或 workload 名称；
9. Prefix shadow hit 不从保守 prompt work 扣除；
10. topology、profile、trace、source 和参数均写入 cell manifest。

## 11. 复杂度与通信

设当前可移动 cohort 大小 N，replica 数 R。

- Mixed WorkYield planner：设不可移动 fixed prefix 共 F 条，复杂度为
  `O(N log N + F log F + N * R + F)`；
- on-time set 的直接实现包含 victim/TBT-set 扫描，最坏
  `O(N^2 * R + sorting)`，但 N 被 `max_num_seqs` 派生的 candidate limit
  限制；
- 每个请求最多一次 pre-KV rebind；
- 不发送逐 token Router 控制事件；
- 生命周期反馈是 first-token 与 completion 的批量/稀疏事件；
- 控制事件总量随请求数增长，而非随生成 token 数增长。

通信复杂度是渐进上界，不等于瞬时带宽永远平滑。Sidecar 必须继续使用
batch、coalescing、token bucket 和 bounded queue 吸收 completion burst。

## 12. 参数来源

### 12.1 系统参数

由部署者声明并写入 fingerprint：

- model path / served model name；
- vLLM version、dtype；
- `max_model_len`；
- `max_num_seqs`；
- `max_num_batched_tokens`；
- block size；
- APC、Chunked Prefill、enforce eager；
- replica layout 与完整 directed RTT matrix。

### 12.2 测量参数

由 service calibration 得到：

- Prefill slope/intercept curve；
- Decode TPOT curve；
- sample count、R-squared 与 residual；
- profile source SHA256。

### 12.3 请求语义参数

由请求/JITServe SLO profile给出：

- request type；
- TTFT/TBT/TTLT SLO；
- stage metadata；
- prompt tokens；
- client region。

### 12.4 容量推导参数

- mobile candidate limit = `max_num_seqs`；
- soft-admission total capacity =
  `replicas * pending_request_limit`；
- default reserve = one replica-equivalent `pending_request_limit`；
- adapter chunk = TBT 与 service-envelope 的 block-aligned 解。

### 12.5 固定 policy 参数

- `risk_epsilon=0.05`：residual rank 的风险水平；
- `residual_window=256`：有界在线诊断窗口；
- `soft_release_policy=slo_deadline`：唯一正式释放语义；
- `soft_reserve_sequences=0`：0 表示按容量自动推导，不表示无 reserve。

不存在按 dataset name 选择的 speedup、release ratio、Prefill rate 或
Decode rate 表。

## 13. 回放与结果有效性

trace speedup 使用 open-loop 绝对目标时间：

```text
target_i = replay_start + (trace_ts_i - trace_ts_0) / speedup
```

生成器只等待到 `target_i`，不会等待上一个请求完成或等待 Router
schedule 返回后再开始计时。若进程因调度延迟错过目标，立即并发提交；
不能通过 Router 计算变慢来自我限流。

每个 cell 至少验证：

- 500/500 request rows；
- prompt token 精确；
- output token 精确；
- TBT coverage 精确；
- endpoint 成功；
- source/input/topology/profile fingerprint；
- cache reset；
- no rejection；
- actual arrival span。
- network delay mode（physical 或 synthetic）。

报告列：

- A/B/C placement；
- SLO 和 Task SLO；
- TTFT mean/p20/p50/p95/p99；
- E2E mean/p20/p50/p95/p99；
- TPOT mean；
- Router wait；
- Prefix tokens/request 与 shadow hit ratio；
- Prompt token/s。

最优值的粗体必须在相同 dataset/speedup cell 内比较，SLO 越高越优，
latency 越低越优；placement 与 Prefix 指标不自动判定“越大越优”。

## 14. 跨服务器部署

推荐流程：

1. clone 仓库并安装 Router 依赖；
2. 在每种 GPU/service class 上启动 vanilla vLLM；
3. 填写完整 directed RTT matrix；
4. 在每个地域本地运行 `calibrate_local_service_profile.py`；
5. 检查 profile quality gate；
6. 用 `build_cluster_topology.py` 将 profile 和 RTT 组装为 topology；
7. 用 `start_vllm_ravel.sh --profile ...` 重启正式 backend；
8. 运行 `validate_deployment.py`；
9. 运行 `RAVEL-Unified`；
10. 任一 engine fingerprint 字段改变后重新校准。

详细命令见 `docs/DEPLOYMENT.md`。

## 15. 部署兼容边界

换成 4090、其他模型、dtype、vLLM 或 engine dimensions 后，旧 profile
和旧效果数字都不再有效。新部署不需要修改算法常量：每个地域本地重新
校准，组装新的 topology，再通过启动与 endpoint 验证即可。

## 16. 已知限制

- Router 没有 engine-certified KV residency，Prefix 只能是 locality hint；
- adapter 依赖 vLLM 的私有 `_schedule_chunked_prefill` 接口，版本升级必须
  重新跑兼容测试；
- completion soft admission 会改善按时完成数，但可能增加等待和 TTFT tail；
- maximum-SLO 与 minimum-flow-time 在 completion-heavy overload 下存在真实
  冲突：提前执行预测必迟到请求会降低它们的 TTFT，却会消耗仍可挽救请求的
  Prefill/Decode capacity；
- `RAVEL-Unified-Standby` 仅为实验性 Pareto 策略，不能替换发布默认值；
- schema-v2 workflow profile 假设训练与部署的 stage semantics 可迁移；
- rolling residual 在 drift 下不提供无条件 coverage theorem；
- 任意性能结论都只适用于其 manifest 绑定的 trace、profile、topology、
  source hash 和回放协议。

## 17. 代码索引

- 统一策略：
  `src/dualmap/scheduler/global_scheduler/ravel_unified_global_scheduler.py`
- 基础 quote / WorkYield / OnTimeSet：
  `src/dualmap/scheduler/global_scheduler/ravel_native_global_scheduler.py`
- engine priority protocol：
  `src/ravel_engine_adapter/protocol.py`
- vLLM feature patch：
  `src/ravel_engine_adapter/vllm_patch.py`
- startup hook：
  `runtime/ravel_vllm_adapter/sitecustomize.py`
- service calibration：
  `scripts/calibrate_service_profile.py`
- same-host calibration helper：
  `scripts/calibrate_three_cluster_profiles.py`
- per-region calibration：
  `scripts/calibrate_local_service_profile.py`
- topology assembly：
  `scripts/build_cluster_topology.py`
- workflow profile：
  `scripts/calibrate_workflow_output_profile.py`
- deployment validation：
  `scripts/validate_deployment.py`
- profile/launch fingerprint validation：
  `scripts/validate_profile_launch.py`
- formal startup：
  `scripts/start_vllm_ravel.sh`
- matrix runner：
  `scripts/run_jitserve_three_cluster_matrix.py`
- SLO evaluator：
  `src/dualmap/cluster/slo.py`
- core tests：
  `src/tests/test_ravel_unified_scheduler.py` and
  `src/tests/test_ravel_engine_adapter.py`
