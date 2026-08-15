# RAVEL-Unified 参数与公式审计

> 审计对象：`RavelUnifiedGlobalScheduler`（mixed WorkYield + dynamic Chunked Prefill；pure workflow OnTimeSet + soft admission）生效路径。更新日期：2026-08-12。
> 参数分为：系统测量量、实验协议、operator policy、控制器计算预算和 legacy fallback。

## 1. 系统测量量

| 参数 | 来源 | 数学含义 |
|---|---|---|
| `prefill_tpot_curve[n]` | `calibrate_service_profile.py` 流式 TTFT 回归 | active Decode sequences 为 n 时，Prefill 每 token 边际时间 |
| `prefill_intercept_curve[n]` | 同一回归 | 每个待 Prefill 请求的固定服务时间 |
| `decode_tpot_curve[n]` | 流式 Decode profile | active sequences 为 n 时的实测 TPOT |
| `RTT(client,cluster)` | directed topology / 网络测量 | Router quote 中的客户端首 token 往返传播时间；physical 模式不再人工注入 |
| `block_size` | vLLM `--block-size` | APC block 的 token 数 |
| `max_num_batched_tokens` | vLLM 配置 | 单次 scheduler iteration 的 token budget |
| `max_num_seqs` | vLLM 配置 | 引擎最大 active sequence 数 |
| `kv_cache_size_per_token` | 模型 config 推导 | `layers × KV_heads × head_dim × K/V × dtype_bytes` |
| `kv_cache_blocks` | 应由引擎报告 | vLLM 实际可用 GPU KV blocks；未提供时仅构造 active-sequence shadow bound |

Topology 支持每个集群通过 `service_profile` 引用独立 JSON，并按 active sequences 线性插值。
`run_cluster.py` 默认拒绝 inline 速率；旧的 `0.0018/0.0185` 只能在 legacy/smoke 运行中
显式传入 `--allow-inline-service-profile` 使用，不能产生正式论文结果。
`network_delay_mode=physical` 用于真实 WAN；`synthetic` 只用于同机
仿真并在请求/响应各 sleep RTT/2。它是实验协议，不是 policy 参数。

## 2. 核心公式

对请求 i、副本 r：

```text
W = engine_prompt_debt + EDF-before prompt debt + newcomer_prompt
N = engine_pending_count + EDF-before request count + 1
mu_ttft = elapsed + RTT + N × intercept_prefill(n_running)
          + W × slope_prefill(n_running)
n_decode^U = 当前 target 上 queued/pending/running/newcomer 的去重上界
mu_e2e = mu_ttft + max(0, q_output - 1) × tpot_decode(n_decode^U)
B_objective = mu_objective + Q_(1-epsilon,type)(actual_objective - mu_objective)
B_latency_full = max(B_ttft, L_ttft × tpot_decode(n_decode^U) / L_tbt)
```

- `tau_prefill(n)` 已经在 n 条 Decode 流下测量，因此不再额外叠加 Decode tax，避免重复计费。
- generic 请求的 `q_output` 来自已完成历史请求的在线分位数；样本不足时使用公开固定 fallback。纯多阶段 workflow 可使用独立 training trace 生成的 `(stage_id, stage_num)` expected/q95 profile。两条路径都不读取当前请求真实输出；`q_output-1` 因 TTFT 已包含首 token。
- Prefix shadow hit 不减少 `w_before`，只作为 locality tie-break。没有 engine telemetry 时不声称 KV residency。
- Residual 按 SLO 类型隔离：latency 在 first-token 时记录 `TTFT-point_TTFT`；throughput/collective 在 completion 时记录 `E2E-point_objective`。不能减去已经包含 guard 的预测。
- Guard 使用 conformal-style order statistic `ceil((N+1)(1-epsilon))`。滚动在线样本只有在残差近似 exchangeable 时才有 conformal coverage；否则它是可审计的经验分位数，不是无条件概率保证。
- rank 尚不可计算时显式记录 `risk_calibrated=false`，并标记为 best effort。
- 正式 latency feasibility 同时要求 `B_ttft<=L_ttft` 与 `tpot_decode<=L_tbt`。TBT 项是
  profile point estimate，尚无 token-level residual，因此不声称 TBT tail 概率保证。

## 3. Operator policy

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `risk_epsilon` | 0.05 | 允许的同类型 objective residual tail risk；论文需做 sensitivity |
| `output_quantile` | 0.90 | throughput/collective Decode demand 的保守分位数 |
| `SLO profile` | JITServe paper | 外部服务目标，不由 RAVEL 学习 |
| fallback output hint | 256 | 冷启动公开先验；必须报告 sensitivity |
| `soft_admission_reserve_sequences` | 0（自动） | 0 表示保留一个 replica 的 `max_num_seqs`；显式值是 operator 容量策略，不允许按数据集选择 |

## 4. 控制器计算预算

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `mobile_candidate_limit` | `max_num_seqs` | 每次线性 cohort assignment 最多考虑的 EDF 候选数 |
| `mobile_beam_width` | 256 | 历史 beam 变体兼容参数；不进入 Unified 的 WorkYield/OnTimeSet 主 planner |
| `residual/output history window` | 256 | 有界 Router 内存预算 |

候选上限由部署 sequence capacity 派生。mixed WorkYield 在排序后摊销扫描
fixed prefix，复杂度为 `O(N log N + F log F + NR + F)`；pure OnTimeSet 的
victim/set 扫描最坏为 `O(N^2 R)`。两者都不在 arrival 路径运行 beam search，
且 `N <= max_num_seqs`；history window 只限制在线统计内存。

## 5. 派发与移动不变量

- 每个已派请求在创建网络任务前原子写入 Replica ledger。
- RAVEL 只有在 `request_work <= current_budget` 时派发，不允许最后一个请求把 budget 变负。
- 默认 `pending_limit=max_num_seqs`。
- 默认 `replica_budget=max_model_len × pending_limit`，是 pending prompt debt 的结构上界；`max_num_batched_tokens` 只是单 iteration budget，不能当总队列容量。
- mixed 初始放置和 cohort 规划使用固定 lexicographic 风险层：feasibility、pending overflow、objective lateness、TTFT lateness、LATENCY token-period virtual finish、profile-service virtual finish，然后才比较 predicted objective、凸边际工作/locality。不存在 Queue/RTT/Prefix 加权总分。
- LATENCY virtual charge 为 `decode_tpot_r(1)/TBT_SLO_i`（无量纲）；service virtual charge 为该请求在 replica 上的 profiled Prefill+Decode 秒数。rebind 时旧 owner 减一次、新 owner 加一次，不随 completion 释放，因为它们是长期 list-scheduling ledger，不是实时 Queue。
- MobileTriggered hold 为 `min(remote_RTT, request_slack)`，仅在观察到连续两次到达间隔不超过远端 RTT时启用。
- Latency 请求不 hold；只有存在 remote SLO-feasible quote 时才允许参与未物化 KV 的重新分配。
- 不可移动固定队列中 EDF 更早的请求，其完整 Prompt debt/request intercept 必须进入 planner 的 `work_before/requests_before`；所有固定队列请求占用 future pending slot。
- 凸边际项为 `((W+s)^2-W^2)/SLO^2`，是二次拥塞势函数的离散增量，不是拟合权重。
- pure workflow 只有在 `outstanding >= total_capacity - one_replica_capacity` 时启用 soft admission；deferred 请求最晚在自己的 objective deadline 释放，所有请求最终执行。

## 6. Prefix 与指标边界

- `prefix_hit_prompt_tokens` 保留为兼容列，语义是 Router shadow estimate。
- 新增 `estimated_prefix_hit_prompt_tokens` 和 `prefix_hit_source=router_shadow_locality_hint`。
- `actual_num_prefill_tokens` 是旧兼容列；新增 `admission_prompt_charge_tokens` 明确表示完整 Prompt 的保守 ledger charge，不是 vLLM cached-token 真值。
- TTFT 与 objective 均分开记录 point/guarded 值；`objective_prediction_residual_s` 记录对应 SLO 目标误差，`risk_calibrated` 表示该副本、该类型的经验边界是否可计算。
- TPOT 使用 `(E2E-TTFT)/(actual_output_tokens-1)`。
- TBT 来源标记为 `stream_chunk_interarrival`；没有 token timestamp telemetry 时不能声称严格逐 token TBT。

## 7. Legacy / 非主路径参数

`candidate_limit/frontier_per_replica/work_compat_ratio/decode_horizon_tokens`、旧
`mobile_*_fraction`、`region_risk_hysteresis`、`mobile_beam_width` 只属于历史
变体、兼容 CLI 或死路径，不应被解释成 Unified 主策略的拟合参数。
内部路由基础类仍只接受一个 scalar Prefill slope；`run_cluster.py` 从同一
topology 中取最慢集群的 idle slope，不使用隐藏服务速率常数。

## 8. 正式 profile 与部署边界

每个 schema-v2 profile 必须：

- 在 endpoint 所在地域内测量，避免将 WAN RTT 混入服务曲线；
- 绑定 GPU、模型、vLLM、dtype、engine dimensions、APC、Chunked Prefill
  和 eager mode；
- 覆盖 prompt 256..12000 tokens 与至少四档并发；
- 每轮先 reset Prefix Cache，并使用 run-unique 首块 nonce；
- Prefill 回归 `R²>=0.90`；
- 保存原始样本、残差、离散度、engine fingerprint 和 calibrator hash。

部署时仍有三项必须明确：

1. 一个 profile 只代表相同 fingerprint 的 service class；异构 replica
   必须分别校准；
2. 未取得 vLLM 实际 KV block count 时，Prefix 只能作为 shadow locality
   hint，完整 Prompt 仍进入 admission/service debt；
3. affine profile 是 point model。必须继续报告 held-out objective residual、
   经验 coverage 和 drift，不能由 `R²` 直接推出 SLO 保证。
