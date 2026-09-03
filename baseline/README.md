# Workload、Baseline 与 Calibration 口径

本文件只整理论文复现口径，不复制原始 trace、profiling log 或实验结果目录。

## Workload

三类 trace 的直接来源都是 [JITServe artifact](https://github.com/UIUC-MLSys/JITServe)。
冻结文件与 SHA-256：

| Workload | 文件 | SHA-256 |
|:--|:--|:--|
| LMSYS | `data/lmsys_first500.json` | `3e511beaa165b70234a5d60329633c1c4efd1d6a839770244fe96e93129508a4` |
| Burst | `data/burst_first500.json` | `37cc55c902ac7001855a619fbec1ff4eb07b099376a549e28f7de7282b39c65c` |
| DeepResearch | `data/deepresearch_flat_first500.json` | `389593fda54288d32e83a4229fc3752705996ffa656710aba7ed671c9a1fcd9c` |

### LMSYS

- 从 JITServe 的 `traces/lmsys.json` 按文件顺序直接取前 500 条：
  `json.loads(... )[:500]`。
- 不是随机采样，没有过滤、分层或重采样。

### Burst

- request content 与 LMSYS 的前 500 条完全相同，只替换 `deliver_time`。
- timestamp 来自 [BurstGPT](https://github.com/HPMLL/BurstGPT) 的
  `BurstGPT_1.csv` 前 500 行 `Timestamp`；不是 Poisson 或其他随机合成流。
- 对第 `i` 条请求和 load speedup `s`，实际 replay 时间为：

```text
t_i(s) = max(0, Timestamp_i - Timestamp_0) / (1000 * s) 秒。
```

BurstGPT 官方 schema 把 `Timestamp` 定义为秒；当前构建器原样保存数值差，
replay clock 则把它当作毫秒，因此上式包含一次固定 1000x 压缩。之后的
`s` 只按比例压缩同一条相对时间线，不改变 burst pattern。

### DeepResearch

- 来源文件是 JITServe 的 `traces/deepresearch_filter.jsonl`。
- 按文件顺序读取 workflow，直到累计 branch 数至少达到 1000。
- 每个 workflow 的 arrival 使用 `BurstGPT_1.csv` 从 offset 100 开始的
  timestamp，并减去第一个选中 timestamp；单位解释与 Burst 相同。
- branch 到达时间：

```text
deliver_time_ms(w,b) = burst_arrival_ms(w)
                     + 1000 * max(0, start_time(w,b) - workflow_start(w))
```

- 所有 branch 按 `(deliver_time, collection_id)` 全局排序后取前 500。
- 这 500 条来自 168 个 workflow，其中 154 个因截断而不完整，所以只报告
  request-level SLO，不报告 workflow-level Task SLO。

Stage metadata 分布：

| 字段 | 分布 |
|:--|:--|
| `stage_id` | 0: 168；1: 248；2: 50；3: 34 |
| `stage_num` | 4: 192；6: 308 |
| `branch_id` | 0: 413；1: 62；2: 21；3: 2；4: 2 |
| `(stage_id, stage_num)` | (0,4): 59；(0,6): 109；(1,4): 98；(1,6): 150；(2,4): 21；(2,6): 29；(3,4): 14；(3,6): 20 |

每个已选 workflow 有 2--8 个 branch，中位数 3、均值 2.976；全局前 500
截断后没有 `stage_id=4/5`。

### Input/output token summary

Input 是实际服务口径：Qwen3-1.7B tokenizer 对单条 user message 应用 chat
template，`add_generation_prompt=True`。Output 是 trace 中请求的
`output_len`，不是模型实际生成量。P95 使用 `(n-1)*0.95` 线性插值。

| Workload | 指标 | Total | Min | Mean | Median | P95 | Max |
|:--|:--|--:|--:|--:|--:|--:|--:|
| LMSYS | input | 87,613 | 9 | 175.226 | 56 | 550 | 10,846 |
| LMSYS | requested output | 136,037 | 2 | 272.074 | 126.5 | 898.95 | 1,025 |
| Burst | input | 87,613 | 9 | 175.226 | 56 | 550 | 10,846 |
| Burst | requested output | 136,037 | 2 | 272.074 | 126.5 | 898.95 | 1,025 |
| DeepResearch | input | 216,313 | 136 | 432.626 | 187 | 1,476.8 | 6,979 |
| DeepResearch | requested output | 190,498 | 13 | 380.996 | 280 | 1,023.05 | 1,024 |

## SLO class 与 multiplier

Base budget 为 TTFT 0.8 s、TBT 0.08 s、TTLT 8 s。

- LMSYS/Burst 声明比例为 `(LATENCY, THROUGHPUT, COLLECTIVE)=(3,5,2)`。
  为匹配 JITServe 的实际可执行实现，COLLECTIVE 权重先变成 `2/7`，因此
  500 条中实际为 181 LATENCY（36.2%）、301 THROUGHPUT（60.2%）和
  18 COLLECTIVE（3.6%）。label 用 `random.Random(42)` shuffle 后按 trace
  顺序赋值。
- DeepResearch 的 500 条全部锁定为 COLLECTIVE。
- 每条请求的 multiplier 为 `m = collection_id % 4 + 1`，同时乘到 TTFT、
  TBT、TTLT。LMSYS/Burst 的 1x/2x/3x/4x 各 125 条；DeepResearch 分别为
  124/128/124/124 条。
- LATENCY 成功条件：TTFT 达标且每个已测 TBT 都达标；THROUGHPUT 和
  COLLECTIVE 成功条件：TTLT 达标。DeepResearch branch TTLT 不再乘
  `stage_num`。

## Baseline 的真实算法定义

- **RR（Round Robin）**：第 `k` 条请求发给配置顺序中的
  `replica[k mod N]`。
- **LL（Least Load）**：发给 pending input tokens 总数最少的 replica，
  相同则取最小 replica ID；不是按 outstanding request 数。
- **RD（Random Dispatch）**：每条请求调用
  `random.Random(trace_seed=42).randrange(N)` 均匀选择 replica，因此是
  可复现的伪随机序列。

## DualMap / SkyWalker

### DualMap

实验代码是依据 [DualMap 官方仓库](https://github.com/ASISys/DualMap)
（核对 commit `24816acc70b8e8f4f6b47bc47b7ce0fdddddf40d`）在统一 harness
中的 reimplementation/adaptation，不是直接运行官方仓库代码。

标准 `cluster_dualmap` 的关键规则：

- prefix 双 hash 到两个候选 cluster，每个 cluster 内最多再取两个候选 replica。
- 预测 TTFT = client RTT + pending prefill intercept/token slope；LATENCY
  最小化 TTFT，其他 class 使用 TTFT + 256-token decode hint。
- cache-affine 候选的 objective 不超过 `0.5*SLO` 时保留，否则在 feasible
  候选中取 objective 最小者。
- rebalance：queued tokens > 32768 或等待 >= 0.5 s；hysteresis 0.02 s；
  每次最多移动 8 条。



完整矩阵中 Burst 4x--16x 和 DeepResearch 2x--16x 实际使用仓库内
**DualMap-CA** adaptation，不是标准 DualMap：overload fraction 0.40、
escape ratio 0.10、cluster shares 0.40/0.45/0.15、type shares
`0.10/0.85/0.05;0.15/0.65/0.20;0.10/0.25/0.65`、warmup 10、
objective ratio `1e9`。Burst 的 share slack 为 0.0（force type 0），
DeepResearch 为 0.1（不 force type）；二者都将 rebalance wait 设为 3600 s、
token threshold 设为 `1e9`，即实际关闭 rebalance。只有两个 hash cluster
都 infeasible 时才检查非 hash cluster，且 escape 目标 objective 不超过
primary 的 0.10。其余 cell 使用标准 DualMap：LMSYS 全部、Burst 1x/2x、
DeepResearch 1x。

### SkyWalker

实验代码是按 [SkyWalker/SkyLB 论文](https://arxiv.org/abs/2505.24095)
写入统一 harness 的 reimplementation，不是官方代码。

实际参数 `pending_threshold=0`。routing 顺序是：先选 pending=0 的本地
replica，再选 pending=0 的远端 replica；集合内依次按 prefix-hit tokens
最大、pending requests 最少、running requests 最少、replica ID 最小排序。
若没有 available replica，则全局按 pending requests、actual pending tokens、
running requests、replica ID 从小到大选择。

## 完整 SLO attainment（108 cells）

每个 cell 都完成 500/500 请求。DM-CA 表示上面的 capacity-aware adaptation。

| Workload | Load | RAVEL | RR | LL | RD | DualMap variant | DualMap | SkyWalker | Best baseline | Gain |
|:--|--:|--:|--:|--:|--:|:--|--:|--:|:--|--:|
| LMSYS | 1x | 97.4 | 93.8 | 90.4 | 94.2 | DM | 96.4 | 94.2 | DM 96.4 | +1.0 |
| LMSYS | 2x | 96.6 | 92.6 | 86.0 | 94.0 | DM | 94.6 | 92.8 | DM 94.6 | +2.0 |
| LMSYS | 4x | 96.4 | 90.8 | 81.4 | 90.4 | DM | 93.6 | 86.4 | DM 93.6 | +2.8 |
| LMSYS | 8x | 96.2 | 90.4 | 81.0 | 89.0 | DM | 92.0 | 81.4 | DM 92.0 | +4.2 |
| LMSYS | 12x | 92.6 | 85.6 | 84.6 | 86.6 | DM | 90.8 | 83.8 | DM 90.8 | +1.8 |
| LMSYS | 16x | 91.6 | 81.6 | 80.6 | 72.6 | DM | 82.8 | 65.4 | DM 82.8 | **+8.8** |
| Burst | 1x | 92.8 | 76.8 | 76.6 | 82.8 | DM | 74.6 | 83.6 | SkyWalker 83.6 | +9.2 |
| Burst | 2x | 93.8 | 67.2 | 58.8 | 76.0 | DM | 69.6 | 70.2 | RD 76.0 | +17.8 |
| Burst | 4x | 87.6 | 52.6 | 54.6 | 59.6 | DM-CA | 73.0 | 62.2 | DM-CA 73.0 | +14.6 |
| Burst | 8x | 80.8 | 48.6 | 53.4 | 54.4 | DM-CA | 70.6 | 70.6 | DM-CA/SkyWalker 70.6 | +10.2 |
| Burst | 12x | 81.2 | 55.2 | 51.2 | 55.6 | DM-CA | 69.0 | 55.6 | DM-CA 69.0 | +12.2 |
| Burst | 16x | 79.2 | 59.6 | 57.4 | 52.2 | DM-CA | 68.0 | 62.0 | DM-CA 68.0 | **+11.2** |
| DeepResearch | 1x | 83.4 | 74.2 | 80.0 | 81.2 | DM | 74.2 | 80.0 | RD 81.2 | +2.2 |
| DeepResearch | 2x | 81.8 | 66.8 | 71.2 | 70.4 | DM-CA | 68.4 | 70.0 | LL 71.2 | +10.6 |
| DeepResearch | 4x | 81.4 | 40.4 | 48.8 | 47.0 | DM-CA | 67.6 | 52.4 | DM-CA 67.6 | +13.8 |
| DeepResearch | 8x | 80.8 | 40.8 | 46.4 | 45.0 | DM-CA | 67.0 | 50.2 | DM-CA 67.0 | +13.8 |
| DeepResearch | 12x | 79.2 | 40.4 | 45.0 | 46.8 | DM-CA | 65.8 | 46.6 | DM-CA 65.8 | +13.4 |
| DeepResearch | 16x | 79.0 | 38.8 | 43.4 | 47.6 | DM-CA | 65.6 | 43.2 | DM-CA 65.6 | **+13.4** |

所以 16x 的三个 headline 数字确实是对每行五个 baseline 取最大值后相减：
`91.6-82.8=8.8`、`79.2-68.0=11.2`、`79.0-65.6=13.4` percentage
points。它们是 canonical single-run cell comparison，不是另一个 Campaign-A
五次重复实验的 mean/CI。

## Calibration

- 对 `max_num_seqs=64`，background concurrency/profile points 为
  `{0,8,16,32,48,63}`，生成规则是
  `{0,N/8,N/4,N/2,3N/4,N-1}` 取整、去重、排序。
- Prefill 在每个 concurrency 下测 256/1024/4096/12000 token 四个 prompt
  size，每点各 5 次，共 20 个样本，并拟合 TTFT affine curve。
- Decode 的 total active sequence points 是 `{1,9,17,33,49,64}`：每次用
  32-token prompt 生成 256-token 的 fresh background cohort，全部进入
  Decode 后加入一个 32-token prompt、128-token output 的 probe；每点 5 次。
- calibration 与 evaluation 独立：它单独运行，只使用由固定单词
  `calibration` 和 run-unique nonce 构造的 token-controlled synthetic
  prompts，不读取三类 evaluation prompt 或其 realized output。prefix cache
  在运行前 reset。
- DeepResearch semantic output profile 也只使用标记为 `training_only=true`
  的 training split，hash 与 evaluation trace 不同。
- 论文只需保留冻结 profile JSON 中的采样点、曲线、engine fingerprint 和
  `release/CALIBRATION_MANIFEST.json` 的 hash；无需整理全部原始 profiling log。
