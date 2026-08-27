# RAVEL 篇幅与引用整合计划（仅供手工修改）

**状态：** 计划稿；本文件和 `RAVEL_compact_66_references.bib` 是新增辅助文件。没有修改 `usenix.tex` 或 `sample.bib`。

## 0. 本轮目标与审计基线

- [ ] 将正文（Abstract 至 Conclusion）压回约 12 页以内，并让 Conclusion 完整结束在参考文献之前。
- [ ] 最终正文使用约 60--70 篇真实、可追溯文献；本方案选定 **66 篇**（52 篇 core + 14 篇 optional）。
- [ ] 把引用集中在“相关工作的事实性陈述”和“实验数据/模型来源”处；RAVEL 自己的算法、公式、数值结果不靠外部文献背书。
- [ ] 保持用户已确定的数据口径：单次主矩阵与五次重跑分开；比较对象固定为 DualMap；不要重新引入 RD。

当前审计（2026-08-27）：

| 项目 | 当前 | 目标 |
|---|---:|---:|
| PDF 总页数（含参考文献和附录） | 16 | 由投稿规则决定 |
| 正文 texcount（含标题/图注约 6,600--6,700 words） | Conclusion 延伸到参考文献页 | 正文约 5,700--6,200 words |
| BibTeX 条目 | `sample.bib` 17 条 | 66 条 |
| 浮动体 | 9 个 | 不增加；只调整顺序和图注长度 |

页数目标按“正文 12 页、参考文献/附录另计”的常见系统论文排版来制定。若具体 venue 把参考文献或附录也计入页数，先确认规则：66 条引用加现有详细附录很难同时维持 12 页，此时优先压缩附录和引用条目，而不是缩小字体或改页边距。

### 建议的正文页面分配（用于编译后对照）

这是浮动体允许上下移动时的目标区间，不是强制分页：

| 正文页 | 主要内容 | 控制点 |
|---:|---|---|
| 1 | Abstract + Introduction P1--P3 + Figure 1 | Figure 1 只承担动机，不在图注重复 Problem 定义 |
| 2 | Introduction P4--P7 + contributions | 贡献 bullet 不跨栏；结果只保留 headline |
| 3 | Problem + operating-point table + Design Overview | 公式保留，解释压短 |
| 4--6 | Design：quotes、accounting、replanning | 每个机制先结论后公式；详细伪代码进 Appendix |
| 7 | Design：placement/frontier/handoff + Implementation 开头 | handoff boundary 一次讲清 |
| 8 | Implementation 结尾 + Evaluation Setup | setup 表不在正文逐项复述 |
| 9--10 | Evaluation：E2E、five-run、mechanism | 图注和正文各承担不同信息 |
| 11 | Evaluation：quote、overhead + Related Work 开头 | 结果段落使用“观察—解释—限制” |
| 12 | Related Work 结尾 + Conclusion | Conclusion 完整结束；下一页再开始 References |

如果 Figure 1 或 Figure 6/7 浮动到不理想位置，先尝试调整 `[t]`/`[tb]` 和相邻段落顺序；不要用负间距强行把图压进某页。

## 1. 独立 BibTeX 文件的使用方式

文件：`RAVEL_compact_66_references.bib`

- [ ] 手工确认文件已和 `usenix.tex` 位于同一目录（当前已放入 `paper` 目录）。
- [ ] 在 `usenix.tex` 的末尾把

  ```latex
  \bibliography{sample}
  ```

  改为

  ```latex
  \bibliography{RAVEL_compact_66_references}
  ```

- [ ] 不要同时写 `\bibliography{sample,RAVEL_compact_66_references}`：两份文件含有相同的 17 个 key，会产生重复 BibTeX key。
- [ ] 如果必须保留 `sample.bib` 文件名，则由用户手工将 compact 文件内容合并进去；本轮不改 `sample.bib`。
- [ ] 第一次合并后运行 `pdflatex → bibtex → pdflatex → pdflatex`，检查 `undefined citation`、重复 key 和参考文献排序。

## 2. 全文压缩原则（先执行，再补引用）

- [ ] 每段前 1--2 句先给结论/段落作用；删去“先铺背景、末句才点题”的过渡。
- [ ] 同一事实只保留一次：Introduction 给直觉，Problem 给形式化，Design 给机制，Evaluation 给证据，Related Work 给边界比较。
- [ ] 结果段落遵循“观察 → 解释 → 限定”三句结构；不要在图注和正文重复完整数字。
- [ ] 一组同类工作用一个 grouped citation；不要为每篇论文单独造一句背景句。
- [ ] 不把引用塞入公式、状态机、伪代码的每一行；这些是 RAVEL 的原创定义或实现事实。
- [ ] 不用 `\\vspace`、负 `\\vspace`、缩小字号或改变 `\\textwidth` 解决超页；先删重复文字、缩短图注、调整浮动体位置。
- [ ] 只在 `\clearpage` 前后检查浮动体；不要为了“填满”某一栏而增加无信息的段落。

## 3. 分章篇幅与重组方案

### Abstract（目标 190--205 words；当前约 217）

**回答的问题：** 为什么问题重要？RAVEL 的一句话洞见是什么？部署和最强结果是什么？

- [ ] 保留场景、one-shot 缺口、immediate accounting/deferred commitment、部署、三项 headline 数字。
- [ ] 删除 Abstract 中对机制的第二次解释；不要放引用。
- [ ] 修正当前标点错误：

  ```latex
  ... P95 end-to-end latency by 15.5--56.8\% at $16\times$ load.
  ```

  （删掉 `\%.,` 中多余的句点。）
- [ ] 不在 Abstract 增加五次重跑的 17.4 pp；该数字留在 Evaluation/Appendix B。

### Introduction（目标 730--780 words；当前约 877）

保留 7 段，每段只承担一个任务：

1. **场景和后果（约 100--115 words）。** 先说“capacity is request-dependent”，再给 WAN、cache、TTFT/TBT/completion 背景。引用 `b4,swan,skylb,gorgo,solyx,distserve,jitserve,andes,tail_at_scale`。
2. **现有方法的结构性缺口（约 95--110 words）。** 先说现有 router 优化 where，再立刻提出 when/irreversibility。引用 `preble,dualmap,quartz`；缺口本身用 RAVEL 自己的措辞，不要归因给任何一篇论文。
3. **两请求反例（约 75--90 words）。** 只保留 A/B、`r_1` 可选两地、`r_2` 只能 A 和一句结论；删掉与 Problem 中 `\mathcal F_i` 等价的形式化内容。
4. **Information--slack tradeoff（约 90--105 words）。** 保留 Figure 1 的观察和“等待消耗同一份 SLO slack”；不要再次定义完整 feasibility set。可在段末加 `delayscheduling,liu_layland_edf,sparrow`。
5. **核心洞见（约 105--120 words）。** 只讲三步：arrival 立即 tentative/account → Router-owned 时可转移 → handoff 后 final。把 KV state 的细节移到 Design/Related Work。
6. **三个挑战（约 90--105 words）。** 每个挑战最多两句：replica-specific quotes、transferable accounting、bounded replanning/handoff boundary。不要枚举实现字段。
7. **实现、证据和贡献（约 130--150 words + 3 bullets）。** 保留 five replicas/three clusters、`8.8/11.2/13.4 pp`、TTFT `49.1--80.5\%`、`2.10 ms`；Burst 的绝对延迟和 quote 误差下沉 Evaluation。

**建议删/并：**

- [ ] 删除 P3 末尾关于“independently optimizing requests reduces aggregate ...”的重复解释，或压为一句。
- [ ] 将 P4 的 fixed-delay 解释压为 3 句；不要复述 Problem 表格。
- [ ] 将 P5 的 KV-cache/migration 解释移到 Design 的 commitment 小节。
- [ ] 将 P6 的具体策略名、阈值、队列名移到 Implementation/Appendix。

### Problem and Design Opportunity（目标 600--660 words；当前约 781）

**回答的问题：** “request-dependent feasibility”如何定义？为什么 online state 让 placement timing 变成独立问题？RAVEL 的四个设计约束是什么？

- [ ] 开头三句保留 placement/accounting/commitment 三分法；删去后文再次解释的同义句。
- [ ] `\Phi_{i,r}(t)` 和 `\mathcal F_i(t)` 公式保留，这是全文后续机制的语义基础。
- [ ] 公式后的“new arrivals change state”和“waiting consumes slack”合并成一个短段。
- [ ] 表格保留；表格前只用一句引导，表格后只用一段解释三种 operating point。
- [ ] 最后一段保留四项 requirements，但改成一个紧凑的 numbered sentence。
- [ ] 可在 online/deadline precedent 处插入一次：

  ```latex
  The tension is related to locality-aware delay scheduling and deadline-driven
  online scheduling, but those settings do not expose RAVEL's
  request-dependent feasible-region semantics~\cite{delayscheduling,liu_layland_edf,
  sparrow,flow_imprecise}.
  ```

  这句只说明概念背景，不把经典结果当作 RAVEL 的理论保证。

### RAVEL Design（目标 1,900--2,000 words；当前约 2,148）

**回答的问题：** RAVEL 如何把“立即可见”和“延迟最终化”做成可执行、可证明不重复计费的控制流程？

- [ ] Overview 由“五个 invariant”长列表改为“一句总览 + 四个机制小段”；详细 invariant、边界表和伪代码下沉 Appendix A。
- [ ] Replica SLO Quotes：保留 quote tuple、causal inputs、风险/期望 demand 的区别；删掉已经在 Implementation 解释过的 profile 读取细节。插入 `vidur,llm_learning_to_rank,lodestar,flow_imprecise`。
- [ ] Immediate and Transferable Demand Accounting：保留 transfer 公式和 exactly-once invariant；删除两次解释“later arrivals see the charge”。插入 `omega,firmament,sparrow`。
- [ ] Bounded Observation and Replanning：保留触发条件、cohort bound、mobility hold、复杂度；把所有配置值和异常分支表移到 Appendix A.4。插入 `delayscheduling,sparrow,firmament`。
- [ ] Unified Placement/Dispatch：保留 deterministic key、frontier 和 local TBT guard 的因果关系；不要在主文展开每个 tie-break 字段。插入 `power_of_two_choices,consistent_hashing,fairness_llm,revisit_slo_goodput`。
- [ ] Commitment and Engine Handoff：只保留 handoff 成功才 final、拒绝回滚、stale callback 三个关键事实。把完整状态转换表移到 Appendix A.8；插入 `llumnix,spotserve,serverlessllm,dejavu,dlora,mooncake,cachedattention,infinigen`，并明确这些是对比背景，不是 RAVEL 的实现来源。
- [ ] Prefill/Decode 的局部控制只保留“为何需要”和“边界是什么”；实现 hook 名称、block 对齐公式和 runtime 开关放 Implementation/Appendix。插入 `sarathi,distserve,splitwise,shuffleinfer`。

### Implementation（目标 430--470 words；当前约 472）

**回答的问题：** 设计是否真实落地到一个可复现的控制层？串行化、profile、vLLM hook、handoff 的边界是什么？

- [ ] 第一段保留 Python/asyncio lock、vLLM 版本和 serialized path；删除与 Design 重复的“why”解释。
- [ ] 第二段保留 schema-v2、插值/端点 clamp、conservative Prefill envelope；把 profile 的动机压成一句。
- [ ] 第三段保留 feature detection、priority range、temporary token cap 和 `finally` restoration；不要解释算法正确性。
- [ ] 第四段保留 registration + posting-task 成功才 handoff、拒绝恢复、attempt ID 和 terminal release；异常路径细节放 Appendix。
- [ ] 可直接加入以下 grouped citations（每组只出现一次）：

  ```latex
  We implement RAVEL in Python as an asynchronous control layer in front of
  geo-distributed vLLM replicas~\cite{vllm,orca,sglang,pope_scaling_inference}.
  ```

  ```latex
  The external profile design follows the broader practice of
  profile-based inference modeling and engine-level optimization~\cite{vidur,
  lodestar,deepspeed_inference,flashattention2}.
  ```

  ```latex
  The adapter's local phase-control path is complementary to prior
  Prefill/Decode scheduling and speculative-execution systems~\cite{sarathi,
  distserve,splitwise,specinfer}.
  ```

  ```latex
  Request-visible workflow metadata is consistent with systems that expose
  structure across dependent LLM calls~\cite{parrot,agentix,libra,ayo}.
  ```

  这些引用只支撑“背景/对照”，锁、callback、attempt ID、具体 hook 名称仍必须写成 RAVEL 的代码事实。

### Evaluation（目标 1,350--1,450 words；当前约 1,552）

**回答的问题：** RAVEL 是否有效、是否可复现、机制是否必要、quote 是否够准、控制开销是否可接受？

保留现有 RQ1--RQ5，但压缩每个 RQ 的解释：

- [ ] Setup：把 Testbed、Workloads/SLOs、Replay protocol 合并为两个段落；表格保留，不再在段落逐项重复表格。
- [ ] Workload provenance 句末加入：

  ```latex
  The conversational trace is based on LMSYS-Chat-1M~\cite{lmsys_chat_1m};
  the evaluated Qwen3 model family is described in its technical report~\cite{qwen3}.
  ```

  若 WildChat 只作为背景，不要写成数据来源：

  ```latex
  Public chat traces such as WildChat illustrate the heterogeneity of
  real-world interactions, but the replay used here follows the LMSYS trace
  construction described above~\cite{wildchat,lmsys_chat_1m}.
  ```

- [ ] Policies 段删除 `RD`，改为：

  ```latex
  We compare RAVEL with RoundRobin (RR), LeastLoad (LL), DualMap, and SkyWalker.
  ```

- [ ] 五次重跑的三处 comparator 文字统一为：

  ```latex
  The repeated-run campaign uses DualMap as the fixed comparator for every
  workload and run.
  ```

  ```latex
  For each workload, every run contains a complete 500-request replay, and the
  RAVEL result is paired with DualMap under the same run conditions.
  ```

- [ ] End-to-End 只保留 headline `8.8/11.2/13.4 pp`、TTFT/E2E 范围和一段解释；Burst 的 `2.684/13.767/18.780/25.098` 等绝对值只在图或表出现一次。
- [ ] Repeated-Run 保留 4.1/7.5/17.4 pp、五个 paired runs 的统计单位和 Appendix 引用；删除“different batch”解释的第二次重复。
- [ ] Mechanism、Quote Accuracy、Planner Overhead 各压成“问题—数字—限制”三段；图注不要重复正文的整段解释。
- [ ] 指标定义可加一次：

  ```latex
  We report request-level SLO attainment together with tail latency because
  percentile behavior and token-delivery QoE can diverge from aggregate means
  ~\cite{revisit_slo_goodput,tail_at_scale,andes}.
  ```

- [ ] 不给 RAVEL 的 8.8/11.2/13.4、17.4、6.2、1.0、2.10 等数字加外部引用；它们只能由实验 artifact 支撑。

### Related Work（目标 400--470 words；当前约 445）

**回答的问题：** RAVEL 与最接近的 routing、SLO、cache、migration、engine 和经典调度工作分别差在哪里？

压成五个短段，每段最后一句都明确 RAVEL 的差异，不写论文清单：

1. **Engine execution and phase scheduling：** `orca,vllm,pope_scaling_inference,deepspeed_inference,flashattention2,sglang,specinfer,loongserve,helix,cassini,slora`。结尾强调 RAVEL 是 Router/control-plane 的 pre-handoff 机制。
2. **Cache-aware/distributed routing：** `preble,dualmap,mooncake,cachedattention,cachecraft,chunkattention,cacheblend,kvlink,infinigen`。结尾强调 RAVEL 的 novelty 是 immediate accounting + revisable ownership，不是 cache primitive。
3. **Geo/WAN-aware placement：** `skylb,gorgo,solyx,swan,b4,quartz,power_of_two_choices,consistent_hashing`。结尾强调 RAVEL 用 request-specific feasible regions 和 handoff boundary 连接 WAN 与 SLO。
4. **SLO and application scheduling：** `jitserve,sarathi,distserve,splitwise,shuffleinfer,andes,revisit_slo_goodput,fairness_llm,vidur,lodestar,alpaserve,parrot,agentix,libra,ayo,autogen,gaia,toolkengpt,agentbench,webarena,wildchat,lmsys_chat_1m,qwen3,simple_is_better`。结尾强调 RAVEL 的决策层和目标不同。
5. **Reassignment, migration, and classical scheduling：** `llumnix,spotserve,serverlessllm,dejavu,dlora,delayscheduling,sparrow,liu_layland_edf,flow_imprecise,omega,firmament`。结尾明确 RAVEL 在 engine state 物化前停止迁移；不声称替代 post-handoff migration。

为节省篇幅，每段使用 1--2 个长 grouped citation；同一 key 若已在 Introduction/Design 出现，不必再次逐篇解释，但覆盖表仍要保持可核对。

### Conclusion（目标 110--130 words；当前约 138）

**回答的问题：** 论文最终证明了什么、适用边界是什么？

- [ ] 合并前两句背景，保留 immediate visibility/deferred commitment 的一句总结。
- [ ] 保留三类 workload、SLO improvement、millisecond-scale overhead 的结论性描述，不重复所有数字。
- [ ] 加一句边界：结果来自 five-replica physical-WAN deployment，不是任意 fleet 的 formal guarantee。
- [ ] 不在 Conclusion 新增引用。

### Appendix A/B（不纳入正文预算时保留细节；若计入总页数则压缩）

- [ ] Appendix A 逐节替换 `\AppendixPlan`：state table、quote equation、accounting invariant、replanning bound、handoff transition table 是必须项。
- [ ] A.2/A.3 的完整伪代码和事件表只出现一次；正文只引用小节和结论。
- [ ] Appendix B 保留 single-run `13.4 pp vs DualMap` 与 separate five-run `17.4 pp vs DualMap` 的区分，不再出现 RD、47.6、64.2、46.2、16.6、18.0。
- [ ] 五次重跑 figure caption 明确 5 runs、500 requests/run、DualMap fixed comparator、CI 方法；不要把五次均值替换主矩阵单次值。
- [ ] 若附录计入页数，优先将 A.4--A.7 的配置/伪代码移到 artifact/supplementary，正文只保留状态机和不变量。

## 4. 可直接粘贴的引用插入块

下面代码块是“最少句子、最大覆盖”的版本。插入前先确认相邻句子的事实确实与文献相符；不要为了凑数量强行引用。

### Introduction

```latex
Geo-distributed serving relies on heterogeneous private-WAN paths and
cross-region traffic engineering~\cite{b4,swan,skylb,gorgo,solyx}.
Interactive serving also exposes distinct TTFT, TBT/TPOT, and completion
objectives~\cite{distserve,jitserve,andes,tail_at_scale}.
```

```latex
Existing routers combine load, locality, and queueing signals when choosing a
destination~\cite{preble,dualmap,quartz}; the unresolved question is when that
choice should become irreversible.
```

```latex
Multi-stage and agentic applications make request demand dependent on
application-visible workflow structure~\cite{parrot,agentix,ayo,autogen,gaia,
toolkengpt,agentbench,webarena}.
```

### Problem

```latex
This timing problem has conceptual precedents in locality-aware delay
scheduling, deadline-driven scheduling, and online scheduling with imprecise
information~\cite{delayscheduling,liu_layland_edf,sparrow,flow_imprecise}.
```

### Design/Implementation

```latex
Replica quotes draw on profile-based latency prediction and state-dependent
scheduling models~\cite{vidur,llm_learning_to_rank,lodestar}.
```

```latex
The local execution guard complements prior work on iteration-level scheduling,
Chunked Prefill, phase disaggregation, and interference reduction
~\cite{orca,sarathi,distserve,splitwise,shuffleinfer,sglang,specinfer}.
```

```latex
Cache-centric and memory-management systems motivate the distinction between
Router-owned tentative work and engine-managed state~\cite{mooncake,
cachedattention,infinigen,cachecraft,chunkattention,cacheblend,kvlink}.
```

```latex
Prior migration systems move already materialized execution state or adapt
placement after admission~\cite{llumnix,spotserve,serverlessllm,dejavu,dlora};
RAVEL stops reassignment at the successful handoff boundary.
```

```latex
The deterministic placement key is intentionally simpler than a learned or
weighted objective and is related to classical sampling, hashing, and shared
state scheduling primitives~\cite{simple_is_better,power_of_two_choices,
consistent_hashing,omega,firmament}.
```

```latex
The implementation sits on established transformer/LLM execution systems and
resource-placement techniques~\cite{vllm,pope_scaling_inference,
deepspeed_inference,flashattention2,alpaserve,loongserve,helix,cassini,slora}.
```

### Evaluation/Related Work

```latex
We use LMSYS-Chat-1M and Qwen3 as workload/model provenance references
~\cite{lmsys_chat_1m,qwen3}; WildChat and general agent benchmarks provide
context for heterogeneous interactive traces~\cite{wildchat,agentbench,webarena}.
```

```latex
We report request-level SLO attainment together with percentile latency and
token-delivery metrics, since tail behavior and aggregate means can diverge
~\cite{revisit_slo_goodput,tail_at_scale,andes,fairness_llm}.
```

```latex
Related distributed placement systems also study topology-aware scheduling and
WAN-aware capacity allocation~\cite{skylb,gorgo,solyx,swan,b4,quartz}.
```

```latex
The closest SLO/application systems include adaptive serving, workload
placement, and structured multi-call execution~\cite{jitserve,adaserve,
alpaserve,parrot,agentix,libra,ayo,autogen,gaia,toolkengpt}.
```

## 5. 66 条引用覆盖检查表

勾选表示该 key 已在正文某处实际出现（不是仅存在于 `.bib`）。同一 key 可以在多个位置出现，但至少要有一个与其用途匹配的句子。

### Geo/routing/cache（18）

- [ ] `skylb` — Introduction P1 / Related Work geo routing
- [ ] `gorgo` — Introduction P1/P2 / Related Work WAN routing
- [ ] `solyx` — Introduction P1 / Related Work telemetry (optional)
- [ ] `preble` — Introduction P2 / cache-aware routing
- [ ] `dualmap` — Introduction P2 / baseline and closest comparator
- [ ] `quartz` — Introduction P2 / TTFT-SLO routing
- [ ] `mooncake` — Design handoff/cache context
- [ ] `cachedattention` — Design/Related Work cache locality
- [ ] `infinigen` — Design/Related Work KV management
- [ ] `cachecraft` — Related Work reusable chunk cache
- [ ] `chunkattention` — Related Work prefix-aware execution
- [ ] `cacheblend` — Related Work RAG/prefix cache
- [ ] `kvlink` — Related Work KV reuse
- [ ] `swan` — WAN foundation
- [ ] `b4` — WAN foundation
- [ ] `power_of_two_choices` — placement foundation
- [ ] `consistent_hashing` — routing/cache foundation
- [ ] `helix` — distributed/network-aware serving

### SLO/execution/prediction（20）

- [ ] `distserve` — Introduction/Design Prefill--Decode
- [ ] `jitserve` — Introduction/Related Work heterogeneous SLOs
- [ ] `sarathi` — local TBT/Chunked Prefill
- [ ] `adaserve` — Related Work per-request SLO adaptation (optional)
- [ ] `andes` — token-delivery/QoE metrics
- [ ] `revisit_slo_goodput` — Evaluation metric interpretation
- [ ] `fairness_llm` — Related Work SLO/fairness boundary
- [ ] `vidur` — quote/profile prediction
- [ ] `llm_learning_to_rank` — scheduling prediction contrast
- [ ] `flow_imprecise` — online uncertainty precedent
- [ ] `lodestar` — recent profile/scheduler context
- [ ] `orca` — iteration-level execution
- [ ] `vllm` — implementation/testbed engine
- [ ] `pope_scaling_inference` — inference scaling context
- [ ] `deepspeed_inference` — engine background (optional)
- [ ] `flashattention2` — engine/profile variability (optional)
- [ ] `alpaserve` — bursty placement/statistical multiplexing
- [ ] `splitwise` — phase disaggregation
- [ ] `shuffleinfer` — interference/phase scheduling
- [ ] `sglang` — structured LLM execution

### Applications/workloads（10）

- [ ] `parrot` — structured dependent calls
- [ ] `agentix` — agent-program scheduling
- [ ] `libra` — cooperating micro-requests
- [ ] `ayo` — application-level optimization
- [ ] `autogen` — multi-agent workload context
- [ ] `gaia` — multi-step assistant tasks
- [ ] `toolkengpt` — tool-oriented workflows (optional)
- [ ] `agentbench` — agent benchmark context (optional)
- [ ] `webarena` — web-agent context (optional)
- [ ] `wildchat` — public chat trace context (optional)

### Migration/commitment/cache execution（7）

- [ ] `llumnix` — post-admission migration contrast
- [ ] `spotserve` — live migration contrast
- [ ] `serverlessllm` — dynamic placement/loading (optional)
- [ ] `dejavu` — KV-state movement contrast
- [ ] `dlora` — dynamic request/adapter migration
- [ ] `loongserve` — long-context serving
- [ ] `slora` — heterogeneous adapter serving (optional)

### Classical scheduling/state（9）

- [ ] `delayscheduling` — locality vs waiting
- [ ] `sparrow` — low-latency online scheduling
- [ ] `liu_layland_edf` — deadline scheduling precedent
- [ ] `omega` — shared-state scheduling
- [ ] `firmament` — repeated/global placement
- [ ] `tail_at_scale` — tail-latency motivation
- [ ] `simple_is_better` — deterministic rule motivation (optional)
- [ ] `specinfer` — speculative execution context (optional)
- [ ] `qwen3` — evaluated model provenance

> 注：上面分组按用途而非 BibTeX 文件顺序排列；`qwen3` 属于 Evaluation provenance，`specinfer` 属于 execution context。最终以 BibTeX parser 的 66 个唯一 key 为准。

## 6. 编译与页数验收清单

- [ ] `rg -n "RD|47\.6|64\.2|46\.2|16\.6|18\.0|workload-specific best baseline|designated best baseline" usenix.tex` 只留下经确认的历史说明（最好为 0）。
- [ ] `rg -n "\\cite\{" usenix.tex` 后，把每个 selected key 与第 5 节覆盖表逐项核对。
- [ ] `bibtex` 无 duplicate key、undefined citation、missing author/title/year 警告。
- [ ] Abstract 中不再出现 `\%.,`；全文百分号和 percentage points 口径一致。
- [ ] 五次重跑所有地方都写 DualMap fixed comparator；single-run 的 13.4 pp 不与 five-run 的 17.4 pp 混用。
- [ ] 主矩阵图注只保留实验条件、boxplot 定义和主要观察；不要再放一个重复的大表。
- [ ] 正文 Conclusion 完整结束后再出现 `\bibliography`；正文目标不超过 12 页。
- [ ] 编译后检查每页顶部/底部是否有孤立标题、过大的图下空白或 caption 跨栏异常；优先移动浮动体，不改字体参数。

## 7. 本轮已完成的非正式验证

- [x] `RAVEL_compact_66_references.bib` 解析出 66 个唯一 key；与 `RAVEL_expanded_references.bib` 的选定条目一致。
- [x] 独立 `pdflatex + BibTeX` smoke test 使用 `\nocite{*}` 成功生成 66 个 `\bibitem`，BibTeX warnings 为 0。
- [x] 临时完整论文副本用 66 条条目编译成功；该副本约 18 页（包含完整参考文献），不能直接当作最终页数结论。
- [x] 正式 `usenix.tex` 和 `sample.bib` 在本轮未写入；临时编译文件均位于 `overleaf-main/tmp`。

临时 18 页的原因主要是把 66 条文献全部用 `\nocite{*}` 强制打印出来；完成正文 grouped citations 后，实际参考文献页数会取决于真正被引用的条目和 venue 的排版规则。正文仍需按第 3 节的预算压缩约 0.5--1 页。
