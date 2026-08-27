# RAVEL citation placement and usage map

> Generated at 2026-08-27T04:08:48.403274+00:00. The library contains **80 verified entries**: **52 core** and **28 optional/grouped** references.
> Discovery follows reference chains from JITServe, DualMap, Preble, Mooncake, SkyWalker, and GORGO; final metadata is checked against the existing bibliography, official proceedings/publisher pages, Crossref DOI records, or arXiv's official BibTeX endpoint.

## How to use this file

- `core` means the paper can carry a distinct claim in the main text.
- `optional` means it should normally appear in a grouped Related Work citation, not receive a standalone sentence.
- Project-specific numbers, testbed RTTs, ablation values, and five-run confidence intervals must come from RAVEL artifacts, never from these external papers.
- Do not cite all 80 merely to reach a count. A strong USENIX-style draft will usually cite the 50 core items plus selected optional items where the prose actually discusses them.

## Recommended citation bundles

- Geo-distributed routing: `\cite{skylb,gorgo,solyx,preble,dualmap,quartz}`.
- Engine execution and Prefill/Decode behavior: `\cite{orca,vllm,sarathi,distserve,splitwise,sglang}`.
- Cache locality and reuse: `\cite{preble,dualmap,mooncake,cachedattention,cachecraft,chunkattention,cacheblend,kvlink}`.
- SLO-aware scheduling: `\cite{jitserve,distserve,sarathi,andes,revisit_slo_goodput,fairness_llm}`.
- Reassignment and migration contrast: `\cite{llumnix,spotserve,dlora,serverlessllm,dejavu}`.
- Agent and multi-stage workloads: `\cite{parrot,agentix,libra,ayo,autogen,gaia,webarena}`.
- Classical foundations: `\cite{delayscheduling,sparrow,liu_layland_edf,power_of_two_choices,consistent_hashing,omega,firmament,swan,b4}`.

## Existing direct references

| Key | Tier | Seed/reference chain | Recommended location | How to use | Do not overclaim |
|---|---|---|---|---|---|
| `skylb` (SkyWalker) | core | existing bibliography; SkyWalker seed | Introduction P1 and Related Work: geo-distributed routing | Support the claim that recent LLM routers explicitly route across regions while considering locality and load. | Do not use it to claim pre-handoff revisability or immediate prospective accounting. |
| `gorgo` (GORGO) | core | existing bibliography; GORGO seed | Introduction P1/P2 and Related Work: cross-region routing | Support network-aware routing with WAN delay, queue state, and serving cost. | It is a 2026 preprint; label it as such and do not attribute RAVEL's mechanism to it. |
| `solyx` (Solyx AI Grid) | optional | existing bibliography | Introduction P1 and Related Work: cross-site telemetry | Use as a recent example of hardware-telemetry-aware cross-site placement. | Preprint only; avoid presenting it as established production evidence. |
| `preble` (Preble) | core | existing bibliography; Preble seed; DualMap References | Introduction P2 and Related Work: prefix-aware distributed routing | Support joint reasoning about reusable KV state and replica computation load. | Do not imply that Preble delays commitment or transfers accounting with ownership. |
| `dualmap` (DualMap) | core | existing bibliography; DualMap seed | Introduction P2; Evaluation baselines; Related Work | Describe the closest cache-affinity/load-balancing comparator and its hotspot-aware rebalancing. | Keep the paper's exact experimental comparator wording separate from the general related-work description. |
| `quartz` (QUARTZ) | core | existing bibliography | Introduction P2 and Related Work: TTFT-SLO routing | Support quantile-aware routing and queueing for request-level TTFT SLOs. | Do not use it as evidence for completion-deadline or multi-stage semantics unless the paper explicitly covers them. |
| `distserve` (DistServe) | core | existing bibliography; JITServe ref. 86; DualMap References | Introduction P1; Design local control; Related Work | Support the TTFT/TPOT distinction and the importance of Prefill--Decode interference and phase-aware provisioning. | Do not describe DistServe as a pre-handoff reassignment system. |
| `jitserve` (JITServe) | core | existing bibliography; JITServe seed | Introduction P1; Evaluation metrics; Related Work | Support heterogeneous SLO semantics and scheduling under imprecise request information. | Do not use JITServe to validate RAVEL's numerical results or geo-distributed claims. |
| `sarathi` (Sarathi-Serve) | core | existing bibliography; JITServe ref. 10 | Design engine-local TBT protection; Implementation; Related Work | Support chunked Prefill as a mechanism for reducing Prefill--Decode interference. | Do not imply identical chunk-bound equations or vLLM hook implementations. |
| `adaserve` (AdaServe) | optional | existing bibliography; JITServe ref. 39 | Related Work: per-request SLO adaptation | Use as a complementary per-request SLO mechanism based on customized speculative decoding. | Do not group it with geographic routing baselines. |
| `parrot` (Parrot) | core | existing bibliography; JITServe ref. 40 | Introduction agent motivation; Implementation metadata; Related Work | Support exposing application-level structure across dependent LLM calls. | Do not claim RAVEL implements Parrot's semantic-variable API. |
| `agentix` (Agentix) | core | existing bibliography; JITServe ref. 42 under earlier title Autellix | Introduction multi-stage motivation; Related Work | Support treating agent programs as scheduling entities with program-level latency objectives. | Use the final published title/key consistently; do not cite the preprint and final paper as separate works. |
| `libra` (Libra) | core | existing bibliography | Related Work: multi-request partitioning and SLO-aware batching | Support global request partitioning combined with local SLO-aware batching for dynamic workloads. | Do not present it as a geographic router. |
| `llumnix` (Llumnix) | core | existing bibliography; JITServe ref. 63 | Design commitment boundary; Evaluation ablation context; Related Work migration | Contrast RAVEL's pre-handoff reassignment with live migration after execution state exists. | State complementarity; do not claim RAVEL dominates post-handoff migration. |
| `delayscheduling` (Delay Scheduling) | core | existing bibliography; Preble/SkyWalker reference chains | Problem information--slack tradeoff; Related Work foundations | Provide the classic locality-versus-waiting precedent for bounded deferral. | Do not equate cluster locality delay with request-level SLO slack without explaining the analogy. |
| `sparrow` (Sparrow) | core | existing bibliography; cluster-scheduling reference chain | Problem online decisions; Design bounded cohort; Planner overhead | Support low-latency online scheduling and stale-state concerns in distributed schedulers. | Do not use it to claim LLM-specific service prediction. |
| `vllm` (vLLM / PagedAttention) | core | existing bibliography; JITServe ref. 34; DualMap References | Implementation opening; Evaluation setup; Related Work engines | Cite the serving engine and PagedAttention basis of the implementation and testbed. | Version-specific behavior must still be supported by RAVEL's code/configuration, not the vLLM paper. |

## LLM serving, routing, execution, and cache systems

| Key | Tier | Seed/reference chain | Recommended location | How to use | Do not overclaim |
|---|---|---|---|---|---|
| `orca` (Orca) | core | JITServe ref. 79 | Implementation engine scheduling; Related Work engines | Support iteration-level scheduling and selective batching as foundational LLM serving mechanisms. | Do not imply Orca provides request-level geographic replanning. |
| `deepspeed_inference` (DeepSpeed-Inference) | optional | JITServe ref. 12 | Related Work engines | Use as an early system for scalable transformer inference and kernel/model-parallel optimization. | Keep it background; it does not address RAVEL's commitment problem. |
| `turbotransformers` (TurboTransformers) | optional | JITServe ref. 21 | Related Work engines | Support dynamic batching and memory management for variable-length transformer serving. | Do not treat pre-LLM transformer serving assumptions as equivalent to modern autoregressive scheduling. |
| `flashattention` (FlashAttention) | optional | JITServe ref. 18 | Implementation service-profile context or Related Work engines | Cite IO-aware exact attention as an engine-level kernel optimization that changes calibrated service time. | Do not claim RAVEL implements or modifies FlashAttention. |
| `flashattention2` (FlashAttention-2) | optional | JITServe ref. 17 | Implementation service-profile context | Use to motivate why RAVEL loads measured profiles rather than assuming a universal token rate across engine kernels. | It supports engine variability, not Router-level scheduling claims. |
| `pope_scaling_inference` (Efficiently Scaling Transformer Inference) | core | DualMap References | Related Work engines and Implementation profiles | Support the observation that inference efficiency depends on batching, parallelism, and memory behavior. | Do not cite it for WAN routing or request SLO guarantees. |
| `flexgen` (FlexGen) | optional | Preble and Mooncake reference chains | Related Work engines and memory hierarchy | Use as an example of throughput-oriented offloading across GPU, CPU, and storage. | Explicitly distinguish its latency-insensitive setting from RAVEL's online SLO objective. |
| `alpaserve` (AlpaServe) | core | JITServe ref. 37 | Introduction burst motivation; Related Work placement | Support statistical multiplexing and model placement under bursty request arrivals and latency constraints. | It places models/devices, not individual geo-distributed requests with deferred commitment. |
| `splitwise` (Splitwise) | core | JITServe ref. 54 | Design local Prefill/Decode discussion; Related Work phase disaggregation | Support distinct Prefill and Decode resource characteristics and phase splitting. | Do not imply Splitwise uses RAVEL's quote or ownership model. |
| `shuffleinfer` (Inference without Interference / ShuffleInfer) | core | JITServe ref. 27 | Related Work phase disaggregation | Support disaggregating LLM inference for mixed downstream workloads to reduce interference. | Name the cited arXiv version consistently; do not silently combine it with a differently titled final record. |
| `serverlessllm` (ServerlessLLM) | optional | Preble/JITServe reference chains | Related Work dynamic placement and migration | Use as a model-locality-aware scheduler with fast loading and live migration. | Its model-loading locality is different from request-prefix locality and pre-handoff reassignment. |
| `sglang` (SGLang) | core | JITServe ref. 84; DualMap References | Implementation structured workflows; Related Work engines | Support efficient execution of structured language-model programs and reusable state across calls. | Do not claim the DeepResearch trace is generated by SGLang unless that is true in the experiment artifacts. |
| `specinfer` (SpecInfer) | optional | Preble/Mooncake reference chains | Related Work engine execution | Use as a representative speculative-inference system that changes per-request service behavior. | Do not cite it as a routing baseline. |
| `infinigen` (InfiniGen) | core | JITServe ref. 35; DualMap References | Design/Implementation KV-state context; Related Work cache | Support dynamic KV-cache management and offloading for long generation. | Do not imply RAVEL evicts or migrates KV state before handoff. |
| `mooncake` (Mooncake) | core | DualMap References; Mooncake seed | Introduction capacity/cache context; Design; Related Work | Support a KVCache-centric disaggregated architecture, global cache, and SLO-aware scheduling. | Do not transfer Mooncake's production scale or performance numbers to RAVEL. |
| `dejavu` (DéjàVu) | core | DualMap References | Design commitment boundary; Related Work migration | Contrast KV-cache streaming and fault-tolerant state movement with RAVEL's decision to stop Router reassignment before KV state materializes. | Do not call RAVEL fault tolerant on the basis of this citation. |
| `vattention` (vAttention) | optional | DualMap References | Implementation memory-management context; Related Work cache | Use as an alternative dynamic memory-management design to PagedAttention. | Do not suggest the evaluated vLLM version uses vAttention. |
| `loongserve` (LoongServe) | core | DualMap References | Related Work long-context scheduling | Support elastic sequence parallelism and the scheduling challenges of long-context serving. | It is not a geo-routing or commitment comparator. |
| `powerinfer` (PowerInfer) | optional | Preble/Mooncake reference chains | Related Work heterogeneous inference | Use as an example of exploiting heterogeneous GPU/CPU resources during LLM inference. | Do not use it to support WAN heterogeneity. |
| `helix` (Helix) | optional | GORGO/SkyWalker reference chains | Related Work distributed inference | Use as a distributed LLM serving system that jointly reasons about model execution and network placement. | Verify any finer mechanism claim against the paper before adding prose beyond the placement note. |
| `apparate` (Apparate) | optional | JITServe/serving reference chain | Related Work adaptive inference | Use as an example of online adaptation to latency--throughput tension during inference. | Do not categorize it as cache-aware routing. |
| `spotserve` (SpotServe) | core | JITServe ref. 45 | Design commitment boundary; Related Work migration | Support live migration and dynamic orchestration when serving on preemptible instances. | Use only for the post-placement migration contrast. |
| `slora` (S-LoRA) | optional | JITServe and DualMap reference chains | Related Work heterogeneous serving | Support unified paging and heterogeneous batching across many LoRA adapters. | It is peripheral; keep it in the engine/heterogeneity paragraph rather than the main novelty comparison. |
| `dlora` (dLoRA) | core | JITServe ref. 73 | Related Work reassignment and migration | Support dynamic request/adapter migration across replicas under skewed demand. | Distinguish adapter co-migration from RAVEL's pre-handoff placement transfer. |
| `fairness_llm` (Fairness in Serving LLMs) | core | JITServe ref. 61 | Problem objective boundary; Related Work SLO/fairness | Use to show that unpredictable lengths and continuous batching complicate fair work-conserving scheduling. | RAVEL optimizes SLO attainment, not the cited paper's formal fairness objective. |
| `cachedattention` (CachedAttention) | core | DualMap References | Introduction cache locality; Related Work cache-aware serving | Support cross-turn KV reuse and scheduler-aware cache movement in multi-turn conversations. | Do not cite it as evidence for RAVEL's tentative accounting. |
| `vidur` (Vidur) | core | DualMap References | Design quote construction; Evaluation methodology; Appendix quote calibration | Support profile-based prediction/simulation of LLM inference latency across dynamic batch states. | Do not imply RAVEL's quote model is Vidur or inherits its reported error bounds. |
| `llm_learning_to_rank` (Efficient LLM Scheduling by Learning to Rank) | core | JITServe ref. 23 | Design quote ordering; Related Work scheduling | Support learning/predicting scheduling decisions under heterogeneous request lengths and runtime state. | Do not claim RAVEL uses a learned ranker; use it as contrast to RAVEL's deterministic tuple. |
| `flow_imprecise` (Flow Scheduling with Imprecise Knowledge) | core | JITServe ref. 36 | Problem online uncertainty; Design quotes; Appendix | Provide a systems precedent for scheduling from bounded or imprecise job information. | Network-flow size uncertainty is an analogy, not direct validation of LLM latency quotes. |
| `cassini` (CASSINI) | optional | JITServe ref. 58 | Related Work network-aware scheduling | Support topology/network-aware placement of ML jobs in shared clusters. | Its periodic communication-job model is not the same as request-level WAN inference. |
| `muserve` (mu-Serve) | optional | JITServe ref. 56 | Related Work SLO-aware serving | Use as an example of jointly optimizing model-serving configuration and resource controls under SLOs. | Power management is outside RAVEL's scope; do not overstate similarity. |
| `andes` (Andes) | core | JITServe ref. 41 | Introduction SLO semantics; Evaluation metrics; Related Work | Support QoE-aware text streaming and the importance of token-delivery behavior beyond aggregate E2E latency. | Do not equate Andes's QoE metric with RAVEL's exact TBT success predicate. |
| `revisit_slo_goodput` (Revisiting SLO and Goodput Metrics) | core | JITServe ref. 71 | Evaluation metrics and interpretation boundaries; Related Work | Use to justify reporting request-level SLO attainment alongside latency distributions. | Do not use it to support RAVEL's measured gains or confidence intervals. |
| `lodestar` (Lodestar) | core | GORGO and recent serving reference chains | Design replica quotes; Appendix calibration; Related Work | Use as a recent predictor/scheduler reference for causal service estimates under changing LLM-serving state. | It is a 2026 preprint; phrase mechanism comparisons conservatively. |
| `simple_is_better` (Simple is Better) | optional | recent serving reference chain; user-supplied PDF | Design deterministic placement key; Related Work | Use to motivate simple, interpretable scheduling rules as an alternative to fitted weighted objectives. | Preprint status must be explicit; verify any detailed algorithm comparison before use. |
| `cachecraft` (Cache-Craft) | core | DualMap References | Related Work RAG/prefix caches | Support managing reusable chunk caches for retrieval-augmented generation. | RAVEL does not implement chunk-cache fusion or RAG-specific cache admission. |
| `snapkv` (SnapKV) | optional | DualMap References | Related Work KV-cache reduction | Use as a representative selective KV retention method for long-context inference. | It concerns within-request cache reduction, not cross-replica routing. |
| `chunkattention` (ChunkAttention) | core | DualMap References | Related Work prefix-aware execution | Support prefix-aware KV-cache sharing and attention execution across requests. | Do not describe it as a global router. |
| `cacheblend` (CacheBlend) | core | DualMap References | Related Work RAG/prefix caches | Support fusing cached knowledge for RAG serving and the value of reusable prefixes. | Avoid claiming its cache mechanism is present in RAVEL. |
| `pqcache` (PQCache) | optional | DualMap References | Related Work KV-cache compression | Use as a recent long-context KV-cache compression/offloading design. | Peripheral to RAVEL; one grouped citation is sufficient. |
| `nacl` (NACL) | optional | DualMap References | Related Work KV-cache eviction | Use as a representative KV-cache eviction framework. | Do not imply eviction policy affects RAVEL's pre-handoff accounting. |
| `arkvale` (ArkVale) | optional | DualMap References | Related Work KV-cache eviction | Support recallable KV eviction as another way to trade memory and recomputation. | Keep it in a grouped cache paragraph. |
| `kvlink` (KVLink) | core | DualMap References | Related Work KV reuse | Support efficient KV-cache reuse/linking across requests. | Do not call it a baseline unless it was actually implemented in the evaluation. |
| `shadowkv` (ShadowKV) | optional | DualMap References | Related Work long-context cache | Use as a high-throughput long-context KV-cache design. | Preprint status and scope should be explicit. |
| `megascale_infer` (MegaScale-Infer) | optional | DualMap References | Related Work disaggregated inference | Use as a recent large-scale MoE-serving and disaggregated expert-parallel system. | Its MoE scale is not evidence for RAVEL fleet scalability. |

## Agent applications, workloads, and evaluated model

| Key | Tier | Seed/reference chain | Recommended location | How to use | Do not overclaim |
|---|---|---|---|---|---|
| `ayo` (Ayo) | core | JITServe ref. 64 | Introduction multi-stage motivation; Related Work applications | Support end-to-end optimization across dependent calls in LLM applications. | Do not claim RAVEL implements Ayo's application optimizer. |
| `toolkengpt` (ToolkenGPT) | optional | DualMap References | Introduction tool/agent workload motivation | Support repeated tool instructions and tool-oriented LLM application structure as a source of shared prefixes. | Do not cite it as the provenance of RAVEL's trace unless the trace was derived from it. |
| `autogen` (AutoGen) | core | JITServe ref. 74 | Introduction multi-agent/multi-stage motivation; Related Work | Support multi-agent conversation as a common source of chained LLM calls. | Use for workload motivation only, not serving-system performance claims. |
| `agentbench` (AgentBench) | optional | agent-system reference chain | Introduction agent workload motivation; Evaluation workload discussion | Use to establish the diversity of interactive agent tasks. | Do not imply the DeepResearch workload is AgentBench-derived without artifact evidence. |
| `webarena` (WebArena) | optional | agent-system reference chain | Introduction and Evaluation: web-agent motivation | Support realistic web environments requiring multi-step agent interaction. | It is a benchmark, not a serving scheduler. |
| `gaia` (GAIA) | core | JITServe ref. 44 | Introduction DeepResearch motivation; Evaluation workload discussion | Support general-assistant tasks that require multi-step reasoning and tool use. | Do not equate GAIA completion accuracy with RAVEL's request-level SLO attainment. |
| `wildchat` (WildChat) | optional | JITServe ref. 82 | Evaluation workload provenance/alternatives | Use as evidence that public chat workloads contain heterogeneous real-world interactions. | If RAVEL uses LMSYS rather than WildChat, label WildChat as related trace evidence, not the source dataset. |
| `lmsys_chat_1m` (LMSYS-Chat-1M) | core | JITServe ref. 83 | Evaluation Workloads paragraph | Cite the provenance and characteristics of the LMSYS conversational workload. | State the exact preprocessing and replay construction from RAVEL artifacts separately. |
| `qwen3` (Qwen3 Technical Report) | core | JITServe ref. 66 | Evaluation Testbed paragraph and setup table | Cite the evaluated Qwen3 model family and architecture provenance. | Do not use the report to support RAVEL's latency or throughput measurements. |

## Classical scheduling, hashing, cluster, and WAN foundations

| Key | Tier | Seed/reference chain | Recommended location | How to use | Do not overclaim |
|---|---|---|---|---|---|
| `tail_at_scale` (The Tail at Scale) | core | JITServe/Preble/SkyWalker reference chains | Introduction tail-latency motivation; Evaluation metric interpretation | Support why tail latency matters at service scale and why percentile metrics complement means. | Do not treat it as a statistical-method citation for RAVEL's five-run confidence intervals. |
| `liu_layland_edf` (Liu--Layland EDF) | core | JITServe scheduling-theory context | Problem information--slack tradeoff; Related Work foundations | Introduce deadline-driven scheduling as a classical reference point. | Hard-real-time schedulability results do not transfer directly to stochastic, batched LLM serving. |
| `power_of_two_choices` (Power of Two Choices) | core | DualMap References | Problem substitutable capacity; Related Work foundations | Support the classic load-balancing benefit of sampling two candidates. | Do not use it to explain request-dependent SLO feasibility or cache affinity without the additional DualMap citation. |
| `consistent_hashing` (Consistent Hashing) | core | DualMap References | Related Work cache-aware routing foundations | Support stable key-to-node mapping and hotspot-relief motivation in distributed caching. | Clarify that RAVEL does not base its novelty on a hashing primitive. |
| `borg` (Borg) | optional | cluster-scheduling reference chain | Related Work cluster scheduling foundations | Use as production evidence for centralized cluster state, placement constraints, and mixed workloads. | Do not imply Borg offers request-level LLM SLO quotes. |
| `omega` (Omega) | core | cluster-scheduling reference chain | Design serialized ownership/accounting; Related Work foundations | Support shared-state scheduling and consistency challenges when multiple decisions update placement state. | RAVEL uses serialized control, not Omega's optimistic concurrency architecture. |
| `firmament` (Firmament) | core | cluster-scheduling reference chain | Design bounded replanning; Planner overhead; Related Work | Support centralized global placement and repeated rescheduling with explicit scheduler-latency concerns. | Do not claim RAVEL solves min-cost max-flow or scales to Firmament's cluster sizes. |
| `swan` (SWAN) | core | SkyWalker/GORGO WAN reference chains | Introduction P1 and Related Work WAN foundations | Support centrally controlled inter-datacenter traffic engineering and changing WAN demand. | Do not use SWAN's utilization results as RAVEL testbed evidence. |
| `b4` (B4) | core | SkyWalker/GORGO WAN reference chains | Introduction P1 and Related Work WAN foundations | Support the reality of globally deployed private WANs with heterogeneous paths and traffic engineering. | RAVEL's measured RTT range must come only from its own testbed. |

## Section-level insertion order

1. **Introduction P1:** cite geo-distributed routing/WAN and heterogeneous SLOs; keep engine/cache details to one grouped citation.
2. **Introduction P2:** cite one-shot/cache-aware routers, then state the uncited RAVEL gap in the authors' own words.
3. **Problem section:** use classical delay/deadline/online scheduling only as conceptual precedent; the request-dependent feasible-set formulation is RAVEL's own.
4. **Design:** use prediction papers around quote construction, cache/migration papers at the handoff boundary, and Sarathi/DistServe/Splitwise around Prefill--Decode protection.
5. **Implementation:** cite vLLM for PagedAttention/engine architecture and cite profiling papers only to motivate externally calibrated profiles. Implementation facts must be grounded in the repository.
6. **Evaluation:** cite dataset/model provenance and metric definitions. Do not add citations to RAVEL result sentences.
7. **Related Work:** expand to five paragraphs: engines; distributed/cache-aware routing; SLO-aware scheduling; application/agent scheduling; reassignment/migration and classical foundations.
8. **Appendix:** use Vidur/Lodestar/learning-to-rank for quote calibration context and Omega/Firmament/Sparrow for state/ownership context, but retain RAVEL-specific equations and invariants as uncited original design details.

## Provenance and validation artifacts

- `tmp/seed_reference_sections.json`: reference sections extracted from the six user-supplied seed PDFs.
- `tmp/RAVEL_source_candidates.json`: per-entry retrieval provider, timestamp, verification status, seed chain, and relevance evidence.
- `tmp/RAVEL_crossref_verified.json`: raw Crossref metadata used for DOI-backed entries.
- `tmp/arxiv_bibtex_verified.json`: official arXiv BibTeX responses.
- `tmp/RAVEL_bib_validation.json`: entry-count and duplicate-key/DOI/arXiv/title audit.
- `tmp/RAVEL_compile_validation.json`: independent pdfLaTeX + BibTeX smoke-test result for all 80 entries.
