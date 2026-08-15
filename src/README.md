# RAVEL source tree

The importable router package, RAVEL engine adapter, tests and single-cell
entrypoint live here.

## Release implementation

- `dualmap/scheduler/global_scheduler/ravel_unified_global_scheduler.py`:
  the released RAVEL-Unified policy.
- `dualmap/scheduler/global_scheduler/ravel_native_global_scheduler.py`:
  shared quote, service-ledger and placement primitives.
- `dualmap/cluster/topology.py`: topology and schema-v2 service curves.
- `dualmap/client/open_ai.py`: streaming request path and exact metrics.
- `dualmap/request_generator/replay_clock.py`: open-loop absolute replay.
- `ravel_engine_adapter/`: priority wire protocol and feature-detected
  vLLM patch.
- `run_cluster.py`: one RAVEL policy/workload/speedup cell.

The report label `RAVEL-Unified` maps to scheduler id `ravel_unified`.
The inherited cluster-routing base is an internal RAVEL implementation
dependency, not a separately exposed policy in this release.

## Run one cell

```bash
python src/run_cluster.py \
  --replicas a0:8000,a1:8000,b0:8000,b1:8000,c0:8000 \
  --cluster-topology configs/three_cluster_2_2_1.local.json \
  --scheduler ravel_unified \
  --model-path /models/Qwen3-1.7B \
  --model-name qwen \
  --trace data/lmsys_first500.json \
  --result-path results/single_cell \
  --request-num 500 \
  --arrival-speedup 12 \
  --slo-profile paper_e2e \
  --request-ratio 3,5,2 \
  --network-delay-mode physical \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --block-size 16 \
  --ravel-mobile-candidate-limit 64 \
  --ravel-collective-stage-output-profile \
    configs/output_profiles/jitserve_collective_stage_context_q95.schema-v2.json
```
