#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import asyncio
from pathlib import Path
from types import SimpleNamespace

from dualmap.cluster.topology import ClusterTopology
from dualmap.launcher import SystemLauncher
from dualmap.logger import init_logger

logger = init_logger(__name__)


def derive_kv_cache_size_per_token(model_path: str, dtype_bytes: int) -> int:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if dtype_bytes <= 0:
        dtype_name = str(getattr(config, "torch_dtype", "")).lower()
        dtype_sizes = {
            "torch.float16": 2,
            "float16": 2,
            "torch.bfloat16": 2,
            "bfloat16": 2,
            "torch.float32": 4,
            "float32": 4,
        }
        if dtype_name not in dtype_sizes:
            raise ValueError(
                "cannot derive KV dtype bytes from model config; pass "
                "--kv-cache-dtype-bytes explicitly"
            )
        dtype_bytes = dtype_sizes[dtype_name]
    layers = int(getattr(config, "num_hidden_layers"))
    attention_heads = int(getattr(config, "num_attention_heads"))
    kv_heads = int(getattr(config, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // attention_heads))
    return layers * kv_heads * head_dim * 2 * dtype_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAVEL-Unified over regional vLLM clusters"
    )
    parser.add_argument("--replicas", required=True, help="comma-separated host:port endpoints")
    parser.add_argument("--cluster-topology", required=True, help="cluster topology JSON")
    parser.add_argument(
        "--allow-inline-service-profile",
        action="store_true",
        help=(
            "allow legacy inline prefill/decode rates; formal runs must use "
            "schema-v2 service_profile files with engine fingerprints"
        ),
    )
    parser.add_argument(
        "--scheduler",
        choices=("ravel_unified",),
        default="ravel_unified",
    )
    parser.add_argument("--model-path", required=True, help="local tokenizer/model path")
    parser.add_argument("--model-name", required=True, help="model name sent to vLLM")
    parser.add_argument("--trace", required=True, help="JITServe lmsys JSON or DeepResearch JSONL")
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--request-num", type=int, default=500)
    parser.add_argument("--qps", type=float, default=3.0)
    parser.add_argument(
        "--arrival-speedup",
        type=float,
        default=0.0,
        help="preserve trace gaps and divide them by this factor; 0 uses --qps normalization",
    )
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--slo-profile", choices=("paper_e2e", "paper_collective"), default="paper_e2e")
    parser.add_argument("--request-ratio", default="3,5,2", help="latency,throughput,collective")
    parser.add_argument("--client-region", default="region_a")
    parser.add_argument(
        "--network-delay-mode",
        choices=("synthetic", "physical"),
        default="synthetic",
        help=(
            "synthetic injects topology RTT for same-host emulation; physical "
            "uses real transport latency and never sleeps an extra RTT"
        ),
    )
    parser.add_argument("--routing-output-tokens-hint", type=int, default=256)
    parser.add_argument("--trace-seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument(
        "--kv-cache-blocks",
        type=int,
        default=0,
        help="engine-reported GPU KV block count; 0 uses an active-sequence shadow bound",
    )
    parser.add_argument(
        "--kv-cache-dtype-bytes",
        type=int,
        default=0,
        help="0 derives model dtype; use 1 for an explicitly configured FP8 KV cache",
    )
    parser.add_argument(
        "--kv-cache-size-per-token",
        type=int,
        default=0,
        help="override derived KV bytes/token; 0 derives it from model config",
    )
    parser.add_argument(
        "--replica-slo-budget-tokens",
        type=int,
        default=0,
        help="0 derives max_model_len * pending_request_limit",
    )
    parser.add_argument(
        "--pending-request-limit",
        type=int,
        default=0,
        help="0 uses the engine max_num_seqs",
    )
    parser.add_argument("--rebalance-token-threshold", type=int, default=32768)
    parser.add_argument("--rebalance-wait-s", type=float, default=0.5)
    parser.add_argument("--rebalance-hysteresis-s", type=float, default=0.02)
    parser.add_argument("--max-rebalances-per-event", type=int, default=8)
    parser.add_argument("--request-timeout-s", type=float, default=3600.0)
    parser.add_argument("--ravel-risk-epsilon", type=float, default=0.05)
    parser.add_argument("--ravel-residual-window", type=int, default=256)
    parser.add_argument("--ravel-output-quantile", type=float, default=0.9)
    parser.add_argument("--ravel-output-history-window", type=int, default=256)
    parser.add_argument("--ravel-output-min-samples", type=int, default=20)
    parser.add_argument("--ravel-mobile-candidate-limit", type=int, default=16)
    parser.add_argument("--ravel-mobile-beam-width", type=int, default=256)
    parser.add_argument("--ravel-mobile-hold-fraction", type=float, default=None)
    parser.add_argument("--ravel-mobile-slack-factor", type=float, default=None)
    parser.add_argument(
        "--ravel-collective-stage-output-profile",
        default="",
        help=(
            "optional held-out semantic output profile; never built from the "
            "evaluated request stream"
        ),
    )
    parser.add_argument(
        "--ravel-soft-admission-release-policy",
        choices=("slo_deadline",),
        default="slo_deadline",
        help="latest release boundary for deferred completion-only requests",
    )
    parser.add_argument(
        "--ravel-soft-admission-reserve-sequences",
        type=int,
        default=0,
        help=(
            "outstanding-sequence reserve before activation; 0 derives one "
            "replica-equivalent max_num_seqs cohort"
        ),
    )
    parser.add_argument("--ravel-assignment-file", default="")
    parser.add_argument("--ravel-assignment-arm", choices=("future_initial", "future_insertion", "future_combined", "selected"), default="future_initial")
    return parser.parse_args()


def build_system_args(cli: argparse.Namespace) -> SimpleNamespace:
    replica_count = len(cli.replicas.split(","))
    positive_fields = {
        "replica_count": replica_count,
        "max_model_len": cli.max_model_len,
        "max_num_seqs": cli.max_num_seqs,
        "max_num_batched_tokens": cli.max_num_batched_tokens,
        "block_size": cli.block_size,
        "request_timeout_s": cli.request_timeout_s,
    }
    invalid = {name: value for name, value in positive_fields.items() if value <= 0}
    if invalid:
        raise ValueError(f"runtime dimensions must be positive: {invalid}")
    if cli.pending_request_limit < 0 or cli.replica_slo_budget_tokens < 0:
        raise ValueError("derived-limit overrides must be non-negative")
    if cli.ravel_soft_admission_reserve_sequences < 0:
        raise ValueError("soft-admission reserve cannot be negative")
    if (
        cli.kv_cache_blocks < 0
        or cli.kv_cache_dtype_bytes < 0
        or cli.kv_cache_size_per_token < 0
    ):
        raise ValueError("KV cache overrides must be non-negative")
    topology = ClusterTopology.from_json(cli.cluster_topology)
    topology.validate_replica_count(replica_count)
    topology.validate_service_profiles(
        allow_inline=cli.allow_inline_service_profile
    )
    # The internal routing base accepts one scalar service time. Derive a conservative
    # value from the same measured topology instead of a hidden constant.
    legacy_prefill_tpot = max(
        cluster.prefill_tpot_for(0) for cluster in topology.clusters
    )
    legacy_busy_prefill_interval = max(
        cluster.prefill_intercept_for(cli.max_num_seqs)
        + cli.max_num_batched_tokens
        * cluster.prefill_tpot_for(cli.max_num_seqs)
        for cluster in topology.clusters
    )
    pending_limit = cli.pending_request_limit or cli.max_num_seqs
    replica_budget = (
        cli.replica_slo_budget_tokens
        or cli.max_model_len * pending_limit
    )
    kv_bytes_per_token = (
        cli.kv_cache_size_per_token
        or derive_kv_cache_size_per_token(cli.model_path, cli.kv_cache_dtype_bytes)
    )
    shadow_blocks = cli.kv_cache_blocks or (
        cli.max_num_seqs
        * ((cli.max_model_len + cli.block_size - 1) // cli.block_size)
    )
    cache_capacity = shadow_blocks * cli.block_size * kv_bytes_per_token
    return SimpleNamespace(
        replicas_ip_port=cli.replicas,
        model=cli.model_path,
        model_path=cli.model_path,
        model_name=cli.model_name,
        replica_num=replica_count,
        replica_dram=cache_capacity / (1 << 30),
        cache_capacity=cache_capacity,
        block_size=cli.block_size,
        kv_cache_size_per_token=kv_bytes_per_token,
        kv_cache_blocks=shadow_blocks,
        max_num_seqs=cli.max_num_seqs,
        max_num_batched_tokens=cli.max_num_batched_tokens,
        max_model_len=cli.max_model_len,
        request_generate_qps=cli.qps,
        arrival_speedup=cli.arrival_speedup,
        request_num=cli.request_num,
        warm_up_requests_num=0,
        warm_up_qps=0.5,
        requests_num_dataset_start=0,
        request_active_timeout=cli.request_timeout_s,
        dataset_type="jitserve",
        dataset_file=cli.trace,
        request_dataset_dir=str(Path(cli.trace).parent),
        trace_format="jitserve",
        slo_profile=cli.slo_profile,
        jitserve_request_ratio=cli.request_ratio,
        trace_seed=cli.trace_seed,
        client_region=cli.client_region,
        network_delay_mode=cli.network_delay_mode,
        routing_output_tokens_hint=cli.routing_output_tokens_hint,
        ravel_risk_epsilon=cli.ravel_risk_epsilon,
        ravel_collective_stage_output_profile=(
            cli.ravel_collective_stage_output_profile
        ),
        ravel_soft_admission_release_policy=cli.ravel_soft_admission_release_policy,
        ravel_soft_admission_reserve_sequences=cli.ravel_soft_admission_reserve_sequences,
        ravel_soft_admission_enabled=True,
        ravel_residual_window=cli.ravel_residual_window,
        ravel_output_quantile=cli.ravel_output_quantile,
        ravel_output_history_window=cli.ravel_output_history_window,
        ravel_output_min_samples=cli.ravel_output_min_samples,
        ravel_mobile_candidate_limit=cli.ravel_mobile_candidate_limit,
        ravel_mobile_beam_width=cli.ravel_mobile_beam_width,
        ravel_assignment_file=cli.ravel_assignment_file,
        ravel_assignment_arm=cli.ravel_assignment_arm,
        global_scheduler_type=cli.scheduler,
        balance_type="dualmap" if cli.scheduler == "dualmap" else "",
        cluster_topology=cli.cluster_topology,
        cluster_overload_fraction=0.5,
        cluster_rebalance_hysteresis_s=cli.rebalance_hysteresis_s,
        cluster_max_rebalances_per_event=cli.max_rebalances_per_event,
        replica_slo_budget=replica_budget,
        dh_rebalance_thredhold=cli.rebalance_token_threshold,
        dh_rebalance_waiting_latency_thredhold=cli.rebalance_wait_s,
        dh_replica_pending_req_threshold=pending_limit,
        result_path=cli.result_path,
        update_replica_info=True,
        window_duration=30,
        dh_recompute_punish_ratio=1.0,
        dh_cancel_rebalance_req=False,
        prefill_tpot=legacy_prefill_tpot,
        busy_prefill_interval=legacy_busy_prefill_interval,
        enable_scale=False,
    )


def queued_request_count(scheduler, replica_count: int) -> int:
    queue = getattr(scheduler, "global_request_queue", None)
    if queue is None and hasattr(scheduler, "double_hash_util"):
        queue = scheduler.double_hash_util.global_request_queue
    if queue is None:
        return 0
    return sum(queue.get_queue_len(replica_id) for replica_id in range(replica_count))


async def run(cli: argparse.Namespace) -> None:
    system_args = build_system_args(cli)
    launcher = SystemLauncher(system_args)
    await launcher.initialize(system_args)
    try:
        await launcher.run()
        idle_rounds = 0
        while idle_rounds < 4:
            scheduler = launcher._request_router_proxy.global_scheduler
            queued = queued_request_count(scheduler, system_args.replica_num)
            posting = len(launcher.shared_state.posting_request_tasks)
            if queued == 0 and posting == 0:
                idle_rounds += 1
            else:
                idle_rounds = 0
            await asyncio.sleep(0.5)
        await launcher.metric_store.sync_cache()
        expected = int(getattr(cli, "request_num", 0))
        rows = []
        try:
            with open(
                f"{system_args.result_path}/request_metrics.csv",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))
        except FileNotFoundError:
            rows = []
        unique_ids = {row.get("request_id") for row in rows}
        successes = sum(
            1
            for row in rows
            if float(row.get("time_to_first_token", 3600000)) < 3600000
        )
        if expected > 0 and (
            len(rows) < expected or successes < expected or len(unique_ids) < expected
        ):
            logger.error(
                "Validation failed: expected %d requests, got rows=%d "
                "successes=%d unique=%d; exiting non-zero",
                expected,
                len(rows),
                successes,
                len(unique_ids),
            )
            raise SystemExit(1)
        logger.info(
            "Validation passed: rows=%d successes=%d unique=%d",
            len(rows),
            successes,
            len(unique_ids),
        )
    finally:
        await launcher.stop()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
