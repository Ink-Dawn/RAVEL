#!/usr/bin/env python3
"""Run a resumable JITServe routing matrix on the three-cluster testbed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence


# Environment variables may override paths; fingerprints bind actual inputs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_jitserve_cluster_matrix import (
    POLICIES,
    Workload,
    atomic_write,
    format_float,
    prepare_workloads,
    summarize_csv,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("RAVEL_ROOT", str(REPO_ROOT)))
RAVEL_SRC = Path(os.environ.get("RAVEL_SRC_DIR", str(ROOT / "src")))
PYTHON = os.environ.get("RAVEL_PYTHON", sys.executable)
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "RAVEL_MODEL_PATH", str(ROOT / "models/Qwen3-1.7B")
    )
)
DEFAULT_SERVED_MODEL_NAME = os.environ.get("RAVEL_SERVED_MODEL_NAME", "qwen")
DEFAULT_RESULT = Path(
    os.environ.get(
        "RAVEL_RESULT_ROOT",
        str(ROOT / "results/ravel_three_cluster_run"),
    )
)
DEFAULT_TOPOLOGY = Path(
    os.environ.get(
        "RAVEL_TOPOLOGY",
        str(ROOT / "configs/three_cluster_2_2_1.local.json"),
    )
)
DEFAULT_COLLECTIVE_STAGE_OUTPUT_PROFILE = Path(
    os.environ.get(
        "RAVEL_COLLECTIVE_STAGE_OUTPUT_PROFILE",
        str(ROOT / "configs/output_profiles/jitserve_collective_stage_context_q95.schema-v2.json"),
    )
)
DEFAULT_ENDPOINTS = tuple(
    item.strip()
    for item in os.environ.get(
        "RAVEL_ENDPOINTS",
        "127.0.0.1:8000,127.0.0.1:8001,127.0.0.1:18000,127.0.0.1:18001,127.0.0.1:28000",
    ).split(",")
    if item.strip()
)
DEFAULT_POLICIES = ("RAVEL-Unified",)
DEFAULT_NETWORK_DELAY_MODE = os.environ.get(
    "RAVEL_NETWORK_DELAY_MODE", ""
)

RAVEL_UNIFIED_POLICIES = {
    "RAVEL-Unified",
}

# Policy parameters stay fixed across deployments. Engine dimensions and the
# client region are loaded from the calibrated topology at runtime.
POLICY_RUN_CONFIG: dict[str, int | float | str] = {
    "ttft_scale": float(os.environ.get("RAVEL_TTFT_SCALE", "1.0")),
    "kv_cache_blocks": 0,
    "kv_cache_dtype_bytes": 0,
    "kv_cache_size_per_token": 0,
    "replica_slo_budget_tokens": 0,
    "pending_request_limit": 0,
    "routing_output_tokens_hint": int(
        os.environ.get("RAVEL_ROUTING_OUTPUT_TOKENS_HINT", "256")
    ),
    "rebalance_token_threshold": 32768,
    "rebalance_wait_s": 0.5,
    "rebalance_hysteresis_s": 0.02,
    "max_rebalances_per_event": 8,
    "request_timeout_s": 3600.0,
    "ravel_risk_epsilon": 0.05,
    "ravel_residual_window": 256,
    "ravel_output_quantile": 0.9,
    "ravel_output_history_window": 256,
    "ravel_output_min_samples": 20,
    "ravel_mobile_beam_width": 256,
}


ENGINE_CONTRACT_FIELDS = (
    "dtype",
    "engine_impl",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "block_size",
    "apc",
    "chunked_prefill",
    "enforce_eager",
)


def parse_engine_fingerprint(raw: str) -> dict[str, str]:
    parts = [part.strip() for part in raw.split("|") if part.strip()]
    if len(parts) < 3:
        raise ValueError(f"invalid engine fingerprint: {raw!r}")
    fields = {"model": parts[0], "vllm": parts[1]}
    for part in parts[2:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    missing = [
        field for field in ENGINE_CONTRACT_FIELDS if field not in fields
    ]
    if missing:
        raise ValueError(
            f"engine fingerprint lacks required fields {missing}: {raw!r}"
        )
    return fields


def deployment_run_config(topology: Path) -> dict[str, int | float | str]:
    """Derive the engine contract from profiles, never source constants."""

    topology_payload = json.loads(topology.read_text(encoding="utf-8"))
    clusters = topology_payload.get("clusters", [])
    if not clusters:
        raise ValueError(f"topology has no clusters: {topology}")

    contracts = []
    for cluster in clusters:
        profile_value = cluster.get("service_profile")
        if not profile_value:
            raise ValueError(
                f"cluster {cluster.get('id')!r} has no calibrated "
                "service_profile"
            )
        profile_path = Path(str(profile_value))
        if not profile_path.is_absolute():
            profile_path = topology.parent / profile_path
        profile = json.loads(
            profile_path.resolve().read_text(encoding="utf-8")
        )
        contracts.append(
            (
                str(cluster.get("id")),
                parse_engine_fingerprint(
                    str(profile.get("engine_fingerprint", ""))
                ),
            )
        )

    reference_cluster, reference = contracts[0]
    common_fields = ("model", "vllm", *ENGINE_CONTRACT_FIELDS)
    for cluster_id, contract in contracts[1:]:
        differences = {
            field: (reference[field], contract[field])
            for field in common_fields
            if reference[field] != contract[field]
        }
        if differences:
            raise ValueError(
                "all replicas in one matrix must share an engine contract; "
                f"{reference_cluster} vs {cluster_id}: {differences}"
            )
    for field in ("apc", "chunked_prefill", "enforce_eager"):
        if reference[field] != "on":
            raise ValueError(f"formal runs require {field}=on")

    config = dict(POLICY_RUN_CONFIG)
    for field in (
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "block_size",
    ):
        config[field] = int(reference[field])
    config["client_region"] = str(
        topology_payload.get("default_client_region", "region_a")
    )
    config["model_fingerprint_name"] = reference["model"]
    config["vllm_fingerprint"] = reference["vllm"]
    config["dtype"] = reference["dtype"]
    dtype_bytes = {
        "float": 4,
        "float32": 4,
        "fp32": 4,
        "half": 2,
        "float16": 2,
        "fp16": 2,
        "bfloat16": 2,
        "bf16": 2,
        "fp8": 1,
    }
    normalized_dtype = str(reference["dtype"]).lower()
    if normalized_dtype not in dtype_bytes:
        raise ValueError(
            "cannot derive KV dtype bytes from calibrated dtype "
            f"{reference['dtype']!r}"
        )
    config["kv_cache_dtype_bytes"] = dtype_bytes[normalized_dtype]
    # run_cluster derives a zero pending limit from max_num_seqs. The joint
    # planner still needs the explicit calibrated capacity.
    config["ravel_mobile_candidate_limit"] = int(reference["max_num_seqs"])
    return config


def parse_csv_list(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest() -> str:
    paths = [RAVEL_SRC / "run_cluster.py"]
    paths.extend(sorted((RAVEL_SRC / "dualmap").rglob("*.py")))
    paths.extend(
        sorted(
            (REPO_ROOT / "src" / "ravel_engine_adapter").glob("*.py")
        )
    )
    paths.append(
        REPO_ROOT / "runtime" / "ravel_vllm_adapter" / "sitecustomize.py"
    )
    paths.extend(
        REPO_ROOT / "scripts" / name
        for name in (
            "run_jitserve_cluster_matrix.py",
            "run_jitserve_three_cluster_matrix.py",
        )
    )
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(resolved)))
    return digest.hexdigest()


def topology_profile_digests(topology: Path) -> dict[str, str]:
    raw = json.loads(topology.read_text(encoding="utf-8"))
    profiles: dict[str, str] = {}
    for cluster in raw.get("clusters", []):
        cluster_id = cluster.get("id")
        profile_value = cluster.get("service_profile")
        if not profile_value:
            raise ValueError(
                f"cluster {cluster_id!r} lacks an external service_profile"
            )
        profile = Path(str(profile_value))
        if not profile.is_absolute():
            profile = topology.parent / profile
        profiles[str(cluster["id"])] = sha256_file(profile.resolve())
    return profiles


def cell_fingerprint(
    *,
    source_sha256: str,
    topology: Path,
    workload: Workload,
    speedup: float,
    policy: str,
    expected: int,
    endpoints: Sequence[str],
    model_path: Path,
    served_model_name: str,
    output_profile_path: Path,
    run_config: Mapping[str, int | float | str],
) -> tuple[str, dict[str, object]]:
    output_profile = None
    if policy in RAVEL_UNIFIED_POLICIES:
        output_profile_path = output_profile_path.resolve()
        if not output_profile_path.is_file():
            raise FileNotFoundError(output_profile_path)
        output_profile = {
            "path": str(output_profile_path),
            "sha256": sha256_file(output_profile_path),
        }
    effective_engine_priority = (
        policy in RAVEL_UNIFIED_POLICIES
        or os.environ.get("RAVEL_ENGINE_PRIORITY", "0") == "1"
    )
    payload: dict[str, object] = {
        "schema": "ravel-three-cluster-cell-v2",
        "source_sha256": source_sha256,
        "topology_sha256": sha256_file(topology.resolve()),
        "service_profile_sha256": topology_profile_digests(topology),
        "trace_sha256": sha256_file(workload.trace.resolve()),
        "model_path": str(model_path.resolve()),
        "served_model_name": served_model_name,
        "endpoints": list(endpoints),
        "dataset": workload.name,
        "slo_profile": workload.slo_profile,
        "request_ratio": workload.request_ratio,
        "request_count": expected,
        "trace_seed": 42,
        "speedup": float(speedup),
        "policy": policy,
        "scheduler": POLICIES[policy],
        "apc": True,
        "chunked_prefill": True,
        "engine_scheduling_policy": os.environ.get(
            "RAVEL_ENGINE_SCHEDULING_POLICY", "priority"
        ),
        "ravel_engine_priority": effective_engine_priority,
        "ravel_engine_priority_mode": os.environ.get(
            "RAVEL_ENGINE_PRIORITY_MODE", "selected_edf"
        ),
        "ravel_engine_adapter_mode": os.environ.get(
            "RAVEL_ENGINE_ADAPTER_MODE", "feature_detected"
        ),
        "network_delay_mode": str(run_config["network_delay_mode"]),
        "ravel_sidecar_enable_abort": (
            os.environ.get("RAVEL_SIDECAR_ENABLE_ABORT", "0") == "1"
        ),
        "ravel_sidecar_engine_buffer": (
            os.environ.get("RAVEL_SIDECAR_ENGINE_BUFFER", "0") == "1"
        ),
        "collective_stage_output_profile": output_profile,
        "diagnostic_realized_output": (
            os.environ.get(
                "RAVEL_DIAGNOSTIC_ALLOW_REALIZED_OUTPUT", "0"
            )
            == "1"
        ),
        "diagnostic_completion_order": os.environ.get(
            "RAVEL_DIAGNOSTIC_COMPLETION_ORDER", "edf"
        ),
        "formal_run_config": dict(run_config),
        "metric_schema": "jitserve-token-exact-flat-first500-v5",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload


def direct_request(url: str, method: str = "GET", timeout: float = 15.0) -> bytes:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {}
    api_key = os.environ.get("RAVEL_API_KEY") or os.environ.get(
        "VLLM_API_KEY"
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, method=method, headers=headers)
    for attempt in range(3):
        try:
            with opener.open(request, timeout=timeout) as response:
                if response.status >= 300:
                    raise RuntimeError(
                        f"{method} {url}: HTTP {response.status}"
                    )
                return response.read()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt >= 2:
                raise
            time.sleep(0.25 * (2**attempt))
    raise AssertionError("unreachable")


def endpoint_url(endpoint: str, path: str) -> str:
    base = str(endpoint).rstrip("/")
    if "://" not in base:
        base = f"http://{base}"
    return f"{base}{path}"


def ensure_ready(
    endpoints: Sequence[str], expected_model: str = ""
) -> None:
    failures = []
    for endpoint in endpoints:
        try:
            payload = json.loads(
                direct_request(endpoint_url(endpoint, "/v1/models"))
            )
            models = [
                str(row.get("id", ""))
                for row in payload.get("data", [])
                if row.get("id")
            ]
            if not models:
                failures.append(f"{endpoint}: no served models")
            elif expected_model and expected_model not in models:
                failures.append(
                    f"{endpoint}: expected model {expected_model!r}, "
                    f"found {models}"
                )
        except Exception as error:  # noqa: BLE001
            failures.append(f"{endpoint}: {error}")
    if failures:
        raise RuntimeError("endpoint readiness failed: " + "; ".join(failures))


def reset_caches(endpoints: Sequence[str]) -> None:
    failures = []
    for endpoint in endpoints:
        try:
            direct_request(
                endpoint_url(endpoint, "/reset_prefix_cache"),
                method="POST",
            )
        except Exception as error:  # noqa: BLE001
            failures.append(f"{endpoint}: {error}")
    if failures:
        raise RuntimeError("prefix cache reset failed: " + "; ".join(failures))


def cluster_counts(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    counts = {"region_a": 0, "region_b": 0, "region_c": 0}
    for row in rows:
        cluster = row.get("cluster_id", "")
        counts[cluster] = counts.get(cluster, 0) + 1
    return counts


def discover(result_root: Path, expected: int) -> list[dict[str, object]]:
    summaries = []
    for path in sorted((result_root / "cells").glob("*/*/*/request_metrics.csv")):
        dataset, speedup_label, policy = path.relative_to(result_root / "cells").parts[:3]
        row = summarize_csv(path, expected)
        row.update(
            dataset=dataset,
            speedup=float(speedup_label.removeprefix("speedup_")),
            policy=policy,
            cluster_counts=cluster_counts(path),
            path=str(path),
        )
        # The frozen DeepResearch first-500 trace intentionally truncates
        # workflows. Request metrics remain valid, but workflow Task SLO does
        # not; LMSYS/Burst have no workflow identity either.
        row["task_slo"] = math.nan
        summaries.append(row)
    return summaries


def best_value(
    group: Sequence[Mapping[str, object]], key: str, maximize: bool
) -> float:
    values = [
        float(row[key])
        for row in group
        if isinstance(row.get(key), (int, float))
        and math.isfinite(float(row[key]))
    ]
    if not values:
        return math.nan
    return max(values) if maximize else min(values)


def metric(
    row: Mapping[str, object],
    key: str,
    optimum: float,
    digits: int = 3,
    scale: float = 1.0,
    suffix: str = "",
) -> str:
    value = float(row[key])
    rendered = format_float(value * scale, digits) + suffix
    if math.isfinite(optimum) and math.isclose(value, optimum, rel_tol=1e-9, abs_tol=1e-12):
        return f"**{rendered}**"
    return rendered


def topology_summary(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    client_region = str(payload.get("default_client_region", "region_a"))
    values = []
    for cluster in payload.get("clusters", []):
        cluster_id = str(cluster.get("id", ""))
        row = cluster.get("rtt_ms_by_client_region", {})
        if client_region not in row:
            raise ValueError(
                f"cluster {cluster_id!r} has no RTT from {client_region!r}"
            )
        values.append(f"{cluster_id}={float(row[client_region]):g} ms")
    return f"client={client_region}; " + ", ".join(values)


def render_report(
    result_root: Path,
    expected: int,
    workloads: Mapping[str, Workload],
    speedups: Sequence[float],
    policies: Sequence[str],
    endpoints: Sequence[str],
    topology: Path,
    rows: Sequence[Mapping[str, object]],
) -> str:
    completed = sum(bool(row["complete"]) for row in rows)
    total = len(workloads) * len(speedups) * len(policies)
    lines = [
        "# JITServe Three-Cluster Routing Matrix",
        "",
        f"- Progress: **{completed}/{total} complete cells**",
        f"- Requests per cell: **{expected}**",
        f"- Speedups: **{', '.join(format_float(value, 0) for value in speedups)}x**",
        f"- Policies: **{', '.join(policies)}**",
        f"- Endpoints: **{len(endpoints)} replicas**; APC and Chunked Prefill enabled.",
        f"- Topology: `{topology}`; {topology_summary(topology)}.",
        "- Cache protocol: every endpoint cache is reset before every cell.",
        "- Bold values are best within the same dataset and speedup.",
        "",
        "## Primary Performance",
        "",
        "| Dataset | Speedup | Policy | A/B/C | SLO | Task SLO | TTFT mean (s) | TTFT p95 (s) | E2E mean (s) | E2E p95 (s) |",
        "|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["dataset"]),
            float(row["speedup"]),
            str(row["policy"]),
        ),
    )
    best_by_group = {}
    for row in ordered:
        group_key = (row["dataset"], row["speedup"])
        if group_key not in best_by_group:
            group = [
                item
                for item in rows
                if item["dataset"] == row["dataset"]
                and item["speedup"] == row["speedup"]
            ]
            best_by_group[group_key] = {
                key: best_value(group, key, maximize)
                for key, maximize in (
                    ("slo", True),
                    ("task_slo", True),
                    ("ttft_mean", False),
                    ("ttft_p95", False),
                    ("e2e_mean", False),
                    ("e2e_p95", False),
                    ("tpot_mean_ms", False),
                    ("wait_mean", False),
                    ("prefix_tokens_per_request", True),
                    ("prefix_hit_ratio", True),
                    ("prompt_tokens_per_s", True),
                )
            }
        best = best_by_group[group_key]
        counts = row["cluster_counts"]
        task_slo = (
            metric(row, "task_slo", best["task_slo"], 1, 100.0, "%")
            if math.isfinite(float(row["task_slo"]))
            else "-"
        )
        lines.append(
            "| {dataset} | {speedup}x | {policy} | {a}/{b}/{c} | {slo} | {task} | "
            "{ttft_mean} | {ttft_p95} | {e2e_mean} | {e2e_p95} |".format(
                dataset=row["dataset"],
                speedup=format_float(row["speedup"], 0),
                policy=row["policy"],
                a=counts.get("region_a", 0),
                b=counts.get("region_b", 0),
                c=counts.get("region_c", 0),
                slo=metric(row, "slo", best["slo"], 1, 100.0, "%"),
                task=task_slo,
                ttft_mean=metric(row, "ttft_mean", best["ttft_mean"]),
                ttft_p95=metric(row, "ttft_p95", best["ttft_p95"]),
                e2e_mean=metric(row, "e2e_mean", best["e2e_mean"]),
                e2e_p95=metric(row, "e2e_p95", best["e2e_p95"]),
            )
        )
    lines.extend(
        [
            "",
            "## Efficiency And Locality",
            "",
            "| Dataset | Speedup | Policy | TPOT (ms) | Wait (s) | Prefix tok/req | Prefix hit | Prompt tok/s |",
            "|:---|---:|:---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ordered:
        best = best_by_group[(row["dataset"], row["speedup"])]
        lines.append(
            "| {dataset} | {speedup}x | {policy} | {tpot} | {wait} | "
            "{prefix} | {prefix_ratio} | {throughput} |".format(
                dataset=row["dataset"],
                speedup=format_float(row["speedup"], 0),
                policy=row["policy"],
                tpot=metric(row, "tpot_mean_ms", best["tpot_mean_ms"], 2),
                wait=metric(row, "wait_mean", best["wait_mean"]),
                prefix=metric(
                    row,
                    "prefix_tokens_per_request",
                    best["prefix_tokens_per_request"],
                    1,
                ),
                prefix_ratio=metric(
                    row,
                    "prefix_hit_ratio",
                    best["prefix_hit_ratio"],
                    1,
                    100.0,
                    "%",
                ),
                throughput=metric(
                    row,
                    "prompt_tokens_per_s",
                    best["prompt_tokens_per_s"],
                    1,
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- The 2/2/1 pilot is capacity-aware but not the final symmetric 2/2/2 result.",
            "- All policies use the same endpoints, RTT injection, trace, SLO, and cache-reset protocol.",
            "- Realized output length, future arrivals, and dataset identity are not available to RAVEL online decisions.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(
    result_root: Path,
    expected: int,
    workloads: Mapping[str, Workload],
    speedups: Sequence[float],
    policies: Sequence[str],
    endpoints: Sequence[str],
    topology: Path,
) -> None:
    rows = discover(result_root, expected)
    atomic_write(result_root / "summary.json", json.dumps(rows, indent=2, sort_keys=True, allow_nan=True))
    atomic_write(
        result_root / "RESULTS.md",
        render_report(result_root, expected, workloads, speedups, policies, endpoints, topology, rows),
    )


def cell_complete(path: Path, expected: int) -> bool:
    csv_path = path / "request_metrics.csv"
    return csv_path.exists() and bool(summarize_csv(csv_path, expected)["complete"])


def cell_reusable(path: Path, expected: int, fingerprint: str) -> bool:
    if not cell_complete(path, expected):
        return False
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("returncode") == 0
        and manifest.get("experiment_fingerprint") == fingerprint
    )


def run_cell(
    result_root: Path,
    workload: Workload,
    speedup: float,
    policy: str,
    expected: int,
    timeout_s: float,
    endpoints: Sequence[str],
    topology: Path,
    source_sha256: str,
    model_path: Path,
    served_model_name: str,
    output_profile_path: Path,
    run_config: Mapping[str, int | float | str],
) -> None:
    cell = result_root / "cells" / workload.name / f"speedup_{speedup:g}" / policy
    fingerprint, fingerprint_inputs = cell_fingerprint(
        source_sha256=source_sha256,
        topology=topology,
        workload=workload,
        speedup=speedup,
        policy=policy,
        expected=expected,
        endpoints=endpoints,
        model_path=model_path,
        served_model_name=served_model_name,
        output_profile_path=output_profile_path,
        run_config=run_config,
    )
    if cell_reusable(cell, expected, fingerprint):
        return
    if cell_complete(cell, expected):
        raise RuntimeError(
            f"completed cell has a stale fingerprint: {cell}; use a new result root"
        )
    cell.mkdir(parents=True, exist_ok=True)
    reset_caches(endpoints)
    ensure_ready(endpoints, served_model_name)
    time.sleep(1.0)
    command = [
        str(PYTHON),
        str(RAVEL_SRC / "run_cluster.py"),
        "--replicas",
        ",".join(endpoints),
        "--cluster-topology",
        str(topology),
        "--scheduler",
        POLICIES[policy],
        "--model-path",
        str(model_path),
        "--model-name",
        served_model_name,
        "--trace",
        str(workload.trace),
        "--result-path",
        str(cell),
        "--request-num",
        str(expected),
        "--arrival-speedup",
        str(speedup),
        "--slo-profile",
        workload.slo_profile,
        "--request-ratio",
        workload.request_ratio,
        "--trace-seed",
        "42",
        "--client-region",
        str(run_config["client_region"]),
        "--network-delay-mode",
        str(run_config["network_delay_mode"]),
        "--max-model-len",
        str(run_config["max_model_len"]),
        "--max-num-seqs",
        str(run_config["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(run_config["max_num_batched_tokens"]),
        "--block-size",
        str(run_config["block_size"]),
        "--kv-cache-blocks",
        str(run_config["kv_cache_blocks"]),
        "--kv-cache-dtype-bytes",
        str(run_config["kv_cache_dtype_bytes"]),
        "--kv-cache-size-per-token",
        str(run_config["kv_cache_size_per_token"]),
        "--replica-slo-budget-tokens",
        str(run_config["replica_slo_budget_tokens"]),
        "--pending-request-limit",
        str(run_config["pending_request_limit"]),
        "--routing-output-tokens-hint",
        str(run_config["routing_output_tokens_hint"]),
        "--rebalance-token-threshold",
        str(run_config["rebalance_token_threshold"]),
        "--rebalance-wait-s",
        str(run_config["rebalance_wait_s"]),
        "--rebalance-hysteresis-s",
        str(run_config["rebalance_hysteresis_s"]),
        "--max-rebalances-per-event",
        str(run_config["max_rebalances_per_event"]),
        "--request-timeout-s",
        str(run_config["request_timeout_s"]),
        "--ravel-risk-epsilon",
        str(run_config["ravel_risk_epsilon"]),
        "--ravel-residual-window",
        str(run_config["ravel_residual_window"]),
        "--ravel-output-quantile",
        str(run_config["ravel_output_quantile"]),
        "--ravel-output-history-window",
        str(run_config["ravel_output_history_window"]),
        "--ravel-output-min-samples",
        str(run_config["ravel_output_min_samples"]),
        "--ravel-mobile-candidate-limit",
        str(run_config["ravel_mobile_candidate_limit"]),
        "--ravel-mobile-beam-width",
        str(run_config["ravel_mobile_beam_width"]),
    ]
    if policy in RAVEL_UNIFIED_POLICIES:
        command.extend(
            [
                "--ravel-collective-stage-output-profile",
                str(output_profile_path.resolve()),
            ]
        )
    manifest = {
        "command": command,
        "dataset": workload.name,
        "endpoints": list(endpoints),
        "policy": policy,
        "speedup": speedup,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "topology": str(topology),
        "experiment_fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
    }
    atomic_write(cell / "manifest.json", json.dumps(manifest, indent=2))
    child_environment = os.environ.copy()
    child_environment["RAVEL_TTFT_SCALE"] = str(
        run_config["ttft_scale"]
    )
    if policy in RAVEL_UNIFIED_POLICIES:
        child_environment["RAVEL_ENGINE_PRIORITY"] = "1"
        child_environment["RAVEL_COLLECTIVE_STAGE_OUTPUT_PROFILE"] = str(
            output_profile_path.resolve()
        )
    manifest["environment"] = {
        "RAVEL_TTFT_SCALE": child_environment["RAVEL_TTFT_SCALE"],
        "RAVEL_ENGINE_PRIORITY": child_environment.get(
            "RAVEL_ENGINE_PRIORITY", "0"
        ),
        "RAVEL_ENGINE_SCHEDULING_POLICY": child_environment.get(
            "RAVEL_ENGINE_SCHEDULING_POLICY", "priority"
        ),
        "RAVEL_ENGINE_PRIORITY_MODE": child_environment.get(
            "RAVEL_ENGINE_PRIORITY_MODE", "selected_edf"
        ),
        "RAVEL_ENGINE_ADAPTER_MODE": child_environment.get(
            "RAVEL_ENGINE_ADAPTER_MODE", "feature_detected"
        ),
        "RAVEL_SIDECAR_ENABLE_ABORT": child_environment.get(
            "RAVEL_SIDECAR_ENABLE_ABORT", "0"
        ),
        "RAVEL_SIDECAR_ENGINE_BUFFER": child_environment.get(
            "RAVEL_SIDECAR_ENGINE_BUFFER", "0"
        ),
        "RAVEL_COLLECTIVE_STAGE_OUTPUT_PROFILE": child_environment.get(
            "RAVEL_COLLECTIVE_STAGE_OUTPUT_PROFILE", ""
        ),
        "RAVEL_DIAGNOSTIC_ALLOW_REALIZED_OUTPUT": child_environment.get(
            "RAVEL_DIAGNOSTIC_ALLOW_REALIZED_OUTPUT", "0"
        ),
        "RAVEL_DIAGNOSTIC_COMPLETION_ORDER": child_environment.get(
            "RAVEL_DIAGNOSTIC_COMPLETION_ORDER", "edf"
        ),
        "RAVEL_ROUTING_OUTPUT_TOKENS_HINT": str(
            run_config["routing_output_tokens_hint"]
        ),
    }
    atomic_write(cell / "manifest.json", json.dumps(manifest, indent=2))
    with (cell / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=RAVEL_SRC,
            env=child_environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    manifest["returncode"] = result.returncode
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_write(cell / "manifest.json", json.dumps(manifest, indent=2))
    if result.returncode != 0:
        raise RuntimeError(f"cell failed: {workload.name} {speedup:g}x {policy}")
    if not cell_complete(cell, expected):
        raise RuntimeError(f"cell incomplete: {workload.name} {speedup:g}x {policy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--served-model-name", default=DEFAULT_SERVED_MODEL_NAME
    )
    parser.add_argument(
        "--collective-stage-output-profile",
        type=Path,
        default=DEFAULT_COLLECTIVE_STAGE_OUTPUT_PROFILE,
    )
    parser.add_argument("--endpoints", default=",".join(DEFAULT_ENDPOINTS))
    parser.add_argument(
        "--network-delay-mode",
        choices=("synthetic", "physical"),
        default=DEFAULT_NETWORK_DELAY_MODE,
        help=(
            "required: physical for real WAN endpoints; synthetic only for "
            "same-host RTT emulation"
        ),
    )
    parser.add_argument("--workloads", default="lmsys,burst,deepresearch")
    parser.add_argument("--speedups", default="1,2,4,8,12,16")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--request-count", type=int, default=500)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Child cells run with RAVEL_SRC as their cwd. Resolve the result root before
    # preparing workloads or constructing per-cell paths so a caller-provided
    # relative path cannot be written under src/ and checked under the repo.
    args.result_root = args.result_root.resolve()
    endpoints = parse_csv_list(args.endpoints)
    speedups = tuple(float(item) for item in parse_csv_list(args.speedups))
    policies = parse_csv_list(args.policies)
    if args.network_delay_mode not in {"synthetic", "physical"}:
        raise ValueError(
            "set --network-delay-mode or RAVEL_NETWORK_DELAY_MODE to "
            "physical (real WAN) or synthetic (same-host emulation)"
        )
    unknown = [policy for policy in policies if policy not in POLICIES]
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    all_workloads = prepare_workloads(args.result_root, args.request_count)
    names = parse_csv_list(args.workloads)
    workloads = {name: all_workloads[name] for name in names}
    args.topology = args.topology.resolve()
    topology_profile_digests(args.topology)
    run_config = deployment_run_config(args.topology)
    run_config["network_delay_mode"] = args.network_delay_mode
    args.model_path = args.model_path.resolve()
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    if args.model_path.name != run_config["model_fingerprint_name"]:
        raise ValueError(
            "model path does not match calibrated profiles: "
            f"path={args.model_path.name!r}, "
            f"profile={run_config['model_fingerprint_name']!r}"
        )
    ensure_ready(endpoints, args.served_model_name)
    args.collective_stage_output_profile = (
        args.collective_stage_output_profile.resolve()
    )
    current_source_digest = source_digest()
    args.result_root.mkdir(parents=True, exist_ok=True)
    for workload in workloads.values():
        for speedup in speedups:
            for policy in policies:
                run_cell(
                    args.result_root,
                    workload,
                    speedup,
                    policy,
                    args.request_count,
                    args.timeout_s,
                    endpoints,
                    args.topology,
                    current_source_digest,
                    args.model_path,
                    args.served_model_name,
                    args.collective_stage_output_profile,
                    run_config,
                )
                write_reports(
                    args.result_root,
                    args.request_count,
                    workloads,
                    speedups,
                    policies,
                    endpoints,
                    args.topology,
                )


if __name__ == "__main__":
    main()
