#!/usr/bin/env python3
"""Run and summarize the two-cluster JITServe routing matrix.

The campaign is intentionally self-contained and resumable. Every cell uses
the same four APC-on/chunked-prefill vLLM endpoints and resets their prefix
cache before dispatching requests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("RAVEL_ROOT", str(REPO_ROOT)))
RAVEL_SRC = Path(os.environ.get("RAVEL_SRC_DIR", str(ROOT / "src")))
if str(RAVEL_SRC) not in sys.path:
    sys.path.insert(0, str(RAVEL_SRC))
from dualmap.cluster.slo import collective_task_deadline_s
PYTHON = os.environ.get("RAVEL_PYTHON", sys.executable)
FROZEN_DATA = ROOT / "data"
MODEL_PATH = Path(
    os.environ.get(
        "RAVEL_MODEL_PATH", str(ROOT / "models/Qwen3-1.7B")
    )
)
TOPOLOGY = Path(
    os.environ.get("RAVEL_TOPOLOGY", str(ROOT / "configs/two_cluster.example.json"))
)
DEFAULT_RESULT = Path(
    os.environ.get(
        "RAVEL_RESULT_ROOT", str(ROOT / "results/jitserve_cluster_matrix_v1")
    )
)
ENDPOINTS = tuple(
    item.strip()
    for item in os.environ.get(
        "RAVEL_ENDPOINTS",
        "127.0.0.1:8000,127.0.0.1:8001,127.0.0.1:18000,127.0.0.1:18001",
    ).split(",")
    if item.strip()
)
DEFAULT_NETWORK_DELAY_MODE = os.environ.get(
    "RAVEL_NETWORK_DELAY_MODE", ""
)


POLICIES = {
    "RAVEL-Unified": "ravel_unified",
}

DEFAULT_POLICY_NAMES = ("RAVEL-Unified",)

@dataclass(frozen=True)
class Workload:
    name: str
    trace: Path
    slo_profile: str
    request_ratio: str
    description: str


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def prepare_workloads(result_root: Path, limit: int) -> dict[str, Workload]:
    del result_root
    if not 0 < limit <= 500:
        raise ValueError("frozen first-500 workloads support 1 through 500 requests")

    traces = {
        "lmsys": FROZEN_DATA / "lmsys_first500.json",
        "burst": FROZEN_DATA / "burst_first500.json",
        "deepresearch": FROZEN_DATA / "deepresearch_flat_first500.json",
    }
    for name, trace_path in traces.items():
        if not trace_path.exists():
            raise FileNotFoundError(f"missing frozen {name} workload: {trace_path}")
        records = json.loads(trace_path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or len(records) != 500:
            raise ValueError(
                f"frozen {name} workload must contain exactly 500 requests"
            )

    return {
        "lmsys": Workload(
            "lmsys",
            traces["lmsys"],
            "paper_e2e",
            "3,5,2",
            "Frozen first 500 flat LMSYS requests with native arrivals.",
        ),
        "burst": Workload(
            "burst",
            traces["burst"],
            "paper_e2e",
            "3,5,2",
            "Frozen LMSYS request content with the first 500 BurstGPT arrivals.",
        ),
        "deepresearch": Workload(
            "deepresearch",
            traces["deepresearch"],
            "paper_collective",
            "",
            "Frozen globally sorted DeepResearch branches, first 500 requests; "
            "workflows may be truncated and Task SLO is not reported.",
        ),
    }


def http_request(url: str, method: str = "GET", timeout: float = 10.0) -> bytes:
    headers = {}
    api_key = os.environ.get("RAVEL_API_KEY") or os.environ.get(
        "VLLM_API_KEY"
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, method=method, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        if response.status >= 300:
            raise RuntimeError(f"{method} {url}: HTTP {response.status}")
        return response.read()


def endpoint_url(endpoint: str, path: str) -> str:
    base = str(endpoint).rstrip("/")
    if "://" not in base:
        base = f"http://{base}"
    return f"{base}{path}"


def ensure_endpoints_ready() -> None:
    failures = []
    for endpoint in ENDPOINTS:
        try:
            payload = json.loads(
                http_request(endpoint_url(endpoint, "/v1/models"))
            )
            if not payload.get("data"):
                failures.append(f"{endpoint}: no served models")
        except Exception as error:  # noqa: BLE001 - aggregate readiness failures
            failures.append(f"{endpoint}: {error}")
    if failures:
        raise RuntimeError("endpoint readiness failed: " + "; ".join(failures))


def reset_prefix_caches() -> None:
    failures = []
    for endpoint in ENDPOINTS:
        try:
            http_request(
                endpoint_url(endpoint, "/reset_prefix_cache"),
                method="POST",
            )
        except Exception as error:  # noqa: BLE001 - aggregate reset failures
            failures.append(f"{endpoint}: {error}")
    if failures:
        raise RuntimeError("prefix cache reset failed: " + "; ".join(failures))


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def numeric(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def summarize_csv(path: Path, expected: int) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    valid = [
        row
        for row in rows
        if numeric(row, "time_to_first_token", 3_600_000) < 3_600_000
        and numeric(row, "request_latency", 3_600_000) < 3_600_000
    ]
    output_exact = all(
        int(numeric(row, "actual_output_tokens", -1)) == int(numeric(row, "output_len", -2))
        for row in valid
    )
    prompt_exact = all(
        int(numeric(row, "actual_prompt_tokens", -1))
        == int(numeric(row, "input_len", -2))
        for row in valid
    )
    tbt_exact = all(
        int(numeric(row, "tbt_measurement_valid", 0)) == 1
        for row in valid
    )
    ttft = [numeric(row, "time_to_first_token") for row in valid]
    e2e = [numeric(row, "request_latency") for row in valid]
    tpot = [numeric(row, "tpot(ms)") for row in valid]
    waits = [numeric(row, "wait_queue_s") for row in valid]
    input_tokens = sum(numeric(row, "input_len") for row in valid)
    prefix_tokens = sum(numeric(row, "prefix_hit_prompt_tokens") for row in valid)
    arrivals = [numeric(row, "req_arrived_at") for row in valid]
    endings = [numeric(row, "request_end_time") for row in valid]
    duration = max(endings, default=0.0) - min(arrivals, default=0.0)
    cluster_a = sum(row.get("cluster_id") == "region_a" for row in valid)
    cluster_b = sum(row.get("cluster_id") == "region_b" for row in valid)
    slo_no_tbt = (
        sum(
            int(numeric(row, "ttft_slo_met"))
            if int(numeric(row, "request_type", -1)) == 0
            else int(numeric(row, "request_slo_met"))
            for row in valid
        )
        / len(valid)
        if valid
        else math.nan
    )
    by_type = {}
    for request_type, label in ((0, "latency"), (1, "throughput"), (2, "collective")):
        subset = [row for row in valid if int(numeric(row, "request_type", -1)) == request_type]
        if subset:
            by_type[label] = sum(int(numeric(row, "request_slo_met")) for row in subset) / len(subset)
    task_slo = math.nan
    task_groups_complete = True
    task_group_count = 0
    collective_rows = [row for row in valid if int(numeric(row, "request_type", -1)) == 2]
    if collective_rows:
        groups: dict[str, list[Mapping[str, str]]] = {}
        for row in collective_rows:
            groups.setdefault(row.get("collection_id", ""), []).append(row)
        successes = 0
        task_group_count = len(groups)
        for collection_id, group in groups.items():
            stage_num = max(int(numeric(row, "stage_num", 1)) for row in group)
            expected_branches = max(
                int(numeric(row, "task_branch_count", 1)) for row in group
            )
            observed_stages = {
                int(numeric(row, "stage_id", 0)) for row in group
            }
            group_complete = (
                len(group) == expected_branches
                and observed_stages == set(range(stage_num))
            )
            task_groups_complete = task_groups_complete and group_complete
            if not group_complete:
                continue
            task_latency = max(numeric(row, "request_end_time") for row in group) - min(
                numeric(row, "req_arrived_at") for row in group
            )
            request_deadline = max(
                numeric(row, "slo_ttlt_s", 0.0) for row in group
            )
            task_deadline = collective_task_deadline_s(
                request_deadline,
                int(float(collection_id or 0)),
                stage_num,
            )
            successes += task_deadline > 0 and task_latency <= task_deadline
        if task_groups_complete:
            task_slo = successes / len(groups)
    return {
        "rows": len(rows),
        "valid": len(valid),
        "complete": (
            len(rows) == expected
            and len(valid) == expected
            and output_exact
            and prompt_exact
            and tbt_exact
        ),
        "output_exact": output_exact,
        "prompt_exact": prompt_exact,
        "tbt_exact": tbt_exact,
        "task_groups_complete": task_groups_complete,
        "task_group_count": task_group_count,
        "a_count": cluster_a,
        "b_count": cluster_b,
        "a_ratio": cluster_a / len(valid) if valid else math.nan,
        "slo": sum(int(numeric(row, "request_slo_met")) for row in valid) / len(valid) if valid else math.nan,
        "slo_no_tbt": slo_no_tbt,
        "task_slo": task_slo,
        "slo_by_type": by_type,
        "ttft_mean": sum(ttft) / len(ttft) if ttft else math.nan,
        "ttft_p20": percentile(ttft, 0.20),
        "ttft_p50": percentile(ttft, 0.50),
        "ttft_p95": percentile(ttft, 0.95),
        "ttft_p99": percentile(ttft, 0.99),
        "e2e_mean": sum(e2e) / len(e2e) if e2e else math.nan,
        "e2e_p20": percentile(e2e, 0.20),
        "e2e_p50": percentile(e2e, 0.50),
        "e2e_p95": percentile(e2e, 0.95),
        "e2e_p99": percentile(e2e, 0.99),
        "tpot_mean_ms": sum(tpot) / len(tpot) if tpot else math.nan,
        "wait_mean": sum(waits) / len(waits) if waits else math.nan,
        "prefix_tokens_per_request": prefix_tokens / len(valid) if valid else math.nan,
        "prefix_hit_ratio": prefix_tokens / input_tokens if input_tokens else 0.0,
        "prompt_tokens_per_s": input_tokens / duration if duration > 0 else math.nan,
        "duration_s": duration,
    }


def format_float(value: object, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or math.isnan(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def discover_results(result_root: Path, expected: int) -> list[dict[str, object]]:
    summaries = []
    cells = result_root / "cells"
    if not cells.exists():
        return summaries
    for csv_path in sorted(cells.glob("*/*/*/request_metrics.csv")):
        relative = csv_path.relative_to(cells)
        dataset, speedup_label, policy = relative.parts[:3]
        summary = summarize_csv(csv_path, expected)
        summary.update(
            {
                "dataset": dataset,
                "speedup": float(speedup_label.removeprefix("speedup_")),
                "policy": policy,
                "path": str(csv_path),
            }
        )
        if dataset != "deepresearch":
            summary["task_slo"] = math.nan
        summaries.append(summary)
    return summaries


def render_report(
    result_root: Path,
    expected: int,
    workloads: Mapping[str, Workload],
    speedups: Sequence[float],
    summaries: Sequence[Mapping[str, object]],
) -> str:
    completed = sum(bool(item["complete"]) for item in summaries)
    total = len(workloads) * len(speedups) * len(POLICIES)
    lines = [
        "# JITServe Two-Cluster Routing Matrix",
        "",
        f"- Progress: **{completed}/{total} complete cells**",
        f"- Requests per cell: **{expected}**",
        f"- Speedups: **{', '.join(format_float(value, 0) for value in speedups)}x**",
        "- Policy: **RAVEL-Unified**",
        "- Backends: four identical vLLM replicas; APC and Chunked Prefill enabled.",
        "- Topology: region A replicas 0-1, region B replicas 2-3; "
        "client in A; data-plane RTT 0.5 ms to A and 100 ms to B.",
        "- Cache protocol: all four prefix caches are reset before every cell.",
        "- SLO: JITServe base (0.8 s TTFT, 0.08 s TBT, 8 s TTLT), "
        "multiplied by `collection_id % 4 + 1`.",
        "",
        "## Workloads",
        "",
    ]
    for workload in workloads.values():
        lines.append(f"- **{workload.name}**: {workload.description}")
    lines.extend(
        [
            "",
            "> DeepResearch request-level SLO and reconstructed workflow SLO are reported "
            "separately. The flattened harness preserves recorded branch timing but does not "
            "claim to reproduce JITServe's internal DAG scheduler.",
            "",
            "## Results",
            "",
            "| Dataset | Speedup | Policy | Valid | A/B requests | SLO attainment | "
            "Task SLO | TTFT mean/p95 (s) | E2E mean/p95 (s) | TPOT mean (ms) | "
            "Wait mean (s) | Estimated prefix tok/req | Estimated prefix locality | Prompt tok/s |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in sorted(summaries, key=lambda value: (str(value["dataset"]), float(value["speedup"]), str(value["policy"]))):
        lines.append(
            "| {dataset} | {speedup}x | {policy} | {valid}/{expected} | {a}/{b} | "
            "{slo} | {task_slo} | {ttft_mean}/{ttft_p95} | {e2e_mean}/{e2e_p95} | "
            "{tpot} | {wait} | {prefix} | {prefix_ratio} | {throughput} |".format(
                dataset=item["dataset"],
                speedup=format_float(item["speedup"], 0),
                policy=item["policy"],
                valid=item["valid"],
                expected=expected,
                a=item["a_count"],
                b=item["b_count"],
                slo=format_float(100 * float(item["slo"]), 1) + "%",
                task_slo=(
                    format_float(100 * float(item["task_slo"]), 1) + "%"
                    if not math.isnan(float(item["task_slo"]))
                    else "-"
                ),
                ttft_mean=format_float(item["ttft_mean"]),
                ttft_p95=format_float(item["ttft_p95"]),
                e2e_mean=format_float(item["e2e_mean"]),
                e2e_p95=format_float(item["e2e_p95"]),
                tpot=format_float(item["tpot_mean_ms"], 2),
                wait=format_float(item["wait_mean"]),
                prefix=format_float(item["prefix_tokens_per_request"], 1),
                prefix_ratio=format_float(100 * float(item["prefix_hit_ratio"]), 1) + "%",
                throughput=format_float(item["prompt_tokens_per_s"], 1),
            )
        )
    lines.extend(
        [
            "",
            "## Metric Semantics",
            "",
            "- A/B requests: final request placement across the two regions.",
            "- SLO attainment: latency requests require TTFT and every observed TBT to pass; "
            "throughput/collective requests require TTLT to pass.",
            "- Wait: central admission/rebinding queue time before data-plane dispatch.",
            "- Prefix hit tokens: prompt tokens avoided according to the shared router-side "
            "block cache metadata; this is not a direct vLLM KV-residency oracle.",
            "- Prompt tok/s: total prompt tokens divided by wall-clock cell duration.",
            "",
            "## Policy Ports",
            "",
            "- RAVEL-Unified: SLO-aware placement, service ledger and protected admission.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(
    result_root: Path,
    expected: int,
    workloads: Mapping[str, Workload],
    speedups: Sequence[float],
) -> list[dict[str, object]]:
    summaries = discover_results(result_root, expected)
    atomic_write(
        result_root / "summary.json",
        json.dumps(summaries, indent=2, sort_keys=True, allow_nan=True),
    )
    atomic_write(
        result_root / "RESULTS.md",
        render_report(result_root, expected, workloads, speedups, summaries),
    )
    return summaries


def cell_complete(path: Path, expected: int) -> bool:
    csv_path = path / "request_metrics.csv"
    return csv_path.exists() and bool(summarize_csv(csv_path, expected)["complete"])


def run_cell(
    result_root: Path,
    workload: Workload,
    speedup: float,
    policy: str,
    request_count: int,
    timeout_s: float,
    network_delay_mode: str,
) -> None:
    cell = result_root / "cells" / workload.name / f"speedup_{speedup:g}" / policy
    if cell_complete(cell, request_count):
        return
    cell.mkdir(parents=True, exist_ok=True)
    reset_prefix_caches()
    command = [
        str(PYTHON),
        str(RAVEL_SRC / "run_cluster.py"),
        "--replicas",
        ",".join(ENDPOINTS),
        "--cluster-topology",
        str(TOPOLOGY),
        "--scheduler",
        POLICIES[policy],
        "--model-path",
        str(MODEL_PATH),
        "--model-name",
        "qwen",
        "--trace",
        str(workload.trace),
        "--result-path",
        str(cell),
        "--request-num",
        str(request_count),
        "--arrival-speedup",
        str(speedup),
        "--slo-profile",
        workload.slo_profile,
        "--request-ratio",
        workload.request_ratio,
        "--trace-seed",
        "42",
        "--network-delay-mode",
        network_delay_mode,
    ]
    manifest = {
        "dataset": workload.name,
        "speedup": speedup,
        "policy": policy,
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_write(cell / "manifest.json", json.dumps(manifest, indent=2))
    with (cell / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=RAVEL_SRC,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    manifest["returncode"] = result.returncode
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_write(cell / "manifest.json", json.dumps(manifest, indent=2))
    if result.returncode != 0:
        raise RuntimeError(f"cell failed with return code {result.returncode}: {cell}")
    summary = summarize_csv(cell / "request_metrics.csv", request_count)
    if not summary["complete"]:
        raise RuntimeError(f"cell incomplete: {cell}: {summary}")


def parse_csv_values(value: str, cast) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--request-count", type=int, default=500)
    parser.add_argument("--speedups", default="1,2,4,8")
    parser.add_argument("--datasets", default="lmsys,burst,deepresearch")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICY_NAMES))
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--network-delay-mode",
        choices=("synthetic", "physical"),
        default=DEFAULT_NETWORK_DELAY_MODE,
    )
    args = parser.parse_args()

    speedups = parse_csv_values(args.speedups, float)
    args.result_root.mkdir(parents=True, exist_ok=True)
    workloads = prepare_workloads(args.result_root, args.request_count)
    dataset_names = parse_csv_values(args.datasets, str)
    policy_names = parse_csv_values(args.policies, str)
    unknown_datasets = set(dataset_names) - set(workloads)
    unknown_policies = set(policy_names) - set(POLICIES)
    if unknown_datasets or unknown_policies:
        raise ValueError(
            f"unknown datasets={sorted(unknown_datasets)}, policies={sorted(unknown_policies)}"
        )
    if not args.report_only and args.network_delay_mode not in {
        "synthetic",
        "physical",
    }:
        raise ValueError(
            "set --network-delay-mode or RAVEL_NETWORK_DELAY_MODE"
        )

    manifest = {
        "request_count": args.request_count,
        "speedups": speedups,
        "datasets": dataset_names,
        "policies": policy_names,
        "endpoints": ENDPOINTS,
        "topology": str(TOPOLOGY),
        "model": str(MODEL_PATH),
        "apc": True,
        "chunked_prefill": True,
        "network_delay_mode": args.network_delay_mode,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_write(args.result_root / "campaign_manifest.json", json.dumps(manifest, indent=2))
    write_reports(args.result_root, args.request_count, workloads, speedups)
    if args.report_only:
        return 0

    ensure_endpoints_ready()
    cells = [
        (dataset, speedup, policy)
        for dataset in dataset_names
        for speedup in speedups
        for policy in policy_names
    ]
    random.Random(42).shuffle(cells)
    failed_cells = []
    completed_now = 0
    for dataset, speedup, policy in cells:
        cell = args.result_root / "cells" / dataset / f"speedup_{speedup:g}" / policy
        if cell_complete(cell, args.request_count):
            continue
        print(f"RUN dataset={dataset} speedup={speedup:g} policy={policy}", flush=True)
        error = None
        for attempt in range(args.retries + 1):
            try:
                ensure_endpoints_ready()
                run_cell(
                    args.result_root,
                    workloads[dataset],
                    speedup,
                    policy,
                    args.request_count,
                    args.timeout_s,
                    args.network_delay_mode,
                )
                error = None
                break
            except Exception as caught:
                error = caught
                print(
                    f"RETRY dataset={dataset} speedup={speedup:g} policy={policy} "
                    f"attempt={attempt + 1}: {caught}",
                    flush=True,
                )
                time.sleep(min(60.0, 5.0 * (attempt + 1)))
        if error is not None:
            failed_cells.append({
                "dataset": dataset,
                "speedup": speedup,
                "policy": policy,
                "error": str(error),
            })
            atomic_write(
                args.result_root / "failed_cells.json",
                json.dumps(failed_cells, indent=2),
            )
            write_reports(args.result_root, args.request_count, workloads, speedups)
            continue
        completed_now += 1
        write_reports(args.result_root, args.request_count, workloads, speedups)
        if args.max_cells and completed_now >= args.max_cells:
            break
    write_reports(args.result_root, args.request_count, workloads, speedups)
    return 1 if failed_cells else 0


if __name__ == "__main__":
    sys.exit(main())
