#!/usr/bin/env python3
"""Summarize Campaign A per-run SLO and 95% run-level confidence intervals."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


WORKLOADS = {"lmsys", "burst", "deepresearch"}


def number(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def identify(path: Path, root: Path) -> tuple[str, str, int]:
    parts = path.relative_to(root).parts
    workload = next((part for part in parts if part in WORKLOADS), "unknown")
    repeat = next(
        (int(part.split("_", 1)[1]) for part in parts if part.startswith("run_")),
        -1,
    )
    if "cells" in parts:
        index = parts.index("cells")
        policy = parts[index + 3] if len(parts) > index + 3 else "unknown"
    else:
        policy = "unknown"
    return workload, policy, repeat


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize_csv(path: Path, root: Path) -> dict[str, object]:
    workload, policy, repeat = identify(path, root)
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    valid = [row for row in rows if number(row, "request_latency", 3_600_000) < 3_600_000]
    slo = [number(row, "request_slo_met", 0.0) for row in valid]
    protected = [number(row, "ravel_soft_admission_protected", 0.0) for row in valid]
    deferred = [number(row, "ravel_yield_deferred", 0.0) for row in valid]
    active = [number(row, "ravel_soft_admission_active", 0.0) for row in valid]
    deferral_times = [
        number(row, "ravel_soft_admission_deferral_s", 0.0)
        for row in valid
        if number(row, "ravel_soft_admission_deferral_s", 0.0) > 0.0
    ]
    release_rows = [
        number(row, "ravel_soft_admission_deadline_release", 0.0)
        for row in valid
        if number(row, "ravel_soft_admission_deferral_s", 0.0) > 0.0
    ]
    admission_deferred = [
        number(row, "ravel_soft_admission_first_deferred_at", 0.0) > 0.0
        for row in valid
    ]
    return {
        "repeat": repeat,
        "workload": workload,
        "policy": policy,
        "requests": len(rows),
        "valid": len(valid),
        "slo_attainment": sum(slo) / len(slo) if slo else math.nan,
        "p95_e2e_s": percentile([number(row, "request_latency") for row in valid], 0.95),
        "protected_pct": sum(protected) / len(protected) if protected else math.nan,
        "deferred_pct": sum(deferred) / len(deferred) if deferred else math.nan,
        "admission_deferred_pct": (
            sum(admission_deferred) / len(admission_deferred)
            if admission_deferred
            else math.nan
        ),
        "soft_admission_active_pct": sum(active) / len(active) if active else math.nan,
        "median_deferral_s": statistics.median(deferral_times) if deferral_times else math.nan,
        "deadline_release_pct": sum(release_rows) / len(release_rows) if release_rows else math.nan,
        "source": str(path),
    }


def mean_ci(values: list[float]) -> tuple[float, float]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan, math.nan
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, math.nan
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.result_root.resolve()
    output_dir = (args.output_dir or root / "summary").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Archived incomplete retries are retained for auditability but must not be
    # treated as campaign cells (their directory names intentionally include a
    # timestamp rather than the canonical ``run_NN`` layout).
    metric_paths = [
        path
        for path in sorted(root.rglob("request_metrics.csv"))
        if "failed_attempts" not in path.relative_to(root).parts
    ]
    rows = [summarize_csv(path, root) for path in metric_paths]
    fields = list(rows[0].keys()) if rows else ["repeat", "workload", "policy"]
    with (output_dir / "campaign_a_runs.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    grouped: list[dict[str, object]] = []
    baselines = {"lmsys": "DualMap", "burst": "DualMap", "deepresearch": "RD"}
    for workload in sorted({str(row["workload"]) for row in rows}):
        policies = sorted({str(row["policy"]) for row in rows if row["workload"] == workload})
        for policy in policies:
            values = [float(row["slo_attainment"]) for row in rows if row["workload"] == workload and row["policy"] == policy]
            mean, ci = mean_ci(values)
            grouped.append({"workload": workload, "policy": policy, "runs": len(values), "slo_mean": mean, "slo_ci95": ci})
        baseline = baselines.get(workload)
        paired = []
        if baseline:
            for repeat in sorted({int(row["repeat"]) for row in rows if row["workload"] == workload}):
                ravel = next((row for row in rows if row["workload"] == workload and row["policy"] == "RAVEL-Unified" and int(row["repeat"]) == repeat), None)
                base = next((row for row in rows if row["workload"] == workload and row["policy"] == baseline and int(row["repeat"]) == repeat), None)
                if ravel and base:
                    paired.append(float(ravel["slo_attainment"]) - float(base["slo_attainment"]))
            gain, gain_ci = mean_ci(paired)
            grouped.append({"workload": workload, "policy": f"RAVEL-Unified minus {baseline}", "runs": len(paired), "slo_mean": gain, "slo_ci95": gain_ci})
    gfields = ["workload", "policy", "runs", "slo_mean", "slo_ci95"]
    with (output_dir / "campaign_a_ci.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=gfields)
        writer.writeheader()
        writer.writerows(grouped)
    with (output_dir / "campaign_a_summary.md").open("w", encoding="utf-8") as output:
        output.write("# Campaign A (16x)\n\n")
        output.write("Confidence intervals treat each 500-request run as one independent unit; requests within a run are not treated as independent samples.\n\n")
        output.write("| workload | policy | runs | mean SLO | 95% CI half-width |\n|---|---|---:|---:|---:|\n")
        for row in grouped:
            output.write(f"| {row['workload']} | {row['policy']} | {row['runs']} | {row['slo_mean']:.4f} | {row['slo_ci95']:.4f} |\n")
    print(f"Campaign A summary: {output_dir}")


if __name__ == "__main__":
    main()
