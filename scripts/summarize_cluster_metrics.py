#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def metric_summary(values):
    return {
        "mean": statistics.fmean(values) if values else None,
        "p20": percentile(values, 20),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def summarize(path: Path):
    with path.open("r", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    completed = [
        row
        for row in rows
        if float(row["request_latency"]) < 3600000
        and float(row["time_to_first_token"]) < 3600000
    ]
    ttft = [float(row["time_to_first_token"]) for row in completed]
    e2e = [float(row["request_latency"]) for row in completed]
    slo_successes = sum(int(row["request_slo_met"]) for row in completed)
    if completed:
        start = min(float(row["request_start_time"]) for row in completed)
        end = max(float(row["request_end_time"]) for row in completed)
        elapsed_s = max(end - start, 1e-9)
    else:
        elapsed_s = 0.0
    return {
        "requests": len(rows),
        "completed": len(completed),
        "request_slo_successes": slo_successes,
        "request_slo_success_rate": slo_successes / len(rows) if rows else 0.0,
        "slo_goodput_requests_per_s": slo_successes / elapsed_s if elapsed_s else 0.0,
        "ttft_s": metric_summary(ttft),
        "e2e_s": metric_summary(e2e),
        "cluster_counts": Counter(row["cluster_id"] for row in completed),
        "request_type_counts": Counter(row["request_type"] for row in completed),
        "route_reason_counts": Counter(row["cluster_route_reason"] for row in completed),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.metrics_csv), indent=2, sort_keys=True))
