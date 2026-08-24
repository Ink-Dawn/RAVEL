#!/usr/bin/env python3
"""Analyze final-handoff quote accuracy from RAVEL request CSVs."""

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


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def workload_for(path: Path) -> str:
    for part in path.parts:
        if part in WORKLOADS:
            return part
    return "unknown"


def analyze(path: Path, request_rows: list[dict[str, object]], workload: str) -> dict[str, object]:
    valid_rows = [row for row in request_rows if row["valid"]]
    feasible = [row for row in valid_rows if row["predicted_feasible"]]
    infeasible = [row for row in valid_rows if not row["predicted_feasible"]]
    false_feasible = sum(not row["actual_success"] for row in feasible)
    false_infeasible = sum(row["actual_success"] for row in infeasible)
    latency_errors = [
        float(row["normalized_error"])
        for row in valid_rows
        if row["request_type"] == 0 and row["normalized_error"] is not None
    ]
    completion_errors = [
        float(row["normalized_error"])
        for row in valid_rows
        if row["request_type"] != 0 and row["normalized_error"] is not None
    ]
    return {
        "source": str(path),
        "workload": workload,
        "rows": len(request_rows),
        "handoff_valid": len(valid_rows),
        "handoff_valid_rate": len(valid_rows) / len(request_rows) if request_rows else math.nan,
        "predicted_feasible": len(feasible),
        "actual_success_given_feasible": (
            1.0 - false_feasible / len(feasible) if feasible else math.nan
        ),
        "false_feasible_rate": false_feasible / len(feasible) if feasible else math.nan,
        "predicted_infeasible": len(infeasible),
        "actual_success_given_infeasible": (
            false_infeasible / len(infeasible) if infeasible else math.nan
        ),
        "false_infeasible_rate": false_infeasible / len(infeasible) if infeasible else math.nan,
        "latency_error_p50": quantile(latency_errors, 0.5),
        "latency_error_p95_abs": quantile(sorted(abs(v) for v in latency_errors), 0.95),
        "completion_error_p50": quantile(completion_errors, 0.5),
        "completion_error_p95_abs": quantile(sorted(abs(v) for v in completion_errors), 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="result root or request_metrics.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: <input>/quote_accuracy when input is a result root",
    )
    args = parser.parse_args()
    input_path = args.input.resolve()
    if input_path.name == "request_metrics.csv":
        csv_paths = [input_path]
    else:
        # Quote fields are emitted by RAVEL-Unified only.  Baseline CSVs have
        # no final-handoff quote to evaluate, and archived failed retries are
        # not part of the completed campaign denominator.
        csv_paths = [
            path
            for path in sorted(input_path.rglob("request_metrics.csv"))
            if "failed_attempts" not in path.relative_to(input_path).parts
            and "RAVEL-Unified" in path.parts
        ]
    output_dir = (args.output_dir or (input_path.parent / "quote_accuracy" if input_path.name == "request_metrics.csv" else input_path / "quote_accuracy")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    cdf_rows: list[dict[str, object]] = []
    for csv_path in csv_paths:
        rows: list[dict[str, object]] = []
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                for raw in reader:
                    valid = number(raw, "quote_handoff_valid", 0.0) == 1.0
                    request_type = int(number(raw, "request_type", -1))
                    actual_success = number(raw, "request_slo_met", 0.0) == 1.0
                    predicted_feasible = number(raw, "quote_handoff_feasible", 0.0) == 1.0
                    if request_type == 0:
                        predicted = number(raw, "quote_handoff_predicted_ttft_s")
                        actual = number(raw, "time_to_first_token")
                        budget = number(raw, "slo_ttft_s")
                        metric = "latency_ttft"
                    else:
                        predicted = number(raw, "quote_handoff_point_completion_s")
                        actual = number(raw, "request_latency")
                        budget = number(raw, "slo_ttlt_s")
                        metric = "completion_e2e"
                    error = (
                        (predicted - actual) / budget
                        if valid and math.isfinite(predicted) and math.isfinite(actual) and budget > 0
                        else None
                    )
                    row = {
                        "source": str(csv_path),
                        "workload": workload_for(csv_path),
                        "request_id": raw.get("request_id", ""),
                        "request_type": request_type,
                        "valid": valid,
                        "predicted_feasible": predicted_feasible,
                        "actual_success": actual_success,
                        "metric": metric,
                        "predicted": predicted,
                        "actual": actual,
                        "budget": budget,
                        "normalized_error": error,
                    }
                    rows.append(row)
                    detail_rows.append(row)
                    if error is not None:
                        cdf_rows.append({**row, "cdf": 0.0})
        except (OSError, csv.Error) as error:
            print(f"skip {csv_path}: {error}")
            continue
        summaries.append(analyze(csv_path, rows, workload_for(csv_path)))

    # Add empirical CDF ranks per source/metric.  Keeping the raw rows makes it
    # easy to replot without rerunning the experiment.
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in cdf_rows:
        groups.setdefault((str(row["source"]), str(row["metric"])), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: float(row["normalized_error"]))
        n = len(group)
        for index, row in enumerate(group, 1):
            row["cdf"] = index / n

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "quote_accuracy_requests.csv", detail_rows)
    write_csv(output_dir / "quote_accuracy_cdf.csv", cdf_rows)
    write_csv(output_dir / "quote_accuracy_summary.csv", summaries)
    with (output_dir / "quote_accuracy_summary.md").open("w", encoding="utf-8") as output:
        output.write("# Quote accuracy\n\n")
        output.write("False-feasible is the miss rate among placements predicted feasible; false-infeasible is the success rate among placements predicted infeasible. Errors are normalized by the applicable SLO budget.\n\n")
        if summaries:
            fields = [
                "workload", "handoff_valid", "handoff_valid_rate",
                "false_feasible_rate", "false_infeasible_rate",
                "latency_error_p95_abs", "completion_error_p95_abs",
            ]
            output.write("| " + " | ".join(fields) + " |\n")
            output.write("|" + "---|" * len(fields) + "\n")
            for row in summaries:
                output.write("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |\n")
    print(f"quote accuracy outputs: {output_dir}")


if __name__ == "__main__":
    main()
