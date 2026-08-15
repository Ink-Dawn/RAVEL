#!/usr/bin/env python3
"""Rebuild the public RAVEL-only result tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CAMPAIGN = "qwen3-1.7b-vllm085-v0-ravel-unified-hybrid-final-v3-20260814"
DEFAULT_ROOT = REPO / "results" / CAMPAIGN
DATASETS = ("lmsys", "burst", "deepresearch")
SPEEDUPS = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
POLICY = "RAVEL-Unified"
QUANTILES = (10, 25, 50, 75, 90)

BASE_COLUMNS = [
    "Dataset", "Speedup", "Policy", "A/B/C", "SLO",
    "TTFT Mean (s)", "TTFT P95 (s)", "E2E Mean (s)", "E2E P95 (s)",
    "TPOT Mean (ms)", "Wait Mean (s)", "Prefix Tok/Req", "Prefix Hit %",
    "Prompt Tok/s",
]
QUANTILE_COLUMNS = [
    f"{metric} P{q:02d}{' / Median' if q == 50 else ''} ({unit})"
    for metric, unit in (("TTFT", "s"), ("E2E", "s"), ("TPOT", "ms"))
    for q in QUANTILES
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_rows(root: Path) -> list[dict[str, Any]]:
    indexed = {
        (str(row["dataset"]), float(row["speedup"]), str(row["policy"])): row
        for row in read_json(root / "summary.json")
    }
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for speedup in SPEEDUPS:
            key = (dataset, speedup, POLICY)
            if key not in indexed:
                raise RuntimeError(f"missing RAVEL cell: {key}")
            cell = root / "cells" / dataset / f"speedup_{speedup:g}" / POLICY
            raw_path = cell / "request_metrics.csv"
            raw = read_csv(raw_path)
            manifest = read_json(cell / "manifest.json")
            if len(raw) != 500 or len({item["request_id"] for item in raw}) != 500:
                raise RuntimeError(f"invalid request coverage: {raw_path}")
            if manifest.get("returncode") != 0 or not indexed[key].get("complete"):
                raise RuntimeError(f"incomplete RAVEL cell: {cell}")
            row = dict(indexed[key])
            for metric, column in (
                ("ttft", "time_to_first_token"),
                ("e2e", "request_latency"),
                ("tpot", "tpot(ms)"),
            ):
                values = np.asarray([float(item[column]) for item in raw], dtype=float)
                for q in QUANTILES:
                    row[f"{metric}_p{q:02d}"] = float(np.quantile(values, q / 100.0))
            row["source_campaign"] = CAMPAIGN
            row["source_path"] = str(raw_path.resolve().relative_to(REPO.resolve()))
            rows.append(row)
    return rows


def display_record(row: dict[str, Any]) -> dict[str, str]:
    counts = row["cluster_counts"]
    item = {
        "Dataset": str(row["dataset"]),
        "Speedup": f"{float(row['speedup']):g}x",
        "Policy": POLICY,
        "A/B/C": f"{counts.get('region_a', 0)}/{counts.get('region_b', 0)}/{counts.get('region_c', 0)}",
        "SLO": f"{100 * float(row['slo']):.1f}%",
        "TTFT Mean (s)": f"{float(row['ttft_mean']):.3f}",
        "TTFT P95 (s)": f"{float(row['ttft_p95']):.3f}",
        "E2E Mean (s)": f"{float(row['e2e_mean']):.3f}",
        "E2E P95 (s)": f"{float(row['e2e_p95']):.3f}",
        "TPOT Mean (ms)": f"{float(row['tpot_mean_ms']):.2f}",
        "Wait Mean (s)": f"{float(row['wait_mean']):.3f}",
        "Prefix Tok/Req": f"{float(row['prefix_tokens_per_request']):.1f}",
        "Prefix Hit %": f"{100 * float(row['prefix_hit_ratio']):.1f}%",
        "Prompt Tok/s": f"{float(row['prompt_tokens_per_s']):.1f}",
    }
    for label, prefix, digits in (
        ("TTFT", "ttft", 3), ("E2E", "e2e", 3), ("TPOT", "tpot", 2)
    ):
        unit = "ms" if label == "TPOT" else "s"
        for q in QUANTILES:
            column = f"{label} P{q:02d}{' / Median' if q == 50 else ''} ({unit})"
            item[column] = f"{float(row[f'{prefix}_p{q:02d}']):.{digits}f}"
    return item


def table(records: list[dict[str, str]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|:---|---:|:---|:---|" + "|".join("---:" for _ in columns[4:]) + "|",
    ]
    lines.extend(
        "| " + " | ".join(record[column] for column in columns) + " |"
        for record in records
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravel-root", type=Path, default=DEFAULT_ROOT)
    root = parser.parse_args().ravel_root.resolve()
    rows = load_rows(root)
    records = [display_record(row) for row in rows]
    columns = BASE_COLUMNS + QUANTILE_COLUMNS

    with (root / "RESULTS_OVERVIEW_WITH_QUANTILES.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    overview = [
        "# RAVEL Final Results Overview with Latency Quantiles", "",
        f"- Source campaign: `{CAMPAIGN}`.",
        "- Coverage: **18/18 complete RAVEL cells**; **500/500 valid requests per cell**.",
        "- P50 is the median; percentiles use linear interpolation at `(n - 1) × q`.",
        "- TTFT and E2E are seconds; TPOT is milliseconds.", "",
        *table(records, columns), "",
    ]
    (root / "RESULTS_OVERVIEW_WITH_QUANTILES.md").write_text(
        "\n".join(overview), encoding="utf-8"
    )

    summary = [
        "# RAVEL Final Results", "",
        "- Coverage: **18/18 complete cells** (3 datasets × 6 speedups).",
        "- Requests per cell: **500**.",
        f"- Policy: **{POLICY}**.",
        "- This public release contains no baseline measurements or comparison rows.", "",
        *table(records, BASE_COLUMNS), "",
        "See `RESULTS_OVERVIEW_WITH_QUANTILES.md` for P10/P25/P50/P75/P90.", "",
    ]
    (root / "RESULTS.md").write_text("\n".join(summary), encoding="utf-8")
    (root / "summary.release.json").write_text(
        json.dumps(rows, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"built RAVEL-only artifacts from 18 cells: {root}")


if __name__ == "__main__":
    main()
