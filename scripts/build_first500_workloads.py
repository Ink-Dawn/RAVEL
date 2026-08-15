#!/usr/bin/env python3
"""Build the frozen request-level first-500 evaluation workloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent



def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def read_burst_timestamps(path: Path, limit: int, offset: int = 0) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for index, row in enumerate(csv.DictReader(source)):
            if index < offset:
                continue
            values.append(float(row["Timestamp"]))
            if len(values) == limit:
                break
    if len(values) != limit:
        raise ValueError(f"BurstGPT provided {len(values)} of {limit} timestamps")
    baseline = values[0]
    return [value - baseline for value in values]


def flatten_deepresearch(
    trace: Path, burst_trace: Path, limit: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    workflows: list[dict[str, object]] = []
    branch_count = 0
    with trace.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            workflow = json.loads(line)
            workflows.append(workflow)
            branch_count += sum(len(stage) for stage in workflow.get("stages", []))
            if branch_count >= limit * 2:
                break

    arrivals = read_burst_timestamps(burst_trace, len(workflows), offset=100)
    records: list[dict[str, object]] = []
    for collection_id, (workflow, arrival_ms) in enumerate(zip(workflows, arrivals)):
        stages = workflow.get("stages", [])
        starts = [
            float(request.get("start_time", request.get("timestamp", 0.0)))
            for stage in stages
            for request in stage
        ]
        workflow_start = min(starts, default=0.0)
        for stage_id, stage in enumerate(stages):
            for branch_id, request in enumerate(stage):
                relative_ms = 1000.0 * max(
                    0.0,
                    float(request.get("start_time", workflow_start)) - workflow_start,
                )
                records.append(
                    {
                        "prompt": request["prompt"],
                        "output_len": int(request["output_tokens"]),
                        "collection_id": collection_id,
                        "request_type": 2,
                        "deliver_time": arrival_ms + relative_ms,
                        "stage_id": stage_id,
                        "branch_id": branch_id,
                        "stage_num": int(workflow.get("stage_num", len(stages))),
                    }
                )
    records.sort(
        key=lambda item: (float(item["deliver_time"]), int(item["collection_id"]))
    )
    if len(records) < limit:
        raise ValueError(f"DeepResearch provided only {len(records)} branch requests")
    return records[:limit], workflows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_incomplete_workflows(
    records: Iterable[dict[str, object]], workflows: list[dict[str, object]]
) -> tuple[int, int]:
    observed: dict[int, set[tuple[int, int]]] = {}
    for record in records:
        collection_id = int(record["collection_id"])
        observed.setdefault(collection_id, set()).add(
            (int(record["stage_id"]), int(record["branch_id"]))
        )

    incomplete = 0
    for collection_id, positions in observed.items():
        expected = {
            (stage_id, branch_id)
            for stage_id, stage in enumerate(workflows[collection_id].get("stages", []))
            for branch_id, _ in enumerate(stage)
        }
        incomplete += positions != expected
    return len(observed), incomplete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=REPO_ROOT / "data" / "traces"
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    if args.limit != 500:
        raise ValueError("formal workload artifacts are fixed at exactly 500 requests")

    lmsys = json.loads(
        (args.source / "lmsys.json").read_text(encoding="utf-8")
    )[: args.limit]
    burst = [dict(record) for record in lmsys]
    burst_times = read_burst_timestamps(
        args.source / "burst/BurstGPT_1.csv", args.limit
    )
    for record, timestamp in zip(burst, burst_times):
        record["deliver_time"] = timestamp
    deepresearch, workflows = flatten_deepresearch(
        args.source / "deepresearch_filter.jsonl",
        args.source / "burst/BurstGPT_1.csv",
        args.limit,
    )

    paths = {
        "lmsys": args.output / "lmsys_first500.json",
        "burst": args.output / "burst_first500.json",
        "deepresearch": args.output / "deepresearch_flat_first500.json",
    }
    for name, records in (
        ("lmsys", lmsys),
        ("burst", burst),
        ("deepresearch", deepresearch),
    ):
        atomic_write(paths[name], json.dumps(records, ensure_ascii=True))

    workflow_count, incomplete_count = count_incomplete_workflows(
        deepresearch, workflows
    )
    manifest = {
        "artifact_sha256": {name: sha256(path) for name, path in paths.items()},
        "deepresearch": {
            "task_slo_eligible": False,
            "workflow_ids": workflow_count,
            "workflows_missing_stages": incomplete_count,
        },
        "request_count": args.limit,
        "schema": "ravel-request-level-first500-v1",
        "selection": {
            "burst": "same LMSYS records with first 500 BurstGPT timestamps",
            "deepresearch": "legacy globally time-sorted flattened branches, first 500",
            "lmsys": "first 500 flat LMSYS records",
        },
    }
    atomic_write(
        args.output / "first500_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
