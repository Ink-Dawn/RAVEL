from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def load_jitserve_records(path: str | Path) -> List[Dict[str, object]]:
    trace_path = Path(path)
    with trace_path.open("r", encoding="utf-8") as trace_file:
        first_character = trace_file.read(1)
        trace_file.seek(0)
        if first_character == "[":
            raw = json.load(trace_file)
        else:
            raw = [json.loads(line) for line in trace_file if line.strip()]

    records: List[Dict[str, object]] = []
    for collection_id, item in enumerate(raw):
        if "stages" not in item:
            record = dict(item)
            record.setdefault("collection_id", collection_id)
            records.append(record)
            continue

        stage_offset_ms = 0.0
        for stage_id, stage in enumerate(item["stages"]):
            for branch_id, request in enumerate(stage):
                records.append(
                    {
                        "prompt": request["prompt"],
                        "output": "",
                        "output_len": int(request["output_tokens"]),
                        "collection_id": collection_id,
                        "request_type": 2,
                        "request_type_locked": True,
                        "deliver_time": (
                            float(request["start_time"]) * 1000.0
                            if "start_time" in request
                            else stage_offset_ms
                        ),
                        "stage_id": stage_id,
                        "branch_id": branch_id,
                        "stage_num": int(item.get("stage_num", len(item["stages"]))),
                    }
                )
            if stage:
                starts = [float(request.get("start_time", 0.0)) for request in stage]
                finishes = [float(request.get("timestamp", 0.0)) for request in stage]
                stage_offset_ms += max(0.0, max(finishes) - min(starts)) * 1000.0
    return records


def parse_request_ratio(value: str) -> Tuple[float, float, float]:
    ratios = tuple(float(part.strip()) for part in value.split(","))
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or sum(ratios) <= 0:
        raise ValueError("request ratio must be three non-negative values")
    return ratios


def assign_request_types(
    records: Sequence[Mapping[str, object]],
    ratio: Optional[Tuple[float, float, float]],
    seed: int,
) -> List[Dict[str, object]]:
    copied = [dict(record) for record in records]
    if ratio is None:
        return copied

    assignable = [
        record for record in copied if not record.get("request_type_locked", False)
    ]
    latency_ratio, throughput_ratio, collective_ratio = ratio
    # Exact compatibility with JITServe's released
    # benchmark/trace/tools/create_trace.py. Its prose comment says /10, but
    # the executable calculate_request_counts() implementation uses /7; the
    # executable rule is the reproducibility source of truth here.
    effective_collective_ratio = collective_ratio / 7.0
    total = latency_ratio + throughput_ratio + effective_collective_ratio
    latency_count = int(len(assignable) * latency_ratio / total)
    throughput_count = int(len(assignable) * throughput_ratio / total)
    collective_count = len(assignable) - latency_count - throughput_count
    request_types = [0] * latency_count + [1] * throughput_count + [2] * collective_count
    random.Random(seed).shuffle(request_types)
    for record, request_type in zip(assignable, request_types):
        record["request_type"] = request_type
    return copied


def stable_prefix_block_hashes(input_ids: Sequence[int], block_tokens: int) -> List[str]:
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive")
    hashes = []
    for start in range(0, len(input_ids), block_tokens):
        block = input_ids[start : start + block_tokens]
        payload = ",".join(str(token_id) for token_id in block).encode("ascii")
        hashes.append(hashlib.sha256(payload).hexdigest()[:16])
    return hashes


def trace_time_ms(record: Mapping[str, object], fallback_index: int) -> float:
    value = record.get("deliver_time", record.get("timestamp", fallback_index))
    return max(0.0, float(value))
