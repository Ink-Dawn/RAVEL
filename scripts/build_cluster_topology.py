#!/usr/bin/env python3
"""Assemble local service profiles and a measured RTT matrix into topology."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REQUIRED_CONTRACT_FIELDS = (
    "model",
    "vllm",
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


def parse_named_paths(raw: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in raw.split(","):
        name, separator, value = item.strip().partition("=")
        if not separator or not name or not value:
            raise ValueError(
                "--profiles must use region=path comma-separated entries"
            )
        if name in result:
            raise ValueError(f"duplicate profile region: {name}")
        result[name] = Path(value).resolve()
    if len(result) < 2:
        raise ValueError("a geo topology requires at least two clusters")
    return result


def parse_contract(profile: dict[str, Any]) -> dict[str, str]:
    if int(profile.get("schema_version", 0)) != 2:
        raise ValueError("service profiles must use schema_version=2")
    parts = [
        part.strip()
        for part in str(profile.get("engine_fingerprint", "")).split("|")
        if part.strip()
    ]
    if len(parts) < 3:
        raise ValueError("invalid engine_fingerprint")
    contract = {"model": parts[0], "vllm": parts[1]}
    for part in parts[2:]:
        if "=" in part:
            key, value = part.split("=", 1)
            contract[key] = value
    missing = [
        field for field in REQUIRED_CONTRACT_FIELDS if field not in contract
    ]
    if missing:
        raise ValueError(f"engine_fingerprint lacks fields: {missing}")
    return contract


def parse_layout(raw: str, regions: list[str]) -> dict[str, list[int]]:
    counts = [int(value.strip()) for value in raw.split(",")]
    if len(counts) != len(regions) or any(value <= 0 for value in counts):
        raise ValueError(
            "--replica-layout needs one positive count per profile region"
        )
    cursor = 0
    result = {}
    for region, count in zip(regions, counts):
        result[region] = list(range(cursor, cursor + count))
        cursor += count
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        required=True,
        help="ordered region=profile entries, e.g. region_a=a.json,...",
    )
    parser.add_argument("--replica-layout", required=True)
    parser.add_argument("--rtt-matrix", type=Path, required=True)
    parser.add_argument("--default-client-region", default="region_a")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profiles = parse_named_paths(args.profiles)
    regions = list(profiles)
    if args.default_client_region not in profiles:
        raise ValueError("default client region is absent from --profiles")
    replica_ids = parse_layout(args.replica_layout, regions)

    contracts = {}
    for region, path in profiles.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        contracts[region] = parse_contract(payload)

    reference_region = regions[0]
    reference = contracts[reference_region]
    for region in regions[1:]:
        differences = {
            field: (reference[field], contracts[region][field])
            for field in REQUIRED_CONTRACT_FIELDS
            if reference[field] != contracts[region][field]
        }
        if differences:
            raise ValueError(
                "profiles use incompatible engine contracts; "
                f"{reference_region} vs {region}: {differences}"
            )
    for field in ("apc", "chunked_prefill", "enforce_eager"):
        if reference[field] != "on":
            raise ValueError(f"formal topology requires {field}=on")

    rtt_payload = json.loads(args.rtt_matrix.read_text(encoding="utf-8"))
    rtt = rtt_payload.get("rtt_ms_by_cluster", rtt_payload)
    clusters = []
    output_parent = args.output.resolve().parent
    for region in regions:
        if region not in rtt:
            raise ValueError(f"RTT matrix lacks destination row {region}")
        row = {source: float(rtt[region][source]) for source in regions}
        if any(value < 0 for value in row.values()):
            raise ValueError("RTT values must be non-negative")
        row["default"] = float(
            rtt[region].get("default", max(row.values()))
        )
        relative_profile = os.path.relpath(profiles[region], output_parent)
        clusters.append(
            {
                "id": region,
                "replica_ids": replica_ids[region],
                "rtt_ms_by_client_region": row,
                "service_profile": relative_profile,
            }
        )

    topology = {
        "default_client_region": args.default_client_region,
        "clusters": clusters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(topology, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
