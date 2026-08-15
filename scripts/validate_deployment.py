#!/usr/bin/env python3
"""Validate a calibrated RAVEL deployment before running experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON {path}: {error}")
    if not isinstance(payload, dict):
        fail(f"{path} must contain a JSON object")
    return payload


def positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def validate_profile(path: Path) -> dict[str, Any]:
    profile = load_json(path)
    if profile.get("schema_version") != 2:
        fail(f"{path}: expected service-profile schema_version=2")
    fingerprint = str(profile.get("engine_fingerprint", ""))
    required_fingerprint_terms = (
        "engine_impl=v0",
        "max_model_len=",
        "max_num_seqs=",
        "max_num_batched_tokens=",
        "block_size=",
        "apc=on",
        "chunked_prefill=on",
    )
    missing = [term for term in required_fingerprint_terms if term not in fingerprint]
    if missing:
        fail(f"{path}: incomplete/incompatible engine fingerprint: {missing}")

    prefill = profile.get("prefill_curve")
    decode = profile.get("decode_curve")
    if not isinstance(prefill, list) or not prefill:
        fail(f"{path}: prefill_curve must be non-empty")
    if not isinstance(decode, list) or not decode:
        fail(f"{path}: decode_curve must be non-empty")
    if any(
        not positive_number(point.get("seconds_per_token"))
        or float(point.get("intercept_s", -1.0)) < 0.0
        or float(point.get("r_squared", -1.0)) < 0.90
        for point in prefill
    ):
        fail(f"{path}: prefill curve failed the positive/r_squared>=0.90 gate")
    if any(
        not positive_number(point.get("tpot_s_median"))
        or not positive_number(point.get("tpot_s_p95"))
        for point in decode
    ):
        fail(f"{path}: decode curve contains an invalid point")
    decode_levels = [int(point.get("active_sequences", -1)) for point in decode]
    if decode_levels != sorted(set(decode_levels)):
        fail(f"{path}: decode concurrency levels must strictly increase")
    decode_cvs = []
    for point in decode:
        samples = [
            float(sample["tpot_s"])
            for sample in point.get("samples", [])
            if positive_number(sample.get("tpot_s"))
        ]
        if len(samples) < 2:
            fail(f"{path}: decode curve requires at least two raw samples per point")
        mean = sum(samples) / len(samples)
        variance = sum((value - mean) ** 2 for value in samples) / (len(samples) - 1)
        decode_cvs.append(math.sqrt(variance) / mean)
    if max(decode_cvs) > 0.15:
        fail(f"{path}: decode calibration max coefficient of variation exceeds 0.15")
    configuration = profile.get("configuration", {})
    prompt_targets = configuration.get("prompt_token_targets", [])
    if not prompt_targets or min(prompt_targets) > 256 or max(prompt_targets) < 12000:
        fail(f"{path}: calibration does not cover prompt lengths 256..12000")
    prefill_levels = configuration.get("prefill_levels", [])
    configured_decode_levels = configuration.get(
        "decode_total_active_sequences", []
    )
    if len(prefill_levels) < 4 or len(configured_decode_levels) < 4:
        fail(f"{path}: calibration concurrency coverage is too sparse")
    return profile


def endpoint_models(endpoint: str, timeout_s: float) -> list[str]:
    url = endpoint if "://" in endpoint else f"http://{endpoint}"
    headers = {}
    api_key = os.environ.get("RAVEL_API_KEY") or os.environ.get(
        "VLLM_API_KEY"
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/models", headers=headers
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        fail(f"endpoint {endpoint} is not ready: {error}")
    models = [str(row.get("id", "")) for row in payload.get("data", [])]
    models = [model for model in models if model]
    if not models:
        fail(f"endpoint {endpoint} returned no model IDs")
    return models


def parse_endpoints(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology",
        type=Path,
        default=root / "configs" / "three_cluster_2_2_1.local.json",
    )
    parser.add_argument(
        "--endpoints",
        required=True,
        help="comma-separated endpoints ordered by contiguous replica ID",
    )
    parser.add_argument("--expected-model", default="")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    topology_path = args.topology.resolve()
    topology = load_json(topology_path)
    clusters = topology.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        fail("topology.clusters must be a non-empty list")

    cluster_ids: set[str] = set()
    replica_ids: list[int] = []
    profiles: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        cluster_id = str(cluster.get("id", ""))
        if not cluster_id or cluster_id in cluster_ids:
            fail(f"invalid or duplicate cluster id: {cluster_id!r}")
        cluster_ids.add(cluster_id)
        ids = cluster.get("replica_ids")
        if not isinstance(ids, list) or not ids:
            fail(f"cluster {cluster_id} has no replica_ids")
        replica_ids.extend(int(replica_id) for replica_id in ids)

        rtt = cluster.get("rtt_ms_by_client_region")
        if not isinstance(rtt, dict) or not rtt:
            fail(f"cluster {cluster_id} has no RTT row")
        if any(float(value) < 0.0 for value in rtt.values()):
            fail(f"cluster {cluster_id} has a negative RTT")

        profile_value = cluster.get("service_profile")
        if not profile_value:
            fail(f"cluster {cluster_id} has no service_profile")
        profile_path = Path(str(profile_value))
        if not profile_path.is_absolute():
            profile_path = topology_path.parent / profile_path
        profiles[cluster_id] = validate_profile(profile_path.resolve())

    if sorted(replica_ids) != list(range(len(replica_ids))):
        fail("replica IDs must be unique and contiguous starting at zero")

    endpoints = parse_endpoints(args.endpoints)
    if len(endpoints) != len(replica_ids):
        fail(
            f"endpoint count {len(endpoints)} does not match "
            f"replica count {len(replica_ids)}"
        )
    endpoint_inventory = {
        endpoint: endpoint_models(endpoint, args.timeout_s)
        for endpoint in endpoints
    }
    model_sets = {tuple(models) for models in endpoint_inventory.values()}
    if len(model_sets) != 1:
        fail(f"endpoints expose different model inventories: {endpoint_inventory}")
    if args.expected_model and any(
        args.expected_model not in models
        for models in endpoint_inventory.values()
    ):
        fail(
            f"expected model {args.expected_model!r} is not exposed by every endpoint"
        )

    result = {
        "status": "ready",
        "topology": str(topology_path),
        "clusters": sorted(cluster_ids),
        "replica_ids": sorted(replica_ids),
        "endpoints": endpoint_inventory,
        "service_profile_fingerprints": {
            cluster_id: profile["engine_fingerprint"]
            for cluster_id, profile in profiles.items()
        },
        "invariants": {
            "apc": True,
            "chunked_prefill": True,
            "profile_schema": 2,
            "contiguous_replica_ids": True,
            "uniform_model_inventory": True,
        },
    }
    rendered = json.dumps(result, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json_output.with_suffix(args.json_output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.json_output)
    print(rendered)


if __name__ == "__main__":
    main()
