#!/usr/bin/env python3
"""Same-host helper to calibrate three endpoints and build a topology.

Do not use this helper against geo-distributed endpoints: WAN delay would enter
the service curves and then be counted again by the RTT topology. Cross-region
deployments must run calibrate_local_service_profile.py in each region and use
build_cluster_topology.py on the Router.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import subprocess
import time
from pathlib import Path


CLUSTERS = {
    "region_a": "127.0.0.1:8000",
    "region_b": "127.0.0.1:18000",
    "region_c": "127.0.0.1:28000",
}
RTT_MS = {
    "region_a": {
        "region_a": 0.5,
        "region_b": 34.1,
        "region_c": 16.6,
        "default": 34.1,
    },
    "region_b": {
        "region_a": 34.1,
        "region_b": 0.5,
        "region_c": 16.0,
        "default": 34.1,
    },
    "region_c": {
        "region_a": 16.6,
        "region_b": 16.0,
        "region_c": 0.5,
        "default": 16.6,
    },
}

def replica_ids_from_layout(raw: str) -> dict[str, list[int]]:
    counts = tuple(int(value.strip()) for value in raw.split(","))
    if len(counts) != len(CLUSTERS) or any(value <= 0 for value in counts):
        raise ValueError(
            "--replica-layout must contain three positive counts, e.g. 2,2,1"
        )
    next_id = 0
    result: dict[str, list[int]] = {}
    for cluster_id, count in zip(CLUSTERS, counts):
        result[cluster_id] = list(range(next_id, next_id + count))
        next_id += count
    return result


def load_rtt_matrix(path: Path | None) -> dict[str, dict[str, float]]:
    payload = RTT_MS if path is None else json.loads(
        path.read_text(encoding="utf-8")
    )
    raw = payload.get("rtt_ms_by_cluster", payload)
    matrix: dict[str, dict[str, float]] = {}
    for destination in CLUSTERS:
        if destination not in raw:
            raise ValueError(f"RTT matrix lacks destination {destination}")
        row = {
            source: float(raw[destination][source])
            for source in CLUSTERS
        }
        if any(value < 0 for value in row.values()):
            raise ValueError("RTT values cannot be negative")
        row["default"] = float(
            raw[destination].get(
                "default",
                max(row.values()),
            )
        )
        matrix[destination] = row
    return matrix


def installed_vllm_version(python: str) -> str:
    command = [
        python,
        "-c",
        (
            "import importlib.metadata; "
            "print(importlib.metadata.version('vllm'))"
        ),
    ]
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot determine vLLM version from {python!r}"
        ) from error

def valid_profile(
    path: Path,
    fingerprint: str,
    repeats: int,
    calibrator_sha256: str,
    prompt_targets: tuple[int, ...],
    levels: tuple[int, ...],
) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        configuration = payload["configuration"]
        prefill = payload["prefill_curve"]
        decode = payload["decode_curve"]
    except (KeyError, OSError, TypeError, ValueError):
        return False

    expected_prefill_levels = list(levels)
    expected_decode_levels = [level + 1 for level in levels]
    prefill_samples = repeats * len(prompt_targets)
    return (
        payload.get("schema_version") == 2
        and payload.get("engine_fingerprint") == fingerprint
        and payload.get("calibrator_sha256") == calibrator_sha256
        and configuration.get("repeats") == repeats
        and configuration.get("prompt_token_targets") == list(prompt_targets)
        and [point.get("active_sequences") for point in prefill]
        == expected_prefill_levels
        and [point.get("active_sequences") for point in decode]
        == expected_decode_levels
        and all(
            point.get("sample_count") == prefill_samples
            and point.get("prompt_tokens_min") <= min(prompt_targets)
            and point.get("prompt_tokens_max") >= max(prompt_targets)
            and point.get("seconds_per_token", 0.0) > 0.0
            and point.get("intercept_s", -1.0) >= 0.0
            and point.get("r_squared", 0.0) >= 0.90
            and len(point.get("samples", [])) == prefill_samples
            for point in prefill
        )
        and all(
            point.get("sample_count") == repeats
            and point.get("tpot_s_median", 0.0) > 0.0
            and point.get("tpot_s_p95", 0.0) > 0.0
            and len(point.get("samples", [])) == repeats
            for point in decode
        )
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate three same-host/same-LAN endpoints. For geo endpoints, "
            "calibrate locally per region and assemble the topology separately."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "configs" / "service_profiles" / "local",
    )
    parser.add_argument(
        "--topology-output",
        type=Path,
        default=root / "configs" / "three_cluster_2_2_1.local.json",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("RAVEL_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "RAVEL_MODEL_PATH",
            str(root / "models" / "Qwen3-1.7B"),
        ),
    )
    parser.add_argument(
        "--served-model-name",
        default=os.environ.get("RAVEL_SERVED_MODEL_NAME", "qwen"),
        help="model ID exposed by each endpoint's OpenAI-compatible API",
    )
    parser.add_argument(
        "--endpoints",
        default=os.environ.get(
            "RAVEL_CALIBRATION_ENDPOINTS",
            ",".join(CLUSTERS.values()),
        ),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--replica-layout",
        default="2,2,1",
        help="replica counts for region_a,region_b,region_c",
    )
    parser.add_argument(
        "--rtt-matrix",
        type=Path,
        help="JSON destination-by-client RTT matrix; defaults to 0.5/34.1/16.6ms",
    )
    parser.add_argument(
        "--vllm-version",
        default=None,
        help=(
            "optional assertion; the actual version is read from --python"
        ),
    )
    parser.add_argument("--dtype", default="float")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--apc", choices=("on", "off"), default="on")
    parser.add_argument(
        "--chunked-prefill", choices=("on", "off"), default="on"
    )
    parser.add_argument(
        "--enforce-eager", choices=("on", "off"), default="on"
    )
    args = parser.parse_args()
    endpoint_values = tuple(
        item.strip() for item in args.endpoints.split(",") if item.strip()
    )
    if len(endpoint_values) != len(CLUSTERS):
        raise ValueError(
            "--endpoints requires one representative endpoint for each of "
            f"{tuple(CLUSTERS)}"
        )
    cluster_endpoints = dict(zip(CLUSTERS, endpoint_values))
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if min(
        args.max_model_len,
        args.max_num_seqs,
        args.max_num_batched_tokens,
        args.block_size,
    ) <= 0:
        raise ValueError("engine dimensions must be positive")
    if args.max_num_seqs < 2:
        raise ValueError("--max-num-seqs must be at least 2")

    detected_vllm_version = installed_vllm_version(args.python)
    if (
        args.vllm_version is not None
        and args.vllm_version != detected_vllm_version
    ):
        raise ValueError(
            "--vllm-version does not match --python: "
            f"declared={args.vllm_version!r}, "
            f"installed={detected_vllm_version!r}"
        )
    args.vllm_version = detected_vllm_version

    prompt_targets = (256, 1024, 4096, 12000)
    levels = tuple(
        sorted(
            {
                0,
                max(1, args.max_num_seqs // 8),
                max(1, args.max_num_seqs // 4),
                max(1, args.max_num_seqs // 2),
                max(1, 3 * args.max_num_seqs // 4),
                args.max_num_seqs - 1,
            }
        )
    )
    replica_ids = replica_ids_from_layout(args.replica_layout)
    rtt_matrix = load_rtt_matrix(args.rtt_matrix)

    calibrator = root / "scripts/calibrate_service_profile.py"
    calibrator_sha256 = hashlib.sha256(calibrator.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.topology_output.parent.mkdir(parents=True, exist_ok=True)

    common_fingerprint = (
        f"{Path(args.model_path).name}|vllm={args.vllm_version}|"
        f"dtype={args.dtype}|max_model_len={args.max_model_len}|"
        f"max_num_seqs={args.max_num_seqs}|"
        f"max_num_batched_tokens={args.max_num_batched_tokens}|"
        f"block_size={args.block_size}|apc={args.apc}|"
        f"chunked_prefill={args.chunked_prefill}|enforce_eager={args.enforce_eager}"
    )
    processes: list[tuple[str, str, Path, subprocess.Popen, object]] = []
    manifest: dict[str, object] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "calibrator_sha256": calibrator_sha256,
        "engine_configuration": {
            "model_path": str(Path(args.model_path).resolve()),
            "served_model_name": args.served_model_name,
            "vllm_version": args.vllm_version,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "block_size": args.block_size,
            "apc": args.apc,
            "chunked_prefill": args.chunked_prefill,
            "enforce_eager": args.enforce_eager,
        },
        "replica_ids": replica_ids,
        "rtt_ms_by_cluster": rtt_matrix,
        "quality_gate": {
            "repeats": args.repeats,
            "prefill_r_squared_min": 0.90,
            "prompt_tokens_min": min(prompt_targets),
            "prompt_tokens_max": max(prompt_targets),
            "prefill_levels": list(levels),
            "decode_total_active_sequences": [
                level + 1 for level in levels
            ],
        },
        "clusters": {},
    }

    for cluster_id, endpoint in cluster_endpoints.items():
        fingerprint = f"{common_fingerprint}|cluster={cluster_id}|endpoint={endpoint}"
        output = args.output_dir / f"{cluster_id}.schema-v2.json"
        if valid_profile(
            output,
            fingerprint,
            args.repeats,
            calibrator_sha256,
            prompt_targets,
            levels,
        ):
            manifest["clusters"][cluster_id] = {
                "endpoint": endpoint,
                "engine_fingerprint": fingerprint,
                "output": str(output),
                "reused": True,
                "returncode": 0,
            }
            continue
        command = [
            args.python,
            str(calibrator),
            "--endpoint",
            endpoint,
            "--model",
            args.served_model_name,
            "--tokenizer-path",
            args.model_path,
            "--prompt-sizes",
            ",".join(str(value) for value in prompt_targets),
            "--probe-prompt-tokens",
            "32",
            "--probe-tokens",
            "128",
            "--background-prompt-tokens",
            "32",
            "--background-tokens",
            "256",
            "--repeats",
            str(args.repeats),
            "--levels",
            ",".join(str(value) for value in levels),
            "--prefill-levels",
            ",".join(str(value) for value in levels),
            "--timeout-s",
            "900",
            "--engine-fingerprint",
            fingerprint,
            "--output",
            str(output),
        ]
        log_path = args.output_dir / f"{cluster_id}.log"
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((cluster_id, fingerprint, output, process, log))
        manifest["clusters"][cluster_id] = {
            "endpoint": endpoint,
            "engine_fingerprint": fingerprint,
            "output": str(output),
            "command": command,
            "log": str(log_path),
            "reused": False,
        }

    failure = False
    for cluster_id, fingerprint, output, process, log in processes:
        returncode = process.wait()
        log.close()
        manifest["clusters"][cluster_id]["returncode"] = returncode
        if returncode != 0 or not valid_profile(
            output, fingerprint, args.repeats, calibrator_sha256,
            prompt_targets, levels,
        ):
            failure = True

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failure:
        raise SystemExit(f"calibration failed; inspect {manifest_path}")

    clusters = []
    for cluster_id in cluster_endpoints:
        profile = (args.output_dir / f"{cluster_id}.schema-v2.json").resolve()
        relative_profile = Path(os.path.relpath(
            profile, args.topology_output.parent.resolve()
        ))
        clusters.append(
            {
                "id": cluster_id,
                "replica_ids": replica_ids[cluster_id],
                "rtt_ms_by_client_region": rtt_matrix[cluster_id],
                "service_profile": str(relative_profile),
            }
        )
    topology = {
        "default_client_region": "region_a",
        "clusters": clusters,
    }
    temporary = args.topology_output.with_suffix(
        args.topology_output.suffix + ".tmp"
    )
    temporary.write_text(json.dumps(topology, indent=2), encoding="utf-8")
    temporary.replace(args.topology_output)
    print(f"wrote {args.topology_output}")


if __name__ == "__main__":
    main()
