#!/usr/bin/env python3
"""Calibrate one vLLM service class from a host in the same region."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def probe(
    python: str, expression: str, environment: dict[str, str]
) -> str:
    try:
        return subprocess.check_output(
            [python, "-c", expression],
            text=True,
            stderr=subprocess.STDOUT,
            env=environment,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot inspect serving environment with {python!r}"
        ) from error


def concurrency_levels(max_num_seqs: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                0,
                max(1, max_num_seqs // 8),
                max(1, max_num_seqs // 4),
                max(1, max_num_seqs // 2),
                max(1, 3 * max_num_seqs // 4),
                max_num_seqs - 1,
            }
        )
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run service calibration against a same-region endpoint. Do not "
            "run this command across the WAN because network delay would be "
            "counted again by the topology RTT model."
        )
    )
    parser.add_argument("--endpoint", default="127.0.0.1:8000")
    parser.add_argument(
        "--python", default=os.environ.get("RAVEL_PYTHON", sys.executable)
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--served-model-name", default="qwen")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster-id", default="")
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA_VISIBLE_DEVICES value used by the calibrated endpoint",
    )
    parser.add_argument("--dtype", default="float")
    parser.add_argument(
        "--engine-implementation", choices=("v0",), default="v0"
    )
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--vllm-version", default="")
    args = parser.parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    probe_environment = os.environ.copy()
    probe_environment["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.max_num_seqs < 2:
        raise ValueError("--max-num-seqs must be at least 2")
    if min(
        args.max_model_len,
        args.max_num_batched_tokens,
        args.block_size,
        args.repeats,
    ) <= 0:
        raise ValueError("engine dimensions and repeats must be positive")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")

    detected_vllm = probe(
        args.python,
        "import importlib.metadata; "
        "print(importlib.metadata.version('vllm'))",
        probe_environment,
    )
    if args.vllm_version and args.vllm_version != detected_vllm:
        raise ValueError(
            f"declared vLLM {args.vllm_version!r} != detected "
            f"{detected_vllm!r}"
        )
    gpu_name = probe(
        args.python,
        "import torch; print(torch.cuda.get_device_name(0))",
        probe_environment,
    )
    dimensions = {
        "dtype": args.dtype,
        "engine_impl": args.engine_implementation,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "block_size": args.block_size,
        "apc": "on",
        "chunked_prefill": "on",
        "enforce_eager": "on",
        "gpu": gpu_name,
    }
    fields = "|".join(f"{key}={value}" for key, value in dimensions.items())
    fingerprint = (
        f"{args.model_path.name}|vllm={detected_vllm}|{fields}"
    )
    if args.cluster_id:
        fingerprint += f"|cluster={args.cluster_id}"

    levels = concurrency_levels(args.max_num_seqs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(root / "scripts" / "calibrate_service_profile.py"),
        "--endpoint",
        args.endpoint,
        "--model",
        args.served_model_name,
        "--tokenizer-path",
        str(args.model_path),
        "--prompt-sizes",
        "256,1024,4096,12000",
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
        str(args.output),
    ]
    subprocess.run(
        command, cwd=root, env=probe_environment, check=True
    )


if __name__ == "__main__":
    main()
