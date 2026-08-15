#!/usr/bin/env python3
"""Reject a vLLM launch that does not match its calibrated service profile."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def parse_fingerprint(value: str) -> tuple[str, str, dict[str, str]]:
    parts = [part.strip() for part in value.split("|") if part.strip()]
    if len(parts) < 3:
        raise ValueError("invalid engine_fingerprint")
    if parts[1].startswith("vllm-"):
        vllm_version = parts[1][len("vllm-") :]
    elif parts[1].startswith("vllm="):
        vllm_version = parts[1][len("vllm=") :]
    else:
        raise ValueError("engine_fingerprint has no vLLM version")
    fields: dict[str, str] = {}
    for part in parts[2:]:
        if "=" in part:
            key, field_value = part.split("=", 1)
            fields[key] = field_value
    return parts[0], vllm_version, fields


def normalize_dtype(value: str) -> str:
    aliases = {
        "float32": "float",
        "fp32": "float",
        "float16": "half",
        "fp16": "half",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
    }
    lowered = value.strip().lower()
    return aliases.get(lowered, lowered)


def validate(args: argparse.Namespace) -> list[str]:
    payload = json.loads(args.profile.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 2:
        return ["service profile must use schema_version=2"]
    try:
        model_name, profile_vllm, fields = parse_fingerprint(
            str(payload.get("engine_fingerprint", ""))
        )
    except ValueError as error:
        return [str(error)]

    actual_vllm = importlib.metadata.version("vllm")
    expected = {
        "model": model_name,
        "vllm": profile_vllm,
        "dtype": normalize_dtype(fields.get("dtype", "")),
        "engine_impl": fields.get("engine_impl", ""),
        "max_model_len": fields.get("max_model_len", ""),
        "max_num_seqs": fields.get("max_num_seqs", ""),
        "max_num_batched_tokens": fields.get(
            "max_num_batched_tokens", ""
        ),
        "block_size": fields.get("block_size", ""),
        "apc": fields.get("apc", ""),
        "chunked_prefill": fields.get("chunked_prefill", ""),
        "enforce_eager": fields.get("enforce_eager", ""),
        "gpu": fields.get("gpu", ""),
    }
    actual_gpu = ""
    if expected["gpu"]:
        try:
            import torch

            actual_gpu = str(torch.cuda.get_device_name(0))
        except (ImportError, RuntimeError) as error:
            return [f"cannot inspect launch GPU: {error}"]
    actual = {
        # The calibrator fingerprints the lexical model-path basename. Keep
        # the same rule so a stable symlink remains a valid deployment path.
        "model": args.model.name,
        "vllm": actual_vllm,
        "dtype": normalize_dtype(args.dtype),
        "engine_impl": args.engine_implementation,
        "max_model_len": str(args.max_model_len),
        "max_num_seqs": str(args.max_num_seqs),
        "max_num_batched_tokens": str(args.max_num_batched_tokens),
        "block_size": str(args.block_size),
        "apc": "on",
        "chunked_prefill": "on",
        "enforce_eager": "on",
        "gpu": actual_gpu,
    }
    return [
        f"{key}: profile={expected[key]!r}, launch={actual[key]!r}"
        for key in expected
        if expected[key] != actual[key]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument(
        "--engine-implementation", choices=("v0",), default="v0"
    )
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    args = parser.parse_args()

    mismatches = validate(args)
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise SystemExit(
            "launch does not match calibrated service profile:\n  - "
            + details
        )
    print("service profile matches launch configuration")


if __name__ == "__main__":
    main()
