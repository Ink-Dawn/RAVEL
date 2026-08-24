#!/usr/bin/env python3
"""Campaign A: repeated 16x headline cells for RAVEL and best baselines.

The script deliberately delegates each cell to the existing resumable
three-cluster runner.  RAVEL cells use this frozen campaign worktree; baseline
cells use the separately maintained DualMap/RD repository.  Results are placed
under a new campaign root and never overwrite final-v3 artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAVEL_ROOT = ROOT
DEFAULT_BASELINE_ROOT = Path("/root/autodl-tmp/RAVEL")
DEFAULT_MODEL = Path("/root/autodl-tmp/models/Qwen3-1.7B")
DEFAULT_TOPOLOGY = ROOT / "configs/topologies/ravel-final-numapinned-v3.2-2-1.json"
DEFAULT_ENDPOINTS = os.environ.get(
    "RAVEL_ENDPOINTS",
    "127.0.0.1:8000,127.0.0.1:8001,127.0.0.1:18000,127.0.0.1:18001,127.0.0.1:28000",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ravel-root", type=Path, default=DEFAULT_RAVEL_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    parser.add_argument(
        "--network-delay-mode",
        choices=("physical", "synthetic"),
        default=os.environ.get("RAVEL_NETWORK_DELAY_MODE", "physical"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results/campaign-a-16x-20260823",
    )
    parser.add_argument("--request-count", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_for(
    source_root: Path,
    result_root: Path,
    workload: str,
    policy: str,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, str]]:
    runner = source_root / "scripts/run_jitserve_three_cluster_matrix.py"
    command = [
        args.python,
        str(runner),
        "--result-root",
        str(result_root),
        "--topology",
        str(args.topology.resolve()),
        "--model-path",
        str(args.model_path.resolve()),
        "--endpoints",
        args.endpoints,
        "--network-delay-mode",
        args.network_delay_mode,
        "--workloads",
        workload,
        "--speedups",
        "16",
        "--policies",
        policy,
        "--request-count",
        str(args.request_count),
        "--timeout-s",
        str(args.timeout_s),
    ]
    env = os.environ.copy()
    env.update(
        {
            "RAVEL_ROOT": str(source_root.resolve()),
            "RAVEL_SRC_DIR": str((source_root / "src").resolve()),
            "RAVEL_MODEL_PATH": str(args.model_path.resolve()),
            "RAVEL_TOPOLOGY": str(args.topology.resolve()),
            "RAVEL_ENDPOINTS": args.endpoints,
        }
    )
    return command, env


def main() -> None:
    args = parse_args()
    if args.request_count != 500:
        raise ValueError("Campaign A is frozen to 500 requests per cell")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    args.result_root = args.result_root.resolve()
    args.ravel_root = args.ravel_root.resolve()
    args.baseline_root = args.baseline_root.resolve()
    args.model_path = args.model_path.resolve()
    args.topology = args.topology.resolve()

    # (workload, policy, source tree), with the workload-specific best
    # baseline requested for the headline comparison.
    cells = (
        ("lmsys", "RAVEL-Unified", args.ravel_root),
        ("lmsys", "DualMap", args.baseline_root),
        ("burst", "RAVEL-Unified", args.ravel_root),
        ("burst", "DualMap", args.baseline_root),
        ("deepresearch", "RAVEL-Unified", args.ravel_root),
        ("deepresearch", "RD", args.baseline_root),
    )
    manifest = {
        "campaign": "A",
        "speedup": 16,
        "request_count": args.request_count,
        "repeats": args.repeats,
        "cells": [],
    }
    args.result_root.mkdir(parents=True, exist_ok=True)
    for repeat in range(1, args.repeats + 1):
        for workload, policy, source_root in cells:
            cell_root = args.result_root / f"run_{repeat:02d}" / policy
            command, env = command_for(
                source_root, cell_root, workload, policy, args
            )
            row = {
                "repeat": repeat,
                "workload": workload,
                "policy": policy,
                "result_root": str(cell_root),
                "command": command,
                "command_shell": shlex.join(command),
            }
            manifest["cells"].append(row)
            print(f"[campaign-a] run={repeat:02d} {workload} {policy}")
            print(f"  {shlex.join(command)}")
            if not args.dry_run:
                subprocess.run(command, cwd=source_root, env=env, check=True)
    (args.result_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"campaign manifest: {args.result_root / 'campaign_manifest.json'}")


if __name__ == "__main__":
    main()
