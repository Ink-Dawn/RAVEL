#!/usr/bin/env python3
"""Build a request-visible workflow-stage output profile from training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        raise ValueError("cannot compute a quantile from no samples")
    rank = max(1, math.ceil(quantile * len(values)))
    return sorted(values)[rank - 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--request-type", type=int, default=2)
    parser.add_argument("--min-stage-num", type=int, default=2)
    parser.add_argument("--fallback-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.5 <= args.quantile < 1.0:
        raise ValueError("--quantile must be in [0.5, 1.0)")
    if args.min_stage_num < 2:
        raise ValueError("--min-stage-num must be at least 2")
    if args.fallback_tokens <= 0:
        raise ValueError("--fallback-tokens must be positive")

    source_bytes = args.input.read_bytes()
    workflows = json.loads(source_bytes)
    if not isinstance(workflows, list):
        raise ValueError("training input must be a JSON list of workflows")

    by_stage: dict[int, list[int]] = defaultdict(list)
    by_context: dict[tuple[int, int], list[int]] = defaultdict(list)
    for workflow in workflows:
        stages = workflow.get("stages", ())
        if len(stages) < args.min_stage_num:
            continue
        stage_num = len(stages)
        for stage in stages:
            stage_id = int(stage["stage_number"])
            for request in stage.get("requests", ()):
                output_tokens = int(request["output_tokens"])
                if output_tokens <= 0:
                    raise ValueError(
                        f"stage {stage_id} contains non-positive output_tokens"
                    )
                by_stage[stage_id].append(output_tokens)
                by_context[(stage_id, stage_num)].append(output_tokens)

    if not by_stage:
        raise ValueError("no eligible workflow-stage samples found")

    payload = {
        "schema_version": 2,
        "kind": "semantic-collective-stage-output-profile",
        "source": {
            "name": args.input.name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "training_only": True,
        },
        "selector": {
            "request_type": args.request_type,
            "min_stage_num": args.min_stage_num,
        },
        "context_features": ["stage_id", "stage_num"],
        "quantile": args.quantile,
        "fallback_tokens": args.fallback_tokens,
        "stages": {
            str(stage_id): {
                "samples": len(values),
                "mean_tokens": sum(values) / len(values),
                "q95_tokens": nearest_rank(values, args.quantile),
            }
            for stage_id, values in sorted(by_stage.items())
        },
        "contexts": {
            f"{stage_id}:{stage_num}": {
                "samples": len(values),
                "mean_tokens": sum(values) / len(values),
                "q95_tokens": nearest_rank(values, args.quantile),
            }
            for (stage_id, stage_num), values in sorted(by_context.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
