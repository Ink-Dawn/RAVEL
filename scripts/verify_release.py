#!/usr/bin/env python3
"""Validate the frozen RAVEL-only GitHub release without contacting endpoints."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_NAME = "qwen3-1.7b-vllm085-v0-ravel-unified-hybrid-final-v3-20260814"
DATASETS = ("lmsys", "burst", "deepresearch")
SPEEDUPS = (1, 2, 4, 8, 12, 16)
POLICY = "RAVEL-Unified"
SOURCE_DIGEST = "c9108ee504f900153c7ec0a071c51d771e76a399fcb5e55ccf93a88c6794661b"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def validate_cell(path: Path) -> None:
    data = csv_rows(path / "request_metrics.csv")
    if len(data) != 500 or len({row["request_id"] for row in data}) != 500:
        raise RuntimeError(f"invalid rows: {path}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("returncode") != 0:
        raise RuntimeError(f"failed manifest: {path}")
    actual = manifest.get("fingerprint_inputs", {}).get("source_sha256")
    if actual != SOURCE_DIGEST:
        raise RuntimeError(f"source mismatch: {path}: {actual}")


def main() -> None:
    result_dirs = sorted(
        path.name for path in (ROOT / "results").iterdir() if path.is_dir()
    )
    if result_dirs != [RESULT_NAME]:
        raise RuntimeError(f"unexpected results: {result_dirs}")
    result_root = ROOT / "results" / RESULT_NAME
    for dataset in DATASETS:
        for speedup in SPEEDUPS:
            validate_cell(
                result_root / "cells" / dataset / f"speedup_{speedup}" / POLICY
            )

    release_rows = json.loads(
        (result_root / "summary.release.json").read_text(encoding="utf-8")
    )
    expected = {
        (dataset, float(speedup), POLICY)
        for dataset in DATASETS for speedup in SPEEDUPS
    }
    actual = {
        (row["dataset"], float(row["speedup"]), row["policy"])
        for row in release_rows
    }
    if actual != expected or len(release_rows) != 18:
        raise RuntimeError("RAVEL-only summary is not 18-cell complete")
    if any(row.get("source_campaign") != RESULT_NAME for row in release_rows):
        raise RuntimeError("summary contains a foreign campaign")

    calibration = json.loads(
        (ROOT / "release" / "CALIBRATION_MANIFEST.json").read_text(encoding="utf-8")
    )
    expected_contract = {
        "prompt_tokens": 32, "background_tokens": 256,
        "probe_tokens": 128, "repeats": 5,
    }
    if calibration["contract"] != expected_contract:
        raise RuntimeError("calibration contract changed")
    if set(calibration["profile_sets"]) != {"ravel_final"}:
        raise RuntimeError("non-RAVEL calibration set is present")

    forbidden = [
        ROOT / "baseline",
        ROOT / "release" / "baselines",
        ROOT / "release" / "recorded-inputs" / "baseline-formal-20260812",
        ROOT / "configs" / "service_profiles" / "qwen3-1.7b-vllm-0.8.5-v0",
        ROOT / "configs" / "topologies" / "baseline-formal-v0.2-2-1.json",
    ]
    present = [path for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"baseline artifacts present: {present}")
    if "vllm==0.8.5.post1" not in (
        ROOT / "requirements-serving.txt"
    ).read_text(encoding="utf-8"):
        raise RuntimeError("serving version is not frozen")
    debris = [
        path for path in ROOT.rglob("*")
        if path.is_file()
        and (path.name.endswith(".orig") or "__pycache__" in path.parts)
    ]
    if debris:
        raise RuntimeError(f"release debris: {debris[:5]}")
    large = [
        path for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size >= 100_000_000
    ]
    if large:
        raise RuntimeError(f"GitHub-incompatible files: {large}")
    print("release validation passed: RAVEL-only, 18 cells, 500 unique requests each")


if __name__ == "__main__":
    main()
