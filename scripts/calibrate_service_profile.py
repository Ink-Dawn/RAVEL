#!/usr/bin/env python3
"""Profile the explainable service curves consumed by RAVEL.

The profiler measures streaming TTFT and TPOT using API-reported token counts.
Every Decode repeat owns a fresh, fixed-context background cohort and exactly
one probe. Cohorts are joined before the next repeat, preventing context-phase
and cross-level contamination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from transformers import AutoTokenizer


PROMPT_WORD = "calibration"
DECODE_MAX_CV = 0.15


@dataclass(frozen=True)
class Sample:
    wall_s: float
    ttft_s: float
    tpot_s: float
    max_tbt_s: float
    tbt_token_count: int
    prompt_tokens: int
    completion_tokens: int


def reset_prefix_cache(endpoint: str, timeout_s: float) -> None:
    request = urllib.request.Request(
        f"http://{endpoint}/reset_prefix_cache",
        data=b"",
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout_s) as response:
        if response.status >= 300:
            raise RuntimeError(
                f"prefix cache reset returned HTTP {response.status}"
            )


def streaming_completion(
    endpoint: str,
    prompt: str,
    max_tokens: int,
    model: str,
    timeout_s: float,
    first_token_event: Optional[threading.Event] = None,
) -> Sample:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": 1,
    }
    request = urllib.request.Request(
        f"http://{endpoint}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.perf_counter()
    first_token_at = None
    last_token_at = None
    prompt_tokens = 0
    completion_tokens = 0
    token_intervals_s: list[float] = []
    with opener.open(request, timeout=timeout_s) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            event = json.loads(body)
            usage = event.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
            completion_tokens = int(
                usage.get("completion_tokens", completion_tokens)
            )
            choices = event.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.perf_counter()
                choice = choices[0]
                logprobs = choice.get("logprobs") or {}
                chunk_tokens = len(logprobs.get("tokens") or []) or 1
                if first_token_at is None:
                    first_token_at = now
                    if chunk_tokens > 1:
                        token_intervals_s.extend([0.0] * (chunk_tokens - 1))
                    if first_token_event is not None:
                        first_token_event.set()
                elif last_token_at is not None:
                    token_intervals_s.extend(
                        [(now - last_token_at) / chunk_tokens] * chunk_tokens
                    )
                last_token_at = now
    finished = time.perf_counter()
    if first_token_at is None or prompt_tokens <= 0 or completion_tokens <= 0:
        raise RuntimeError("stream did not return token timestamps and usage")
    tpot_s = 0.0
    if completion_tokens > 1 and last_token_at is not None:
        tpot_s = (last_token_at - first_token_at) / (completion_tokens - 1)
    return Sample(
        wall_s=finished - started,
        ttft_s=first_token_at - started,
        tpot_s=tpot_s,
        max_tbt_s=max(token_intervals_s, default=0.0),
        tbt_token_count=len(token_intervals_s),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


class BackgroundLoad:
    def __init__(
        self,
        count: int,
        endpoint: str,
        model: str,
        prompt: str,
        max_tokens: int,
        timeout_s: float,
    ) -> None:
        self.count = max(0, count)
        self.endpoint = endpoint
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.stop = threading.Event()
        self.threads: list[threading.Thread] = []
        self.ready: list[threading.Event] = []

    def _worker(self, ready: threading.Event) -> None:
        while not self.stop.is_set():
            try:
                streaming_completion(
                    self.endpoint,
                    self.prompt,
                    self.max_tokens,
                    self.model,
                    self.timeout_s,
                    first_token_event=ready,
                )
            except Exception:
                if self.stop.wait(0.05):
                    return

    def __enter__(self) -> "BackgroundLoad":
        for _ in range(self.count):
            ready = threading.Event()
            thread = threading.Thread(
                target=self._worker, args=(ready,), daemon=True
            )
            thread.start()
            self.ready.append(ready)
            self.threads.append(thread)
        deadline = time.monotonic() + self.timeout_s
        for ready in self.ready:
            if not ready.wait(max(0.0, deadline - time.monotonic())):
                self.stop.set()
                for thread in self.threads:
                    thread.join(timeout=1.0)
                raise RuntimeError(
                    "background calibration workers did not all reach Decode"
                )
        if self.count:
            time.sleep(0.25)
        return self

    def __exit__(self, *_args) -> None:
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=self.timeout_s)
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("background calibration workers did not stop cleanly")


def prompt_with_tokens(tokenizer, target_tokens: int, nonce: str) -> str:
    if target_tokens <= 0:
        raise ValueError("target prompt tokens must be positive")
    prefix = f"{nonce} "
    text = prefix + ((PROMPT_WORD + " ") * target_tokens)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    while len(token_ids) < target_tokens:
        text += (PROMPT_WORD + " ") * (target_tokens - len(token_ids))
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    prompt = tokenizer.decode(
        token_ids[:target_tokens],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not prompt.strip():
        raise RuntimeError("tokenizer produced an empty calibration prompt")
    return prompt


def nearest_rank(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("nearest-rank quantile requires samples")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def fit_line(points: Iterable[tuple[int, float]]) -> tuple[float, float, float]:
    rows = list(points)
    if len(rows) < 2 or len({x for x, _ in rows}) < 2:
        raise ValueError("prefill regression requires at least two prompt sizes")
    x_mean = statistics.mean(x for x, _ in rows)
    y_mean = statistics.mean(y for _, y in rows)
    denominator = sum((x - x_mean) ** 2 for x, _ in rows)
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in rows) / denominator
    )
    intercept = y_mean - slope * x_mean
    if intercept < 0:
        intercept = 0.0
        slope = sum(x * y for x, y in rows) / sum(x * x for x, _ in rows)
    if slope <= 0:
        raise ValueError("prefill regression produced a non-positive slope")
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in rows)
    total = sum((y - y_mean) ** 2 for _, y in rows)
    r_squared = 1.0 - residual / total if total > 0 else 1.0
    return slope, intercept, r_squared


def profile_prefill(
    args: argparse.Namespace,
    tokenizer,
    levels: list[int],
    prompt_sizes: list[int],
    run_nonce: str,
) -> list[dict]:
    curve = []
    background_prompt = prompt_with_tokens(
        tokenizer, args.background_prompt_tokens, f"{run_nonce}-bg-prefill"
    )
    for level in levels:
        samples = []
        with BackgroundLoad(
            level,
            args.endpoint,
            args.model,
            background_prompt,
            args.background_tokens,
            args.timeout_s,
        ):
            for words in prompt_sizes:
                for repeat in range(args.repeats):
                    sample = streaming_completion(
                        args.endpoint,
                        prompt_with_tokens(
                            tokenizer,
                            words,
                            f"{run_nonce}-p-{level}-{words}-{repeat}",
                        ),
                        1,
                        args.model,
                        args.timeout_s,
                    )
                    samples.append(sample)
        slope, intercept, r_squared = fit_line(
            (sample.prompt_tokens, sample.ttft_s) for sample in samples
        )
        absolute_residuals = [
            abs(sample.ttft_s - (intercept + slope * sample.prompt_tokens))
            for sample in samples
        ]
        curve.append(
            {
                "active_sequences": level,
                "seconds_per_token": slope,
                "intercept_s": intercept,
                "r_squared": r_squared,
                "residual_mae_s": statistics.mean(absolute_residuals),
                "residual_p95_abs_s": nearest_rank(absolute_residuals, 0.95),
                "sample_count": len(samples),
                "prompt_tokens_min": min(s.prompt_tokens for s in samples),
                "prompt_tokens_max": max(s.prompt_tokens for s in samples),
                "samples": [
                    {
                        "prompt_tokens": sample.prompt_tokens,
                        "ttft_s": sample.ttft_s,
                    }
                    for sample in samples
                ],
            }
        )
    return curve


def profile_decode(
    args: argparse.Namespace,
    tokenizer,
    levels: list[int],
    run_nonce: str,
) -> list[dict]:
    curve = []
    for level in levels:
        samples = []
        for repeat in range(args.repeats):
            # A fresh prompt gives every background sequence the configured KV
            # context at probe start. One probe per cohort prevents later
            # repeats from sampling a restarted background stream.
            background_prompt = prompt_with_tokens(
                tokenizer,
                args.background_prompt_tokens,
                f"{run_nonce}-bg-decode-{level}-{repeat}",
            )
            with BackgroundLoad(
                level,
                args.endpoint,
                args.model,
                background_prompt,
                args.background_tokens,
                args.timeout_s,
            ):
                sample = streaming_completion(
                    args.endpoint,
                    prompt_with_tokens(
                        tokenizer,
                        args.probe_prompt_tokens,
                        f"{run_nonce}-d-{level}-{repeat}",
                    ),
                    args.probe_tokens,
                    args.model,
                    args.timeout_s,
                )
                if sample.completion_tokens > 1 and sample.tpot_s > 0:
                    samples.append(sample)
        if not samples:
            raise RuntimeError(f"decode level {level} produced no valid samples")
        values = [sample.tpot_s for sample in samples]
        max_tbt_values = [sample.max_tbt_s for sample in samples]
        mean_tpot = statistics.mean(values)
        stdev_tpot = statistics.stdev(values) if len(values) > 1 else 0.0
        curve.append(
            {
                "active_sequences": level + 1,
                "background_sequences": level,
                "tpot_s_mean": mean_tpot,
                "tpot_s_median": statistics.median(values),
                "tpot_s_stdev": stdev_tpot,
                "tpot_s_cv": stdev_tpot / mean_tpot,
                "tpot_s_p95": nearest_rank(values, 0.95),
                "max_tbt_s_mean": statistics.mean(max_tbt_values),
                "max_tbt_s_median": statistics.median(max_tbt_values),
                "max_tbt_s_p95": nearest_rank(max_tbt_values, 0.95),
                "sample_count": len(values),
                "samples": [
                    {
                        "completion_tokens": sample.completion_tokens,
                        "tpot_s": sample.tpot_s,
                        "max_tbt_s": sample.max_tbt_s,
                        "tbt_token_count": sample.tbt_token_count,
                    }
                    for sample in samples
                ],
            }
        )
    return curve


def validate_decode_curve_quality(curve: list[dict]) -> dict:
    """Reject phase-noisy curves before they can steer formal routing."""

    levels = [int(point["active_sequences"]) for point in curve]
    if levels != sorted(set(levels)):
        raise RuntimeError("decode concurrency levels must strictly increase")
    cvs = [float(point["tpot_s_cv"]) for point in curve]
    max_cv = max(cvs, default=0.0)
    if max_cv > DECODE_MAX_CV:
        raise RuntimeError(
            "decode calibration is unstable: "
            f"max coefficient of variation {max_cv:.3f} > {DECODE_MAX_CV:.3f}"
        )
    medians = [float(point["tpot_s_median"]) for point in curve]
    relative_drops = [
        max(0.0, (left - right) / left)
        for left, right in zip(medians, medians[1:])
    ]
    max_relative_drop = max(relative_drops, default=0.0)
    return {
        "max_cv": max_cv,
        "max_relative_drop": max_relative_drop,
        "max_cv_allowed": DECODE_MAX_CV,
        "monotonicity_required": False,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="127.0.0.1:8000")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--prompt-sizes", default="256,1024,4096")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--probe-prompt-tokens", type=int, default=32)
    parser.add_argument("--probe-tokens", type=int, default=64)
    parser.add_argument("--background-prompt-tokens", type=int, default=32)
    parser.add_argument("--background-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--levels", default="0,8,16,24,32,40,48,56")
    parser.add_argument("--prefill-levels", default="0,8,24,40,56")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--engine-fingerprint",
        required=True,
        help="model/GPU/vLLM/APC/chunked-prefill deployment fingerprint",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    levels = [int(value) for value in args.levels.split(",") if value.strip()]
    prefill_levels = [
        int(value) for value in args.prefill_levels.split(",") if value.strip()
    ]
    prompt_sizes = [
        int(value) for value in args.prompt_sizes.split(",") if value.strip()
    ]
    if args.repeats < 2 or any(level < 0 for level in levels + prefill_levels):
        raise ValueError("repeats must be >=2 and concurrency levels non-negative")
    if not levels or not prefill_levels or len(set(prompt_sizes)) < 2:
        raise ValueError("profile requires non-empty levels and two prompt sizes")
    if any(size <= 0 for size in prompt_sizes):
        raise ValueError("prompt sizes must be positive")
    if args.probe_tokens <= 1 or args.background_tokens <= 1:
        raise ValueError("decode probe/background tokens must exceed one")
    if args.probe_prompt_tokens <= 0 or args.background_prompt_tokens <= 0:
        raise ValueError("probe/background prompt tokens must be positive")
    if args.timeout_s <= 0:
        raise ValueError("timeout must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model, trust_remote_code=True
    )
    run_nonce = str(time.time_ns())
    reset_prefix_cache(args.endpoint, args.timeout_s)
    prefill_curve = profile_prefill(
        args, tokenizer, prefill_levels, prompt_sizes, run_nonce
    )
    decode_curve = profile_decode(args, tokenizer, levels, run_nonce)
    try:
        decode_quality = validate_decode_curve_quality(decode_curve)
    except RuntimeError:
        rejected = Path(args.output).with_suffix(
            Path(args.output).suffix + ".rejected"
        )
        rejected.write_text(
            json.dumps(
                {
                    "engine_fingerprint": args.engine_fingerprint,
                    "decode_curve": decode_curve,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    report = {
        "schema_version": 2,
        "calibrator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "endpoint": args.endpoint,
        "model": args.model,
        "engine_fingerprint": args.engine_fingerprint,
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prefill_curve": prefill_curve,
        "decode_curve": decode_curve,
        "decode_quality": decode_quality,
        "prefill_tpot_s": prefill_curve[0]["seconds_per_token"],
        "prefill_intercept_s": prefill_curve[0]["intercept_s"],
        "decode_base_tpot_s": decode_curve[0]["tpot_s_median"],
        "configuration": {
            "prefill_levels": prefill_levels,
            "decode_total_active_sequences": [level + 1 for level in levels],
            "prompt_token_targets": prompt_sizes,
            "tokenizer_path": args.tokenizer_path or args.model,
            "repeats": args.repeats,
            "probe_prompt_tokens": args.probe_prompt_tokens,
            "probe_tokens": args.probe_tokens,
            "background_prompt_tokens": args.background_prompt_tokens,
            "background_tokens": args.background_tokens,
            "timeout_s": args.timeout_s,
            "run_nonce": run_nonce,
            "prefix_cache_reset_before_run": True,
        },
        "method": {
            "prefill": (
                "streaming TTFT affine regression on API prompt token counts; "
                "negative intercepts are projected to zero and refit through origin"
            ),
            "decode": (
                "streaming post-first-token mean TPOT plus the maximum "
                "per-token interarrival gap measured from completion logprobs"
            ),
            "quantile": "finite-sample nearest-rank order statistic",
            "background": (
                "one fresh fixed-context cohort per Decode repeat; every worker "
                "emits its first token before the single probe, and the cohort "
                "is joined before the next repeat"
            ),
            "cache_isolation": (
                "POST /reset_prefix_cache before run plus run-unique first-block nonce"
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
