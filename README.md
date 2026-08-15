# RAVEL

RAVEL is an SLO-aware cross-cluster router for vLLM. The released policy,
`RAVEL-Unified`, combines measured service curves, reversible Router-owned
placement before engine submission, and SLO-semantic admission. It does not
migrate materialized KV and does not modify the installed vLLM source tree.

This directory is the RAVEL-only public release prepared on 2026-08-15.
Baseline implementations, measurements and comparison artifacts are not part
of this bundle.

## Frozen release scope

- Model: Qwen3-1.7B, bfloat16, vLLM 0.8.5.post1 V0.
- Engine: max model length 16384, 64 sequences, 16384 batched tokens,
  block size 16, APC on, Chunked Prefill on, eager mode on.
- Policy: `RAVEL-Unified` (scheduler id `ravel_unified`).
- Result campaign:
  `qwen3-1.7b-vllm085-v0-ravel-unified-hybrid-final-v3-20260814`.
- Calibration: 32-token prompt, 256-token background, 128-token probe,
  5 repeats.

## Start here

- Deployment: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- Algorithm and formulas: [docs/RAVEL_TECHNICAL_REFERENCE.md](docs/RAVEL_TECHNICAL_REFERENCE.md)
- Parameter provenance: [docs/PARAMETER_AUDIT.md](docs/PARAMETER_AUDIT.md)
- Release provenance: [release/README.md](release/README.md)
- Final results: [results/qwen3-1.7b-vllm085-v0-ravel-unified-hybrid-final-v3-20260814/RESULTS.md](results/qwen3-1.7b-vllm085-v0-ravel-unified-hybrid-final-v3-20260814/RESULTS.md)

## Repository layout

```text
configs/    frozen RAVEL service/output profiles and portable topology
data/       frozen first-500 workloads and provenance
docs/       deployment, algorithm and parameter documentation
release/    RAVEL calibration, recorded inputs and release manifests
runtime/    opt-in vLLM startup adapter
scripts/    calibration, validation and RAVEL experiment tools
src/        RAVEL router, engine adapter and tests
results/    the single final RAVEL campaign
```

## Installation

```bash
python -m pip install -r requirements-router.txt
python -m pip install -r requirements-dev.txt
# On each serving node, in its own environment:
python -m pip install -r requirements-serving.txt
```

The serving process must use V0 (`VLLM_USE_V1=0`) and match the selected
schema-v2 service-profile fingerprint.

## Verify and rebuild

```bash
python scripts/build_release_artifacts.py
python scripts/verify_release.py
PYTHONPATH=src pytest -q src/tests
```

The builder validates exactly 18 RAVEL cells (3 workloads × 6 speedups), with
500 unique requests per cell, and regenerates the 14-column and quantile
reports.

## Run a new RAVEL matrix

Copy `.env.example` to `.env.local`, set deployment-local values, and run:

```bash
set -a
source .env.local
set +a

python scripts/run_jitserve_three_cluster_matrix.py \
  --topology configs/topologies/ravel-final-numapinned-v3.2-2-1.json \
  --model-path "$RAVEL_MODEL_PATH" \
  --served-model-name qwen \
  --endpoints "$RAVEL_ENDPOINTS" \
  --network-delay-mode physical \
  --result-root results/local_run \
  --workloads lmsys,burst,deepresearch \
  --speedups 1,2,4,8,12,16 \
  --policies RAVEL-Unified \
  --request-count 500
```

A different GPU, model, vLLM version, dtype or engine contract requires a new
calibration.

## License

MIT. See [LICENSE](LICENSE).
