#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 --profile FILE --python PYTHON --model MODEL --port PORT --gpu GPU [options]" \
    "" \
    "Options:" \
    "  --served-model-name NAME       Default: qwen" \
    "  --host HOST                    Default: 0.0.0.0" \
    "  --dtype DTYPE                  Default: float" \
    "  --max-model-len N              Default: 16384" \
    "  --max-num-seqs N               Default: 64" \
    "  --max-num-batched-tokens N     Default: 16384" \
    "  --block-size N                 Default: 16" \
    "  --gpu-memory-utilization X     Default: 0.9" \
    "  --cpu-set CPULIST              Optional taskset CPU list" \
    "  --                             Extra vLLM arguments"
}

PROFILE=""
PYTHON=""
MODEL=""
PORT=""
GPU=""
SERVED_MODEL_NAME="qwen"
HOST="0.0.0.0"
DTYPE="float"
MAX_MODEL_LEN="16384"
MAX_NUM_SEQS="64"
MAX_NUM_BATCHED_TOKENS="16384"
BLOCK_SIZE="16"
GPU_MEMORY_UTILIZATION="0.9"
CPU_SET=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --cpu-set) CPU_SET="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in PROFILE PYTHON MODEL PORT GPU; do
  if [[ -z "${!required}" ]]; then
    printf 'Missing required argument for %s\n' "$required" >&2
    usage >&2
    exit 2
  fi
done
if [[ ! -f "$PROFILE" ]]; then
  printf 'Service profile not found: %s\n' "$PROFILE" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  printf 'Python is not executable: %s\n' "$PYTHON" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"
"$PYTHON" "$ROOT/scripts/validate_profile_launch.py" \
  --profile "$PROFILE" \
  --model "$MODEL" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --block-size "$BLOCK_SIZE" \
  --engine-implementation v0

export PYTHONPATH="$ROOT/runtime/ravel_vllm_adapter:$ROOT/src:${PYTHONPATH:-}"
export RAVEL_VLLM_ADAPTER=1
export RAVEL_SERVICE_PROFILE="$(cd "$(dirname "$PROFILE")" && pwd)/$(basename "$PROFILE")"
export VLLM_USE_V1=0

VLLM_COMMAND=("$PYTHON" -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --block-size "$BLOCK_SIZE" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --scheduling-policy priority \
  --enforce-eager \
  --disable-log-requests \
  "${EXTRA_ARGS[@]}")

if [[ -n "$CPU_SET" ]]; then
  if ! command -v taskset >/dev/null 2>&1; then
    printf 'taskset is required when --cpu-set is provided\n' >&2
    exit 2
  fi
  exec taskset --cpu-list "$CPU_SET" "${VLLM_COMMAND[@]}"
fi

exec "${VLLM_COMMAND[@]}"
