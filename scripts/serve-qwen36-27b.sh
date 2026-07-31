#!/usr/bin/env bash

set -euo pipefail

runtime_home="${SGLANG_RUNTIME_HOME:-/mnt/ss32t/SGLang}"
venv_dir="${SGLANG_VENV:-$runtime_home/venv}"
model_path="${MODEL_PATH:-/mnt/ss32t/LLM-models/Qwen3.6-27B-INT8-W8A8}"
profile="${PROFILE:-mtp}"
host="${HOST:-127.0.0.1}"
port="${PORT:-8000}"

die() {
  echo "error: $*" >&2
  exit 1
}

[[ $EUID -eq 0 ]] || die "run as root so vm.overcommit_memory can be restored"
[[ -x "$venv_dir/bin/sglang" ]] || die "SGLang runtime not found at $venv_dir"
[[ -f "$model_path/config.json" ]] || die "model config not found at $model_path"
[[ -f "$model_path/model.safetensors" ]] || die "model weights not found at $model_path"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
command -v curl >/dev/null || die "curl is required"

compute_cap="$({ nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1; } 2>/dev/null)"
if [[ "$compute_cap" != "8.0" && "${ALLOW_NON_SM80:-0}" != "1" ]]; then
  die "this profile requires SM80 (detected compute capability: ${compute_cap:-unknown})"
fi

if curl -fsS "http://$host:$port/model_info" >/dev/null 2>&1; then
  die "an SGLang server is already responding at http://$host:$port"
fi

case "$profile" in
  mtp)
    context_length="${CONTEXT_LENGTH:-24576}"
    mem_fraction_static="${MEM_FRACTION_STATIC:-0.94}"
    max_total_tokens="${MAX_TOTAL_TOKENS:-$context_length}"
    max_running_requests="${MAX_RUNNING_REQUESTS:-1}"
    chunked_prefill_size="${CHUNKED_PREFILL_SIZE:-2048}"
    max_prefill_tokens="${MAX_PREFILL_TOKENS:-4096}"
    cuda_graph_max_bs_decode="${CUDA_GRAPH_MAX_BS_DECODE:-1}"
    [[ -f "$model_path/mtp.safetensors" ]] || die "MTP weights not found"
    profile_args=(
      --disable-prefill-cuda-graph
      --disable-radix-cache
      --page-size "${PAGE_SIZE:-64}"
      --speculative-algorithm EAGLE
      --speculative-num-steps "${MTP_STEPS:-3}"
      --speculative-eagle-topk "${MTP_TOPK:-1}"
      --speculative-num-draft-tokens "${MTP_DRAFT_TOKENS:-4}"
    )
    ;;
  baseline)
    context_length="${CONTEXT_LENGTH:-24576}"
    mem_fraction_static="${MEM_FRACTION_STATIC:-0.88}"
    max_total_tokens="${MAX_TOTAL_TOKENS:-$context_length}"
    max_running_requests="${MAX_RUNNING_REQUESTS:-1}"
    chunked_prefill_size="${CHUNKED_PREFILL_SIZE:-2048}"
    max_prefill_tokens="${MAX_PREFILL_TOKENS:-4096}"
    cuda_graph_max_bs_decode="${CUDA_GRAPH_MAX_BS_DECODE:-1}"
    profile_args=(
      --disable-prefill-cuda-graph
      --disable-radix-cache
      --page-size "${PAGE_SIZE:-64}"
    )
    ;;
  mtp-c4)
    context_length="${CONTEXT_LENGTH:-24576}"
    mem_fraction_static="${MEM_FRACTION_STATIC:-0.98}"
    max_total_tokens="${MAX_TOTAL_TOKENS:-20480}"
    max_running_requests="${MAX_RUNNING_REQUESTS:-4}"
    chunked_prefill_size="${CHUNKED_PREFILL_SIZE:-2048}"
    max_prefill_tokens="${MAX_PREFILL_TOKENS:-4096}"
    cuda_graph_max_bs_decode="${CUDA_GRAPH_MAX_BS_DECODE:-4}"
    [[ -f "$model_path/mtp.safetensors" ]] || die "MTP weights not found"
    profile_args=(
      --disable-prefill-cuda-graph
      --disable-radix-cache
      --page-size "${PAGE_SIZE:-64}"
      --speculative-algorithm EAGLE
      --speculative-num-steps "${MTP_STEPS:-3}"
      --speculative-eagle-topk "${MTP_TOPK:-1}"
      --speculative-num-draft-tokens "${MTP_DRAFT_TOKENS:-4}"
    )
    ;;
  baseline-c4)
    context_length="${CONTEXT_LENGTH:-24576}"
    mem_fraction_static="${MEM_FRACTION_STATIC:-0.88}"
    max_total_tokens="${MAX_TOTAL_TOKENS:-20480}"
    max_running_requests="${MAX_RUNNING_REQUESTS:-4}"
    chunked_prefill_size="${CHUNKED_PREFILL_SIZE:-2048}"
    max_prefill_tokens="${MAX_PREFILL_TOKENS:-4096}"
    cuda_graph_max_bs_decode="${CUDA_GRAPH_MAX_BS_DECODE:-4}"
    profile_args=(
      --disable-prefill-cuda-graph
      --disable-radix-cache
      --page-size "${PAGE_SIZE:-64}"
    )
    ;;
  prefill)
    context_length="${CONTEXT_LENGTH:-32768}"
    mem_fraction_static="${MEM_FRACTION_STATIC:-0.88}"
    max_total_tokens="${MAX_TOTAL_TOKENS:-$context_length}"
    max_running_requests="${MAX_RUNNING_REQUESTS:-1}"
    chunked_prefill_size="${CHUNKED_PREFILL_SIZE:-4096}"
    max_prefill_tokens="${MAX_PREFILL_TOKENS:-8192}"
    cuda_graph_max_bs_decode="${CUDA_GRAPH_MAX_BS_DECODE:-1}"
    profile_args=()
    ;;
  throughput)
    context_length="${CONTEXT_LENGTH:-32768}"
    mem_fraction_static="${MEM_FRACTION_STATIC:-0.88}"
    max_total_tokens="${MAX_TOTAL_TOKENS:-32768}"
    max_running_requests="${MAX_RUNNING_REQUESTS:-4}"
    chunked_prefill_size="${CHUNKED_PREFILL_SIZE:-2048}"
    max_prefill_tokens="${MAX_PREFILL_TOKENS:-4096}"
    cuda_graph_max_bs_decode="${CUDA_GRAPH_MAX_BS_DECODE:-4}"
    profile_args=(--disable-prefill-cuda-graph --enable-mixed-chunk)
    ;;
  *)
    die "unknown PROFILE=$profile (expected mtp, baseline, mtp-c4, baseline-c4, prefill, throughput)"
    ;;
esac

(( max_total_tokens <= context_length )) || \
  die "MAX_TOTAL_TOKENS cannot exceed CONTEXT_LENGTH"

original_overcommit="$(sysctl -n vm.overcommit_memory)"
overcommit_restored=0
server_pid=""

restore_overcommit() {
  if [[ "$overcommit_restored" -eq 0 ]]; then
    sysctl -q -w "vm.overcommit_memory=$original_overcommit"
    overcommit_restored=1
  fi
}

# shellcheck disable=SC2317 # Invoked indirectly by trap.
cleanup() {
  restore_overcommit
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ -f "$runtime_home/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$runtime_home/env.sh"
else
  export VIRTUAL_ENV="$venv_dir"
  export PATH="$venv_dir/bin:$PATH"
fi
export PYTHONNOUSERSITE=1

if [[ -n "${SGLANG_SOURCE:-}" ]]; then
  [[ -d "$SGLANG_SOURCE/python/sglang" ]] || \
    die "SGLANG_SOURCE does not contain python/sglang: $SGLANG_SOURCE"
  export PYTHONPATH="$SGLANG_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}"
fi

storage_path="${FILE_STORAGE_PATH:-$runtime_home/storage}"
mkdir -p "$storage_path"
sysctl -q -w vm.overcommit_memory=1

echo "profile=$profile gpu=SM$compute_cap context=$context_length max_total_tokens=$max_total_tokens"
"$venv_dir/bin/sglang" serve \
  --model-path "$model_path" \
  --served-model-name "${SERVED_MODEL_NAME:-qwen3.6-27b-int8-w8a8}" \
  --host "$host" \
  --port "$port" \
  --tp-size 1 \
  --context-length "$context_length" \
  --mem-fraction-static "$mem_fraction_static" \
  --max-total-tokens "$max_total_tokens" \
  --max-running-requests "$max_running_requests" \
  --chunked-prefill-size "$chunked_prefill_size" \
  --max-prefill-tokens "$max_prefill_tokens" \
  --cuda-graph-max-bs-decode "$cuda_graph_max_bs_decode" \
  --attention-backend flashinfer \
  --linear-attn-backend triton \
  --sampling-backend flashinfer \
  --mamba-radix-cache-strategy extra_buffer \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --quantization compressed-tensors \
  --file-storage-path "$storage_path" \
  "${profile_args[@]}" \
  "$@" &
server_pid=$!

for _ in $(seq 1 "${STARTUP_TIMEOUT:-600}"); do
  if curl -fsS "http://$host:$port/model_info" >/dev/null 2>&1; then
    restore_overcommit
    echo "SGLang ready at http://$host:$port (profile=$profile)"
    wait "$server_pid"
    exit $?
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    exit $?
  fi
  sleep 1
done

die "SGLang did not become ready within ${STARTUP_TIMEOUT:-600} seconds"
