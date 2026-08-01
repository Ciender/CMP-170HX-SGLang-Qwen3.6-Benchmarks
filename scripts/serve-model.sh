#!/usr/bin/env bash

set -euo pipefail

runtime_home="${SGLANG_RUNTIME_HOME:?SGLANG_RUNTIME_HOME is required}"
venv_dir="${SGLANG_VENV:-$runtime_home/venv}"
model_path="${MODEL_PATH:?MODEL_PATH is required}"
host="${HOST:-127.0.0.1}"
port="${PORT:-8000}"
context_length="${CONTEXT_LENGTH:?CONTEXT_LENGTH is required}"
max_running_requests="${MAX_RUNNING_REQUESTS:-1}"
mem_fraction_static="${MEM_FRACTION_STATIC:-0.90}"
max_total_tokens="${MAX_TOTAL_TOKENS:-auto}"
mtp_state="${MTP_STATE:-disabled}"

die() {
  echo "error: $*" >&2
  exit 1
}

[[ -x "$venv_dir/bin/sglang" ]] || die "SGLang runtime not found at $venv_dir"
[[ -f "$model_path/config.json" ]] || die "model config not found at $model_path"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
command -v curl >/dev/null || die "curl is required"

server_args=(
  serve
  --model-path "$model_path"
  --served-model-name "${SERVED_MODEL_NAME:-${MODEL_NAME:-$(basename "$model_path")}}"
  --host "$host"
  --port "$port"
  --tp-size "${TP_SIZE:-1}"
  --mem-fraction-static "$mem_fraction_static"
  --max-running-requests "$max_running_requests"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-2048}"
  --max-prefill-tokens "${MAX_PREFILL_TOKENS:-4096}"
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE:-$max_running_requests}"
  --page-size "${PAGE_SIZE:-64}"
  --disable-prefill-cuda-graph
  --disable-radix-cache
)

if [[ "$context_length" != "auto" ]]; then
  server_args+=(--context-length "$context_length")
fi
if [[ "$max_total_tokens" != "auto" ]]; then
  server_args+=(--max-total-tokens "$max_total_tokens")
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  server_args+=(--trust-remote-code)
fi
if [[ -n "${QUANTIZATION:-}" ]]; then
  server_args+=(--quantization "$QUANTIZATION")
fi
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  server_args+=(--attention-backend "$ATTENTION_BACKEND")
fi
if [[ -n "${LINEAR_ATTN_BACKEND:-}" ]]; then
  server_args+=(--linear-attn-backend "$LINEAR_ATTN_BACKEND")
fi
if [[ -n "${SAMPLING_BACKEND:-}" ]]; then
  server_args+=(--sampling-backend "$SAMPLING_BACKEND")
fi
if [[ -n "${REASONING_PARSER:-}" ]]; then
  server_args+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ -n "${TOOL_CALL_PARSER:-}" ]]; then
  server_args+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi

case "$mtp_state" in
  enabled)
    server_args+=(
      --speculative-algorithm "${MTP_ALGORITHM:-EAGLE}"
      --speculative-num-steps "${MTP_STEPS:-3}"
      --speculative-eagle-topk "${MTP_TOPK:-1}"
      --speculative-num-draft-tokens "${MTP_DRAFT_TOKENS:-4}"
    )
    ;;
  disabled) ;;
  *) die "MTP_STATE must be enabled or disabled" ;;
esac

if [[ -f "$runtime_home/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$runtime_home/env.sh"
else
  export VIRTUAL_ENV="$venv_dir"
  export PATH="$venv_dir/bin:$PATH"
fi
export PYTHONNOUSERSITE=1
if [[ -n "${SGLANG_SOURCE:-}" ]]; then
  [[ -d "$SGLANG_SOURCE/python/sglang" ]] || die "invalid SGLANG_SOURCE=$SGLANG_SOURCE"
  export PYTHONPATH="$SGLANG_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}"
fi

extra_args=()
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
  # EXTRA_SERVER_ARGS is an explicit operator-supplied argument string.
  read -r -a extra_args <<<"$EXTRA_SERVER_ARGS"
fi

exec "$venv_dir/bin/sglang" "${server_args[@]}" "${extra_args[@]}" "$@"
