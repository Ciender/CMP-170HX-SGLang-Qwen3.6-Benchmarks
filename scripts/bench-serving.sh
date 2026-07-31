#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
runtime_home="${SGLANG_RUNTIME_HOME:-/mnt/ss32t/SGLang}"
if [[ -f "$runtime_home/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$runtime_home/env.sh"
fi
venv_dir="${SGLANG_VENV:-$runtime_home/venv}"
model_path="${MODEL_PATH:-/mnt/ss32t/LLM-models/Qwen3.6-27B-INT8-W8A8}"
base_url="${BASE_URL:-http://127.0.0.1:8000}"
run_tag="${RUN_TAG:-run-$(date +%Y%m%d-%H%M%S)}"
results_dir="${RESULTS_DIR:-$repo_root/results/raw}"
mode="${1:-all}"
prefill_input_len="${PREFILL_INPUT_LEN:-4096}"
prefill_output_len="${PREFILL_OUTPUT_LEN:-1}"
decode_input_len="${DECODE_INPUT_LEN:-128}"
decode_output_len="${DECODE_OUTPUT_LEN:-256}"
mixed_input_len="${MIXED_INPUT_LEN:-1024}"
mixed_output_len="${MIXED_OUTPUT_LEN:-256}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

[[ -x "$venv_dir/bin/python" ]] || { echo "runtime not found at $venv_dir" >&2; exit 1; }
[[ -f "$model_path/config.json" ]] || { echo "model not found at $model_path" >&2; exit 1; }
mkdir -p "$results_dir"

run_case() {
  local name="$1"
  local input_len="$2"
  local output_len="$3"
  local concurrency="$4"
  local prompts="$5"
  local warmup_requests="${WARMUP_REQUESTS:-0}"
  local output_file="$results_dir/${run_tag}-${name}-c${concurrency}.jsonl"

  "$venv_dir/bin/python" -m sglang.benchmark.serving \
    --backend sglang-oai \
    --base-url "$base_url" \
    --dataset-name random \
    --model "$model_path" \
    --tokenizer "$model_path" \
    --num-prompts "$prompts" \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --random-range-ratio 1.0 \
    --max-concurrency "$concurrency" \
    --warmup-requests "$warmup_requests" \
    --flush-cache \
    --seed 42 \
    --temperature 0.0 \
    --top-p 1.0 \
    --tag "$run_tag" \
    --output-details \
    --output-file "$output_file" \
    --disable-tqdm
}

run_dataset_case() {
  local name="$1"
  local dataset_name="$2"
  local dataset_path="$3"
  local concurrency="$4"
  local prompts="$5"
  local context_len="$6"
  local fixed_output_len="$7"
  local warmup_requests="${WARMUP_REQUESTS:-0}"
  local output_file="$results_dir/${run_tag}-${name}-c${concurrency}.jsonl"
  local dataset_args=(
    --dataset-name "$dataset_name"
    --sharegpt-context-len "$context_len"
  )

  if [[ -n "$dataset_path" ]]; then
    dataset_args+=(--dataset-path "$dataset_path")
  fi
  if [[ -n "$fixed_output_len" ]]; then
    dataset_args+=(--sharegpt-output-len "$fixed_output_len")
  fi

  "$venv_dir/bin/python" -m sglang.benchmark.serving \
    --backend sglang-oai \
    --base-url "$base_url" \
    "${dataset_args[@]}" \
    --model "$model_path" \
    --tokenizer "$model_path" \
    --num-prompts "$prompts" \
    --max-concurrency "$concurrency" \
    --warmup-requests "$warmup_requests" \
    --flush-cache \
    --seed 42 \
    --temperature 0.0 \
    --top-p 1.0 \
    --tag "$run_tag" \
    --output-details \
    --output-file "$output_file" \
    --disable-tqdm
}

case "$mode" in
  prefill)
    run_case "prefill-$prefill_input_len-$prefill_output_len" \
      "$prefill_input_len" "$prefill_output_len" 1 "${PREFILL_PROMPTS:-5}"
    ;;
  decode)
    run_case "decode-$decode_input_len-$decode_output_len" \
      "$decode_input_len" "$decode_output_len" 1 "${DECODE_PROMPTS:-5}"
    ;;
  mixed)
    concurrency="${MAX_CONCURRENCY:-4}"
    prompts="${MIXED_PROMPTS:-$((concurrency * 5))}"
    run_case "mixed-$mixed_input_len-$mixed_output_len" \
      "$mixed_input_len" "$mixed_output_len" "$concurrency" "$prompts"
    ;;
  dataset)
    dataset_name="${DATASET_NAME:?DATASET_NAME is required for dataset mode}"
    dataset_path="${DATASET_PATH:-}"
    concurrency="${MAX_CONCURRENCY:-1}"
    prompts="${DATASET_PROMPTS:-30}"
    context_len="${DATASET_CONTEXT_LEN:-24576}"
    fixed_output_len="${DATASET_OUTPUT_LEN:-}"
    run_dataset_case "$dataset_name${fixed_output_len:+-fixed-$fixed_output_len}" \
      "$dataset_name" "$dataset_path" "$concurrency" "$prompts" \
      "$context_len" "$fixed_output_len"
    ;;
  all)
    run_case "prefill-$prefill_input_len-$prefill_output_len" \
      "$prefill_input_len" "$prefill_output_len" 1 "${PREFILL_PROMPTS:-5}"
    run_case "decode-$decode_input_len-$decode_output_len" \
      "$decode_input_len" "$decode_output_len" 1 "${DECODE_PROMPTS:-5}"
    concurrency="${MAX_CONCURRENCY:-4}"
    prompts="${MIXED_PROMPTS:-$((concurrency * 5))}"
    run_case "mixed-$mixed_input_len-$mixed_output_len" \
      "$mixed_input_len" "$mixed_output_len" "$concurrency" "$prompts"
    ;;
  *)
    echo "usage: $0 {prefill|decode|mixed|dataset|all}" >&2
    exit 2
    ;;
esac
