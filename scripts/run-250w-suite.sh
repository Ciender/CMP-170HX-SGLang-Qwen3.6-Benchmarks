#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
serve_script="$script_dir/serve-qwen36-27b.sh"
bench_script="$script_dir/bench-serving.sh"
monitor_script="$script_dir/monitor-gpu.sh"
results_dir="${RESULTS_DIR:-$repo_root/results/raw}"
telemetry_dir="${TELEMETRY_DIR:-$repo_root/results/telemetry}"
logs_dir="${LOGS_DIR:-$repo_root/results/logs}"
base_url="${BASE_URL:-http://127.0.0.1:8000}"
expected_power_limit="${EXPECTED_POWER_LIMIT:-250}"
server_pid=""
monitor_pid=""

die() {
  echo "error: $*" >&2
  exit 1
}

stop_profile() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill -INT "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  monitor_pid=""

  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}

cleanup() {
  stop_profile
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start_profile() {
  local profile="$1"
  local log_file="$logs_dir/$profile.log"
  local telemetry_file="$telemetry_dir/power-$profile-250w.csv"
  local ready=0

  PROFILE="$profile" "$serve_script" >"$log_file" 2>&1 &
  server_pid=$!

  for _ in $(seq 1 "${STARTUP_TIMEOUT:-600}"); do
    if curl -fsS "$base_url/model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -n 80 "$log_file" >&2
      die "$profile server exited during startup"
    fi
    sleep 1
  done
  [[ "$ready" -eq 1 ]] || die "$profile server did not become ready"

  "$monitor_script" "$telemetry_file" &
  monitor_pid=$!
  sleep 1
}

wait_for_vram_release() {
  local used
  for _ in $(seq 1 60); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)"
    if [[ "$used" == "0" ]]; then
      return
    fi
    sleep 1
  done
  die "GPU memory was not released after stopping the server"
}

[[ $EUID -eq 0 ]] || die "run as root; the serving profile manages vm.overcommit_memory"
command -v curl >/dev/null || die "curl is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"

current_power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n 1)"
if [[ "$current_power_limit" != "$expected_power_limit" && \
      "$current_power_limit" != "$expected_power_limit.00" ]]; then
  die "expected ${expected_power_limit}W power limit, found ${current_power_limit}W"
fi

compute_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ -z "$compute_apps" ]] || die "another GPU compute process is active: $compute_apps"
curl -fsS "$base_url/model_info" >/dev/null 2>&1 && \
  die "an SGLang server is already running at $base_url"

mkdir -p "$results_dir" "$telemetry_dir" "$logs_dir" "$repo_root/environment"
"$script_dir/collect-environment.sh" >"$repo_root/environment/latest-run.txt"

start_profile mtp
for output_len in 128 256 512; do
  RUN_TAG=250w-mtp WARMUP_REQUESTS=0 \
    DECODE_INPUT_LEN=128 DECODE_OUTPUT_LEN="$output_len" DECODE_PROMPTS=5 \
    RESULTS_DIR="$results_dir" "$bench_script" decode
done
for input_len in 10240 20480; do
  RUN_TAG=250w-mtp-long WARMUP_REQUESTS=0 \
    DECODE_INPUT_LEN="$input_len" DECODE_OUTPUT_LEN=128 DECODE_PROMPTS=3 \
    RESULTS_DIR="$results_dir" "$bench_script" decode
done
stop_profile
wait_for_vram_release

start_profile baseline
for output_len in 128 256 512; do
  RUN_TAG=250w-baseline WARMUP_REQUESTS=0 \
    DECODE_INPUT_LEN=128 DECODE_OUTPUT_LEN="$output_len" DECODE_PROMPTS=5 \
    RESULTS_DIR="$results_dir" "$bench_script" decode
done
for input_len in 10240 20480; do
  RUN_TAG=250w-baseline-long WARMUP_REQUESTS=0 \
    DECODE_INPUT_LEN="$input_len" DECODE_OUTPUT_LEN=128 DECODE_PROMPTS=3 \
    RESULTS_DIR="$results_dir" "$bench_script" decode
done
stop_profile
wait_for_vram_release

start_profile prefill
for input_len in 128 256 512 1024 2048 4096 8192; do
  RUN_TAG=250w-prefill WARMUP_REQUESTS=0 \
    PREFILL_INPUT_LEN="$input_len" PREFILL_OUTPUT_LEN=1 PREFILL_PROMPTS=5 \
    RESULTS_DIR="$results_dir" "$bench_script" prefill
done
stop_profile
wait_for_vram_release

start_profile throughput
RUN_TAG=250w-throughput-c4 WARMUP_REQUESTS=0 \
  MAX_CONCURRENCY=4 MIXED_INPUT_LEN=1024 MIXED_OUTPUT_LEN=256 MIXED_PROMPTS=20 \
  RESULTS_DIR="$results_dir" "$bench_script" mixed
stop_profile
wait_for_vram_release

echo "Benchmark suite completed."
echo "Results: $results_dir"
echo "Telemetry: $telemetry_dir"
