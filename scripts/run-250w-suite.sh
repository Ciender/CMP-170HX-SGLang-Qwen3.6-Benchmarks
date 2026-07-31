#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
serve_script="$script_dir/serve-qwen36-27b.sh"
bench_script="$script_dir/bench-serving.sh"
monitor_script="$script_dir/monitor-gpu.sh"
run_id="${RUN_ID:-expanded-250w-$(date +%Y%m%d-%H%M%S)}"
results_dir="${RESULTS_DIR:-$repo_root/results/raw/$run_id}"
telemetry_dir="${TELEMETRY_DIR:-$repo_root/results/telemetry/$run_id}"
logs_dir="${LOGS_DIR:-$repo_root/results/logs/$run_id}"
manifest_file="${MANIFEST_FILE:-$repo_root/results/case-manifest-$run_id.csv}"
base_url="${BASE_URL:-http://127.0.0.1:8000}"
expected_power_limit="${EXPECTED_POWER_LIMIT:-250}"
sharegpt_path="${SHAREGPT_PATH:-}"
longbench_path="${LONGBENCH_PATH:-}"
server_pid=""
monitor_pid=""
active_profile=""
active_telemetry=""

timestamp_iso() {
  date '+%Y-%m-%dT%H:%M:%S.%3N%:z'
}

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
  active_profile=""
  active_telemetry=""
}

cleanup() {
  stop_profile
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start_profile() {
  local profile="$1"
  local log_file="$logs_dir/server-$profile.log"
  local segment_id
  local ready=0

  segment_id="$(date '+%Y%m%d-%H%M%S-%3N')"
  active_telemetry="$telemetry_dir/power-$profile-$segment_id.csv"
  PROFILE="$profile" "$serve_script" >"$log_file" 2>&1 &
  server_pid=$!

  for _ in $(seq 1 "${STARTUP_TIMEOUT:-600}"); do
    if curl -fsS "$base_url/model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      wait "$server_pid" 2>/dev/null || true
      server_pid=""
      tail -n 100 "$log_file" >&2
      return 1
    fi
    sleep 1
  done

  if [[ "$ready" -ne 1 ]]; then
    tail -n 100 "$log_file" >&2
    return 1
  fi

  "$monitor_script" "$active_telemetry" &
  monitor_pid=$!
  active_profile="$profile"
  sleep 1
  echo "profile ready: $profile"
}

validate_profile_capacity() {
  local expected_tokens="$1"
  local expected_requests="$2"
  "$SGLANG_VENV/bin/python" - "$base_url" "$expected_tokens" "$expected_requests" <<'PY'
import json
import sys
import urllib.request

base_url, expected_tokens, expected_requests = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with urllib.request.urlopen(f"{base_url}/server_info", timeout=30) as response:
    info = json.load(response)
actual_tokens = info.get("max_total_num_tokens")
internal = info.get("internal_states", [{}])
effective_requests = None
for state in internal:
    if isinstance(state, dict) and "effective_max_running_requests_per_dp" in state:
        effective_requests = state["effective_max_running_requests_per_dp"]
        break
if actual_tokens != expected_tokens or effective_requests != expected_requests:
    raise SystemExit(
        f"capacity mismatch: tokens={actual_tokens} expected={expected_tokens}, "
        f"requests={effective_requests} expected={expected_requests}"
    )
print(f"validated server capacity: tokens={actual_tokens}, running_requests={effective_requests}")
PY
}

wait_for_vram_release() {
  local used
  for _ in $(seq 1 90); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)"
    if [[ "$used" == "0" ]]; then
      return
    fi
    sleep 1
  done
  die "GPU memory was not released after stopping the server"
}

append_manifest() {
  local profile="$1"
  local workload="$2"
  local input_len="$3"
  local output_len="$4"
  local concurrency="$5"
  local requests="$6"
  local started_iso="$7"
  local ended_iso="$8"
  local started_ms="$9"
  local ended_ms="${10}"
  local status="${11}"
  local result_file="${12}"
  local note="${13}"
  local result_field="$result_file"
  local telemetry_field="$active_telemetry"

  result_field="${result_field#"$repo_root/"}"
  telemetry_field="${telemetry_field#"$repo_root/"}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_id" "$profile" "$workload" "$input_len" "$output_len" \
    "$concurrency" "$requests" "$started_iso" "$ended_iso" \
    "$started_ms" "$ended_ms" "$status" "$result_field" \
    "$telemetry_field" "$note" >>"$manifest_file"
}

result_is_complete() {
  local path="$1"
  local expected="$2"
  [[ -s "$path" ]] || return 1
  "$SGLANG_VENV/bin/python" - "$path" "$expected" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
with open(path) as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
raise SystemExit(0 if rows and rows[-1].get("completed") == expected else 1)
PY
}

run_fixed_case() {
  local profile="$1"
  local mode="$2"
  local input_len="$3"
  local output_len="$4"
  local concurrency="$5"
  local requests="$6"
  local label="${7:-official}"
  local tag="$run_id-$profile-$label"
  local name
  local result_file
  local log_file="$logs_dir/case-$profile-$label-$mode-$input_len-$output_len-c$concurrency.log"
  local started_iso started_ms ended_iso ended_ms status

  [[ "$profile" == "$active_profile" ]] || die "profile mismatch: expected $active_profile, got $profile"
  case "$mode" in
    prefill) name="prefill-$input_len-$output_len" ;;
    decode) name="decode-$input_len-$output_len" ;;
    mixed) name="mixed-$input_len-$output_len" ;;
    *) die "unknown fixed workload: $mode" ;;
  esac
  result_file="$results_dir/$tag-$name-c$concurrency.jsonl"

  if result_is_complete "$result_file" "$requests"; then
    echo "reuse complete case: $profile $mode $input_len/$output_len c$concurrency"
    return 0
  fi

  started_iso="$(timestamp_iso)"
  started_ms="$(date +%s%3N)"
  echo "start: $profile $mode $input_len/$output_len c$concurrency n=$requests"

  if RUN_TAG="$tag" WARMUP_REQUESTS=0 \
      DECODE_INPUT_LEN="$input_len" DECODE_OUTPUT_LEN="$output_len" \
      DECODE_PROMPTS="$requests" PREFILL_INPUT_LEN="$input_len" \
      PREFILL_OUTPUT_LEN="$output_len" PREFILL_PROMPTS="$requests" \
      MIXED_INPUT_LEN="$input_len" MIXED_OUTPUT_LEN="$output_len" \
      MIXED_PROMPTS="$requests" MAX_CONCURRENCY="$concurrency" \
      RESULTS_DIR="$results_dir" "$bench_script" "$mode" >"$log_file" 2>&1; then
    status="passed"
  else
    status="failed"
  fi

  ended_ms="$(date +%s%3N)"
  ended_iso="$(timestamp_iso)"
  if [[ "$status" == "passed" ]] && ! result_is_complete "$result_file" "$requests"; then
    status="incomplete"
  fi
  append_manifest "$profile" "$mode" "$input_len" "$output_len" \
    "$concurrency" "$requests" "$started_iso" "$ended_iso" \
    "$started_ms" "$ended_ms" "$status" "$result_file" "$label;flush-cache-required"
  tail -n 18 "$log_file" || true
  [[ "$status" == "passed" ]]
}

run_dataset_case() {
  local profile="$1"
  local dataset_name="$2"
  local dataset_path="$3"
  local output_len="$4"
  local concurrency="$5"
  local requests="$6"
  local tag="$run_id-$profile-real"
  local fixed_suffix="${output_len:+-fixed-$output_len}"
  local result_file="$results_dir/$tag-$dataset_name$fixed_suffix-c$concurrency.jsonl"
  local log_file="$logs_dir/case-$profile-$dataset_name$fixed_suffix-c$concurrency.log"
  local started_iso started_ms ended_iso ended_ms status

  if result_is_complete "$result_file" "$requests"; then
    echo "reuse complete dataset case: $profile $dataset_name$fixed_suffix c$concurrency"
    return 0
  fi

  started_iso="$(timestamp_iso)"
  started_ms="$(date +%s%3N)"
  echo "start: $profile $dataset_name$fixed_suffix c$concurrency n=$requests"
  if RUN_TAG="$tag" WARMUP_REQUESTS=0 DATASET_NAME="$dataset_name" \
      DATASET_PATH="$dataset_path" DATASET_OUTPUT_LEN="$output_len" \
      DATASET_CONTEXT_LEN=24576 DATASET_PROMPTS="$requests" \
      MAX_CONCURRENCY="$concurrency" RESULTS_DIR="$results_dir" \
      "$bench_script" dataset >"$log_file" 2>&1; then
    status="passed"
  else
    status="failed"
  fi
  ended_ms="$(date +%s%3N)"
  ended_iso="$(timestamp_iso)"
  if [[ "$status" == "passed" ]] && ! result_is_complete "$result_file" "$requests"; then
    status="incomplete"
  fi
  append_manifest "$profile" "$dataset_name" "dataset" "${output_len:-natural}" \
    "$concurrency" "$requests" "$started_iso" "$ended_iso" \
    "$started_ms" "$ended_ms" "$status" "$result_file" "flush-cache-required"
  tail -n 18 "$log_file" || true
  [[ "$status" == "passed" ]]
}

record_capacity_skip() {
  local profile="$1"
  local workload="$2"
  local input_len="$3"
  local output_len="$4"
  local concurrency="$5"
  local capacity="${6:-24576}"
  local total_tokens=$((concurrency * (input_len + output_len)))
  local now_iso now_ms

  if awk -F, -v profile="$profile" -v workload="$workload" \
      -v input_len="$input_len" -v output_len="$output_len" \
      -v concurrency="$concurrency" '
        NR > 1 && $2 == profile && $3 == workload && $4 == input_len &&
          $5 == output_len && $6 == concurrency && $12 == "skipped-capacity" {
            found = 1
          }
        END { exit(found ? 0 : 1) }
      ' "$manifest_file"; then
    echo "reuse capacity skip: $profile $workload $input_len/$output_len c$concurrency"
    return 0
  fi

  now_iso="$(timestamp_iso)"
  now_ms="$(date +%s%3N)"
  append_manifest "$profile" "$workload" "$input_len" "$output_len" \
    "$concurrency" 0 "$now_iso" "$now_iso" "$now_ms" "$now_ms" \
    "skipped-capacity" "" \
    "required-total-tokens-$total_tokens-not-supported-by-capacity-$capacity"
}

run_single_profile() {
  local profile="$1"
  local input_len output_len

  for input_len in 512 2048 4096 8192 12288 20480; do
    run_fixed_case "$profile" prefill "$input_len" 1 1 3 || true
  done

  for input_len in 1024 4096 8192 20480; do
    for output_len in 1024 4096 8192; do
      if (( input_len + output_len >= 24576 )); then
        record_capacity_skip "$profile" decode "$input_len" "$output_len" 1
      else
        run_fixed_case "$profile" decode "$input_len" "$output_len" 1 3 || true
      fi
    done
  done

  if [[ "${RUN_REAL_DATASETS:-0}" == "1" ]]; then
    run_dataset_case "$profile" sharegpt "$sharegpt_path" "" 1 30 || true
    run_dataset_case "$profile" longbench_v2 "$longbench_path" "" 1 20 || true
    run_dataset_case "$profile" longbench_v2 "$longbench_path" 512 1 20 || true
  fi
}

run_c4_profile() {
  local profile="$1"
  local input_len output_len

  for input_len in 2048 4096; do
    for output_len in 512 1024 2048; do
      if (( 4 * (input_len + output_len) > 20480 )); then
        record_capacity_skip "$profile" mixed "$input_len" "$output_len" 4 20480
      else
        run_fixed_case "$profile" mixed "$input_len" "$output_len" 4 20 || true
      fi
    done
  done

  if [[ "${RUN_REAL_DATASETS:-0}" == "1" ]]; then
    run_dataset_case "$profile" sharegpt "$sharegpt_path" "" 4 20 || true
  fi
}

[[ $EUID -eq 0 ]] || die "run as root; the serving profile manages vm.overcommit_memory"
command -v curl >/dev/null || die "curl is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
: "${SGLANG_VENV:=/mnt/ss32t/SGLang/venv}"
export SGLANG_VENV

current_power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n 1)"
if [[ "$current_power_limit" != "$expected_power_limit" && \
      "$current_power_limit" != "$expected_power_limit.00" ]]; then
  die "expected ${expected_power_limit}W power limit, found ${current_power_limit}W"
fi
[[ -z "$(swapon --noheadings --show=NAME)" ]] || die "swap must be disabled for this run"
compute_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ -z "$compute_apps" ]] || die "another GPU compute process is active: $compute_apps"
curl -fsS "$base_url/model_info" >/dev/null 2>&1 && \
  die "an SGLang server is already running at $base_url"

mkdir -p "$results_dir" "$telemetry_dir" "$logs_dir" "$repo_root/environment"
if [[ ! -f "$manifest_file" ]]; then
  echo "run_id,profile,workload,input_tokens,output_tokens,max_concurrency,requests,start_iso,end_iso,start_epoch_ms,end_epoch_ms,status,result_file,telemetry_file,note" >"$manifest_file"
fi
if [[ ! -f "$repo_root/environment/$run_id.txt" ]]; then
  "$script_dir/collect-environment.sh" >"$repo_root/environment/$run_id.txt"
fi
if [[ "${RUN_REAL_DATASETS:-0}" == "1" ]]; then
  [[ -f "$sharegpt_path" ]] || die "SHAREGPT_PATH must name a local file"
  [[ -f "$longbench_path" ]] || die "LONGBENCH_PATH must name a local file"
fi

if [[ "${ONLY_DATASET_C4:-0}" != "1" ]]; then
  start_profile mtp || die "MTP single-request profile failed to start"
  validate_profile_capacity 24576 1 || die "MTP single-request capacity validation failed"
  run_single_profile mtp
  stop_profile
  wait_for_vram_release

  start_profile baseline || die "baseline single-request profile failed to start"
  validate_profile_capacity 24576 1 || die "baseline single-request capacity validation failed"
  run_single_profile baseline
  stop_profile
  wait_for_vram_release
fi

if start_profile mtp-c4; then
  validate_profile_capacity 20480 4 || die "MTP c4 capacity validation failed"
  run_c4_profile mtp-c4
  stop_profile
  wait_for_vram_release
else
  echo "MTP c4 profile could not start; c4 tests will be recorded as unavailable" >&2
fi

if start_profile baseline-c4; then
  validate_profile_capacity 20480 4 || die "baseline c4 capacity validation failed"
  run_c4_profile baseline-c4
  stop_profile
  wait_for_vram_release
else
  echo "baseline c4 profile could not start" >&2
fi

echo "Benchmark suite completed."
echo "Run ID: $run_id"
echo "Results: $results_dir"
echo "Telemetry: $telemetry_dir"
echo "Manifest: $manifest_file"
