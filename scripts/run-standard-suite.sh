#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
config_file="${1:-}"

die() {
  echo "error: $*" >&2
  exit 1
}

[[ -n "$config_file" && -f "$config_file" ]] || \
  die "usage: scripts/run-standard-suite.sh /path/to/benchmark.env"

set -a
# shellcheck disable=SC1090
source "$config_file"
set +a

: "${RUN_ID:?RUN_ID is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${SGLANG_VENV:?SGLANG_VENV is required}"
: "${CONTEXT_LENGTH:?CONTEXT_LENGTH is required}"

base_url="${BASE_URL:-http://127.0.0.1:8000}"
bundle_dir="${BUNDLE_DIR:-$repo_root/runs/$RUN_ID}"
raw_dir="${RESULTS_DIR:-$repo_root/results/raw/$RUN_ID}"
logs_dir="${LOGS_DIR:-$repo_root/results/logs/$RUN_ID}"
telemetry_dir="${TELEMETRY_DIR:-$repo_root/results/telemetry/$RUN_ID}"
manifest_file="$bundle_dir/case-manifest.csv"
server_pid=""
monitor_pid=""
active_telemetry=""
gpu_device="${CUDA_VISIBLE_DEVICES:-0}"

timestamp_iso() { date '+%Y-%m-%dT%H:%M:%S.%3N%:z'; }

stop_server() {
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
trap stop_server EXIT
trap 'exit 130' INT TERM

wait_for_vram_release() {
  local used
  for _ in $(seq 1 90); do
    used="$(nvidia-smi -i "$gpu_device" --query-gpu=memory.used --format=csv,noheader,nounits)"
    if awk 'NF && $1 != 0 { busy=1 } END { exit busy }' <<<"$used"; then
      return 0
    fi
    sleep 1
  done
  die "GPU memory was not released after server shutdown"
}

start_server() {
  local profile="$1"
  local mtp_state="$2"
  local concurrency="$3"
  local log_file="$logs_dir/server-$profile.log"
  local ready=0

  active_telemetry="$telemetry_dir/power-$profile.csv"
  local mtp_key="MTP_${mtp_state^^}"
  local concurrency_key="C${concurrency}_${mtp_key}"
  local mem_specific="MEM_FRACTION_STATIC_${concurrency_key}"
  local mem_mtp="MEM_FRACTION_STATIC_${mtp_key}"
  local tokens_specific="MAX_TOTAL_TOKENS_${concurrency_key}"
  local tokens_mtp="MAX_TOTAL_TOKENS_${mtp_key}"
  local profile_mem_fraction="${!mem_specific:-${!mem_mtp:-${MEM_FRACTION_STATIC:-0.90}}}"
  local profile_max_tokens="${!tokens_specific:-${!tokens_mtp:-${MAX_TOTAL_TOKENS:-auto}}}"

  MTP_STATE="$mtp_state" MAX_RUNNING_REQUESTS="$concurrency" \
    MEM_FRACTION_STATIC="$profile_mem_fraction" MAX_TOTAL_TOKENS="$profile_max_tokens" \
    CUDA_GRAPH_MAX_BS_DECODE="$concurrency" \
    "$script_dir/serve-model.sh" >"$log_file" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 "${STARTUP_TIMEOUT:-900}"); do
    if curl -fsS "$base_url/model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      wait "$server_pid" 2>/dev/null || true
      server_pid=""
      tail -n 120 "$log_file" >&2
      return 1
    fi
    sleep 1
  done
  [[ "$ready" == "1" ]] || return 1
  GPU_DEVICE="$gpu_device" "$script_dir/monitor-gpu.sh" "$active_telemetry" &
  monitor_pid=$!
  sleep 1
}

append_manifest() {
  local profile="$1" mtp_state="$2" tier="$3" workload="$4"
  local input_tokens="$5" output_tokens="$6" concurrency="$7" requests="$8"
  local required_total="$9" context_length="${10}" token_pool="${11}"
  local start_iso="${12}" end_iso="${13}" start_ms="${14}" end_ms="${15}"
  local status="${16}" result_file="${17}" note="${18}"
  result_file="${result_file#"$repo_root/"}"
  local telemetry_file="${active_telemetry#"$repo_root/"}"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$RUN_ID" "$profile" "$mtp_state" "$tier" "$workload" "$input_tokens" \
    "$output_tokens" "$concurrency" "$requests" "$required_total" "$context_length" \
    "$token_pool" "$start_iso" "$end_iso" "$start_ms" "$end_ms" "$status" \
    "$result_file" "$telemetry_file" "$note" "${SEED:-42}" "${TEMPERATURE:-0.0}" \
    >>"$manifest_file"
}

result_is_complete() {
  local path="$1" expected="$2"
  [[ -s "$path" ]] || return 1
  "$SGLANG_VENV/bin/python" - "$path" "$expected" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
with open(path) as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
raise SystemExit(0 if rows and rows[-1].get("completed") == expected else 1)
PY
}

manifest_case_exists() {
  local profile="$1" tier="$2" workload="$3" input_tokens="$4"
  local output_tokens="$5" concurrency="$6"
  awk -F, -v profile="$profile" -v tier="$tier" -v workload="$workload" \
      -v input_tokens="$input_tokens" -v output_tokens="$output_tokens" \
      -v concurrency="$concurrency" '
    NR > 1 && $2 == profile && $4 == tier && $5 == workload &&
      $6 == input_tokens && $7 == output_tokens && $8 == concurrency {
        found = 1
      }
    END { exit(found ? 0 : 1) }
  ' "$manifest_file"
}

execute_plan() {
  local profile="$1" mtp_state="$2" plan_file="$3"
  while IFS=, read -r schema _plan_profile _plan_mtp tier workload input_tokens \
      output_tokens concurrency requests required_total context_length token_pool status note; do
    [[ "$schema" == "schema_version" ]] && continue
    local now_iso now_ms mode tag name result_file log_file start_iso end_iso start_ms end_ms final_status
    if manifest_case_exists "$profile" "$tier" "$workload" "$input_tokens" \
        "$output_tokens" "$concurrency"; then
      echo "reuse manifest terminal row: $profile $tier $workload $input_tokens/$output_tokens concurrency=$concurrency"
      continue
    fi
    if [[ "$status" != "planned" ]]; then
      now_iso="$(timestamp_iso)"
      now_ms="$(date +%s%3N)"
      append_manifest "$profile" "$mtp_state" "$tier" "$workload" "$input_tokens" \
        "$output_tokens" "$concurrency" 0 "$required_total" "$context_length" \
        "$token_pool" "$now_iso" "$now_iso" "$now_ms" "$now_ms" "$status" "" "$note"
      continue
    fi
    case "$workload" in
      prefill) mode="prefill"; name="prefill-$input_tokens-$output_tokens" ;;
      generation) mode="decode"; name="decode-$input_tokens-$output_tokens" ;;
      concurrent_generation) mode="mixed"; name="mixed-$input_tokens-$output_tokens" ;;
      *) die "unsupported planned workload: $workload" ;;
    esac
    tag="$RUN_ID-$profile-$tier"
    result_file="$raw_dir/$tag-$name-c$concurrency.jsonl"
    log_file="$logs_dir/case-$profile-$tier-$name-c$concurrency.log"
    if result_is_complete "$result_file" "$requests"; then
      die "orphan result without manifest timing: $result_file; use a new RUN_ID"
    fi
    start_iso="$(timestamp_iso)"
    start_ms="$(date +%s%3N)"
    echo "run: $profile $tier $workload $input_tokens/$output_tokens concurrency=$concurrency requests=$requests"
    if RUN_TAG="$tag" WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}" \
        PREFILL_INPUT_LEN="$input_tokens" PREFILL_OUTPUT_LEN="$output_tokens" PREFILL_PROMPTS="$requests" \
        DECODE_INPUT_LEN="$input_tokens" DECODE_OUTPUT_LEN="$output_tokens" DECODE_PROMPTS="$requests" \
        MIXED_INPUT_LEN="$input_tokens" MIXED_OUTPUT_LEN="$output_tokens" MIXED_PROMPTS="$requests" \
        MAX_CONCURRENCY="$concurrency" RESULTS_DIR="$raw_dir" \
        "$script_dir/bench-serving.sh" "$mode" >"$log_file" 2>&1; then
      final_status="passed"
    else
      final_status="failed"
    fi
    end_ms="$(date +%s%3N)"
    end_iso="$(timestamp_iso)"
    if [[ "$final_status" == "passed" ]] && ! result_is_complete "$result_file" "$requests"; then
      final_status="incomplete"
    fi
    append_manifest "$profile" "$mtp_state" "$tier" "$workload" "$input_tokens" \
      "$output_tokens" "$concurrency" "$requests" "$required_total" "$context_length" \
      "$token_pool" "$start_iso" "$end_iso" "$start_ms" "$end_ms" "$final_status" \
      "$result_file" "capacity-validated;flush-cache-required"
  done <"$plan_file"
}

command -v curl >/dev/null || die "curl is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
[[ -x "$SGLANG_VENV/bin/python" ]] || die "invalid SGLANG_VENV=$SGLANG_VENV"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "RUN_ID contains unsafe characters"
[[ "${TP_SIZE:-1}" == "1" ]] || die "standard pipeline v1 supports TP_SIZE=1 only"
[[ "$gpu_device" != *,* ]] || die "standard pipeline v1 requires one CUDA_VISIBLE_DEVICES entry"
[[ -z "$(swapon --noheadings --show=NAME)" ]] || die "swap must be disabled"
[[ -z "$(nvidia-smi -i "$gpu_device" --query-compute-apps=pid --format=csv,noheader)" ]] || \
  die "another GPU compute process is active"
if [[ -n "${SGLANG_SOURCE:-}" && -d "$SGLANG_SOURCE/.git" && \
      -n "$(git -C "$SGLANG_SOURCE" status --porcelain)" && \
      "${ALLOW_DIRTY_SGLANG:-0}" != "1" ]]; then
  die "SGLANG_SOURCE is dirty; commit it or set ALLOW_DIRTY_SGLANG=1"
fi
if [[ -n "${EXPECTED_POWER_LIMIT:-}" ]]; then
  while IFS= read -r actual_power; do
    [[ "$actual_power" == "$EXPECTED_POWER_LIMIT" || "$actual_power" == "$EXPECTED_POWER_LIMIT.00" ]] || \
      die "expected ${EXPECTED_POWER_LIMIT}W power limit, found ${actual_power}W"
  done < <(nvidia-smi -i "$gpu_device" --query-gpu=power.limit --format=csv,noheader,nounits)
fi
if [[ -e "$manifest_file" && "${RESUME:-0}" != "1" ]]; then
  die "$manifest_file already exists; choose a new RUN_ID or set RESUME=1"
fi

mkdir -p "$bundle_dir/profiles" "$bundle_dir/plans" "$raw_dir" "$logs_dir" "$telemetry_dir"
if [[ ! -f "$manifest_file" ]]; then
  echo "run_id,profile,mtp_state,tier,workload,input_tokens,output_tokens,max_concurrency,requests,required_total_tokens,context_length,max_total_tokens,start_iso,end_iso,start_epoch_ms,end_epoch_ms,status,result_file,telemetry_file,note,seed,temperature" >"$manifest_file"
fi

metadata_args=(
  --run-id "$RUN_ID"
  --model-path "$MODEL_PATH"
  --benchmark-source "$repo_root"
  --python "$SGLANG_VENV/bin/python"
  --output "$bundle_dir/metadata.json"
)
if [[ -n "${SGLANG_SOURCE:-}" ]]; then metadata_args+=(--sglang-source "$SGLANG_SOURCE"); fi
if [[ -n "${SHAREGPT_PATH:-}" && -f "${SHAREGPT_PATH:-}" ]]; then metadata_args+=(--dataset "sharegpt=$SHAREGPT_PATH"); fi
if [[ -n "${LONGBENCH_PATH:-}" && -f "${LONGBENCH_PATH:-}" ]]; then metadata_args+=(--dataset "longbench_v2=$LONGBENCH_PATH"); fi
if [[ "${HASH_WEIGHTS:-0}" == "1" ]]; then metadata_args+=(--hash-weights); fi
"$script_dir/collect-run-metadata.py" "${metadata_args[@]}"

mkdir -p "$bundle_dir/source-diffs"
if ! git -C "$repo_root" diff --quiet HEAD --; then
  git -C "$repo_root" diff HEAD --binary >"$bundle_dir/source-diffs/benchmark.diff"
fi
if [[ -n "${SGLANG_SOURCE:-}" && -d "$SGLANG_SOURCE/.git" ]] && \
    ! git -C "$SGLANG_SOURCE" diff --quiet HEAD --; then
  git -C "$SGLANG_SOURCE" diff HEAD --binary >"$bundle_dir/source-diffs/sglang.diff"
fi
if [[ -n "${SGLANG_SOURCE:-}" && -d "$SGLANG_SOURCE/.git" && \
      -n "$(git -C "$SGLANG_SOURCE" status --porcelain)" ]]; then
  git -C "$SGLANG_SOURCE" status --short >"$bundle_dir/source-diffs/sglang-status.txt"
fi

for concurrency in ${MAX_CONCURRENCY_LIST:-1 4}; do
  for mtp_state in ${MTP_MODES:-disabled}; do
    profile="concurrency-$concurrency-mtp-$mtp_state"
    echo "start profile: $profile"
    if ! start_server "$profile" "$mtp_state" "$concurrency"; then
      echo "profile unavailable: $profile" >&2
      "$SGLANG_VENV/bin/python" - "$profile" "$mtp_state" "$bundle_dir/profiles/$profile.json" <<'PY'
import json, sys
profile, mtp_state, output = sys.argv[1:]
with open(output, "w") as handle:
    json.dump({
        "schema_version": "1.0", "profile_status": "unavailable",
        "profile": profile, "mtp_state": mtp_state,
        "reason": "server-startup-failed; inspect the profile server log",
    }, handle, indent=2)
    handle.write("\n")
PY
      stop_server
      wait_for_vram_release
      continue
    fi
    profile_json="$bundle_dir/profiles/$profile.json"
    capture_args=(
      --base-url "$base_url" --profile "$profile" --mtp-state "$mtp_state"
      --server-log "$logs_dir/server-$profile.log"
      --gpu-device "$gpu_device"
      --configured-max-running-requests "$concurrency" --output "$profile_json"
    )
    if [[ "$CONTEXT_LENGTH" != "auto" ]]; then
      capture_args+=(--configured-context-length "$CONTEXT_LENGTH")
    fi
    "$script_dir/capture-profile.py" "${capture_args[@]}"
    read -r actual_context actual_pool actual_concurrency < <(
      "$SGLANG_VENV/bin/python" - "$profile_json" "$concurrency" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(d.get("context_length"), d.get("max_total_tokens"), d.get("effective_max_running_requests") or sys.argv[2])
PY
    )
    [[ "$actual_context" != "None" ]] || die "server did not report context length for $profile"
    [[ "$actual_pool" != "None" ]] || die "server did not report max_total_num_tokens for $profile"
    if (( actual_concurrency < concurrency )); then
      die "profile $profile exposes only concurrency $actual_concurrency; requested $concurrency"
    fi
    plan_file="$bundle_dir/plans/$profile.csv"
    tiers_value="${MATRIX_TIERS:-core extended}"
    plan_args=(
      --profile "$profile" --mtp-state "$mtp_state"
      --context-length "$actual_context" --max-total-tokens "$actual_pool"
      --max-concurrency "$concurrency"
      --tiers "${tiers_value// /,}"
      --prefill-requests "${PREFILL_REQUESTS:-3}"
      --generation-requests "${GENERATION_REQUESTS:-3}"
      --concurrent-waves "${CONCURRENT_WAVES:-5}"
      --output "$plan_file"
    )
    if [[ -n "${MATRIX_FILE:-}" ]]; then plan_args+=(--matrix-file "$MATRIX_FILE"); fi
    "$script_dir/plan-benchmark.py" "${plan_args[@]}"
    execute_plan "$profile" "$mtp_state" "$plan_file"
    stop_server
    wait_for_vram_release
  done
done

"$script_dir/summarize-expanded.py" "$manifest_file" --repo-root "$repo_root" \
  --serving-output "$bundle_dir/serving.csv" --power-output "$bundle_dir/power.csv"
"$script_dir/generate-report.py" --bundle "$bundle_dir" --output "$bundle_dir/report.md" \
  --pdf "$bundle_dir/report.pdf"

echo "standard benchmark complete: $bundle_dir"
