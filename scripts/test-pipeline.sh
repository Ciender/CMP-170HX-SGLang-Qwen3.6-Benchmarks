#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
fixture_dir="$(mktemp -d)"
trap 'rm -rf "$fixture_dir"' EXIT

assert_counts() {
  local file="$1" expected_planned="$2" expected_context="$3" expected_capacity="$4"
  python3 - "$file" "$expected_planned" "$expected_context" "$expected_capacity" <<'PY'
import csv, sys
from collections import Counter
path, expected_planned, expected_context, expected_capacity = sys.argv[1], *map(int, sys.argv[2:])
rows=list(csv.DictReader(open(path)))
counts=Counter(row["status"] for row in rows)
actual=(counts["planned"], counts["skipped-context"], counts["skipped-capacity"])
expected=(expected_planned, expected_context, expected_capacity)
if actual != expected:
    raise SystemExit(f"{path}: expected {expected}, got {actual}")
for row in rows:
    required=int(row["max_concurrency"]) * (int(row["input_tokens"]) + int(row["output_tokens"]))
    if required != int(row["required_total_tokens"]):
        raise SystemExit(f"bad required tokens in {row}")
PY
}

# Representative live profile capacities. The planner never consumes GPU GiB.
python3 "$script_dir/plan-benchmark.py" --profile 32g-like-c1 --mtp-state disabled \
  --context-length 16384 --max-total-tokens 16384 --max-concurrency 1 \
  --tiers core,extended --output "$fixture_dir/32g-c1.csv"
python3 "$script_dir/plan-benchmark.py" --profile 40g-like-c1 --mtp-state enabled \
  --context-length 24576 --max-total-tokens 24576 --max-concurrency 1 \
  --tiers core,extended --output "$fixture_dir/40g-c1.csv"
python3 "$script_dir/plan-benchmark.py" --profile 64g-like-c1 --mtp-state enabled \
  --context-length 65536 --max-total-tokens 65536 --max-concurrency 1 \
  --tiers core,extended --output "$fixture_dir/64g-c1.csv"
python3 "$script_dir/plan-benchmark.py" --profile 80g-like-c1 --mtp-state enabled \
  --context-length 131072 --max-total-tokens 131072 --max-concurrency 1 \
  --tiers core,extended --output "$fixture_dir/80g-c1.csv"
python3 "$script_dir/plan-benchmark.py" --profile 40g-like-c4 --mtp-state enabled \
  --context-length 24576 --max-total-tokens 20480 --max-concurrency 4 \
  --tiers core,extended --output "$fixture_dir/40g-c4.csv"
python3 "$script_dir/plan-benchmark.py" --profile 80g-like-c4 --mtp-state enabled \
  --context-length 65536 --max-total-tokens 131072 --max-concurrency 4 \
  --tiers core,extended --output "$fixture_dir/80g-c4.csv"
python3 "$script_dir/plan-benchmark.py" --profile custom-c1 --mtp-state disabled \
  --context-length 65536 --max-total-tokens 65536 --max-concurrency 1 \
  --tiers core --matrix-file "$repo_root/configs/matrix.example.json" \
  --output "$fixture_dir/custom-c1.csv"

assert_counts "$fixture_dir/32g-c1.csv" 13 25 0
assert_counts "$fixture_dir/40g-c1.csv" 16 22 0
assert_counts "$fixture_dir/64g-c1.csv" 26 12 0
assert_counts "$fixture_dir/80g-c1.csv" 34 4 0
assert_counts "$fixture_dir/40g-c4.csv" 5 3 7
assert_counts "$fixture_dir/80g-c4.csv" 12 0 3
assert_counts "$fixture_dir/custom-c1.csv" 19 0 0

bundle="$fixture_dir/bundle"
mkdir -p "$bundle/profiles"
cat >"$bundle/metadata.json" <<'JSON'
{
  "schema_version": "1.0",
  "run_id": "fixture-run",
  "collected_at": "2026-08-02T12:00:00+08:00",
  "host": {"memory_bytes": 51539607552, "swap_total_bytes": 0},
  "gpus": [{"name": "Fixture GPU", "memory_total_mib": "40960", "power_limit_w": "250"}],
  "software": {"sglang": "0.5.16", "torch": "fixture", "torch_cuda": "fixture"},
  "model": {
    "name": "Fixture Model", "repository": "owner/model", "revision": "deadbeef",
    "architectures": ["FixtureForCausalLM"], "weight_files_total_bytes": 32212254720,
    "quantization_method": "fixture-int8", "quantization_format": "int-quantized",
    "native_context_length": 131072
  }
}
JSON
cat >"$bundle/profiles/concurrency-1-mtp-disabled.json" <<'JSON'
{
  "profile": "concurrency-1-mtp-disabled", "mtp_state": "disabled",
  "context_length": 24576, "max_total_tokens": 24576,
  "effective_max_running_requests": 1,
  "gpu_runtime": {"gpus": [{"memory_used_mib": 32768}]},
  "startup_memory": {
    "weight_load_phase_deltas_gb_reported": [28.48],
    "kv_caches": [{"k_gb_reported": 0.75, "v_gb_reported": 0.75}],
    "recurrent_state_caches": [{"conv_state_gb_reported": 0.01, "ssm_state_gb_reported": 0.28}],
    "cuda_graphs": [{"kind": "target decode", "memory_gb_reported": 0.06}],
    "memory_pool_free_gb_reported": [8.97]
  }
}
JSON
cat >"$bundle/profiles/concurrency-1-mtp-enabled.json" <<'JSON'
{
  "profile": "concurrency-1-mtp-enabled", "mtp_state": "enabled",
  "context_length": 24576, "max_total_tokens": 24576,
  "effective_max_running_requests": 1,
  "gpu_runtime": {"gpus": [{"memory_used_mib": 38912}]},
  "startup_memory": {
    "weight_load_phase_deltas_gb_reported": [28.48, 5.53],
    "kv_caches": [{"k_gb_reported": 0.75, "v_gb_reported": 0.75}, {"k_gb_reported": 0.05, "v_gb_reported": 0.05}],
    "recurrent_state_caches": [{"conv_state_gb_reported": 0.01, "ssm_state_gb_reported": 0.28, "intermediate_ssm_state_gb_reported": 1.12}],
    "cuda_graphs": [{"kind": "verify", "memory_gb_reported": 0.04}, {"kind": "draft", "memory_gb_reported": 0.09}],
    "memory_pool_free_gb_reported": [2.31, 2.21]
  }
}
JSON
cat >"$bundle/case-manifest.csv" <<'CSV'
run_id,profile,mtp_state,tier,workload,input_tokens,output_tokens,max_concurrency,requests,required_total_tokens,context_length,max_total_tokens,start_iso,end_iso,start_epoch_ms,end_epoch_ms,status,result_file,telemetry_file,note,seed,temperature
fixture-run,concurrency-1-mtp-disabled,disabled,core,generation,1024,1024,1,3,2048,24576,24576,a,b,1,2,passed,,,capacity-validated,42,0.0
fixture-run,concurrency-1-mtp-enabled,enabled,core,generation,1024,1024,1,3,2048,24576,24576,a,b,1,2,passed,,,capacity-validated,42,0.0
fixture-run,concurrency-1-mtp-enabled,enabled,extended,generation,32768,8192,1,0,40960,24576,24576,a,a,1,1,skipped-context,,,per-request-40960-must-be-less-than-context-24576,42,0.0
CSV
cat >"$bundle/serving.csv" <<'CSV'
run_id,profile,mtp_state,tier,workload,input_tokens,output_tokens,max_concurrency,requests,status,required_total_tokens,context_length,max_total_tokens,completed,duration_s,actual_concurrency,total_input_tokens,mean_input_tokens,median_input_tokens,max_input_tokens,total_output_tokens,mean_output_tokens,median_output_tokens,max_output_tokens,input_tok_s,e2e_output_tok_s,steady_decode_tok_s,mean_e2e_latency_ms,median_e2e_latency_ms,p95_e2e_latency_ms,mean_ttft_ms,median_ttft_ms,p95_ttft_ms,mean_tpot_ms,median_tpot_ms,p95_tpot_ms,accept_length,result_file,note
fixture-run,concurrency-1-mtp-disabled,disabled,core,generation,1024,1024,1,3,passed,2048,24576,24576,3,73,1,3072,1024,1024,1024,3072,1024,1024,1024,42,42,42,0,0,0,250,250,250,23.8,23.8,23.8,,,capacity-validated
fixture-run,concurrency-1-mtp-enabled,enabled,core,generation,1024,1024,1,3,passed,2048,24576,24576,3,38,1,3072,1024,1024,1024,3072,1024,1024,1024,80,80,82,0,0,0,270,270,270,12.2,12.2,12.2,3.0,,capacity-validated
fixture-run,concurrency-1-mtp-enabled,enabled,extended,generation,32768,8192,1,0,skipped-context,40960,24576,24576,,,,,,,,,,,,,,,,,,,,,,,,,,per-request-40960-must-be-less-than-context-24576
CSV
cat >"$bundle/power.csv" <<'CSV'
run_id,profile,mtp_state,tier,workload,input_tokens,output_tokens,max_concurrency,status,samples_in_window,active_samples,active_mean_power_w,active_median_power_w,active_p95_power_w,active_mean_gpu_util_pct,active_mean_sm_clock_mhz,active_max_temperature_c,min_power_limit_w,max_power_limit_w,telemetry_file
fixture-run,concurrency-1-mtp-disabled,disabled,core,generation,1024,1024,1,passed,100,90,240,241,250,99,1400,80,250,250,
fixture-run,concurrency-1-mtp-enabled,enabled,core,generation,1024,1024,1,passed,100,90,235,236,250,95,1390,82,250,250,
CSV

python3 "$script_dir/generate-report.py" --bundle "$bundle" \
  --output "$bundle/report.md" --pdf "$bundle/report.pdf"
grep -q '^# SGLang Benchmark Report: fixture-run' "$bundle/report.md"
grep -q '1.90x' "$bundle/report.md"
test "$(head -c 5 "$bundle/report.pdf")" = '%PDF-'
grep -a -q '/Type /Pages' "$bundle/report.pdf"
if command -v pdfinfo >/dev/null; then
  pdfinfo "$bundle/report.pdf" | awk '/^Pages:/ { if ($2 < 1) exit 1; found=1 } END { exit(found ? 0 : 1) }'
fi
if command -v pdftotext >/dev/null; then
  pdftotext "$bundle/report.pdf" - | grep -q 'SGLang Benchmark Report: fixture-run'
fi

echo "pipeline fixtures passed"
