#!/usr/bin/env bash

set -euo pipefail

output_file="${1:?usage: monitor-gpu.sh OUTPUT.csv}"
interval_ms="${GPU_SAMPLE_INTERVAL_MS:-200}"

command -v nvidia-smi >/dev/null || {
  echo "error: nvidia-smi is required" >&2
  exit 1
}
mkdir -p "$(dirname "$output_file")"

exec nvidia-smi \
  --query-gpu=timestamp,power.draw,power.limit,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,pstate \
  --format=csv,noheader,nounits \
  -lms "$interval_ms" \
  --filename="$output_file"
