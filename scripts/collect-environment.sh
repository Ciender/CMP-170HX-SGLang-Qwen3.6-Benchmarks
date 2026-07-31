#!/usr/bin/env bash

set -euo pipefail

runtime_home="${SGLANG_RUNTIME_HOME:-/mnt/ss32t/SGLang}"
venv_dir="${SGLANG_VENV:-$runtime_home/venv}"
model_path="${MODEL_PATH:-/mnt/ss32t/LLM-models/Qwen3.6-27B-INT8-W8A8}"

echo "collected_at=$(date --iso-8601=seconds)"
echo "kernel=$(uname -srmo)"
echo "os=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | tr -d '\"')"
echo "cpu=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)"
echo "logical_cpus=$(nproc)"
echo "memory=$(free -h | awk '/^Mem:/ {print $2}')"
echo "swap_total=$(free -b | awk '/^Swap:/ {print $2}')"
echo "swap_used=$(free -b | awk '/^Swap:/ {print $3}')"
echo "vm_overcommit_memory=$(sysctl -n vm.overcommit_memory)"

pci_bus="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -n 1)"
nvidia-smi \
  --query-gpu=name,driver_version,memory.total,compute_cap,pci.bus_id,power.limit,clocks.max.sm,clocks.max.memory,vbios_version \
  --format=csv,noheader,nounits
lspci -nn -s "${pci_bus#0000}" || true

"$venv_dir/bin/python" - <<'PY'
import platform

import flashinfer
import sglang
import torch
import triton

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"sglang={sglang.__version__}")
print(f"triton={triton.__version__}")
print(f"flashinfer={getattr(flashinfer, '__version__', 'unknown')}")
PY

if [[ -f "$model_path/config.json" ]]; then
  sha256sum "$model_path/config.json"
fi
if [[ -f "$model_path/generation_config.json" ]]; then
  sha256sum "$model_path/generation_config.json"
fi

if [[ -n "${SGLANG_SOURCE:-}" ]] && git -C "$SGLANG_SOURCE" rev-parse HEAD >/dev/null 2>&1; then
  echo "sglang_source=$SGLANG_SOURCE"
  echo "sglang_source_commit=$(git -C "$SGLANG_SOURCE" rev-parse HEAD)"
  if [[ -n "$(git -C "$SGLANG_SOURCE" status --porcelain)" ]]; then
    echo "sglang_source_dirty=true"
  else
    echo "sglang_source_dirty=false"
  fi
fi
