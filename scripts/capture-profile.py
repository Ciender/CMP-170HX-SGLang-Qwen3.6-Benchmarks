#!/usr/bin/env python3

"""Capture actual SGLang capacity, GPU residency, and startup memory records."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def effective_requests(server_info: dict):
    direct = server_info.get("effective_max_running_requests_per_dp")
    if direct is not None:
        return direct
    for state in server_info.get("internal_states", []):
        if isinstance(state, dict) and "effective_max_running_requests_per_dp" in state:
            return state["effective_max_running_requests_per_dp"]
    return None


def gpu_memory(device: str | None) -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,power.limit",
        "--format=csv,noheader,nounits",
    ]
    if device:
        command[1:1] = ["-i", device]
    output = subprocess.check_output(command, text=True).strip().splitlines()
    rows = []
    for line in output:
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": float(parts[2]),
                "memory_used_mib": float(parts[3]),
                "memory_free_mib": float(parts[4]),
                "power_limit_w": float(parts[5]),
            }
        )
    return {"gpus": rows}


def parse_startup_log(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    text = path.read_text(errors="replace")
    load_deltas = [float(value) for value in re.findall(r"Load weight end\..*?mem usage=([0-9.]+) GB", text)]
    kv_matches = [
        {
            "dtype": dtype,
            "tokens": int(tokens),
                "k_gb_reported": float(k_size),
                "v_gb_reported": float(v_size),
        }
        for dtype, tokens, k_size, v_size in re.findall(
            r"KV Cache is allocated\. dtype: ([^,]+), #tokens: ([0-9]+), "
            r"K size: ([0-9.]+) GB, V size: ([0-9.]+) GB",
            text,
        )
    ]
    mamba = []
    for match in re.finditer(
        r"Mamba Cache is allocated\..*?conv_state size: ([0-9.]+)GB, "
        r"ssm_state size: ([0-9.]+)GB(?: intermediate_ssm_state_cache size: ([0-9.]+)GB "
        r"intermediate_conv_window_cache size: ([0-9.]+)GB)?",
        text,
    ):
        mamba.append(
            {
                "conv_state_gb_reported": float(match.group(1)),
                "ssm_state_gb_reported": float(match.group(2)),
                "intermediate_ssm_state_gb_reported": float(match.group(3)) if match.group(3) else None,
                "intermediate_conv_state_gb_reported": float(match.group(4)) if match.group(4) else None,
            }
        )
    pool_free = [float(value) for value in re.findall(r"Memory pool end\. avail mem=([0-9.]+) GB", text)]
    graph_deltas = [
        {"kind": kind.strip(), "memory_gb_reported": float(value)}
        for kind, value in re.findall(r"Capture (.*?) CUDA graph end\..*?mem usage=([0-9.]+) GB", text)
    ]
    return {
        "weight_load_phase_deltas_gb_reported": load_deltas,
        "kv_caches": kv_matches,
        "recurrent_state_caches": mamba,
        "memory_pool_free_gb_reported": pool_free,
        "cuda_graphs": graph_deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a live SGLang serving profile")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mtp-state", choices=("enabled", "disabled", "unavailable"), required=True)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--configured-context-length", type=int)
    parser.add_argument("--configured-max-running-requests", type=int)
    parser.add_argument("--gpu-device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    server_info = get_json(f"{args.base_url}/server_info")
    try:
        model_info = get_json(f"{args.base_url}/model_info")
    except Exception:
        model_info = {}
    payload = {
        "schema_version": "1.0",
        "profile_status": "available",
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile": args.profile,
        "mtp_state": args.mtp_state,
        "context_length": server_info.get("context_len") or server_info.get("context_length") or args.configured_context_length,
        "max_total_tokens": server_info.get("max_total_num_tokens"),
        "effective_max_running_requests": effective_requests(server_info) or args.configured_max_running_requests,
        "model_info": model_info,
        "gpu_runtime": gpu_memory(args.gpu_device),
        "startup_memory": parse_startup_log(args.server_log),
        "server_info": server_info,
    }
    if payload["context_length"] is None:
        internal = server_info.get("internal_states", [])
        for state in internal:
            if isinstance(state, dict) and state.get("context_len") is not None:
                payload["context_length"] = state["context_len"]
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(
        f"profile={args.profile} context={payload['context_length']} "
        f"pool={payload['max_total_tokens']} "
        f"running_requests={payload['effective_max_running_requests']} output={args.output}"
    )


if __name__ == "__main__":
    main()
