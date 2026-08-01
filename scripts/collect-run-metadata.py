#!/usr/bin/env python3

"""Collect portable host, GPU, software, checkpoint, and model metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path


DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "float": 4,
    "bfloat16": 2,
    "float16": 2,
    "half": 2,
    "fp8": 1,
    "int8": 1,
}


def command(args: list[str], default="") -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def text_config(config: dict) -> dict:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def model_metadata(model_path: Path, hash_weights: bool) -> dict:
    config_path = model_path / "config.json"
    config = read_json(config_path)
    model = text_config(config)
    layer_types = model.get("layer_types") or []
    full_attention_layers = sum(layer == "full_attention" for layer in layer_types)
    total_layers = model.get("num_hidden_layers")
    if not full_attention_layers and total_layers and not layer_types:
        full_attention_layers = int(total_layers)
    kv_heads = model.get("num_key_value_heads")
    head_dim = model.get("head_dim")
    if not head_dim and model.get("hidden_size") and model.get("num_attention_heads"):
        head_dim = int(model["hidden_size"]) // int(model["num_attention_heads"])
    kv_dtype = model.get("kv_cache_dtype") or model.get("dtype") or config.get("dtype")
    dtype_bytes = DTYPE_BYTES.get(str(kv_dtype).lower())
    kv_bytes_per_token = None
    if full_attention_layers and kv_heads and head_dim and dtype_bytes:
        kv_bytes_per_token = full_attention_layers * kv_heads * head_dim * dtype_bytes * 2

    weight_files = []
    for path in sorted(model_path.glob("*.safetensors")):
        item = {"name": path.name, "bytes": path.stat().st_size}
        if hash_weights:
            item["sha256"] = sha256(path)
        weight_files.append(item)
    critical_files = []
    for name in (
        "config.json", "generation_config.json", "recipe.yaml", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja",
    ):
        path = model_path / name
        if path.exists():
            critical_files.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})

    quantization = config.get("quantization_config") or {}
    return {
        "path": str(model_path.resolve()),
        "name": os.environ.get("MODEL_NAME") or model_path.name,
        "repository": os.environ.get("MODEL_REPOSITORY", ""),
        "revision": os.environ.get("MODEL_REVISION", ""),
        "base_model_repository": os.environ.get("BASE_MODEL_REPOSITORY", ""),
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type"),
        "dtype": config.get("dtype") or model.get("dtype"),
        "quantization_method": quantization.get("quant_method"),
        "quantization_format": quantization.get("format"),
        "native_context_length": model.get("max_position_embeddings"),
        "num_hidden_layers": total_layers,
        "full_attention_layers": full_attention_layers or None,
        "linear_or_other_layers": (
            int(total_layers) - full_attention_layers
            if total_layers is not None and full_attention_layers is not None else None
        ),
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "kv_dtype": kv_dtype,
        "calculated_target_kv_bytes_per_token": kv_bytes_per_token,
        "mtp_hidden_layers": model.get("mtp_num_hidden_layers"),
        "mtp_use_dedicated_embeddings": model.get("mtp_use_dedicated_embeddings"),
        "weight_files": weight_files,
        "weight_files_total_bytes": sum(item["bytes"] for item in weight_files),
        "critical_files": critical_files,
    }


def gpu_metadata() -> list[dict]:
    query = (
        "index,name,uuid,pci.bus_id,driver_version,memory.total,compute_cap,power.limit,"
        "clocks.max.sm,clocks.max.memory,vbios_version"
    )
    output = command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    fields = [
        "index", "name", "uuid", "pci_bus_id", "driver_version", "memory_total_mib",
        "compute_capability", "power_limit_w", "max_sm_clock_mhz", "max_memory_clock_mhz",
        "vbios_version",
    ]
    rows = []
    for line in output.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != len(fields):
            continue
        rows.append(dict(zip(fields, values)))
    return rows


def software_metadata(python_path: Path) -> dict:
    script = """
import json, platform
result = {"python": platform.python_version()}
for name in ("torch", "sglang", "triton", "flashinfer"):
    try:
        module = __import__(name)
        result[name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        result[name] = f"unavailable:{type(exc).__name__}"
try:
    import torch
    result["torch_cuda"] = torch.version.cuda
except Exception:
    pass
print(json.dumps(result))
"""
    output = command([str(python_path), "-c", script], "{}")
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"python_command": str(python_path), "error": output}


def git_metadata(path: Path | None) -> dict:
    if not path or not (path / ".git").exists():
        return {}
    tracked_dirty = subprocess.run(
        ["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0
    untracked = command(
        ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"]
    ).splitlines()
    return {
        "path": str(path.resolve()),
        "commit": command(["git", "-C", str(path), "rev-parse", "HEAD"]),
        "describe": command(["git", "-C", str(path), "describe", "--always", "--dirty"]),
        "dirty": tracked_dirty,
        "untracked_files": untracked,
        "remote": command(["git", "-C", str(path), "remote", "get-url", "origin"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect standardized benchmark metadata")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sglang-source", type=Path)
    parser.add_argument("--benchmark-source", type=Path)
    parser.add_argument("--python", type=Path, default=Path("python3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()

    datasets = []
    for item in args.dataset:
        name, separator, raw_path = item.partition("=")
        if not separator:
            parser.error(f"invalid --dataset {item!r}; expected NAME=PATH")
        path = Path(raw_path)
        datasets.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    memory = command(["free", "-b"])
    swap_total = None
    swap_used = None
    for line in memory.splitlines():
        if line.startswith("Swap:"):
            parts = line.split()
            swap_total, swap_used = int(parts[1]), int(parts[2])
    payload = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "tester": os.environ.get("TESTER", ""),
        "data_author": os.environ.get("DATA_AUTHOR", ""),
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": {
            "hostname": platform.node(),
            "kernel": platform.platform(),
            "os_release": command(["sh", "-c", ". /etc/os-release; echo $PRETTY_NAME"]),
            "cpu": command(["sh", "-c", "lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1"]),
            "logical_cpus": os.cpu_count(),
            "memory_bytes": int(command(["sh", "-c", "free -b | awk '/^Mem:/ {print $2}'"], "0")),
            "swap_total_bytes": swap_total,
            "swap_used_bytes": swap_used,
        },
        "gpus": gpu_metadata(),
        "software": software_metadata(args.python),
        "sglang_source": git_metadata(args.sglang_source),
        "benchmark_source": git_metadata(args.benchmark_source),
        "model": model_metadata(args.model_path, args.hash_weights),
        "datasets": datasets,
        "run_configuration": {
            key: os.environ.get(key, "")
            for key in (
                "CONTEXT_LENGTH", "MAX_TOTAL_TOKENS", "MEM_FRACTION_STATIC",
                "MAX_CONCURRENCY_LIST", "EXPECTED_POWER_LIMIT", "TP_SIZE",
                "MTP_MODES", "MTP_ALGORITHM", "MTP_STEPS", "MTP_TOPK",
                "MTP_DRAFT_TOKENS", "MATRIX_TIERS", "PREFILL_REQUESTS",
                "GENERATION_REQUESTS", "CONCURRENT_WAVES", "QUANTIZATION",
                "CUDA_VISIBLE_DEVICES",
                "ATTENTION_BACKEND", "LINEAR_ATTN_BACKEND", "SAMPLING_BACKEND",
                "REASONING_PARSER", "TOOL_CALL_PARSER", "TRUST_REMOTE_CODE",
                "EXTRA_SERVER_ARGS",
                "ALLOW_DIRTY_SGLANG",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"metadata={args.output}")


if __name__ == "__main__":
    main()
