#!/usr/bin/env python3

"""Generate standardized Markdown and dependency-free PDF benchmark reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_number(value, digits=2, suffix="") -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def gib(value) -> str:
    if value in (None, ""):
        return "-"
    return f"{int(value) / (1024 ** 3):.2f} GiB"


def markdown_table(headers: list[str], rows: list[list]) -> list[str]:
    rendered = [[str(value) if value not in (None, "") else "-" for value in row] for row in rows]
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(row) + " |" for row in rendered)
    return output


def profile_rows(profile_dir: Path) -> tuple[list[list], list[list]]:
    capacity = []
    memory = []
    for path in sorted(profile_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("profile_status") == "unavailable":
            capacity.append([data.get("profile"), data.get("mtp_state"), "-", "-", "-", "unavailable"])
            memory.append([data.get("profile"), "-", "-", "-", "-", "-", "-", "-"])
            continue
        gpu_rows = data.get("gpu_runtime", {}).get("gpus", [])
        gpu_used = sum(float(row.get("memory_used_mib", 0)) for row in gpu_rows)
        capacity.append(
            [
                data.get("profile"), data.get("mtp_state"), data.get("context_length"),
                data.get("max_total_tokens"), data.get("effective_max_running_requests"),
                fmt_number(gpu_used / 1024, 2, " GiB"),
            ]
        )
        startup = data.get("startup_memory", {})
        kv = startup.get("kv_caches", [])
        target_kv = (kv[0].get("k_gb_reported", 0) + kv[0].get("v_gb_reported", 0)) if kv else None
        draft_kv = (kv[1].get("k_gb_reported", 0) + kv[1].get("v_gb_reported", 0)) if len(kv) > 1 else None
        load_deltas = startup.get("weight_load_phase_deltas_gb_reported", [])
        recurrent = startup.get("recurrent_state_caches", [])
        recurrent_total = None
        if recurrent:
            recurrent_total = sum(
                float(value or 0)
                for row in recurrent
                for key, value in row.items()
                if key.endswith("_gb_reported")
            )
        graphs = sum(float(item.get("memory_gb_reported", 0)) for item in startup.get("cuda_graphs", []))
        memory.append(
            [
                data.get("profile"),
                fmt_number(load_deltas[0] if load_deltas else None, 2),
                fmt_number(load_deltas[1] if len(load_deltas) > 1 else None, 2),
                fmt_number(target_kv, 2), fmt_number(draft_kv, 2),
                fmt_number(recurrent_total, 2), fmt_number(graphs, 2),
                fmt_number(min(startup.get("memory_pool_free_gb_reported", []), default=None), 2),
            ]
        )
    return capacity, memory


def result_rows(serving: list[dict], power: list[dict], tier: str | None = None) -> list[list]:
    power_map = {
        (
            row.get("profile"), row.get("tier"), row.get("workload"),
            row.get("input_tokens"), row.get("output_tokens"), row.get("max_concurrency"),
        ): row
        for row in power
    }
    rows = []
    for row in serving:
        if tier is not None and row.get("tier") != tier:
            continue
        key = (
            row.get("profile"), row.get("tier"), row.get("workload"),
            row.get("input_tokens"), row.get("output_tokens"), row.get("max_concurrency"),
        )
        power_row = power_map.get(key, {})
        rows.append(
            [
                row.get("tier"), row.get("workload"), row.get("input_tokens"),
                row.get("output_tokens"), row.get("max_concurrency"), row.get("mtp_state"),
                row.get("status"), fmt_number(row.get("input_tok_s"), 2),
                fmt_number(row.get("e2e_output_tok_s"), 2),
                fmt_number(row.get("median_ttft_ms"), 2), fmt_number(row.get("mean_tpot_ms"), 2),
                fmt_number(row.get("accept_length"), 2),
                fmt_number(power_row.get("active_mean_power_w"), 2),
            ]
        )
    return rows


def speedup_rows(serving: list[dict]) -> list[list]:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in serving:
        key = (
            row.get("tier"), row.get("workload"), row.get("input_tokens"),
            row.get("output_tokens"), row.get("max_concurrency"),
        )
        grouped.setdefault(key, {})[row.get("mtp_state", "")] = row
    output = []
    for key, states in sorted(grouped.items()):
        enabled = states.get("enabled")
        disabled = states.get("disabled")
        if not enabled or not disabled:
            continue
        if enabled.get("status") != "passed" or disabled.get("status") != "passed":
            continue
        try:
            enabled_rate = float(enabled["e2e_output_tok_s"])
            disabled_rate = float(disabled["e2e_output_tok_s"])
            speedup = enabled_rate / disabled_rate
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        output.append([*key, fmt_number(disabled_rate, 2), fmt_number(enabled_rate, 2), fmt_number(speedup, 2, "x")])
    return output


def build_report(bundle: Path) -> str:
    metadata = json.loads((bundle / "metadata.json").read_text())
    manifest = read_csv(bundle / "case-manifest.csv")
    serving = read_csv(bundle / "serving.csv")
    power = read_csv(bundle / "power.csv")
    model = metadata.get("model", {})
    host = metadata.get("host", {})
    software = metadata.get("software", {})
    gpus = metadata.get("gpus") or [{}]
    selected_gpu = metadata.get("run_configuration", {}).get("CUDA_VISIBLE_DEVICES", "0")
    gpu = next(
        (
            item for item in gpus
            if str(item.get("index")) == selected_gpu or item.get("uuid") == selected_gpu
        ),
        gpus[0],
    )
    benchmark_source = metadata.get("benchmark_source", {})
    sglang_source = metadata.get("sglang_source", {})
    datasets = metadata.get("datasets", [])
    run_configuration = metadata.get("run_configuration", {})
    capacity, memory = profile_rows(bundle / "profiles")
    statuses = Counter(row.get("status") for row in manifest)
    partial = bool(statuses.get("failed") or statuses.get("incomplete"))
    title_suffix = " (Partial Run)" if partial else ""

    lines = [
        f"# SGLang Benchmark Report: {metadata.get('run_id', bundle.name)}{title_suffix}",
        "",
        *( ["This is a partial run because failed or incomplete cases are present.", ""] if partial else [] ),
        "## Run summary",
        "",
        *markdown_table(
            ["Field", "Value"],
            [
                ["Run ID", metadata.get("run_id")],
                ["Tester / data author", f"{metadata.get('tester') or '-'} / {metadata.get('data_author') or '-'}"],
                ["Collected at", metadata.get("collected_at")],
                ["Benchmark commit", benchmark_source.get("describe") or benchmark_source.get("commit") or "not recorded"],
                ["SGLang source", sglang_source.get("describe") or sglang_source.get("commit") or software.get("sglang")],
                ["SGLang dirty / untracked", f"{sglang_source.get('dirty', '-')} / {len(sglang_source.get('untracked_files', []))}"],
                ["Model", model.get("name")],
                ["Model repository", model.get("repository") or "local checkpoint"],
                ["Model revision", model.get("revision") or "not recorded"],
                ["Architecture", ", ".join(model.get("architectures", [])) or model.get("model_type")],
                ["Weight files", gib(model.get("weight_files_total_bytes"))],
                ["Quantization", " / ".join(filter(None, [model.get("quantization_method"), model.get("quantization_format")])) or "not declared"],
                ["Native context", model.get("native_context_length")],
                ["Layers (total / full attention)", f"{model.get('num_hidden_layers', '-')} / {model.get('full_attention_layers', '-')}"],
                ["KV heads / head dimension", f"{model.get('num_key_value_heads', '-')} / {model.get('head_dim', '-')}"],
                ["KV dtype", model.get("kv_dtype")],
                ["Calculated target KV", f"{model.get('calculated_target_kv_bytes_per_token', '-')} bytes/token"],
                ["GPU", gpu.get("name")],
                ["GPU memory", fmt_number(gpu.get("memory_total_mib"), 0, " MiB")],
                ["Compute capability / VBIOS", f"{gpu.get('compute_capability', '-')} / {gpu.get('vbios_version', '-')}"],
                ["Driver", gpu.get("driver_version")],
                ["Power limit", fmt_number(gpu.get("power_limit_w"), 2, " W")],
                ["Maximum SM / memory clock", f"{gpu.get('max_sm_clock_mhz', '-')} / {gpu.get('max_memory_clock_mhz', '-')} MHz"],
                ["Host OS / kernel", f"{host.get('os_release', '-')} / {host.get('kernel', '-')}"],
                ["CPU / logical CPUs", f"{host.get('cpu', '-')} / {host.get('logical_cpus', '-')}"],
                ["SGLang", software.get("sglang")],
                ["PyTorch / CUDA", f"{software.get('torch', '-')} / {software.get('torch_cuda', '-')}"],
                ["Host memory / swap", f"{gib(host.get('memory_bytes'))} / {gib(host.get('swap_total_bytes'))}"],
            ],
        ),
        "",
        "## Weight files",
        "",
        *markdown_table(
            ["File", "Bytes", "GiB"],
            [
                [item.get("name"), item.get("bytes"), fmt_number(item.get("bytes", 0) / (1024 ** 3), 2)]
                for item in model.get("weight_files", [])
            ],
        ),
        "",
        "## Run configuration",
        "",
        *markdown_table(
            ["Parameter", "Value"],
            [[key, value] for key, value in run_configuration.items() if value not in (None, "")],
        ),
        "",
        "## Datasets",
        "",
        *markdown_table(
            ["Dataset", "Bytes", "SHA-256"],
            [[item.get("name"), item.get("bytes"), item.get("sha256")] for item in datasets],
        ),
        "",
        "## Capacity and runtime memory",
        "",
        "Capacity is taken from the live SGLang server. It is not inferred from GPU memory size.",
        "",
        *markdown_table(
            ["Profile", "MTP state", "Context", "Token pool", "Max requests", "GPU used after startup"],
            capacity,
        ),
        "",
        *markdown_table(
            ["Profile", "Target load (log GB)", "Draft load (log GB)", "Target KV (log GB)", "Draft KV (log GB)", "Recurrent state (log GB)", "CUDA Graph (log GB)", "Pool-end free (log GB)"],
            memory,
        ),
        "",
        "## Completion",
        "",
        *markdown_table(["Status", "Cases"], [[key, value] for key, value in sorted(statuses.items())]),
        "",
        "Capacity and context skips are expected results, not zero-throughput failures.",
        "",
        "## MTP comparison",
        "",
        "Speedup is MTP-enabled end-to-end output throughput divided by MTP-disabled throughput.",
        "",
        *markdown_table(
            ["Tier", "Workload", "Input", "Output", "Max concurrency", "MTP disabled tok/s", "MTP enabled tok/s", "Speedup"],
            speedup_rows(serving),
        ),
        "",
        "## Core fixed-length results",
        "",
        *markdown_table(
            ["Tier", "Workload", "Input", "Output", "Max concurrency", "MTP", "Status", "Input tok/s", "Output tok/s", "Median TTFT ms", "Mean TPOT ms", "Accept", "Mean power W"],
            result_rows(serving, power, "core"),
        ),
        "",
        "## Extended fixed-length results",
        "",
        *markdown_table(
            ["Tier", "Workload", "Input", "Output", "Max concurrency", "MTP", "Status", "Input tok/s", "Output tok/s", "Median TTFT ms", "Mean TPOT ms", "Accept", "Mean power W"],
            result_rows(serving, power, "extended"),
        ),
        "",
        "## Custom and dataset results",
        "",
        *markdown_table(
            ["Tier", "Workload", "Input", "Output", "Max concurrency", "MTP", "Status", "Input tok/s", "Output tok/s", "Median TTFT ms", "Mean TPOT ms", "Accept", "Mean power W"],
            [row for row in result_rows(serving, power) if row[0] not in ("core", "extended")],
        ),
        "",
        "## Metric definitions",
        "",
        "- `input tok/s`: completed prompt tokens divided by the complete benchmark request window.",
        "- `output tok/s`: completed generated tokens divided by that same wall-clock window.",
        "- `TTFT`: request latency to the first generated token; lower is better.",
        "- `TPOT`: per-request mean time per output token after the first token; lower is better.",
        "- At concurrency greater than one, output tok/s is aggregate server throughput; `1000 / TPOT` is not a replacement.",
    ]
    return "\n".join(lines) + "\n"


def pdf_escape(value: str) -> str:
    ascii_value = value.encode("ascii", "replace").decode("ascii")
    return ascii_value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def report_lines(markdown: str, width=145) -> list[str]:
    output = []
    for raw in markdown.splitlines():
        raw = raw.replace("**", "").replace("`", "")
        if not raw:
            output.append("")
            continue
        output.extend(textwrap.wrap(raw, width=width, replace_whitespace=False, drop_whitespace=False) or [""])
    return output


def write_pdf(markdown: str, path: Path) -> None:
    lines = report_lines(markdown)
    lines_per_page = 72
    pages = [lines[index:index + lines_per_page] for index in range(0, len(lines), lines_per_page)] or [[]]
    objects: list[bytes] = []

    def add(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    catalog_id = add(b"")
    pages_id = add(b"")
    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    page_ids = []
    for page_number, page_lines in enumerate(pages, start=1):
        commands = ["BT", "/F1 6 Tf", "24 570 Td", "7.4 TL"]
        for line in page_lines:
            commands.append(f"({pdf_escape(line.rstrip())}) Tj")
            commands.append("T*")
        commands.extend(["ET", "BT", "/F1 6 Tf", "760 15 Td", f"(Page {page_number}/{len(pages)}) Tj", "ET"])
        stream = "\n".join(commands).encode("ascii")
        content_id = add(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        page_id = add(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 842 595] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        page_ids.append(page_id)
    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{object_id} 0 obj\n".encode())
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark Markdown and PDF")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    report = build_report(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    if args.pdf:
        write_pdf(report, args.pdf)
    print(f"markdown={args.output}" + (f" pdf={args.pdf}" if args.pdf else ""))


if __name__ == "__main__":
    main()
