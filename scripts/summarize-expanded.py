#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path


def read_last_json(path: Path) -> dict:
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"no JSON records in {path}")
    return rows[-1]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * quantile)]


def parse_telemetry(path: Path) -> list[dict]:
    rows = []
    local_zone = datetime.now().astimezone().tzinfo
    with path.open(newline="") as handle:
        for raw in csv.reader(handle, skipinitialspace=True):
            if len(raw) != 8:
                continue
            try:
                timestamp = datetime.strptime(raw[0], "%Y/%m/%d %H:%M:%S.%f")
                timestamp = timestamp.replace(tzinfo=local_zone)
                rows.append(
                    {
                        "epoch_ms": timestamp.timestamp() * 1000,
                        "power_w": float(raw[1]),
                        "power_limit_w": float(raw[2]),
                        "util_pct": float(raw[3]),
                        "sm_mhz": float(raw[4]),
                        "mem_mhz": float(raw[5]),
                        "temperature_c": float(raw[6]),
                    }
                )
            except ValueError:
                continue
    return rows


def fmt(value, digits: int = 3):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def summarize_lengths(record: dict | None, key: str) -> tuple:
    values = record.get(key, []) if record else []
    if not values:
        return None, None, None
    return statistics.mean(values), statistics.median(values), max(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build serving and per-case power CSVs from an expanded benchmark run"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--serving-output", type=Path, required=True)
    parser.add_argument("--power-output", type=Path, required=True)
    parser.add_argument("--include-preflight", action="store_true")
    parser.add_argument("--active-util-threshold", type=float, default=90.0)
    args = parser.parse_args()

    repo_root = (args.repo_root or args.manifest.parent.parent).resolve()
    with args.manifest.open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not args.include_preflight:
        manifest_rows = [row for row in manifest_rows if not row["note"].startswith("preflight;")]

    serving_fields = [
        "run_id",
        "profile",
        "mtp_state",
        "tier",
        "workload",
        "input_tokens",
        "output_tokens",
        "max_concurrency",
        "requests",
        "status",
        "required_total_tokens",
        "context_length",
        "max_total_tokens",
        "completed",
        "duration_s",
        "actual_concurrency",
        "total_input_tokens",
        "mean_input_tokens",
        "median_input_tokens",
        "max_input_tokens",
        "total_output_tokens",
        "mean_output_tokens",
        "median_output_tokens",
        "max_output_tokens",
        "input_tok_s",
        "e2e_output_tok_s",
        "steady_decode_tok_s",
        "mean_e2e_latency_ms",
        "median_e2e_latency_ms",
        "p95_e2e_latency_ms",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p95_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p95_tpot_ms",
        "accept_length",
        "result_file",
        "note",
    ]
    args.serving_output.parent.mkdir(parents=True, exist_ok=True)
    with args.serving_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=serving_fields, lineterminator="\n")
        writer.writeheader()
        for row in manifest_rows:
            record = None
            result_path = row["result_file"]
            if row["status"] == "passed" and result_path:
                record = read_last_json(repo_root / result_path)
            mean_tpot = record.get("mean_tpot_ms") if record else None
            steady_decode = 1000 / mean_tpot if mean_tpot and mean_tpot > 0 else None
            mean_input, median_input, max_input = summarize_lengths(record, "input_lens")
            mean_output, median_output, max_output = summarize_lengths(record, "output_lens")
            writer.writerow(
                {
                    "run_id": row["run_id"],
                    "profile": row["profile"],
                    "mtp_state": row.get("mtp_state", ""),
                    "tier": row.get("tier", ""),
                    "workload": row["workload"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "max_concurrency": row["max_concurrency"],
                    "requests": row["requests"],
                    "status": row["status"],
                    "required_total_tokens": row.get("required_total_tokens", ""),
                    "context_length": row.get("context_length", ""),
                    "max_total_tokens": row.get("max_total_tokens", ""),
                    "completed": record.get("completed", "") if record else "",
                    "duration_s": fmt(record.get("duration") if record else None),
                    "actual_concurrency": fmt(record.get("concurrency") if record else None),
                    "total_input_tokens": record.get("total_input_tokens", "") if record else "",
                    "mean_input_tokens": fmt(mean_input),
                    "median_input_tokens": fmt(median_input),
                    "max_input_tokens": fmt(max_input),
                    "total_output_tokens": record.get("total_output_tokens", "") if record else "",
                    "mean_output_tokens": fmt(mean_output),
                    "median_output_tokens": fmt(median_output),
                    "max_output_tokens": fmt(max_output),
                    "input_tok_s": fmt(record.get("input_throughput") if record else None),
                    "e2e_output_tok_s": fmt(record.get("output_throughput") if record else None),
                    "steady_decode_tok_s": fmt(steady_decode),
                    "mean_e2e_latency_ms": fmt(record.get("mean_e2e_latency_ms") if record else None),
                    "median_e2e_latency_ms": fmt(record.get("median_e2e_latency_ms") if record else None),
                    "p95_e2e_latency_ms": fmt(record.get("p95_e2e_latency_ms") if record else None),
                    "mean_ttft_ms": fmt(record.get("mean_ttft_ms") if record else None),
                    "median_ttft_ms": fmt(record.get("median_ttft_ms") if record else None),
                    "p95_ttft_ms": fmt(record.get("p95_ttft_ms") if record else None),
                    "mean_tpot_ms": fmt(mean_tpot),
                    "median_tpot_ms": fmt(record.get("median_tpot_ms") if record else None),
                    "p95_tpot_ms": fmt(record.get("p95_tpot_ms") if record else None),
                    "accept_length": fmt(record.get("accept_length") if record else None),
                    "result_file": result_path,
                    "note": row["note"],
                }
            )

    telemetry_cache: dict[Path, list[dict]] = {}
    power_fields = [
        "run_id",
        "profile",
        "mtp_state",
        "tier",
        "workload",
        "input_tokens",
        "output_tokens",
        "max_concurrency",
        "status",
        "samples_in_window",
        "active_samples",
        "active_mean_power_w",
        "active_median_power_w",
        "active_p95_power_w",
        "active_mean_gpu_util_pct",
        "active_mean_sm_clock_mhz",
        "active_max_temperature_c",
        "min_power_limit_w",
        "max_power_limit_w",
        "telemetry_file",
    ]
    args.power_output.parent.mkdir(parents=True, exist_ok=True)
    with args.power_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=power_fields, lineterminator="\n")
        writer.writeheader()
        for row in manifest_rows:
            telemetry_name = row["telemetry_file"]
            window = []
            active = []
            if row["status"] == "passed" and telemetry_name:
                telemetry_path = repo_root / telemetry_name
                if telemetry_path not in telemetry_cache:
                    telemetry_cache[telemetry_path] = parse_telemetry(telemetry_path)
                start_ms = int(row["start_epoch_ms"])
                end_ms = int(row["end_epoch_ms"])
                window = [
                    sample
                    for sample in telemetry_cache[telemetry_path]
                    if start_ms <= sample["epoch_ms"] <= end_ms
                ]
                active = [
                    sample
                    for sample in window
                    if sample["util_pct"] >= args.active_util_threshold
                ]
            powers = [sample["power_w"] for sample in active]
            writer.writerow(
                {
                    "run_id": row["run_id"],
                    "profile": row["profile"],
                    "mtp_state": row.get("mtp_state", ""),
                    "tier": row.get("tier", ""),
                    "workload": row["workload"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "max_concurrency": row["max_concurrency"],
                    "status": row["status"],
                    "samples_in_window": len(window),
                    "active_samples": len(active),
                    "active_mean_power_w": fmt(statistics.mean(powers) if powers else None, 2),
                    "active_median_power_w": fmt(percentile(powers, 0.5) if powers else None, 2),
                    "active_p95_power_w": fmt(percentile(powers, 0.95) if powers else None, 2),
                    "active_mean_gpu_util_pct": fmt(
                        statistics.mean(sample["util_pct"] for sample in active) if active else None,
                        1,
                    ),
                    "active_mean_sm_clock_mhz": fmt(
                        statistics.mean(sample["sm_mhz"] for sample in active) if active else None,
                        0,
                    ),
                    "active_max_temperature_c": fmt(
                        max((sample["temperature_c"] for sample in active), default=None), 0
                    ),
                    "min_power_limit_w": fmt(
                        min((sample["power_limit_w"] for sample in window), default=None), 2
                    ),
                    "max_power_limit_w": fmt(
                        max((sample["power_limit_w"] for sample in window), default=None), 2
                    ),
                    "telemetry_file": telemetry_name,
                }
            )


if __name__ == "__main__":
    main()
