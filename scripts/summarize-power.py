#!/usr/bin/env python3

import argparse
import csv
import statistics
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * quantile)]


def load_active(path: Path, threshold: float) -> list[tuple[float, float, float, float]]:
    samples = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            if len(row) != 8:
                continue
            try:
                power = float(row[1])
                utilization = float(row[3])
                sm_clock = float(row[4])
                temperature = float(row[6])
            except ValueError:
                continue
            if utilization >= threshold:
                samples.append((power, utilization, sm_clock, temperature))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize 200 ms NVML power telemetry")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--util-threshold", type=float, default=90.0)
    args = parser.parse_args()

    print("file,samples,mean_power_w,median_power_w,p95_power_w,mean_util_pct,mean_sm_mhz,max_temp_c")
    for path in args.files:
        rows = load_active(path, args.util_threshold)
        if not rows:
            raise ValueError(f"no active samples in {path}")
        powers = [row[0] for row in rows]
        utilizations = [row[1] for row in rows]
        clocks = [row[2] for row in rows]
        temperatures = [row[3] for row in rows]
        print(
            f"{path.name},{len(rows)},{statistics.mean(powers):.2f},"
            f"{percentile(powers, 0.5):.2f},{percentile(powers, 0.95):.2f},"
            f"{statistics.mean(utilizations):.1f},{statistics.mean(clocks):.0f},"
            f"{max(temperatures):.0f}"
        )


if __name__ == "__main__":
    main()
