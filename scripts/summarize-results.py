#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


FIELDS = (
    ("output_throughput", "out tok/s", ".2f"),
    ("input_throughput", "in tok/s", ".2f"),
    ("mean_ttft_ms", "TTFT ms", ".2f"),
    ("mean_tpot_ms", "TPOT ms", ".2f"),
    ("accept_length", "accept", ".2f"),
)


def load_last_record(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"no JSONL records in {path}")
    return records[-1]


def render(value, fmt: str) -> str:
    if value is None:
        return "-"
    return format(value, fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SGLang benchmark JSONL files")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    headers = ["result", "requests", "concurrency"] + [label for _, label, _ in FIELDS]
    rows = []
    for path in args.files:
        record = load_last_record(path)
        row = [
            path.stem,
            str(record.get("completed", "-")),
            render(record.get("concurrency"), ".2f"),
        ]
        row.extend(render(record.get(key), fmt) for key, _, fmt in FIELDS)
        rows.append(row)

    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


if __name__ == "__main__":
    main()
