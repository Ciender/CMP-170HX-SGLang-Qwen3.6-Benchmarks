#!/usr/bin/env python3

"""Build a capacity-aware fixed-length benchmark plan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCHEMA_VERSION = "1.0"

CORE_PREFILL = [512, 2048, 4096, 8192, 12288, 20480]
EXTENDED_PREFILL = [32768, 49152, 65536, 98304, 131072]
CORE_GENERATION = [
    (input_tokens, output_tokens)
    for input_tokens in (1024, 4096, 8192, 20480)
    for output_tokens in (1024, 4096, 8192)
]
EXTENDED_GENERATION = [
    (input_tokens, output_tokens)
    for input_tokens in (32768, 49152, 65536, 98304, 131072)
    for output_tokens in (1024, 4096, 8192)
]
CORE_CONCURRENT = [
    (input_tokens, output_tokens)
    for input_tokens in (2048, 4096)
    for output_tokens in (512, 1024, 2048)
]
EXTENDED_CONCURRENT = [
    (input_tokens, output_tokens)
    for input_tokens in (8192, 16384, 32768)
    for output_tokens in (512, 1024, 2048)
]

FIELDS = [
    "schema_version",
    "profile",
    "mtp_state",
    "tier",
    "workload",
    "input_tokens",
    "output_tokens",
    "max_concurrency",
    "requests",
    "required_total_tokens",
    "context_length",
    "max_total_tokens",
    "status",
    "note",
]


def parse_positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_custom_cases(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    cases = raw.get("cases", raw) if isinstance(raw, dict) else raw
    if not isinstance(cases, list):
        raise ValueError("matrix file must contain a list or a {cases: [...]} object")
    normalized = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"matrix case {index} is not an object")
        normalized.append(
            {
                "tier": str(case.get("tier", "custom")),
                "workload": str(case["workload"]),
                "input_tokens": parse_positive(str(case["input_tokens"])),
                "output_tokens": parse_positive(str(case["output_tokens"])),
                "max_concurrency": parse_positive(str(case.get("max_concurrency", 1))),
                "requests": parse_positive(str(case.get("requests", 3))),
            }
        )
    return normalized


def standard_cases(concurrency: int, tiers: set[str], args: argparse.Namespace) -> list[dict]:
    cases: list[dict] = []
    if concurrency == 1:
        if "core" in tiers:
            for input_tokens in CORE_PREFILL:
                cases.append(
                    dict(tier="core", workload="prefill", input_tokens=input_tokens,
                         output_tokens=1, max_concurrency=1, requests=args.prefill_requests)
                )
            for input_tokens, output_tokens in CORE_GENERATION:
                cases.append(
                    dict(tier="core", workload="generation", input_tokens=input_tokens,
                         output_tokens=output_tokens, max_concurrency=1,
                         requests=args.generation_requests)
                )
        if "extended" in tiers:
            for input_tokens in EXTENDED_PREFILL:
                cases.append(
                    dict(tier="extended", workload="prefill", input_tokens=input_tokens,
                         output_tokens=1, max_concurrency=1, requests=args.prefill_requests)
                )
            for input_tokens, output_tokens in EXTENDED_GENERATION:
                cases.append(
                    dict(tier="extended", workload="generation", input_tokens=input_tokens,
                         output_tokens=output_tokens, max_concurrency=1,
                         requests=args.generation_requests)
                )
    else:
        requests = concurrency * args.concurrent_waves
        if "core" in tiers:
            for input_tokens, output_tokens in CORE_CONCURRENT:
                cases.append(
                    dict(tier="core", workload="concurrent_generation",
                         input_tokens=input_tokens, output_tokens=output_tokens,
                         max_concurrency=concurrency, requests=requests)
                )
        if "extended" in tiers:
            for input_tokens, output_tokens in EXTENDED_CONCURRENT:
                cases.append(
                    dict(tier="extended", workload="concurrent_generation",
                         input_tokens=input_tokens, output_tokens=output_tokens,
                         max_concurrency=concurrency, requests=requests)
                )
    return cases


def evaluate(case: dict, context_length: int, max_total_tokens: int) -> tuple[str, str]:
    per_request = case["input_tokens"] + case["output_tokens"]
    required_total = case["max_concurrency"] * per_request
    if per_request >= context_length:
        return (
            "skipped-context",
            f"per-request-{per_request}-must-be-less-than-context-{context_length}",
        )
    if required_total > max_total_tokens:
        return (
            "skipped-capacity",
            f"required-total-{required_total}-exceeds-token-pool-{max_total_tokens}",
        )
    return "planned", "capacity-validated"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a benchmark matrix from the actual SGLang context and token pool. "
            "GPU memory size alone is intentionally not used as a capacity estimate."
        )
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mtp-state", choices=("enabled", "disabled", "unavailable"), required=True)
    parser.add_argument("--context-length", type=parse_positive, required=True)
    parser.add_argument("--max-total-tokens", type=parse_positive, required=True)
    parser.add_argument("--max-concurrency", type=parse_positive, required=True)
    parser.add_argument("--tiers", default="core,extended")
    parser.add_argument("--prefill-requests", type=parse_positive, default=3)
    parser.add_argument("--generation-requests", type=parse_positive, default=3)
    parser.add_argument("--concurrent-waves", type=parse_positive, default=5)
    parser.add_argument("--matrix-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tiers = {item.strip() for item in args.tiers.split(",") if item.strip()}
    invalid_tiers = tiers - {"core", "extended"}
    if invalid_tiers:
        parser.error(f"unknown tiers: {', '.join(sorted(invalid_tiers))}")
    if args.max_total_tokens > args.context_length * args.max_concurrency:
        # This can be valid for a shared pool, but it is worth preserving rather than rejecting.
        pass

    candidates = standard_cases(args.max_concurrency, tiers, args)
    if args.matrix_file:
        candidates.extend(
            case for case in load_custom_cases(args.matrix_file)
            if case["max_concurrency"] == args.max_concurrency
        )

    rows = []
    seen = set()
    for case in candidates:
        key = (
            case["workload"], case["input_tokens"], case["output_tokens"],
            case["max_concurrency"], case["tier"],
        )
        if key in seen:
            continue
        seen.add(key)
        status, note = evaluate(case, args.context_length, args.max_total_tokens)
        required = case["max_concurrency"] * (case["input_tokens"] + case["output_tokens"])
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "profile": args.profile,
                "mtp_state": args.mtp_state,
                "tier": case["tier"],
                "workload": case["workload"],
                "input_tokens": case["input_tokens"],
                "output_tokens": case["output_tokens"],
                "max_concurrency": case["max_concurrency"],
                "requests": case["requests"] if status == "planned" else 0,
                "required_total_tokens": required,
                "context_length": args.context_length,
                "max_total_tokens": args.max_total_tokens,
                "status": status,
                "note": note,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    counts = {status: sum(row["status"] == status for row in rows) for status in (
        "planned", "skipped-context", "skipped-capacity"
    )}
    print(
        f"profile={args.profile} context={args.context_length} pool={args.max_total_tokens} "
        f"concurrency={args.max_concurrency} planned={counts['planned']} "
        f"skipped_context={counts['skipped-context']} "
        f"skipped_capacity={counts['skipped-capacity']} output={args.output}"
    )


if __name__ == "__main__":
    main()
