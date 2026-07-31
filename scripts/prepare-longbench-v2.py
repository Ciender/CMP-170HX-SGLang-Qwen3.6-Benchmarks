#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from transformers import AutoTokenizer


REQUIRED_FIELDS = (
    "context",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
)


def format_prompt(example: dict) -> str:
    return (
        f"{example['context']}\n\n"
        f"Question: {example['question']}\n"
        f"A. {example['choice_A']}\n"
        f"B. {example['choice_B']}\n"
        f"C. {example['choice_C']}\n"
        f"D. {example['choice_D']}\n"
        "Answer:"
    )


def read_api_rows(paths: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for wrapper in payload.get("rows", []):
            row = wrapper.get("row", {})
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"{path}: row missing required fields: {missing}")
            identity = row.get("_id", wrapper.get("row_idx"))
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a context-safe LongBench-v2 JSONL sample"
    )
    parser.add_argument("api_json", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-context", type=int, default=24576)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source_rows = read_api_rows(args.api_json)
    candidates = []
    for row in source_rows:
        prompt_tokens = len(tokenizer(format_prompt(row)).input_ids)
        if prompt_tokens + args.output_tokens <= args.max_context:
            candidates.append((row, prompt_tokens))

    if len(candidates) < args.num_samples:
        raise SystemExit(
            f"only {len(candidates)} rows fit, but {args.num_samples} are required"
        )

    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.num_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with args.output.open("w", encoding="utf-8") as handle:
        for row, _ in selected:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))

    lengths = [prompt_tokens for _, prompt_tokens in selected]
    print(f"source_rows={len(source_rows)}")
    print(f"eligible_rows={len(candidates)}")
    print(f"selected_rows={len(selected)}")
    print(f"prompt_tokens_min={min(lengths)}")
    print(f"prompt_tokens_median={statistics.median(lengths):.1f}")
    print(f"prompt_tokens_max={max(lengths)}")
    print(f"fixed_output_tokens={args.output_tokens}")
    print(f"output_sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()
