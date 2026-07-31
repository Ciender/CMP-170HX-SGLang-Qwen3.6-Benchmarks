# Result data

`serving-250w.csv` is the machine-readable serving result table used by both
top-level READMEs. `power-250w.csv` was generated from the four files under
`telemetry/` by selecting samples with GPU utilization greater than or equal to
90 percent.

The telemetry files have no header and use this column order:

```text
timestamp,power.draw,power.limit,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,pstate
```

Regenerate the power summary with:

```bash
scripts/summarize-power.py results/telemetry/*.csv
```

Full SGLang JSONL output is intentionally gitignored because it contains large
per-token latency arrays, generated text, absolute local paths, and a complete
server configuration dump. A fresh run writes those records to `results/raw/`.
