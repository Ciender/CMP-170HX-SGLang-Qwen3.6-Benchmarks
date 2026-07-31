# Result data

The canonical public dataset is `expanded-250w-v2-20260731`:

```text
case-manifest-expanded-250w-v2-20260731.csv  case status, exact time windows,
                                             raw/telemetry paths, dataset hashes
serving-expanded-250w.csv                    serving metrics for all 56 rows
power-expanded-250w.csv                      per-case active-load telemetry
telemetry/expanded-250w-v2-20260731/         raw 200 ms NVML samples
```

The serving and power CSVs are generated with:

```bash
scripts/summarize-expanded.py \
  results/case-manifest-expanded-250w-v2-20260731.csv \
  --repo-root . \
  --serving-output results/serving-expanded-250w.csv \
  --power-output results/power-expanded-250w.csv
```

`input_tok_s` is total completed prompt tokens divided by the benchmark
wall-clock request window. `e2e_output_tok_s` is total completed generated
tokens divided by that wall-clock window. `steady_decode_tok_s` is
`1000 / mean_tpot_ms`; it is a steady decode-rate estimate for
maximum-concurrency-one results and a per-request TPOT inverse for concurrent
runs, not aggregate server throughput.

Power fields prefixed with `active_` use only samples within the manifest case
window whose GPU utilization is at least 90%. `samples_in_window` and
`active_samples` make this filter auditable. The two sub-second 512-token
prefill cases have no active samples and intentionally contain empty active
power fields.

Raw telemetry has no header and uses this column order:

```text
timestamp,power.draw,power.limit,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,pstate
```

Files named only `serving-250w.csv`, `power-250w.csv`, or
`telemetry/power-*-250w.csv` are the earlier short-output run retained for
history. They are not used by the top-level v2 READMEs.

Full SGLang JSONL and server logs are gitignored because they contain generated
text, large per-token latency arrays, absolute local paths, and complete server
configuration dumps. A fresh run recreates them under `results/raw/<run-id>/`
and `results/logs/<run-id>/`.
