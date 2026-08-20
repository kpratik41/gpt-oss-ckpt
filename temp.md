# sft_rft3 — BIRD dev temperature-0 results

Run: `outputs/sft_rft3/gemma4_31b_sftckpt56_dapo12_lr2e6_beta_0p005_0p001_0`
Data: `outputs/old-dev-schema-tool-unpatched.jsonl` · async vLLM · tp=2 · 4 shards · ctx 43k

| ckpt | EX | correct |
| ---: | ---: | ---: |
| 5 | 72.69% | 1115 / 1534 |
| **10** | **73.14%** | **1122 / 1534** |
| 15 | 72.82% | 1117 / 1534 |
| 20 | 72.69% | 1115 / 1534 |
| 25 | 72.36% | 1110 / 1534 |
| 30 | 71.71% | 1100 / 1534 |
| 35 | 71.64% | 1099 / 1534 |
| 40 | 71.64% | 1099 / 1534 |
| 45 | 71.58% | 1098 / 1534 |
| 50 | 71.71% | 1100 / 1534 |
| 55 | 71.38% | 1095 / 1534 |
| 60 | 72.36% | 1110 / 1534 |
| 65 | 72.10% | 1106 / 1534 |
| 70 | 72.43% | 1111 / 1534 |

Notes:

- Peak is **checkpoint-10 at 73.14%**; the run declines to 71.38% by step 55 and
  partially recovers over 60-70 without regaining the peak.
- ckpt-60 used `shards1` rather than `shards4`; sharding does not affect accuracy.
- Rows 5-60 are read from each run's `eval_summary.md`. Rows 65 and 70 were
  supplied directly and have no summary file on disk.
- For comparison, `outputs/gemma-best-rl` scores **73.53%** on the same data, so
  nothing in this run beat the existing best gemma checkpoint.
