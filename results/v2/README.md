# v2 run — recorded outputs

Committed snapshot of the evaluation outputs for the v2 run. These are **records of a specific run**, not live artifacts — the pipeline regenerates all of them under `artifacts/eval/` (gitignored; archived in S3).

| Provenance | Value |
|---|---|
| Producing code | commit `6105b73` (training itself ran at `851b0e2`–`0832ac9` across stages) |
| Dataset hash | `c2eaf668...` (3,749 examples, v2 labels, 100% grounding) |
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Hardware | Lambda A100-SXM4 (40 GB train / 80 GB final eval), driver 580.105.08 |
| Run dates | 2026-07-15 → 2026-07-16 |

## Files

- **`metrics.json`** — the machine-readable CI contract: all 7 model/artifact rows, raw + constrained metrics, per-metric pass/fail against thresholds, the n=500 regression report, and the overall `ci_pass` verdict. This is the file a GitOps pipeline gates on.
- **`report.md`** — generated evaluation report.
- **`model_card.md`** — generated model card (training data, hyperparameters, per-artifact eval, licensing).
- **`qualitative_samples.json`** — 20 real model outputs spanning easy / hard / null / failure-mode categories, including the misses. Field F1 is 0.885, not 1.0 — these show what the remaining 0.115 looks like.

Headline: base 0.000 → fine-tuned 0.885 field F1; quantization metrics-neutral through GGUF Q4_K_M; null accuracy 1.000 on every artifact. Full context in the [main README](../../README.md#results-v2).
