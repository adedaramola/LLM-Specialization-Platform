# scripts/

Entry points for each pipeline phase. Run in order — each phase produces
artifacts that the next phase consumes.

## Phase order

```
prepare_dataset.py        Phase 0a — build, split, decontaminate, hash dataset
tokenizer_audit.py        Phase 0b — verify tokenizer handles task syntax
train_sft.py              Phase 1  — QLoRA supervised fine-tuning
generate_preferences.py   Phase 2a — generate DPO preference pairs from SFT model
train_dpo.py              Phase 2b — direct preference optimization
evaluate.py               Phase 3  — eval harness, metrics.json, qualitative samples
export.py                 Phase 4  — adapter → merged BF16 → GGUF + re-verify
```

`tokenizer_audit.py` can be run independently at any time — it only reads the
tokenizer, it does not modify any artifacts.

## Usage

Every script takes `--config` as its only required argument. All configs
inherit from `base_config.yaml` via the `defaults` key — see `configs/README.md`.

```bash
# Full pipeline (also available as: make train && make evaluate && make export)
python scripts/prepare_dataset.py   --config configs/sft_config.yaml --raw-data data/raw/news_extraction.jsonl
python scripts/tokenizer_audit.py   --config configs/sft_config.yaml
python scripts/train_sft.py         --config configs/sft_config.yaml
python scripts/generate_preferences.py --config configs/dpo_config.yaml
python scripts/train_dpo.py         --config configs/dpo_config.yaml
python scripts/evaluate.py          --config configs/eval_config.yaml --mode all
python scripts/export.py            --config configs/dpo_config.yaml
```

## Script reference

### `prepare_dataset.py`
Loads raw JSONL, splits 80/10/10, decontaminates via hash and n-gram overlap
against the frozen test set, validates null-case fraction, content-hashes each
split, and writes `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.

The test set is frozen after this step — never regenerate it mid-experiment.

**Required:** `--raw-data PATH` — path to the source JSONL file.

### `tokenizer_audit.py`
Verifies task-critical characters (`{`, `}`, `"`, `:`, `\n`, etc.) are
represented as single tokens, checks for byte-level fallbacks, and tests
roundtrip fidelity on structured probes. Writes the audit report to
`artifacts/tokenizer_audit.json`.

Tokenizer drift between training and serving silently destroys structured-output
accuracy — run this before training and again if the serving stack changes.

**Optional:** `--output PATH` — override audit report output path.

### `train_sft.py`
Phase 1 QLoRA supervised fine-tuning via TRL's SFTTrainer. Checkpoints
regularly, evaluates each on validation loss, promotes the best (not the last)
checkpoint to `artifacts/sft/best`.

Reads: `data/train.jsonl`, `data/val.jsonl`
Writes: `artifacts/sft/`, `artifacts/sft/best/`, `artifacts/sft/manifest.json`

### `generate_preferences.py`
Loads the best SFT checkpoint, generates `completions_per_prompt` diverse
outputs per prompt at temperature > 0, scores via a deterministic rubric
(schema validity + null-case correctness + entity F1), and emits
chosen/rejected JSONL pairs.

Must run after `train_sft.py` and before `train_dpo.py`.
Requires a CUDA GPU — generation at temperature > 0 is the slow step.

**Optional flags:**
- `--sft-checkpoint PATH` — override SFT checkpoint path
- `--output-dir PATH` — override preference dataset output directory

Reads: `data/train.jsonl`, `artifacts/sft/best/`
Writes: `artifacts/dpo/preference_dataset/preference_pairs.jsonl`

### `train_dpo.py`
Phase 2 DPO from the best SFT checkpoint using TRL's DPOTrainer. Precomputes
reference log-probabilities by default so only the policy model is in memory
during training — required for single-GPU setups with 7B+ models.

**Optional flags:**
- `--skip-precompute` — skip ref log-prob precomputation (A100/H100 with ≥40 GB VRAM only)

Reads: `artifacts/dpo/preference_dataset/`, `artifacts/sft/best/`
Writes: `artifacts/dpo/`, `artifacts/dpo/best/`, `artifacts/dpo/manifest.json`

### `evaluate.py`
Runs the evaluation harness across all models under both raw and constrained
decoding. Emits `artifacts/eval/metrics.json` — the CI/CD contract — with
pass/fail flags against configurable thresholds.

**Mode flags:**
- `--mode all` — evaluate base, SFT, and DPO adapters (default)
- `--mode baseline` — base model only
- `--mode post-export` — export artifacts only (merged BF16, GGUFs)
- `--post-export` — append export artifact results to an existing run
- `--merge-existing PATH` — load prior results from an existing `metrics.json`
  and only run new evaluations, avoiding re-downloading the base model

Writes: `artifacts/eval/metrics.json`, `artifacts/eval/qualitative_samples.json`,
`artifacts/eval/report.md`

### `export.py`
Exports the final model in three formats: LoRA adapter (safetensors), merged
BF16 full weights (safetensors), and GGUF (Q8_0 and Q4_K_M). Each artifact
is re-evaluated on the frozen test set — the numbers in the final report are
the numbers of the artifact that ships, not the adapter during training.

**Optional flags:**
- `--checkpoint PATH` — override checkpoint path (defaults to `artifacts/dpo/best`)

Reads: `artifacts/dpo/best/`
Writes: `artifacts/export/adapter/`, `artifacts/export/merged_bf16/`,
`artifacts/export/gguf/model_q8_0.gguf`, `artifacts/export/gguf/model_q4_k_m.gguf`
