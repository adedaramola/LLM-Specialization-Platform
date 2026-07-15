.PHONY: all setup train evaluate export clean docker-build docker-run lint test smoke-test check generate-preferences

SFT_CONFIG  ?= configs/sft_config.yaml
DPO_CONFIG  ?= configs/dpo_config.yaml
EVAL_CONFIG ?= configs/eval_config.yaml
RAW         ?= data/raw/news_extraction_v2.jsonl

all: train evaluate export

# Local dev environment (Mac/MPS — dataset prep, tests, small-model inference)
# SFT/DPO training requires CUDA: use docker-run instead
setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt
	@echo "Activate with: source .venv/bin/activate"

# Generate DPO preference dataset from best SFT checkpoint
# use_base_for_rejected is read from dpo_config.yaml preference_data section
generate-preferences:
	python scripts/generate_preferences.py --config $(DPO_CONFIG)

# One-command reproduction from clean checkout
# Stage 1-3: SFT config drives data prep, tokenizer audit, and SFT training
# Stage 4-5: DPO config drives preference generation (incl. use_base_for_rejected) and DPO training
train:
	python scripts/prepare_dataset.py --config $(SFT_CONFIG) --raw-data $(RAW)
	python scripts/tokenizer_audit.py --config $(SFT_CONFIG)
	python scripts/train_sft.py --config $(SFT_CONFIG)
	python scripts/generate_preferences.py --config $(DPO_CONFIG)
	python scripts/train_dpo.py --config $(DPO_CONFIG)

# Full eval: base + SFT + DPO on frozen test set (run on training instance with all checkpoints)
evaluate:
	python scripts/evaluate.py --config $(EVAL_CONFIG) --all-artifacts

# Export adapters → merged BF16 → GGUF, then re-verify all artifacts
export:
	python scripts/export.py --config $(SFT_CONFIG)
	python scripts/evaluate.py --config $(EVAL_CONFIG) --mode post-export --post-export --merge-existing artifacts/eval/metrics.json

baseline:
	python scripts/evaluate.py --config $(CONFIG) --mode baseline

tokenizer-audit:
	python scripts/tokenizer_audit.py --config $(CONFIG)

# Fails when entity labels are not grounded verbatim in the input text —
# ungrounded labels are unlearnable and cap field F1 (RAW defaults to the v2 dataset)
label-audit:
	python scripts/audit_labels.py $(RAW)

# Build args: INSTALL_TRAIN=1 adds trl/datasets/bitsandbytes; INSTALL_GGUF=1 adds llama.cpp + llama-cpp-python
docker-build:
	docker build -t llm-specialization:latest \
	  --build-arg INSTALL_TRAIN=1 \
	  --build-arg INSTALL_GGUF=1 \
	  .

docker-build-eval:
	docker build -t llm-specialization:eval \
	  .

docker-run:
	docker run --gpus all --rm \
	  -e SFT_CONFIG=$(SFT_CONFIG) \
	  -e DPO_CONFIG=$(DPO_CONFIG) \
	  -e MAKE_TARGET=train \
	  -v $(PWD)/artifacts:/workspace/artifacts \
	  -v $(PWD)/data:/workspace/data \
	  llm-specialization:latest

test:
	python3 -m pytest tests/ -v --cov=src --cov-report=term-missing

# Verify pipeline logic without GPU or model downloads
smoke-test:
	python3 -m pytest tests/ -v

# Syntax check — find avoids shell glob issues with **
lint:
	find src scripts -name "*.py" | xargs python3 -m py_compile
	@echo "Syntax check passed"

# Primary CI target: syntax + full test suite
check: lint smoke-test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
