.PHONY: all setup train evaluate export clean docker-build docker-run lint test generate-preferences

CONFIG ?= configs/sft_config.yaml

all: train evaluate export

# Local dev environment (Mac/MPS — dataset prep, tests, small-model inference)
# SFT/DPO training requires CUDA: use docker-run instead
setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt
	@echo "Activate with: source .venv/bin/activate"

# Generate DPO preference dataset from best SFT checkpoint (run before train_dpo)
generate-preferences:
	python scripts/generate_preferences.py --config $(CONFIG)

# One-command reproduction from clean checkout
train:
	python scripts/prepare_dataset.py --config $(CONFIG)
	python scripts/tokenizer_audit.py --config $(CONFIG)
	python scripts/train_sft.py --config $(CONFIG)
	python scripts/generate_preferences.py --config $(CONFIG)
	python scripts/train_dpo.py --config $(CONFIG)

evaluate:
	python scripts/evaluate.py --config $(CONFIG) --all-artifacts

export:
	python scripts/export.py --config $(CONFIG)
	python scripts/evaluate.py --config $(CONFIG) --post-export

baseline:
	python scripts/evaluate.py --config $(CONFIG) --mode baseline

tokenizer-audit:
	python scripts/tokenizer_audit.py --config $(CONFIG)

docker-build:
	docker build -t llm-specialization:latest .

docker-run:
	docker run --gpus all --rm -v $(PWD):/workspace llm-specialization:latest make train CONFIG=$(CONFIG)

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	python -m py_compile src/**/*.py scripts/*.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
