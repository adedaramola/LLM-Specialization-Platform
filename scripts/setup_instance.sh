#!/usr/bin/env bash
# Instance setup for eval-only runs (vLLM constrained decoding).
# Run once on a fresh Lambda Labs A100 instance from the repo root.
#
# Usage:
#   bash scripts/setup_instance.sh            # eval-only (vLLM + peft + data libs)
#   bash scripts/setup_instance.sh --train    # eval + full training stack
#   bash scripts/setup_instance.sh --gguf     # eval + llama-cpp-python with CUDA (for GGUF eval)
#   bash scripts/setup_instance.sh --train --gguf  # full stack
#
# After setup, activate the venv before running any script:
#   source ~/vllm-env/bin/activate
#   USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 python3 scripts/evaluate.py ...
set -euo pipefail

TRAIN_MODE=false
GGUF_MODE=false
for arg in "$@"; do
    [[ "$arg" == "--train" ]] && TRAIN_MODE=true
    [[ "$arg" == "--gguf" ]]  && GGUF_MODE=true
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/vllm-env"
EVAL_LOCK="$REPO_ROOT/requirements-eval.lock"

echo "==> Creating isolated venv at $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip -q

echo "==> Installing eval stack from lock file"
pip install -r "$EVAL_LOCK" -q

if [[ "$TRAIN_MODE" == "true" ]]; then
    echo "==> Installing training stack on top of eval stack"
    # trl 1.4.0 --no-deps: avoids downgrading transformers 5.x installed by vLLM
    # datasets and bitsandbytes kept out of eval lock to avoid fsspec conflicts
    pip install "trl==1.4.0" --no-deps -q
    pip install "datasets==4.8.5" -q
    pip install "bitsandbytes==0.49.2" -q
    pip install "wandb>=0.18.0" -q
fi

if [[ "$GGUF_MODE" == "true" ]]; then
    echo "==> Installing llama-cpp-python with CUDA support (for GGUF eval)"
    # Use the prebuilt cu125 wheel — cu125 is the highest CUDA variant published by abetlen
    # and is runtime-compatible with CUDA 12.8 (the GPU driver provides the actual runtime).
    # This avoids a ~5 min source compile and matches the version in the Dockerfile.
    pip install "llama-cpp-python==0.3.23" \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125 -q
fi

echo "==> Setting up pyairports stub (PyPI 0.0.1 is a squatter, not the real package)"
PYAIR="$HOME/.local/lib/python3.10/site-packages/pyairports"
mkdir -p "$PYAIR"
echo ""              > "$PYAIR/__init__.py"
echo "AIRPORT_LIST = []" > "$PYAIR/airports.py"

echo "==> Verifying key packages"
python3 -c "import vllm, peft, boto3, scipy, sklearn, pandas; print('  All imports OK')"
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Quick-start for constrained eval:"
echo "  source $VENV/bin/activate"
echo "  # 1. Start vLLM server"
echo "  USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 nohup python3 -m vllm.entrypoints.openai.api_server \\"
echo "    --model artifacts/export/merged_bf16 --dtype bfloat16 --port 8000 \\"
echo "    > /tmp/vllm_server.log 2>&1 &"
echo "  # 2. Run eval (wait for server ready first)"
echo "  USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 python3 -u scripts/evaluate.py \\"
echo "    --config configs/eval_config.yaml --mode post-export --post-export \\"
echo "    --merge-existing artifacts/eval/metrics.json"
