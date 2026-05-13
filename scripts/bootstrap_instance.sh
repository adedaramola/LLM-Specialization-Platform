#!/usr/bin/env bash
# Pre-flight bootstrap for a fresh Lambda Labs instance.
# Run this ONCE before any training or eval command.
#
# What it does:
#   1. Checks AWS credentials (S3 access required)
#   2. Pulls data/ and artifacts/ from S3
#   3. Kills any stale GPU process (vLLM, training) holding memory
#   4. Validates CUDA, key Python imports, and GPU free memory
#   5. Prints a clear PASS / FAIL summary — stop here on FAIL
#
# Usage:
#   bash scripts/bootstrap_instance.sh                    # data + artifacts sync
#   bash scripts/bootstrap_instance.sh --skip-artifacts   # data only (faster on eval-only instance)
#   bash scripts/bootstrap_instance.sh --kill-gpu         # also kill any GPU process before check
#
# Must be run with the venv active:
#   source ~/vllm-env/bin/activate && bash scripts/bootstrap_instance.sh

set -euo pipefail

SKIP_ARTIFACTS=false
KILL_GPU=false
for arg in "$@"; do
    [[ "$arg" == "--skip-artifacts" ]] && SKIP_ARTIFACTS=true
    [[ "$arg" == "--kill-gpu" ]]       && KILL_GPU=true
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_BUCKET="llm-specialization-artifacts"
PASS=true

# ── Helpers ──────────────────────────────────────────────────────────────────
ok()   { echo "  [OK]   $*"; }
fail() { echo "  [FAIL] $*"; PASS=false; }
info() { echo "  [INFO] $*"; }

# ── 1. AWS credentials ────────────────────────────────────────────────────────
echo ""
echo "==> Checking AWS credentials"
if aws sts get-caller-identity --query Account --output text &>/dev/null; then
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    ok "AWS authenticated (account $ACCOUNT)"
else
    fail "AWS credentials not configured — copy ~/.aws/credentials to this instance"
    fail "  scp ~/.aws/credentials ubuntu@<instance-ip>:~/.aws/credentials"
    PASS=false
fi

# ── 2. Pull data/ from S3 ─────────────────────────────────────────────────────
echo ""
echo "==> Syncing data/ from S3"
mkdir -p "$REPO_ROOT/data"
for split in train val test; do
    LOCAL="$REPO_ROOT/data/${split}.jsonl"
    S3KEY="runs/v1/data/${split}.jsonl"
    if [[ -f "$LOCAL" ]]; then
        ok "data/${split}.jsonl already present ($(wc -l < "$LOCAL") lines)"
    else
        info "Downloading s3://$S3_BUCKET/$S3KEY"
        if aws s3 cp "s3://$S3_BUCKET/$S3KEY" "$LOCAL" 2>/dev/null; then
            ok "data/${split}.jsonl downloaded ($(wc -l < "$LOCAL") lines)"
        else
            fail "data/${split}.jsonl not found at s3://$S3_BUCKET/$S3KEY"
        fi
    fi
done

# ── 3. Pull artifacts/ from S3 ────────────────────────────────────────────────
if [[ "$SKIP_ARTIFACTS" == "false" ]]; then
    echo ""
    echo "==> Syncing artifacts/ from S3"
    mkdir -p "$REPO_ROOT/artifacts"
    # Sync eval outputs (metrics, report, qualitative samples)
    aws s3 sync "s3://$S3_BUCKET/artifacts/eval/" "$REPO_ROOT/artifacts/eval/" \
        --exclude "*.log" 2>/dev/null && ok "artifacts/eval/ synced" || info "artifacts/eval/ not found on S3 (first run?)"
    # Sync export artifacts — large files, skip if already present
    for artifact in export/merged_bf16 export/gguf dpo/best; do
        LOCAL_DIR="$REPO_ROOT/artifacts/$artifact"
        if [[ -d "$LOCAL_DIR" && "$(ls -A "$LOCAL_DIR" 2>/dev/null)" ]]; then
            ok "artifacts/$artifact already present locally"
        else
            info "Syncing artifacts/$artifact from S3 (may be large)"
            aws s3 sync "s3://$S3_BUCKET/artifacts/$artifact/" "$LOCAL_DIR/" 2>/dev/null \
                && ok "artifacts/$artifact synced" \
                || info "artifacts/$artifact not found on S3"
        fi
    done
fi

# ── 4. GPU memory check ───────────────────────────────────────────────────────
echo ""
echo "==> Checking GPU state"
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found — is this an NVIDIA instance?"
else
    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -v "^$" || true)

    if [[ -n "$GPU_PROCS" ]]; then
        if [[ "$KILL_GPU" == "true" ]]; then
            info "Stale GPU processes found — killing:"
            echo "$GPU_PROCS" | while IFS=, read -r pid mem; do
                pid=$(echo "$pid" | tr -d ' ')
                info "  PID $pid using ${mem} MiB"
                kill "$pid" 2>/dev/null && info "  killed $pid" || info "  could not kill $pid (already gone?)"
            done
            sleep 3
            ok "GPU processes cleared"
        else
            fail "GPU memory in use: ${GPU_MEM_USED}/${GPU_MEM_TOTAL} MiB — stale process holding GPU"
            fail "  Re-run with --kill-gpu to clear, or: kill <pid> manually"
            echo "$GPU_PROCS" | while IFS=, read -r pid mem; do
                info "    PID $(echo "$pid" | tr -d ' ') — $(echo "$mem" | tr -d ' ') MiB"
            done
        fi
    else
        ok "GPU free: ${GPU_MEM_USED}/${GPU_MEM_TOTAL} MiB used"
    fi
fi

# ── 5. Python environment validation ─────────────────────────────────────────
echo ""
echo "==> Validating Python environment"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    fail "No venv active — run: source ~/vllm-env/bin/activate"
else
    ok "venv active: $VIRTUAL_ENV"
fi

python3 - <<'PYCHECK'
import sys
results = []

checks = [
    ("torch + CUDA",    "import torch; assert torch.cuda.is_available(), 'CUDA not available'"),
    ("vllm",            "import vllm"),
    ("peft",            "import peft"),
    ("transformers",    "import transformers"),
    ("boto3",           "import boto3"),
    ("scipy/sklearn",   "import scipy, sklearn"),
    ("pandas",          "import pandas"),
]

all_ok = True
for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  [OK]   {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        all_ok = False

try:
    import torch
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f"  [OK]   GPU: {gpu} ({mem} GB)")
except Exception as e:
    print(f"  [FAIL] GPU query: {e}")
    all_ok = False

sys.exit(0 if all_ok else 1)
PYCHECK
PYCHECK_EXIT=$?
[[ $PYCHECK_EXIT -eq 0 ]] || PASS=false

# ── 6. Key file checks ────────────────────────────────────────────────────────
echo ""
echo "==> Checking required files"
REQUIRED_FILES=(
    "configs/sft_config.yaml"
    "configs/dpo_config.yaml"
    "configs/eval_config.yaml"
    "configs/schemas/extraction_schema.json"
    "requirements-eval.lock"
    "data/train.jsonl"
    "data/val.jsonl"
    "data/test.jsonl"
)
for f in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$REPO_ROOT/$f" ]]; then
        ok "$f"
    else
        fail "$f missing"
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
if [[ "$PASS" == "true" ]]; then
    echo "  PRE-FLIGHT: PASS — ready to run"
    echo ""
    echo "  Next steps:"
    echo "    Training:  make train"
    echo "    Eval only: bash scripts/start_eval.sh"
else
    echo "  PRE-FLIGHT: FAIL — fix errors above before continuing"
    exit 1
fi
echo "════════════════════════════════════════"
echo ""
