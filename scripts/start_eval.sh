#!/usr/bin/env bash
# Start vLLM server and run evaluation in sequence.
# Handles the two most common failures: server not ready, and eval running against
# a stale server that has the wrong model loaded.
#
# Usage:
#   bash scripts/start_eval.sh                        # post-export eval (merged_bf16)
#   bash scripts/start_eval.sh --mode post-dpo        # DPO adapter eval (hf_native)
#   bash scripts/start_eval.sh --merge-existing       # merge results into existing metrics.json
#   bash scripts/start_eval.sh --gguf                 # GGUF-only eval (no vLLM)
#   bash scripts/start_eval.sh --baseline             # base model eval
#
# Must be run with the venv active:
#   source ~/vllm-env/bin/activate && bash scripts/start_eval.sh

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
EVAL_MODE="post-export"
MERGE_EXISTING=false
GGUF_ONLY=false
VLLM_PORT=8000
VLLM_TIMEOUT=300   # seconds to wait for vLLM to be ready
EVAL_CONFIG="configs/eval_config.yaml"
GGUF_CONFIG="configs/eval_config_gguf.yaml"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="$REPO_ROOT/artifacts/export/merged_bf16"
METRICS_PATH="$REPO_ROOT/artifacts/eval/metrics.json"

for arg in "$@"; do
    [[ "$arg" == "--merge-existing" ]] && MERGE_EXISTING=true
    [[ "$arg" == "--gguf" ]]           && GGUF_ONLY=true
    [[ "$arg" == "--baseline" ]]       && EVAL_MODE="baseline"
    [[ "$arg" == "--mode" ]]           && true   # handled below
done
# Capture --mode value
prev=""
for i in "$@"; do
    if [[ "$prev" == "--mode" ]]; then EVAL_MODE="$i"; fi
    prev="$i"
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

wait_vllm() {
    local deadline=$(( $(date +%s) + VLLM_TIMEOUT ))
    log "Waiting for vLLM on port $VLLM_PORT (timeout ${VLLM_TIMEOUT}s)..."
    while [[ $(date +%s) -lt $deadline ]]; do
        if curl -sf "http://localhost:$VLLM_PORT/v1/models" &>/dev/null; then
            log "vLLM ready."
            return 0
        fi
        sleep 5
    done
    die "vLLM did not become ready within ${VLLM_TIMEOUT}s. Check /tmp/vllm_server.log"
}

stop_vllm() {
    local pids
    pids=$(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log "Stopping existing vLLM process(es): $pids"
        kill $pids 2>/dev/null || true
        sleep 3
    fi
}

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "$HOME/vllm-env/bin/activate" ]]; then
        source "$HOME/vllm-env/bin/activate"
        log "Activated venv: $HOME/vllm-env"
    else
        die "No venv found. Run: bash scripts/setup_instance.sh first."
    fi
fi

# ── GGUF-only path (no vLLM) ─────────────────────────────────────────────────
if [[ "$GGUF_ONLY" == "true" ]]; then
    log "Running GGUF eval (llama-cpp-python, no vLLM)"
    [[ -f "$GGUF_CONFIG" ]] || die "GGUF eval config not found: $GGUF_CONFIG"

    MERGE_FLAG=""
    [[ "$MERGE_EXISTING" == "true" && -f "$METRICS_PATH" ]] && MERGE_FLAG="--merge-existing $METRICS_PATH"

    USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 python3 -u scripts/evaluate.py \
        --config "$GGUF_CONFIG" \
        --mode post-export \
        --post-export \
        $MERGE_FLAG
    exit 0
fi

# ── vLLM-backed eval ──────────────────────────────────────────────────────────

# Check model path exists before starting server
if [[ "$EVAL_MODE" == "post-export" ]]; then
    [[ -d "$MODEL_PATH" ]] || die "Merged model not found at $MODEL_PATH — run export first"
fi

# Stop any vLLM holding the wrong model or a stale state
RUNNING_MODEL=""
if curl -sf "http://localhost:$VLLM_PORT/v1/models" &>/dev/null; then
    RUNNING_MODEL=$(curl -sf "http://localhost:$VLLM_PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null || true)
fi

if [[ "$EVAL_MODE" == "post-export" ]]; then
    EXPECTED_MODEL="$MODEL_PATH"
    if [[ "$RUNNING_MODEL" == "$EXPECTED_MODEL" ]]; then
        log "vLLM already serving correct model: $RUNNING_MODEL"
    else
        if [[ -n "$RUNNING_MODEL" ]]; then
            log "vLLM serving wrong model ($RUNNING_MODEL) — restarting"
            stop_vllm
        fi
        log "Starting vLLM with $MODEL_PATH"
        USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 nohup python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --dtype bfloat16 \
            --port "$VLLM_PORT" \
            > /tmp/vllm_server.log 2>&1 &
        wait_vllm
    fi
fi

# ── Run evaluate.py ───────────────────────────────────────────────────────────
MERGE_FLAG=""
[[ "$MERGE_EXISTING" == "true" && -f "$METRICS_PATH" ]] && MERGE_FLAG="--merge-existing $METRICS_PATH"

POST_EXPORT_FLAG=""
[[ "$EVAL_MODE" == "post-export" ]] && POST_EXPORT_FLAG="--post-export"

log "Running evaluation (mode=$EVAL_MODE)"
USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 python3 -u scripts/evaluate.py \
    --config "$EVAL_CONFIG" \
    --mode "$EVAL_MODE" \
    $POST_EXPORT_FLAG \
    $MERGE_FLAG

EXIT_CODE=$?

# ── Upload results ────────────────────────────────────────────────────────────
if [[ $EXIT_CODE -eq 0 && -f "$METRICS_PATH" ]]; then
    log "Uploading metrics.json to S3"
    aws s3 cp "$METRICS_PATH" "s3://llm-specialization-artifacts/v2/artifacts/eval/metrics.json" \
        && log "Uploaded metrics.json" \
        || log "WARNING: S3 upload failed (results saved locally)"
fi

exit $EXIT_CODE
