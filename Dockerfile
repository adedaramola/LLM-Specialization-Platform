# Validated environment from run manifests (2026-05-12):
#   NVIDIA driver:  580.105.08  (minimum: 520 for CUDA 12.8)
#   CUDA Toolkit:   12.8 (nvcc)
#   cuDNN:          9.19.0
#   PyTorch:        2.11.0  (torch.version.cuda = 13.0)
#
# Plain CUDA base — avoids version conflicts with the pre-installed PyTorch
# that ships in NGC PyTorch images. All Python packages come from requirements-eval.lock.
#
# Build args:
#   INSTALL_TRAIN=1   also install trl/datasets/bitsandbytes/wandb (training stack)
#   INSTALL_GGUF=1    install llama-cpp-python with CUDA (GGUF eval)
#
# llama-cpp-python notes:
#   - Uses prebuilt cu125 wheels (highest CUDA variant available from abetlen; no cu128 exists).
#   - cu125 wheels are ABI-compatible with CUDA 12.8 at runtime — the GPU driver provides
#     the actual CUDA runtime, not the wheel.
#   - Wheel tag is py3-none-linux_x86_64 (CUDA binary bundled, not a C extension), so it
#     installs on any Python 3.x. No source compilation needed; no libcuda.so.1 build-time
#     dependency.
#   - Building from source with GGML_CUDA=on fails at docker build time because libcuda.so.1
#     is only available at docker run time (mounted by nvidia-container-toolkit), not during
#     the build layer. The stub at /usr/local/cuda/lib64/stubs/libcuda.so is insufficient for
#     the llama.cpp cmake build.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

LABEL maintainer="LLM-Specialization-Platform"
LABEL cuda_version="12.8"
LABEL cudnn_version="9.19"
LABEL pytorch_version="2.11.0"
LABEL validated_driver="580.105.08"

ARG INSTALL_TRAIN=0
ARG INSTALL_GGUF=0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 + pip, then immediately upgrade pip.
# python3-pip on Ubuntu 22.04 ships pip 22.0.2 which has resolver bugs with recent packages.
# Use python3.10 -m pip throughout to make the interpreter explicit and immune to symlink drift.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3.10 -m pip install --no-cache-dir --upgrade pip

WORKDIR /workspace

# Install pinned eval stack from lockfile — this is the reproducibility contract
COPY requirements-eval.lock .
RUN python3.10 -m pip install --no-cache-dir -r requirements-eval.lock

# Training stack: kept separate to avoid fsspec conflicts with eval stack
# trl --no-deps: avoids downgrading transformers 5.x
RUN if [ "$INSTALL_TRAIN" = "1" ]; then \
        python3.10 -m pip install --no-cache-dir "trl==1.4.0" --no-deps && \
        python3.10 -m pip install --no-cache-dir "datasets==4.8.5" && \
        python3.10 -m pip install --no-cache-dir "bitsandbytes==0.49.2" && \
        python3.10 -m pip install --no-cache-dir "wandb==0.19.11"; \
    fi

# llama-cpp-python prebuilt CUDA wheel — installs in seconds, no compiler needed.
# cu125 is the highest CUDA variant published by abetlen (no cu128 wheel index exists).
# cu125 wheels are runtime-compatible with CUDA 12.8.
# The py3-none-linux_x86_64 wheel tag is compatible with Python 3.10, 3.11, and 3.12.
RUN if [ "$INSTALL_GGUF" = "1" ]; then \
        python3.10 -m pip install --no-cache-dir "llama-cpp-python==0.3.23" \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125; \
    fi

# Copy project code — .dockerignore excludes artifacts/, data/, .git/
COPY . .

# Emit hardware fingerprint then run the requested make target.
# Usage:
#   docker run --gpus all -e MAKE_TARGET=evaluate llm-specialization
#   docker run --gpus all -e SFT_CONFIG=configs/sft_config.yaml llm-specialization
CMD ["bash", "-c", \
     "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader && \
      nvcc --version | grep release && \
      python -c 'import torch; print(f\"PyTorch {torch.__version__}  CUDA {torch.version.cuda}\")' && \
      make ${MAKE_TARGET:-train} SFT_CONFIG=${SFT_CONFIG:-configs/sft_config.yaml} DPO_CONFIG=${DPO_CONFIG:-configs/dpo_config.yaml}"]
