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

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# Install pinned eval stack from lockfile — this is the reproducibility contract
COPY requirements-eval.lock .
RUN pip install --no-cache-dir -r requirements-eval.lock

# Training stack: kept separate to avoid fsspec conflicts with eval stack
# trl --no-deps: avoids downgrading transformers 5.x
RUN if [ "$INSTALL_TRAIN" = "1" ]; then \
        pip install --no-cache-dir "trl==1.4.0" --no-deps && \
        pip install --no-cache-dir "datasets==4.8.5" && \
        pip install --no-cache-dir "bitsandbytes==0.49.2" && \
        pip install --no-cache-dir "wandb>=0.18.0"; \
    fi

# llama-cpp-python with CUDA — builds its own embedded llama.cpp.
# Eval harness uses Python bindings only; no CLI build needed.
RUN if [ "$INSTALL_GGUF" = "1" ]; then \
        CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --no-cache-dir "llama-cpp-python==0.3.9"; \
    fi

# Copy project code — .dockerignore excludes artifacts/, data/, .git/
COPY . .

# Emit hardware fingerprint then run the requested make target.
# Usage:
#   docker run --gpus all -e CONFIG=configs/sft_config.yaml llm-specialization make train
#   docker run --gpus all llm-specialization make evaluate
CMD ["bash", "-c", \
     "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader && \
      nvcc --version | grep release && \
      python -c 'import torch; print(f\"PyTorch {torch.__version__}  CUDA {torch.version.cuda}\")' && \
      make ${MAKE_TARGET:-train} SFT_CONFIG=${SFT_CONFIG:-configs/sft_config.yaml} DPO_CONFIG=${DPO_CONFIG:-configs/dpo_config.yaml}"]
