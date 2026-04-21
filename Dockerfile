# Pinned CUDA 12.1 + PyTorch 2.3.1 base
# Validated combination: CUDA 12.1, cuDNN 8.9.x, Driver >= 525.85
FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL maintainer="LLM-Specialization-Platform"
LABEL cuda_version="12.3"
LABEL pytorch_version="2.3.1"
LABEL cudnn_version="8.9"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for llama.cpp GGUF conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install pinned Python deps from lockfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install llama.cpp for GGUF conversion
RUN git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    cmake -B build -DLLAMA_CUBLAS=ON && \
    cmake --build build --config Release -j$(nproc) && \
    pip install --no-cache-dir /opt/llama.cpp/gguf-py

COPY . .

# Hardware fingerprint captured at container start
CMD ["bash", "-c", "nvidia-smi && nvcc --version && python -c 'import torch; print(torch.version.cuda)' && make train CONFIG=${CONFIG:-configs/sft_config.yaml}"]
