FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    clang \
    git \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY 3rdparty/ 3rdparty/
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Build arguments for model configuration
ARG HF_REPO=microsoft/BitNet-b1.58-2B-4T
ARG QUANT_TYPE=i2_s

# Download pre-quantized GGUF model and build
RUN python3 setup_env.py -hr ${HF_REPO} -q ${QUANT_TYPE}

# --- Runtime stage ---
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/bin/ /app/build/bin/
COPY --from=builder /app/models/ /app/models/
COPY --from=builder /app/run_inference.py /app/run_inference.py
COPY --from=builder /app/run_inference_server.py /app/run_inference_server.py

# Default model path (override with -e MODEL_PATH=...)
ENV MODEL_PATH=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
ENV THREADS=4
ENV CTX_SIZE=2048
ENV HOST=0.0.0.0
ENV PORT=8080

EXPOSE 8080

CMD python3 run_inference_server.py \
    -m ${MODEL_PATH} \
    -t ${THREADS} \
    -c ${CTX_SIZE} \
    --host ${HOST} \
    --port ${PORT}
