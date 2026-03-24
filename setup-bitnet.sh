#!/bin/bash
# setup-bitnet.sh — One-command setup for BitNet.cpp with a 3B model
# Clones Microsoft's BitNet.cpp, downloads a model, compiles, and starts the server.

set -e

MODEL_REPO="${1:-microsoft/BitNet-b1.58-3B}"
QUANT_TYPE="${2:-i2_s}"
PORT="${3:-8080}"
THREADS="${4:-8}"

echo "=============================================="
echo "  BitNet.cpp Quick Setup"
echo "=============================================="
echo ""
echo "  Model:      $MODEL_REPO"
echo "  Quant type: $QUANT_TYPE"
echo "  Port:       $PORT"
echo "  Threads:    $THREADS"
echo ""

# Check dependencies
for cmd in git python3 cmake; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is required but not installed."
        exit 1
    fi
done

# Clone if not already present
if [ ! -d "BitNet-cpp" ]; then
    echo "[1/4] Cloning BitNet.cpp..."
    git clone https://github.com/microsoft/BitNet.git BitNet-cpp
else
    echo "[1/4] BitNet-cpp directory already exists, skipping clone."
fi

cd BitNet-cpp

# Setup environment and download model
echo "[2/4] Setting up environment and downloading model..."
python3 setup_env.py --hf-repo "$MODEL_REPO" --quant-type "$QUANT_TYPE"

# Find the compiled model
MODEL_NAME=$(echo "$MODEL_REPO" | sed 's/.*\///' | tr '.-' '_')
MODEL_PATH="models/${MODEL_NAME}/ggml-model-${QUANT_TYPE}.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Expected model at $MODEL_PATH"
    echo "Looking for any .gguf file..."
    MODEL_PATH=$(find models/ -name "*.gguf" | head -1)
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: No .gguf model found."
        exit 1
    fi
fi

echo "[3/4] Model ready at: $MODEL_PATH"

# Test with a quick inference
echo "[4/4] Testing inference..."
python3 run_inference.py -m "$MODEL_PATH" -p "Hello, I am BitNet" -t "$THREADS" -n 32

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "  Run inference:"
echo "    python3 run_inference.py -m $MODEL_PATH -p 'Your prompt' -t $THREADS"
echo ""
echo "  Start API server:"
echo "    python3 run_inference_server.py -m $MODEL_PATH --port $PORT --threads $THREADS"
echo ""
echo "  Then connect with the BitNet Explorer GUI or any OpenAI-compatible client."
