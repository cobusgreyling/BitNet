#!/bin/bash
# Example: Using the BitNet OpenAI-compatible API with curl.
#
# Start the server first:
#   python run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
#
# Then run: bash examples/openai_curl.sh

BASE_URL="http://localhost:8080"

echo "=== Chat Completion ==="
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bitnet",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain BitNet in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }' | python3 -m json.tool

echo ""
echo "=== Text Completion ==="
curl -s "${BASE_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bitnet",
    "prompt": "1-bit LLMs are important because",
    "max_tokens": 128,
    "temperature": 0.7
  }' | python3 -m json.tool

echo ""
echo "=== List Models ==="
curl -s "${BASE_URL}/v1/models" | python3 -m json.tool

echo ""
echo "=== Health Check ==="
curl -s "${BASE_URL}/health" | python3 -m json.tool
