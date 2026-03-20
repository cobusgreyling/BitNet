"""
Example: Using BitNet server with the OpenAI Python client.

Prerequisites:
    pip install openai

Start the server first:
    python run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

Then run this script:
    python examples/openai_client.py
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# Chat completion
print("=== Chat Completion ===")
response = client.chat.completions.create(
    model="bitnet",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 1-bit quantization and why does it matter?"},
    ],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)

# Streaming chat completion
print("\n=== Streaming Chat Completion ===")
stream = client.chat.completions.create(
    model="bitnet",
    messages=[
        {"role": "user", "content": "Write a haiku about efficient AI."},
    ],
    stream=True,
    max_tokens=64,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# Text completion
print("\n=== Text Completion ===")
response = client.completions.create(
    model="bitnet",
    prompt="The advantages of 1-bit large language models are",
    max_tokens=128,
    temperature=0.7,
)
print(response.choices[0].text)
