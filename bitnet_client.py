"""
BitNet API Client — Connect to a running BitNet.cpp server.

The BitNet.cpp server exposes an OpenAI-compatible API at /v1/*.
This client demonstrates chat, completions, streaming, and model listing.

Usage:
    python3 bitnet_client.py                          # Interactive chat
    python3 bitnet_client.py --prompt "Hello"         # Single prompt
    python3 bitnet_client.py --url http://host:8080   # Custom server URL
    python3 bitnet_client.py --stream                 # Streaming mode
"""

import argparse
import sys
from openai import OpenAI


def get_client(base_url: str) -> OpenAI:
    return OpenAI(api_key="bitnet-local", base_url=base_url.rstrip("/") + "/v1")


def list_models(client: OpenAI):
    """List available models on the server."""
    print("Available models:")
    try:
        models = client.models.list()
        for m in models.data:
            print(f"  - {m.id}")
    except Exception as exc:
        print(f"  Error: {exc}")


def single_prompt(client: OpenAI, prompt: str, stream: bool = False):
    """Send a single prompt and print the response."""
    messages = [{"role": "user", "content": prompt}]

    if stream:
        print("Assistant: ", end="", flush=True)
        response = client.chat.completions.create(
            model="bitnet",
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    else:
        response = client.chat.completions.create(
            model="bitnet",
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        print(f"Assistant: {content}")

        if response.usage:
            print(f"\n  Tokens: {response.usage.prompt_tokens} in + "
                  f"{response.usage.completion_tokens} out = {response.usage.total_tokens}")


def interactive_chat(client: OpenAI, stream: bool = False):
    """Interactive chat loop with conversation history."""
    print("BitNet Chat (type 'quit' to exit, 'clear' to reset history)")
    print("-" * 50)

    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            history = []
            print("History cleared.")
            continue

        history.append({"role": "user", "content": user_input})

        try:
            if stream:
                print("Assistant: ", end="", flush=True)
                response = client.chat.completions.create(
                    model="bitnet",
                    messages=history,
                    max_tokens=512,
                    temperature=0.7,
                    stream=True,
                )
                full_response = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        print(text, end="", flush=True)
                        full_response += text
                print()
                history.append({"role": "assistant", "content": full_response})
            else:
                response = client.chat.completions.create(
                    model="bitnet",
                    messages=history,
                    max_tokens=512,
                    temperature=0.7,
                )
                content = response.choices[0].message.content
                print(f"Assistant: {content}")
                history.append({"role": "assistant", "content": content})

                if response.usage:
                    print(f"  [{response.usage.total_tokens} tokens]")

        except Exception as exc:
            print(f"Error: {exc}")
            history.pop()  # Remove failed user message


def main():
    parser = argparse.ArgumentParser(description="BitNet API Client")
    parser.add_argument("--url", default="http://localhost:8080", help="BitNet server URL")
    parser.add_argument("--prompt", type=str, help="Single prompt (non-interactive)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    parser.add_argument("--models", action="store_true", help="List available models")
    args = parser.parse_args()

    client = get_client(args.url)

    if args.models:
        list_models(client)
    elif args.prompt:
        single_prompt(client, args.prompt, stream=args.stream)
    else:
        interactive_chat(client, stream=args.stream)


if __name__ == "__main__":
    main()
