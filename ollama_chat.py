#!/usr/bin/env python3
"""List models on a remote Ollama server and chat with one of them."""

# argparse: parses command-line arguments like --host
# json: used to decode the streamed JSON lines Ollama sends back during chat
# sys: used to exit the program with an error status code
import argparse
import json
import sys

# requests: third-party HTTP library used to talk to the Ollama server's REST API.
# Install it with: pip install requests
import requests


def list_models(base_url: str) -> list[dict]:
    """Ask the Ollama server for its list of installed models.

    Ollama exposes a GET /api/tags endpoint that returns JSON describing
    every model available on that server.
    """
    resp = requests.get(f"{base_url}/api/tags", timeout=10)
    # raise_for_status() throws an exception if the server returned an
    # error status code (like 404 or 500), so problems don't fail silently.
    resp.raise_for_status()
    # The JSON response looks like {"models": [...]}. If "models" is
    # missing for some reason, default to an empty list instead of crashing.
    return resp.json().get("models", [])


def choose_model(models: list[dict]) -> str:
    """Print the available models and let the user pick one by number."""
    print("Available models:")
    # enumerate(..., start=1) gives us a 1-based counter alongside each
    # model, so the menu reads "1.", "2.", etc. instead of starting at 0.
    for i, model in enumerate(models, start=1):
        # Model sizes come back in bytes; convert to gigabytes for readability.
        size_gb = model.get("size", 0) / (1024 ** 3)
        print(f"  {i}. {model['name']} ({size_gb:.1f} GB)")

    # Keep asking until the user enters a valid number.
    while True:
        choice = input(f"Select a model [1-{len(models)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            # Convert the human-facing 1-based choice back to a 0-based
            # list index to look up the chosen model.
            return models[int(choice) - 1]["name"]
        print("Invalid selection, try again.")


def chat(base_url: str, model: str) -> None:
    """Run an interactive back-and-forth chat loop with the chosen model."""
    print(f"\nChatting with '{model}'. Type 'exit' or 'quit' to leave.\n")

    # This list keeps the full conversation history (both your messages and
    # the model's replies). Ollama's /api/chat endpoint expects the entire
    # history on every request so the model has context from earlier turns.
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            # Ctrl+D sends an end-of-file signal instead of text; treat
            # that the same as typing "exit" so the script exits cleanly.
            break

        if not user_input:
            # Ignore blank input and just re-prompt.
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        # Add the user's new message to the running conversation history.
        messages.append({"role": "user", "content": user_input})

        try:
            # stream=True tells requests not to download the whole response
            # at once; instead we read it piece by piece as it arrives,
            # which lets us print the model's reply as it's generated.
            resp = requests.post(
                f"{base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Error contacting Ollama: {e}")
            # Remove the message we just added since it was never
            # successfully sent, so it doesn't pollute future requests.
            messages.pop()
            continue

        print(f"{model}: ", end="", flush=True)
        reply = ""
        # Ollama streams the response as one JSON object per line. Each
        # chunk contains a small piece of the reply plus a "done" flag
        # that's True on the final chunk.
        for line in resp.iter_lines():
            if not line:
                # Skip empty keep-alive lines.
                continue
            chunk = json.loads(line)
            content = chunk.get("message", {}).get("content", "")
            # Print each piece immediately (flush=True) so the reply
            # appears to type itself out, rather than waiting for it all
            # to finish before showing anything.
            print(content, end="", flush=True)
            reply += content
            if chunk.get("done"):
                break
        print("\n")

        # Save the model's full reply to the conversation history so the
        # next request includes it as context.
        messages.append({"role": "assistant", "content": reply})


def main() -> None:
    # Set up command-line argument parsing so the server address can be
    # passed in as `--host http://...` instead of being hardcoded.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Base URL of the Ollama server, e.g. http://192.168.1.50:11434",
    )
    args = parser.parse_args()
    # Strip any trailing slash so URLs built later don't end up with "//".
    base_url = args.host.rstrip("/")

    try:
        models = list_models(base_url)
    except requests.RequestException as e:
        # Covers connection errors, timeouts, DNS failures, etc.
        print(f"Could not reach Ollama server at {base_url}: {e}")
        sys.exit(1)

    if not models:
        print("No models found on the server.")
        sys.exit(1)

    model = choose_model(models)
    chat(base_url, model)


# This guard ensures main() only runs when the script is executed directly
# (e.g. `python3 ollama_chat.py`), not if it's ever imported as a module
# from another script.
if __name__ == "__main__":
    main()
