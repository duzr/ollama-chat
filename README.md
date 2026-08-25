# ollama-chat

A small command-line script for talking to models hosted on a remote [Ollama](https://ollama.com/) server over your local network.

It does two things:

1. Fetches the list of models installed on the target Ollama server and lets you pick one.
2. Opens an interactive chat session with that model, streaming the response as it's generated.

## Requirements

- Python 3.9+
- [requests](https://pypi.org/project/requests/)
- A reachable Ollama server (default API port `11434`) with at least one model pulled

Install the dependency with:

```bash
pip install requests
```

## Usage

```bash
python3 ollama_chat.py --host http://<ollama-host>:<port>
```

For example, to connect to an Ollama server at `192.168.1.50` on port `11434`:

```bash
python3 ollama_chat.py --host http://192.168.1.50:11434
```

If `--host` is omitted, it defaults to `http://localhost:11434`.

You'll be shown a numbered list of available models:

```
Available models:
  1. deepseek_r1:1.5b (2.1 GB)
  2. llama3.2:1b (1.3 GB)
Select a model [1-2]:
```

After selecting a model, type messages at the `You:` prompt and the model's reply will stream back beneath it. Conversation history is kept for the duration of the session so the model retains context between turns.

### Exiting

Type `exit` or `quit` at the prompt, or press `Ctrl+D`, to leave the chat cleanly.

## Notes

Before a model's reply is added to the conversation history, `<think>...</think>` reasoning blocks are stripped and stray backslashes are escaped. This works around a bug in some Ollama-compatible backends (e.g. Hailo's HailoRT runtime) where replaying a reasoning model's raw LaTeX-style output (`\boxed{4}`, `\sqrt{2}`, etc.) as context on a later turn causes the server to fail with a 500 error.
