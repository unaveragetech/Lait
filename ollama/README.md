# lait-granite — Ollama Model

A custom Ollama model powered by LAIT (Latent Attention in Tokens) — 100% lossless text compression adapter integrated with `jessup-sim:granite4.1`.

## Quick Start

```bash
# Pull pre-built model
ollama pull lait-granite

# Or build from source
git clone https://github.com/lait-project/lait.git
cd lait/ollama
bash build.sh
```

## Usage

```bash
# Interactive chat
ollama run lait-granite

# API call
curl http://localhost:11434/api/chat -d '{
  "model": "lait-granite",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

## Python Chat

```bash
python chat.py
```

## API Server

```bash
# Start on default port 11434
python server.py

# Custom port
python server.py --port 8080
```

## Architecture

```
User Input → LAIT Adapter (compress) → Ollama (generate) → LAIT Adapter (decompress) → Response
```

The adapter is transparent — it compresses input before generation and decompresses output after.

## Demo

```bash
python demo.py
```

Runs 33 test prompts with full metrics (latency, speed, reconstruction accuracy).

## Model Details

| Property | Value |
|----------|-------|
| Base model | jessup-sim:granite4.1 (4.1B params) |
| Adapter | LAIT transformer, 2.06M params |
| Quantization | Q4_K_M |
| Context length | 2,048 tokens |
| Max native input | 1,024 bytes (uncapped via chunking) |

## Files

- `Modelfile` — Ollama model definition
- `build.sh` / `build.bat` — Build scripts
- `chat.py` — Interactive chat
- `server.py` — API server
- `demo.py` — Demo with 33 test prompts
- `README.md` — This file
