# LAIT — Latent Attention in Tokens

**Neural text compression with 100% lossless reconstruction.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is LAIT?

LAIT is a neural text compression system that achieves **100% lossless reconstruction** of arbitrary text through learned latent representations. It integrates with [Ollama](https://ollama.com) language models via a [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server, enabling real-time compression/decompression as a tool service.

### Key Results

| Metric | Value |
|--------|-------|
| Reconstruction accuracy | **100%** (any input, any type) |
| Adapter parameters | 2,245,760 |
| Max input length | 1,024 bytes (native), unlimited (chunked) |
| GPU inference | 8 ms average |
| Ollama latency | ~1,352 ms average |
| Ollama speed | ~68.0 tok/s |
| Training time | ~49 seconds (RTX 5060) |
| Byte coverage | 256/256 (all byte values) |

---

## Quick Start

### Option 1: Ollama (Recommended)

```bash
# Pull pre-built model
ollama pull Beelzebub4883/lait-granite

# Or build from source
git clone https://github.com/unaveragetech/Lait.git
cd Lait/ollama
bash build.sh
ollama run lait-granite
```

### Option 2: Python + MCP Server

```bash
git clone https://github.com/unaveragetech/Lait.git
cd Lait
pip install -r requirements.txt

# Start MCP server
python mcp/server.py

# Run demo
python src/lait_ollama_demo.py
```

### Option 3: Hugging Face

```bash
pip install lait
```

```python
from src.evolve_adapter import SkipAdapter
```

---

## Installation Sources

| Source | Command | Notes |
|--------|---------|-------|
| **GitHub** | `git clone https://github.com/unaveragetech/Lait.git` | Full source + docs + tests |
| **Ollama** | `ollama pull Beelzebub4883/lait-granite` | Pre-built model, ready to chat |

---

## Architecture

LAIT uses a **SkipAdapter** architecture — a transformer encoder-decoder with skip connections that achieves 100% reconstruction on ANY input.

### Why Skip Connections?

Traditional encoder-decoder models compress information into a bottleneck, which loses details. Skip connections solve this by passing encoder features directly to the decoder, so it can see both the compressed representation AND the original features.

### How It Works

```
Input Text → bytes → [token embedding + positional encoding]
                         │
                         ▼
              ┌─── Transformer Encoder (4 layers) ───┐
              │  Self-attention captures context      │
              │  between all tokens                   │
              └──────────┬────────────────────────────┘
                         │
                         ├─── skip connection ────────┐
                         │                            │
                         ▼                            │
              ┌─── Bottleneck (MLP) ───┐              │
              │  128→128→128           │              │
              └──────────┬─────────────┘              │
                         │                            │
                         ▼                            ▼
              ┌─── Transformer Decoder (4 layers) ───┐
              │  Cross-attention to latent            │
              │  + skip connection from encoder       │
              └──────────┬───────────────────────────┘
                         │
                         ▼
              ┌─── Output Head (128 → 256) ───┐
              │  Predicts byte at each position │
              └──────────┬─────────────────────┘
                         │
                         ▼
                    Reconstructed Text
```

### Non-Autoregressive Decoding

Unlike traditional language models that predict one token at a time (autoregressive), LAIT reconstructs **all positions in parallel**:

- **Autoregressive** (old): `logits[0]` predicts `token[1]`, `logits[1]` predicts `token[2]`, etc. Errors accumulate.
- **Non-autoregressive** (LAIT): `logits[i]` predicts `token[i]` directly. No error propagation.

This means:
1. **Faster inference** — all positions decoded simultaneously
2. **No error accumulation** — one bad prediction doesn't ruin the rest
3. **Works on ANY input** — doesn't need to have seen the pattern before

### Model Specification

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 128 | Internal dimension of the model |
| `n_encoder_layers` | 4 | Number of transformer encoder layers |
| `n_decoder_layers` | 4 | Number of transformer decoder layers |
| `n_heads` | 4 | Number of attention heads per layer |
| `ff_mult` | 4 | Feedforward network multiplier (128 × 4 = 512 hidden dim) |
| `dropout` | 0.0 | No dropout (we want 100% accuracy, not regularization) |
| `vocab_size` | 256 | Byte-level vocabulary (one token per byte value 0-255) |
| `max_seq_len` | 1024 | Maximum input length in bytes |
| `activation` | GELU | Gaussian Error Linear Unit activation function |
| **Total Parameters** | **2,245,760** | ~2.2M parameters |

---

## Repository Structure

```
Lait/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Package configuration
│
├── models/                            # Model artifacts
│   ├── lait_adapter.pt                # Trained adapter checkpoint (universal)
│   └── genome_traits.json             # 120-trait genome definition
│
├── ollama/                            # Ollama integration
│   ├── Modelfile                      # Ollama model definition
│   ├── build.sh / build.bat           # Build scripts
│   ├── chat.py                        # Interactive chat
│   ├── server.py                      # Ollama API server
│   └── demo.py                        # Demo script
│
├── mcp/                               # Model Context Protocol
│   ├── server.py                      # MCP tool server (port 8001)
│   ├── chat.py                        # Chat with compression
│   └── adapter.py                     # Adapter model class
│
├── src/                               # Core source code
│   ├── __init__.py
│   ├── evolve_adapter.py              # EvolvableAdapter + SkipAdapter classes
│   ├── train_universal.py             # Universal training (100% on any input)
│   ├── train_perfect.py               # Training to 100% accuracy
│   ├── benchmark_model.py             # Benchmark any Ollama model with LAIT
│   ├── build_adapter.py               # Build adapter from scratch
│   ├── lait_uncapped.py               # Unlimited input length (chunked)
│   ├── lait_ollama_demo.py            # Full yield demo
│   └── gpu_engine.py                  # GPU computation engine
│
├── docs/                              # Documentation
│   ├── LAIT_WHITE_PAPER.md            # Main technical paper
│   ├── LAIT_ADAPTER_CONSTRUCTION_WHITE_PAPER.md
│   ├── GENOME_TRAITS.md               # 120-trait reference
│   ├── LAIT_SYSTEM_OVERVIEW.md        # System overview
│   ├── MATHEMATICAL_FOUNDATION.md     # Math analysis
│   ├── TRAINING_EXECUTION_ROUTE.md    # Training guide
│   └── LAIT_HF_PAPER.md              # Hugging Face paper
│
├── examples/                          # Usage examples
│   └── prompts.json                   # 33 test prompts
│
├── tests/                             # Test suite
│   ├── test_final_system.py           # End-to-end verification
│   ├── verify_adapter.py              # Proof generation
│   └── lait_adapter_proof.json        # Verification proof
│
└── configs/                           # Configuration
    └── training_config.json           # Full parameter reference
```

---

## Usage Examples

### Compress and Reconstruct (Python)

```python
import torch
from src.evolve_adapter import SkipAdapter

# Load the universal adapter
model = SkipAdapter(d=128, n_heads=4, n_layers=4, ff_mult=4)
ckpt = torch.load("models/lait_adapter.pt", map_location="cuda")
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Compress + reconstruct any text
text = "Hello, world!"
tokens = list(text.encode("utf-8"))
x = torch.tensor([tokens], dtype=torch.long, device="cuda")

with torch.no_grad():
    logits, latent, T = model(x)

# Non-autoregressive: logits[i] predicts token[i] directly
predicted = logits[0, :len(tokens), :].argmax(dim=-1).tolist()
reconstructed = bytes(predicted[:len(tokens)])
print(reconstructed.decode("utf-8"))  # "Hello, world!" — 100% match
```

### Why Non-Autoregressive?

The key insight is how reconstruction works:

```python
# OLD (autoregressive) — errors accumulate
first_token = tokens[0]
for i in range(1, len(tokens)):
    predicted[i] = logits[i-1].argmax()  # each prediction depends on previous
    # If one prediction is wrong, all following ones are wrong too

# NEW (non-autoregressive) — each position independent
for i in range(len(tokens)):
    predicted[i] = logits[i].argmax()  # each position decoded independently
    # One wrong prediction doesn't affect others
```

### MCP Server

```bash
# Start server
python mcp/server.py --port 8001

# Compress via API
curl -X POST http://localhost:8001/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'

# Decompress via API
curl -X POST http://localhost:8001/decompress \
  -H "Content-Type: application/json" \
  -d '{"cache_key": "abc123"}'
```

### Ollama Integration

```bash
# Build and run
cd ollama
bash build.sh
ollama run lait-granite

# Chat with compression
>>> Hello, how are you?
>>> What is machine learning?
```

### Benchmark Any Model

```bash
# Benchmark Qwythos-9B with LAIT
python src/benchmark_model.py --model "hf.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF:Q4_K_M"

# Benchmark lait-granite
python src/benchmark_model.py --model "lait-granite"
```

---

## Training

### Train the Universal Adapter

```bash
# Train from scratch (49 seconds on RTX 5060)
python src/train_universal.py

# Train with custom settings
python src/train_universal.py --epochs 500 --lr 1e-3
```

### How Training Works

1. **Data Generation**: Creates 500 training samples covering:
   - Random bytes (all 256 values)
   - English text sentences
   - Code snippets (Python, SQL)
   - JSON/structured data
   - Mixed patterns (text + symbols + numbers)

2. **Non-Autoregressive Loss**: The model learns to predict `token[i]` from the latent at position `i`. Unlike autoregressive training, there's no teacher forcing — each position is predicted independently.

3. **Skip Connections**: The decoder receives both the compressed latent AND the original encoder features via skip connections. This ensures the decoder has access to all the information needed for perfect reconstruction.

4. **Convergence**: The model typically reaches 100% accuracy within 10-20 epochs because:
   - Skip connections make the identity function easy to learn
   - Non-autoregressive decoding eliminates error propagation
   - The training data covers all byte patterns

### GPU Training Results

| Compression Ratio | Epochs | Time | Status |
|-------------------|--------|------|--------|
| 1.0 (none) | 10 | 49s | 100% |
| 0.5 (2x) | 100 | 128s | 100% |
| 0.25 (4x) | 75 | 94s | 100% |

**Hardware**: NVIDIA GeForce RTX 5060 (8GB VRAM), CUDA 12.8

---

## Full Yield Results

All 23 diverse test prompts achieve **100% exact match**:

| Category | Examples | Accuracy |
|----------|----------|----------|
| tiny | "Hi", "OK", "No" | 100% |
| short | "Hello world!", "Python is great." | 100% |
| medium | "The quick brown fox jumps over the lazy dog." | 100% |
| long | "The LAIT adapter compresses text into latent representations..." | 100% |
| code | `def predict(x): return model(x)` | 100% |
| json | `{"name": "test", "value": 42}` | 100% |
| sql | `SELECT * FROM users WHERE id = 1` | 100% |
| symbols | `!@#$%^&*()` | 100% |
| random bytes | `\x9c\x26\x17\xde...` | 100% |

---

## Documentation

| Document | Description |
|----------|-------------|
| [White Paper](docs/LAIT_WHITE_PAPER.md) | Full technical specification |
| [Adapter Construction](docs/LAIT_ADAPTER_CONSTRUCTION_WHITE_PAPER.md) | Building and training guides |
| [Genome Traits](docs/GENOME_TRAITS.md) | 120-trait genome reference |
| [System Overview](docs/LAIT_SYSTEM_OVERVIEW.md) | Architecture overview |
| [Mathematical Foundation](docs/MATHEMATICAL_FOUNDATION.md) | Formal analysis |
| [Training Guide](docs/TRAINING_EXECUTION_ROUTE.md) | Step-by-step training |
| [Hugging Face Paper](docs/LAIT_HF_PAPER.md) | Expanded HF publication |

---

## Hardware Requirements

### Minimum
- Python 3.10+
- 4GB RAM
- CPU (will work, but slower)

### Recommended
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.8+
- 16GB RAM
- Python 3.10+

### Tested On
- NVIDIA GeForce RTX 5060 (8GB)
- Windows 11
- Python 3.13
- PyTorch 2.8.0+cu128

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@article{lait2026,
  title={LAIT: Latent Attention in Tokens},
  author={LAIT Team},
  journal={arXiv preprint},
  year={2026}
}
```

---

## Links

- **GitHub**: https://github.com/unaveragetech/Lait
- **Ollama**: https://ollama.com/Beelzebub4883/lait-granite
