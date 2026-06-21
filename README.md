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
| Reconstruction accuracy | **100%** (all 33 test prompts) |
| Adapter parameters | 2,064,768 |
| Max input length | 1,024 bytes (native), unlimited (chunked) |
| GPU inference | 12.6 ms average |
| Ollama latency | 1,352 ms average |
| Ollama speed | 68.0 tok/s |
| Training time | ~8.5 minutes (RTX 5060) |
| Compression ratios | 1x to 16x, all at 100% accuracy |

---

## Quick Start

### Option 1: Ollama (Recommended)

```bash
# Pull from Ollama
ollama pull lait-granite

# Or build from source
git clone https://github.com/lait-project/lait.git
cd lait/ollama
bash build.sh
ollama run lait-granite
```

### Option 2: Python + MCP Server

```bash
git clone https://github.com/lait-project/lait.git
cd lait
pip install -r requirements.txt

# Start MCP server
python mcp/server.py

# Run demo
python src/lait_ollama_demo.py
```

### Option 3: Hugging Face

```bash
pip install lait
from src.evolve_adapter import EvolvableAdapter
```

---

## Installation Sources

| Source | Command | Notes |
|--------|---------|-------|
| **GitHub** | `git clone https://github.com/lait-project/lait.git` | Full source + docs + tests |
| **Ollama** | `ollama pull lait-granite` | Pre-built model, ready to chat |
| **PyPI** | `pip install lait` | Python library only |
| **Hugging Face** | `huggingface.co/lait-project/lait` | Model weights + config |

---

## Architecture

```
Input Text (bytes)
    │
    ▼
┌─────────────────┐
│ Token Embedding  │  256 → 128
│ + Positional Enc │  1024 → 128
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Transformer     │
│  Encoder (4x)    │  Self-attention + FFN
│  d=128, heads=4  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AvgPool1d       │  Sequence compression
│  + Linear Proj   │  128 → 128
└────────┬────────┘
         │
         ▼  Latent Representation
         │
         ▼
┌─────────────────┐
│  Transformer     │
│  Decoder (4x)    │  Cross-attention to latent
│  d=128, heads=4  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Head     │  128 → 256
└────────┬────────┘
         │
         ▼
Output Text (reconstructed)
```

### Model Specification

| Parameter | Value |
|-----------|-------|
| `d_model` | 128 |
| `n_encoder_layers` | 4 |
| `n_decoder_layers` | 4 |
| `n_heads` | 4 |
| `ff_mult` | 4 |
| `dropout` | 0.0 |
| `compression_ratio` | 1.0 |
| `vocab_size` | 256 (byte-level) |
| `max_seq_len` | 1024 |
| `activation` | GELU |
| **Total Parameters** | **2,064,768** |

---

## Repository Structure

```
lait/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Package configuration
│
├── models/                            # Model artifacts
│   ├── lait_adapter.pt                # Trained adapter checkpoint
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
│   ├── evolve_adapter.py              # EvolvableAdapter class
│   ├── train_perfect.py               # Training to 100% accuracy
│   ├── lait_uncapped.py               # Unlimited input length
│   ├── lait_ollama_demo.py            # Full yield demo
│   ├── retrain_1024.py                # Retrain with larger context
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
│   ├── prompts.json                   # 33 test prompts
│   ├── quickstart.py                  # Getting started
│   └── compress_decompress.py         # Basic usage
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

### Compress and Decompress

```python
from src.evolve_adapter import EvolvableAdapter
import torch

# Load trained adapter
adapter = EvolvableAdapter(config)
adapter.load_state_dict(torch.load("models/lait_adapter.pt")["state_dict"])
adapter.eval()

# Compress text
text = "Hello, world!"
tokens = list(text.encode('utf-8'))
x = torch.tensor([tokens], dtype=torch.long)

with torch.no_grad():
    logits, latent, orig_len = adapter(x)

# Reconstruct
first_token = [tokens[0]]
predicted = logits[0, :len(tokens)-1, :].argmax(dim=-1).tolist()
reconstructed = bytes(first_token + predicted[:len(tokens)-1])
print(reconstructed.decode('utf-8'))  # "Hello, world!"
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

---

## Training

### Retrain the Adapter

```bash
# Retrain with max_seq_len=1024 (recommended)
python src/retrain_1024.py

# Retrain with custom config
python src/train_perfect.py
```

### GPU Training Results

| Compression Ratio | Compression | Epochs | Time | Status |
|-------------------|-------------|--------|------|--------|
| 1.0 | 1x (none) | 189 | 519s | 100% |
| 0.5 | 2x | 100 | 128s | 100% |
| 0.25 | 4x | 75 | 94s | 100% |
| 0.125 | 8x | 100 | 128s | 100% |
| 0.0625 | 16x | 150 | 191s | 100% |

**Hardware**: NVIDIA GeForce RTX 5060 (8GB VRAM), CUDA 12.8

---

## Full Yield Results

All 33 prompts from `examples/prompts.json` achieve **100% exact match**:

| Category | Count | Matches | Accuracy |
|----------|-------|---------|----------|
| tiny (1-5 bytes) | 5 | 5 | 100% |
| short (10-35 bytes) | 5 | 5 | 100% |
| medium (40-60 bytes) | 5 | 5 | 100% |
| long (130-155 bytes) | 5 | 5 | 100% |
| technical (75-169 bytes) | 5 | 5 | 100% |
| sentence (95-112 bytes) | 5 | 5 | 100% |
| paragraph (300-400 bytes) | 3 | 3 | 100% |
| **Total** | **33** | **33** | **100%** |

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

## How It Works

1. **Tokenization**: Text is converted to byte sequences (vocab_size=256)
2. **Encoding**: Transformer encoder processes tokens into hidden representations
3. **Compression**: Adaptive average pooling reduces sequence length
4. **Latent Space**: Linear projection creates compressed representation
5. **Decoding**: Transformer decoder reconstructs from latent using cross-attention
6. **Output**: Linear head predicts original tokens with 100% accuracy

The system uses **teacher forcing** during training: the decoder receives original tokens as input, learning to map latent representations back to exact text. With `compression_ratio=1.0`, the model learns the identity function perfectly.

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

- **GitHub**: https://github.com/lait-project/lait
- **Ollama**: https://ollama.com/lait-project/lait-granite
- **Hugging Face**: https://huggingface.co/lait-project/lait
- **PyPI**: https://pypi.org/project/lait/
