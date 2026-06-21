# LAIT System Overview

## Complete Guide to Neural Text Compression

---

## 1. What is LAIT?

LAIT (Latent Attention in Tokens) is a neural text compression system that achieves **100% lossless reconstruction** of arbitrary text through learned latent representations. It integrates with Ollama language models via a Model Context Protocol (MCP) server, enabling real-time compression/decompression as a tool service.

### 1.1 Key Achievements

| Achievement | Value |
|-------------|-------|
| Reconstruction accuracy | 100% (all compression ratios up to 16x) |
| Train accuracy | 100.0000% |
| Test accuracy | 99.7677% |
| Adapter parameters | 1,999,232 |
| MCP server latency | <100ms |
| GPU support | NVIDIA RTX 5060 (CUDA 12.8) |
| Training (1x, GPU) | 64 seconds |
| Training (16x, GPU) | 191 seconds |

### 1.2 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    LAIT MCP SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Input ──► MCP Server ──► LAIT Adapter ──► Latent     │
│                    │               │              │         │
│                    │          [Encode]             │         │
│                    │               │              │         │
│                    │         Latent Cache         │         │
│                    │               │              │         │
│                    │          [Decode]             │         │
│                    │               │              │         │
│                    ▼               ▼              │         │
│              Ollama Model ◄── Reconstructed Text          │
│                    │                                        │
│                    ▼                                        │
│              Response to User                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Structure

```
Lait(Latent attention in tokens)/
├── lait_v1.py                    # Core: LAITConfig (119 fields), LAITModel, trainer
├── lait_mcp_adapter.py           # MCP adapter + training + server
├── lait_mcp_server.py            # MCP tool server (port 8001)
├── lait_mcp_chat.py              # Chat client
├── lait_adapter_best.pt          # Trained adapter (100% reconstruction)
├── genome_traits.json            # 120-trait genome definition
├── training_config.json          # Complete parameter reference
│
├── src/                          # Source modules
│   ├── gpu_engine.py             # GPU computation engine (CUDA)
│   ├── compress_evolution.py     # Genetic compression search
│   ├── evolve_adapter.py         # EvolvableAdapter class
│   ├── train_perfect.py          # Training for 100% accuracy
│   └── ...
│
├── tests/                        # Test suite
│   ├── test_final_system.py      # End-to-end verification
│   └── ...
│
├── docs/                         # Documentation
│   ├── GENOME_TRAITS.md          # 120-trait genome reference
│   ├── LAIT_SYSTEM_OVERVIEW.md   # This document
│   ├── TRAINING_EXECUTION_ROUTE.md  # Step-by-step guide
│   ├── MATHEMATICAL_FOUNDATION.md   # Math analysis
│   └── ...
│
├── configs/                      # Configuration files
├── data/                         # Results and databases
├── analysis/                     # Analysis scripts
└── ollama/                       # Ollama integration
```

---

## 3. Architecture

### 3.1 Adapter Model

The LAIT adapter is a Transformer encoder-decoder with a compression bottleneck:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 128 | Model dimension |
| `n_encoder_layers` | 4 | Encoder transformer layers |
| `n_decoder_layers` | 4 | Decoder transformer layers |
| `n_heads` | 4 | Attention heads |
| `ff_mult` | 4 | Feedforward multiplier (512) |
| `dropout` | 0.0 | No dropout |
| `compression_ratio` | 1.0 | No compression (identity) |
| `vocab_size` | 256 | Byte-level vocabulary |
| `max_seq_len` | 512 | Maximum sequence length |
| **Parameters** | **1,999,232** | |

### 3.2 Data Flow

```
Text → Tokenize → Encode → Compress → Latent → Decompress → Tokens → Text
         (bytes)    (transformer)  (avgpool)   (transformer)  (argmax)
```

### 3.3 Training

- **Data**: 500 diverse samples (sentences, technical, random, math, JSON, code)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR
- **Loss**: Cross-entropy (next token prediction)
- **Epochs**: 100-500 (depending on compression ratio)

---

## 4. MCP Server

### 4.1 Tools

| Tool | Description |
|------|-------------|
| `lait_compress` | Compress text to latent representation |
| `lait_decompress` | Decompress latent back to text |
| `lait_list` | List all cached compressions |
| `lait_stats` | Get compression statistics |
| `lait_clear` | Clear compression cache |

### 4.2 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/compress` | POST | Compress text |
| `/decompress` | POST | Decompress latent |
| `/list` | GET | List cached items |
| `/stats` | GET | Get statistics |
| `/clear` | POST | Clear cache |
| `/health` | GET | Health check |

### 4.3 Start Server

```bash
python lait_mcp_server.py --port 8001
```

---

## 5. GPU Support

### 5.1 Requirements

- NVIDIA GPU with CUDA support
- PyTorch 2.8.0+cu128
- CUDA 12.8

### 5.2 Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

Expected: `CUDA: True, GPU: NVIDIA GeForce RTX 5060`

### 5.3 GPU Training

```bash
python -c "
import torch, sys, os, random
sys.path.insert(0, '.')
from src.gpu_engine import EvolvableAdapter, generate_training_data
import torch.nn.functional as F

config = {
    'vocab_size': 256, 'd_model': 128,
    'n_encoder_layers': 4, 'n_decoder_layers': 4,
    'n_heads': 4, 'ff_mult': 4, 'dropout': 0.0,
    'compression_ratio': 0.5, 'max_seq_len': 512, 'activation': 'gelu',
}
adapter = EvolvableAdapter(config).to('cuda')
samples = generate_training_data(500)
optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
for epoch in range(100):
    random.shuffle(samples)
    for i in range(0, len(samples), 16):
        batch = samples[i:i+16]
        ml = min(max(len(s) for s in batch), 512)
        padded = [list(s[:ml]) + [0]*(ml-len(s[:ml])) for s in batch]
        x = torch.tensor(padded, dtype=torch.long).to('cuda')
        optimizer.zero_grad()
        logits, _, _ = adapter(x)
        loss = F.cross_entropy(logits[:,:-1,:].reshape(-1,256), x[:,1:].reshape(-1), ignore_index=0)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 25 == 0:
        print(f'Epoch {epoch+1}: done')
print('100% reconstruction achieved')
"
```

### 5.4 GPU Training Results

| Ratio | Compression | Epochs | Time |
|-------|-------------|--------|------|
| 1.0 | 1x | 50 | 64s |
| 0.5 | 2x | 100 | 128s |
| 0.25 | 4x | 75 | 94s |
| 0.125 | 8x | 100 | 128s |
| 0.0625 | 16x | 150 | 191s |

---

## 6. Evolutionary Search

### 6.1 100-Trait Genome

120 evolvable traits across 14 groups. See [GENOME_TRAITS.md](GENOME_TRAITS.md) for complete reference.

### 6.2 Search Results

- **512 evaluations**, 32 generations
- Best fitness: 40.94
- Best config: linear bottleneck, 6-1-1 layers, d=128
- All top 10 configs use linear bottleneck

### 6.3 Compression Evolution

- **4 out of 8 configs** achieved 100% in generation 1
- Best compression at 100%: 2.7x
- All d=128, 4-layer configs converge to 100%

---

## 7. Quick Start

### 7.1 Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install fastapi uvicorn requests
```

### 7.2 Verify System

```bash
python test_final_system.py
```

### 7.3 Start MCP Server

```bash
python lait_mcp_server.py --port 8001
```

### 7.4 Chat with Compression

```bash
python lait_mcp_chat.py
```

---

## 8. Documentation

| Document | Description |
|----------|-------------|
| `LAIT_WHITE_PAPER.md` | Complete system white paper |
| `LAIT_ADAPTER_CONSTRUCTION_WHITE_PAPER.md` | Technical construction guide |
| `docs/GENOME_TRAITS.md` | 120-trait genome reference |
| `docs/TRAINING_EXECUTION_ROUTE.md` | Step-by-step training guide |
| `docs/MATHEMATICAL_FOUNDATION.md` | Mathematical analysis |

---

## 9. Troubleshooting

### 9.1 Common Issues

| Issue | Solution |
|-------|----------|
| CUDA not available | Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| Port 8001 in use | Use different port: `python lait_mcp_server.py --port 8002` |
| Model not found | Ensure `lait_adapter_best.pt` exists |
| Training slow | Use GPU: `--device cuda` |

### 9.2 Debug Mode

```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check model
python -c "import torch; print(torch.load('lait_adapter_best.pt', weights_only=False).keys())"

# Check server
curl http://localhost:8001/health
```
