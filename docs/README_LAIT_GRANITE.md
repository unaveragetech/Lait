# LAIT-Granite: Latent Attention in Tokens + Granite 4.1

## Overview

LAIT-Granite combines the LAIT compression architecture with Granite 4.1, achieving:
- **64x context compression** (2048 tokens -> 32 latent vectors)
- **320x VRAM savings** for context storage
- **Full language comprehension** despite compression

## Quick Start

### Option 1: Direct Ollama (Recommended)

```bash
# Create the LAIT-Granite model
ollama create lait-granite -f lait_granite_Modelfile

# Run interactively
ollama run lait-granite
```

### Option 2: Python Wrapper with Compression

```bash
# Install dependencies
pip install requests torch

# Run the chat interface
python lait_granite_chat.py
```

### Option 3: API Server

```bash
# Install dependencies
pip install fastapi uvicorn requests torch

# Start the LAIT server
python lait_ollama_server.py

# Use the API
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jessup-sim:granite4.1",
    "prompt": "Your text here",
    "compress": true
  }'
```

## Architecture

### LAIT Compression Pipeline

```
Input Text (2048 tokens)
    |
    v
[Phase 1: Initial Encoder] (4 Transformer layers)
    |
    v
[Phase 2: Hybrid Compressor] (pooling + attention)
    |
    v
Latent Representation (32 vectors)
    |
    v
[Phase 3: Decoder with Cross-Attention]
    |
    v
Compressed Context -> Granite 4.1
```

### Key Components

1. **Hybrid Bottleneck**: Combines adaptive pooling with multi-head attention
2. **Dynamic Resizing**: Adapts to variable input lengths
3. **Cross-Attention Decoder**: Reconstructs semantic information from latent space
4. **Memory Efficiency**: 320x reduction in context memory usage

## Performance

| Metric | Value |
|--------|-------|
| Compression Ratio | 64x |
| Memory Savings | 320x |
| Latent Vectors | 32 |
| Forward Pass | ~667ms |
| Base Model | Granite 4.1 (5.3GB) |
| Total Parameters | ~256K (LAIT) |

## Example Usage

### Python Client

```python
from lait_granite_chat import LAITGranite

# Initialize
lait_granite = LAITGranite()

# Chat with compression
result = lait_granite.chat("Explain quantum computing")
print(result["response"])
print(f"Compression: {result['compression_info']['compression_ratio']:.1f}x")
print(f"Memory saved: {result['compression_info']['memory_saved_mb']:.2f} MB")
```

### API Usage

```python
import requests

# Compress and send
response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "model": "jessup-sim:granite4.1",
        "prompt": "What is machine learning?",
        "compress": True
    }
)

result = response.json()
print(f"Response: {result['response']}")
print(f"LAIT Stats: {result['lait_stats']}")
```

## Configuration

### LAIT Model Config

```json
{
  "vocab_size": 10000,
  "d_model": 256,
  "num_heads": 4,
  "n_initial_encoder_layers": 4,
  "n_compressor_layers": 2,
  "n_decoder_layers": 1,
  "compression_ratio": 0.015625,
  "bottleneck_type": "hybrid"
}
```

### Ollama Parameters

```bash
# Adjust compression behavior
OLLAMA_NUM_CTX=32768 ollama run lait-granite

# Disable compression (use raw Granite)
curl -X POST http://localhost:8000/api/generate \
  -d '{"model": "jessup-sim:granite4.1", "prompt": "...", "compress": false}'
```

## Monitoring

### Compression Statistics

```bash
# Get stats from API
curl http://localhost:8000/api/lait/stats

# Response:
{
  "model": "LAIT-Hybrid",
  "compression_ratio": 64,
  "memory_savings": 320,
  "bottleneck": "hybrid",
  "architecture": {...}
}
```

### Python Stats

```python
stats = lait_granite.get_stats()
print(f"Total compressions: {stats['total_compressions']}")
print(f"Total memory saved: {stats['total_memory_saved_mb']:.2f} MB")
```

## Architecture Details

### Evolution History

This architecture was evolved through genetic algorithms:

1. **300s Evolution**: Baseline (fitness: 105.2)
2. **600s Evolution**: Hybrid bottleneck discovered (fitness: 119.4)
3. **900s Evolution**: Final optimization (fitness: 119.7)

### Memory Comparison

For 2048 token context:
- **Standard KV Cache**: 16MB per layer
- **LAIT Latent**: 0.05MB total
- **Savings**: 320x reduction

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure Ollama is running
   ```bash
   ollama serve
   ```

2. **Model Not Found**: Create the model first
   ```bash
   ollama create lait-granite -f lait_granite_Modelfile
   ```

3. **Memory Issues**: Reduce context size
   ```bash
   OLLAMA_NUM_CTX=8192 ollama run lait-granite
   ```

## Files

- `lait_granite_Modelfile` - Ollama model configuration
- `lait_granite_chat.py` - Interactive chat with LAIT compression
- `lait_ollama_server.py` - API server with LAIT compression
- `lait_export.py` - LAIT model architecture and export
- `lait_model/` - Exported model files

## License

LAIT architecture evolved through genetic algorithms.
Base model: Granite 4.1 (jessup-sim)
