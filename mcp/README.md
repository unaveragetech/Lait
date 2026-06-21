# LAIT MCP Server

Model Context Protocol (MCP) server exposing LAIT compression as tool services for Ollama and other LLM clients.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python mcp/server.py --port 8001
```

## Available Tools

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `lait_compress` | Compress text to latent | text, compression_ratio | latent tokens, size, stats |
| `lait_decompress` | Decompress latent to text | latent tokens | reconstructed text |
| `lait_list` | List cached compressions | — | list of compression IDs |
| `lait_stats` | Get compression statistics | compression_id | size, ratio, latency |
| `lait_clear` | Clear compression cache | — | confirmation |

## Usage with Ollama

Configure your Ollama client to use the MCP server:

```json
{
  "mcpServers": {
    "lait": {
      "command": "python",
      "args": ["mcp/server.py", "--port", "8001"],
      "transport": "stdio"
    }
  }
}
```

## Usage with Python

```python
import requests

# Compress
r = requests.post("http://localhost:8001/compress", json={
    "text": "Your long document here...",
    "compression_ratio": 0.5
})
latent = r.json()["latent"]

# Decompress
r = requests.post("http://localhost:8001/decompress", json={
    "latent": latent
})
text = r.json()["text"]
```

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/compress` | POST | Compress text |
| `/decompress` | POST | Decompress latent |
| `/list` | GET | List cached compressions |
| `/stats/{id}` | GET | Get compression stats |
| `/clear` | DELETE | Clear cache |
| `/health` | GET | Health check |

## MCP Protocol

The server communicates via stdio using the MCP protocol:

```json
{"jsonrpc": "2.0", "method": "tools/list"}
{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "lait_compress", "arguments": {"text": "Hello!"}}}
```

## Performance

- Compression: ~12ms per 1KB on GPU (RTX 5060)
- Decompression: ~8ms per 1KB
- 100% reconstruction accuracy (all 33 test prompts)

## Files

- `server.py` — Main MCP server
- `chat.py` — Interactive chat with compression
- `adapter.py` — LAIT adapter wrapper
- `requirements.txt` — Dependencies
- `README.md` — This file
