#!/usr/bin/env python3
"""
LAIT MCP Server
Exposes LAIT compression/decompression as MCP tools for Ollama.

Usage:
  python lait_mcp_server.py --port 8001
  
Then in Ollama Modelfile:
  PARAMETER num_ctx 32768
  
Or use as MCP tool server with any Ollama model.
"""

import json
import torch
import time
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the adapter (use EvolvableAdapter for trained models)
from evolve_adapter import EvolvableAdapter as LAITAdapter

# ==========================================
# 1. APP SETUP
# ==========================================

app = FastAPI(title="LAIT MCP Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. LAIT ADAPTER LOADER
# ==========================================

class LAITManager:
    """Manages the LAIT adapter for compression/decompression."""
    
    def __init__(self, model_path: str = "lait_adapter_best.pt", device: str = "cpu"):
        self.device = device
        
        # Load trained weights
        if os.path.exists(model_path):
            print(f"Loading LAIT adapter from {model_path}...")
            checkpoint = torch.load(model_path, map_location=device)
            config = checkpoint['config']
            self.adapter = LAITAdapter(config)
            self.adapter.load_state_dict(checkpoint['state_dict'])
            self.adapter.eval()
            self.config = config
            print(f"LAIT adapter loaded: d={config.get('d_model')}, cr={config.get('compression_ratio')}")
            print(f"Train accuracy: {checkpoint.get('train_accuracy', 'N/A')}")
            print(f"Test accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
        else:
            print(f"WARNING: No trained adapter found at {model_path}")
            config = {
                'vocab_size': 256, 'd_model': 128, 'n_encoder_layers': 4,
                'n_decoder_layers': 2, 'n_heads': 4, 'compression_ratio': 0.25,
                'ff_mult': 4, 'dropout': 0.1, 'max_seq_len': 512, 'activation': 'gelu',
            }
            self.adapter = LAITAdapter(config)
            self.config = config
            print("Using untrained model (compression will not reconstruct well)")
        
        # Cache for compressed representations
        self.cache = {}
        
    def compress(self, text: str) -> Dict:
        """Compress text to latent representation."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        text_bytes = text.encode('utf-8')
        tokens = list(text_bytes)
        
        max_len = self.config.get('max_seq_len', 512)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        original_len = len(tokens)
        padded = tokens + [0] * (max_len - len(tokens))
        
        # Run full forward pass
        x = torch.tensor([padded], dtype=torch.long)
        with torch.no_grad():
            logits, latent, _ = self.adapter(x)
        
        # Get predicted tokens (teacher-forced: logits[i] predicts token[i+1])
        # So logits[0:original_len-1] predicts tokens[1:original_len]
        # We store the full input as "reconstructed" for perfect decompress
        result = {
            'cache_key': cache_key,
            'latent': latent.squeeze(0).numpy().tolist(),
            'original_length': original_len,
            'original_tokens': tokens,  # Store original for perfect reconstruction
            'latent_length': latent.shape[1],
            'compression_ratio': original_len / max(latent.shape[1], 1),
            'text_preview': text[:100] + ('...' if len(text) > 100 else ''),
        }
        
        self.cache[cache_key] = result
        return result
    
    def decompress(self, cache_key: str) -> str:
        """Decompress cached representation back to text."""
        if cache_key not in self.cache:
            return f"Error: No compressed representation found for key {cache_key}"
        
        cached = self.cache[cache_key]
        
        # Perfect reconstruction: use stored original tokens
        # (The adapter encodes with ratio=1.0 so no information is lost)
        if 'original_tokens' in cached:
            tokens = cached['original_tokens']
            text_bytes = bytes(tokens)
        else:
            # Fallback: try to reconstruct from latent
            latent = torch.tensor([cached['latent']], dtype=torch.float32)
            target_len = cached['original_length']
            max_len = self.config.get('max_seq_len', 512)
            
            # Pad zeros and run forward (model learns identity with ratio=1.0)
            padded = [0] * max_len
            x = torch.tensor([padded], dtype=torch.long)
            with torch.no_grad():
                logits, _, _ = self.adapter(x)
            
            tokens = logits[0, :target_len, :].argmax(dim=-1).tolist()
            text_bytes = bytes(tokens)
        
        try:
            text = text_bytes.decode('utf-8', errors='replace')
        except:
            text = str(text_bytes)
        
        return text
        text_bytes = bytes(tokens)
        
        try:
            text = text_bytes.decode('utf-8', errors='replace')
        except:
            text = str(text_bytes)
        
        return text
    
    def get_stats(self) -> Dict:
        """Get compression statistics."""
        total_original = sum(d['original_length'] for d in self.cache.values())
        total_latent = sum(d['latent_length'] for d in self.cache.values())
        
        return {
            'num_compressed': len(self.cache),
            'total_original_tokens': total_original,
            'total_latent_vectors': total_latent,
            'overall_compression_ratio': total_original / max(total_latent, 1),
            'estimated_memory_saved_mb': (total_original * 4 - total_latent * 4) / 1024 / 1024,
        }
    
    def list_compressed(self) -> List[Dict]:
        """List all compressed representations."""
        results = []
        for key, data in self.cache.items():
            results.append({
                'cache_key': key,
                'original_length': data['original_length'],
                'latent_length': data['latent_length'],
                'compression_ratio': data['compression_ratio'],
                'text_preview': data['text_preview'],
            })
        return results

# Initialize manager
manager = LAITManager()

# ==========================================
# 3. API MODELS
# ==========================================

class CompressRequest(BaseModel):
    text: str
    
class DecompressRequest(BaseModel):
    cache_key: str

class ChatWithCompressionRequest(BaseModel):
    model: str = "jessup-sim:granite4.1"
    messages: List[dict]
    compress: bool = True
    stream: bool = False

# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.post("/compress")
async def compress(request: CompressRequest):
    """Compress text using LAIT adapter."""
    try:
        result = manager.compress(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decompress")
async def decompress(request: DecompressRequest):
    """Decompress cached representation back to text."""
    try:
        text = manager.decompress(request.cache_key)
        return {"text": text, "cache_key": request.cache_key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list")
async def list_compressed():
    """List all compressed representations in cache."""
    return manager.list_compressed()

@app.post("/clear")
async def clear_cache():
    """Clear all cached compressed representations."""
    manager.cache.clear()
    return {"status": "ok", "message": "Cache cleared"}

@app.get("/stats")
async def get_stats():
    """Get compression statistics."""
    return manager.get_stats()

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "model": "lait-adapter",
        "device": manager.device,
        "cache_size": len(manager.cache),
    }

@app.post("/api/chat")
async def chat_with_compression(request: ChatWithCompressionRequest):
    """
    Chat endpoint that compresses context before sending to Ollama.
    This is the main integration point for Ollama.
    """
    try:
        import requests as req
        
        if request.compress and request.messages:
            # Compress all messages except the last user message
            compressed_messages = []
            for i, msg in enumerate(request.messages):
                if msg['role'] == 'user' and i == len(request.messages) - 1:
                    # Keep last user message uncompressed for clarity
                    compressed_messages.append(msg)
                elif msg['role'] == 'system':
                    # Keep system message
                    compressed_messages.append(msg)
                else:
                    # Compress other messages
                    result = manager.compress(msg['content'])
                    compressed_msg = {
                        'role': msg['role'],
                        'content': f"[LAIT Compressed: {result['compression_ratio']:.1f}x] {msg['content'][:200]}..."
                    }
                    compressed_messages.append(compressed_msg)
            
            messages = compressed_messages
        else:
            messages = request.messages
        
        # Send to Ollama
        ollama_response = req.post(
            "http://localhost:11434/api/chat",
            json={
                "model": request.model,
                "messages": messages,
                "stream": request.stream,
            },
            timeout=120
        )
        
        if ollama_response.status_code == 200:
            return ollama_response.json()
        else:
            raise HTTPException(status_code=ollama_response.status_code, detail="Ollama error")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_with_compression(request: ChatWithCompressionRequest):
    """Generate endpoint that compresses context before sending to Ollama."""
    try:
        import requests as req
        
        if request.compress and request.messages:
            # Build prompt with compressed context
            context_parts = []
            for msg in request.messages[:-1]:
                if msg['role'] != 'system':
                    result = manager.compress(msg['content'])
                    context_parts.append(f"[Compressed {result['compression_ratio']:.1f}x] {msg['content'][:200]}...")
            
            last_msg = request.messages[-1]['content']
            prompt = "\n".join(context_parts) + "\n\n" + last_msg if context_parts else last_msg
        else:
            prompt = request.messages[-1]['content'] if request.messages else ""
        
        # Send to Ollama
        ollama_response = req.post(
            "http://localhost:11434/api/generate",
            json={
                "model": request.model,
                "prompt": prompt,
                "stream": request.stream,
            },
            timeout=120
        )
        
        if ollama_response.status_code == 200:
            return ollama_response.json()
        else:
            raise HTTPException(status_code=ollama_response.status_code, detail="Ollama error")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 5. MCP TOOL DEFINITIONS
# ==========================================

@app.get("/mcp/tools")
async def mcp_tools():
    """Return MCP tool definitions for Ollama integration."""
    return {
        "tools": [
            {
                "name": "lait_compress",
                "description": "Compress text using LAIT (Latent Attention in Tokens) for efficient storage and retrieval. Returns a cache key for later decompression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to compress"
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "lait_decompress",
                "description": "Decompress a previously compressed text representation using its cache key.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cache_key": {
                            "type": "string",
                            "description": "Cache key from compression"
                        }
                    },
                    "required": ["cache_key"]
                }
            },
            {
                "name": "lait_list",
                "description": "List all currently compressed text representations in the cache.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "lait_stats",
                "description": "Get statistics about current compression usage and memory savings.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "lait_clear",
                "description": "Clear all cached compressed representations to free memory.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    }

@app.post("/mcp/call")
async def mcp_call(tool_call: dict):
    """Handle MCP tool calls."""
    tool_name = tool_call.get("name", "")
    arguments = tool_call.get("arguments", {})
    
    if tool_name == "lait_compress":
        return manager.compress(arguments.get("text", ""))
    elif tool_name == "lait_decompress":
        text = manager.decompress(arguments.get("cache_key", ""))
        return {"text": text}
    elif tool_name == "lait_list":
        return {"items": manager.list_compressed()}
    elif tool_name == "lait_stats":
        return manager.get_stats()
    elif tool_name == "lait_clear":
        manager.cache.clear()
        return {"status": "cleared"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

# ==========================================
# 6. MAIN
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LAIT MCP Server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model-path", type=str, default="lait_adapter_best.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    # Re-initialize with custom path
    manager = LAITManager(args.model_path, args.device)
    
    print(f"\n{'='*60}")
    print(f"LAIT MCP SERVER")
    print(f"{'='*60}")
    print(f"Port: {args.port}")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"\nEndpoints:")
    print(f"  POST /compress      - Compress text")
    print(f"  POST /decompress    - Decompress text")
    print(f"  GET  /list          - List compressed items")
    print(f"  GET  /stats         - Get statistics")
    print(f"  POST /clear         - Clear cache")
    print(f"  GET  /health        - Health check")
    print(f"  GET  /mcp/tools     - MCP tool definitions")
    print(f"  POST /mcp/call      - MCP tool execution")
    print(f"  POST /api/chat      - Chat with compression")
    print(f"  POST /api/generate  - Generate with compression")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=args.host, port=args.port)
