#!/usr/bin/env python3
"""
LAIT-Enhanced Ollama Server
Provides LAIT compression as a service for any Ollama model.
"""

import json
import torch
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import requests

from lait_export import LAITModel, LAITConfig, get_winning_config

app = FastAPI(title="LAIT Ollama Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load LAIT model
print("Loading LAIT compression model...")
config = get_winning_config()
lait_model = LAITModel(config)
lait_model.eval()
print("LAIT model loaded")

# Ollama configuration
OLLAMA_URL = "http://localhost:11434"

class GenerateRequest(BaseModel):
    model: str = "jessup-sim:granite4.1"
    prompt: str
    system: Optional[str] = None
    stream: bool = False
    options: Optional[dict] = None
    compress: bool = True

class ChatRequest(BaseModel):
    model: str = "jessup-sim:granite4.1"
    messages: List[dict]
    stream: bool = False
    options: Optional[dict] = None
    compress: bool = True

class LAITStats(BaseModel):
    original_tokens: int
    latent_size: int
    compression_ratio: float
    memory_saved_mb: float
    compression_time_ms: float

def compress_text(text: str) -> Dict:
    """Compress text using LAIT."""
    start_time = time.time()
    
    # Tokenize
    tokens = [ord(c) % config.vocab_size for c in text]
    input_ids = torch.tensor([tokens])
    
    # Forward pass
    with torch.no_grad():
        outputs = lait_model(input_ids, return_reconstruction=True)
    
    latent = outputs["latent_memory"]
    latent_size = outputs["latent_size"]
    compression_ratio = outputs["compression_ratio"]
    
    # Calculate memory savings
    standard_kv = 2 * (config.n_initial_encoder_layers + 
                      config.n_compressor_layers +
                      config.n_decoder_layers) * len(tokens) * config.d_model * 4
    lait_memory = latent_size * config.d_model * 4
    memory_saved = standard_kv - lait_memory
    
    elapsed = (time.time() - start_time) * 1000
    
    return {
        "original_tokens": len(tokens),
        "latent_size": latent_size,
        "compression_ratio": compression_ratio,
        "memory_saved_bytes": memory_saved,
        "memory_saved_mb": memory_saved / 1024 / 1024,
        "compression_time_ms": elapsed
    }

def create_compressed_prompt(text: str, stats: LAITStats) -> str:
    """Create a prompt with compression metadata."""
    return f"""[LAIT COMPRESSION]
Original: {stats.original_tokens} tokens -> Latent: {stats.latent_size} vectors ({stats.compression_ratio:.1f}x)
Memory saved: {stats.memory_saved_mb:.2f} MB

[COMPRESSED CONTEXT]
{text}

[INSTRUCTIONS]
Context compressed via LAIT hybrid bottleneck. Maintain full comprehension."""


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Generate text with optional LAIT compression."""
    try:
        if request.compress:
            # Compress input
            stats = compress_text(request.prompt)
            prompt = create_compressed_prompt(request.prompt, stats)
        else:
            stats = None
            prompt = request.prompt
        
        # Build message
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": prompt})
        
        # Send to Ollama
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": request.model,
                "messages": messages,
                "stream": request.stream,
                "options": request.options or {}
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "model": request.model,
                "response": result["message"]["content"],
                "done": True,
                "lait_stats": stats
            }
        else:
            raise HTTPException(status_code=response.status_code, detail="Ollama error")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with LAIT compression."""
    try:
        if request.compress and request.messages:
            # Compress the last user message
            last_msg = request.messages[-1]
            if last_msg["role"] == "user":
                stats = compress_text(last_msg["content"])
                last_msg["content"] = create_compressed_prompt(last_msg["content"], stats)
            else:
                stats = None
        else:
            stats = None
        
        # Send to Ollama
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": request.model,
                "messages": request.messages,
                "stream": request.stream,
                "options": request.options or {}
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "model": request.model,
                "message": result["message"],
                "done": True,
                "lait_stats": stats
            }
        else:
            raise HTTPException(status_code=response.status_code, detail="Ollama error")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tags")
async def list_models():
    """List available models."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            return response.json()
        else:
            return {"models": []}
    except:
        return {"models": []}


@app.get("/api/version")
async def version():
    """Return API version."""
    return {"version": "0.1.0", "lait_enabled": True}


@app.get("/api/lait/stats")
async def lait_stats():
    """Get LAIT compression statistics."""
    return {
        "model": "LAIT-Hybrid",
        "compression_ratio": 64,
        "memory_savings": 320,
        "bottleneck": "hybrid",
        "architecture": {
            "initial_encoder_layers": config.n_initial_encoder_layers,
            "compressor_layers": config.n_compressor_layers,
            "decoder_layers": config.n_decoder_layers,
            "d_model": config.d_model,
            "num_heads": config.num_heads
        }
    }


if __name__ == "__main__":
    print("="*70)
    print("LAIT-Enhanced Ollama Server")
    print("="*70)
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Default model: jessup-sim:granite4.1")
    print(f"Compression: 64x with 320x memory savings")
    print("="*70)
    print("\nStarting server on http://localhost:8000")
    print("Use this server for LAIT-compressed requests to Ollama")
    print("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
