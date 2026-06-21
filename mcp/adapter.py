#!/usr/bin/env python3
"""
LAIT MCP Adapter
Teaches Ollama models how to reconstruct text via compressed latent representations.
Uses a trained LAIT encoder-decoder as an MCP tool server.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ==========================================
# 1. LAIT ADAPTER MODEL (Compact Version)
# ==========================================

class LAITAdapter(nn.Module):
    """
    Compact LAIT adapter for text compression/decompression.
    Trained to reconstruct text from compressed latent representations.
    """
    
    def __init__(
        self,
        vocab_size: int = 256,  # ASCII bytes
        d_model: int = 128,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        n_heads: int = 4,
        compression_ratio: float = 0.125,  # 8x compression
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_encoder_layers': n_encoder_layers,
            'n_decoder_layers': n_decoder_layers,
            'n_heads': n_heads,
            'compression_ratio': compression_ratio,
            'max_seq_len': max_seq_len,
        }
        
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        
        # Compression projection
        self.compress_proj = nn.Linear(d_model, d_model)
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu',
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # Output head
        self.output_head = nn.Linear(d_model, vocab_size)
        
        # Compression ratio parameter
        self.compression_ratio = compression_ratio
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tokens to compressed latent representation."""
        B, T = x.shape
        
        # Embed tokens with positional encoding
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_emb(x) + self.pos_emb(positions)
        
        # Encode
        h = self.encoder(h)
        
        # Compress: pool to target size
        target_size = max(1, int(T * self.compression_ratio))
        h = h.transpose(1, 2)  # B, C, T
        h = F.adaptive_avg_pool1d(h, target_size)  # B, C, target_size
        h = h.transpose(1, 2)  # B, target_size, C
        
        # Project
        h = self.compress_proj(h)
        
        return h
    
    def decode(self, latent: torch.Tensor, target_len: int) -> torch.Tensor:
        """Decode latent representation back to token logits."""
        B, L, C = latent.shape
        
        # Create target positions
        positions = torch.arange(target_len, device=latent.device).unsqueeze(0).expand(B, -1)
        target_emb = self.pos_emb(positions)
        
        # Decode latent against target positions
        # Both batch_first=True, so shapes: (B, T, d_model)
        h = self.decoder(target_emb, latent)
        
        # Project to vocabulary
        logits = self.output_head(h)
        
        return logits
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Full forward pass: encode -> decode.
        Returns: (logits, latent, original_len)
        """
        original_len = x.shape[1]
        latent = self.encode(x)
        logits = self.decode(latent, original_len)
        return logits, latent, original_len
    
    def compress(self, text: str) -> Dict:
        """Compress text to latent representation. Returns metadata."""
        # Convert text to bytes
        text_bytes = text.encode('utf-8')
        tokens = list(text_bytes)
        
        # Truncate if needed
        if len(tokens) > self.config['max_seq_len']:
            tokens = tokens[:self.config['max_seq_len']]
        
        # Convert to tensor
        x = torch.tensor([tokens], dtype=torch.long)
        
        # Encode
        with torch.no_grad():
            latent = self.encode(x)
        
        return {
            'latent': latent.squeeze(0).numpy().tolist(),
            'original_length': len(tokens),
            'latent_length': latent.shape[1],
            'compression_ratio': len(tokens) / max(latent.shape[1], 1),
            'text_preview': text[:100] + ('...' if len(text) > 100 else ''),
        }
    
    def decompress(self, latent_list: List[List[float]], target_len: int) -> str:
        """Decompress latent representation back to text."""
        # Convert to tensor
        latent = torch.tensor([latent_list], dtype=torch.float32)
        
        # Decode
        with torch.no_grad():
            logits = self.decode(latent, target_len)
        
        # Get predicted tokens
        tokens = logits.argmax(dim=-1).squeeze(0).tolist()
        
        # Convert bytes to text
        text_bytes = bytes(tokens)
        try:
            text = text_bytes.decode('utf-8', errors='replace')
        except:
            text = str(text_bytes)
        
        return text


# ==========================================
# 2. RECONSTRUCTION TRAINER
# ==========================================

class ReconstructionTrainer:
    """Trains the LAIT adapter to reconstruct text from compressed representations."""
    
    def __init__(self, adapter: LAITAdapter, device: str = 'cpu'):
        self.adapter = adapter.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def generate_training_data(self, num_samples: int = 1000) -> List[str]:
        """Generate training data: diverse text samples."""
        samples = []
        
        # Common English sentences
        sentence_patterns = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming how we process information.",
            "Latent attention mechanisms enable efficient context compression.",
            "Neural networks can learn to reconstruct text from compressed representations.",
            "The cat sat on the mat and looked out the window.",
            "Artificial intelligence is changing the world in profound ways.",
            "Compression ratios of 8x or higher are achievable with modern architectures.",
            "The early bird catches the worm, but the second mouse gets the cheese.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "In the beginning was the Word, and the Word was with God.",
            "The only thing we have to fear is fear itself.",
            "Ask not what your country can do for you, ask what you can do for your country.",
            "I think therefore I am.",
            "Knowledge is power.",
            "Time flies like an arrow; fruit flies like a banana.",
            "The quick brown fox jumps over the lazy dog near the riverbank.",
            "Scientists have discovered a new method for compressing neural network activations.",
            "The adaptation layer learns to map between token space and latent space.",
            "Reconstruction accuracy improves with more training epochs.",
        ]
        
        # Generate variations
        for _ in range(num_samples // 20):
            base = sentence_patterns[_ % len(sentence_patterns)]
            # Add variation
            variations = [
                base,
                base.lower(),
                base.upper(),
                f"Note: {base}",
                f"Important: {base}",
                f"Context: {base}",
                f"Summary: {base}",
                f"Question: {base}?",
                f"Answer: {base}.",
                f"Point 1: {base} Point 2: {base}",
            ]
            samples.extend(variations)
        
        # Add random technical text
        technical_samples = [
            "d_model=128, n_heads=4, compression_ratio=0.125",
            "Loss: 0.2345, Accuracy: 0.8901, Latent size: 64",
            "Epoch 50/100, Batch 128/256, Learning rate: 0.001",
            "Memory usage: 256MB, Compression time: 12ms",
            "Input tokens: 512, Latent vectors: 64, Output tokens: 512",
            "Transformer encoder: 4 layers, Decoder: 2 layers",
            "Adam optimizer with weight decay 0.01",
            "Gradient clipping at 1.0",
            "Cosine annealing schedule with warmup",
            "Mixed precision training enabled",
        ]
        samples.extend(technical_samples * (num_samples // 20))
        
        return samples[:num_samples]
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.adapter.train()
        
        # Generate training data
        texts = self.generate_training_data(200)
        
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        for i in range(0, len(texts), 8):
            batch_texts = texts[i:i+8]
            
            # Tokenize
            batch_tokens = []
            for text in batch_texts:
                tokens = list(text.encode('utf-8'))[:self.adapter.config['max_seq_len']]
                batch_tokens.append(tokens)
            
            # Pad to same length
            max_len = max(len(t) for t in batch_tokens)
            padded = [t + [0] * (max_len - len(t)) for t in batch_tokens]
            x = torch.tensor(padded, dtype=torch.long).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, latent, orig_len = self.adapter(x)
            
            # Reconstruction loss (predict next token)
            targets = x[:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            preds = logits.argmax(dim=-1)
            mask = targets != 0
            acc = (preds[mask] == targets[mask]).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
        
        self.scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'lr': self.scheduler.get_last_lr()[0],
        }
    
    def train(self, num_epochs: int = 50) -> Dict:
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"LAIT ADAPTER RECONSTRUCTION TRAINING")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.adapter.parameters()):,}")
        print(f"{'='*60}\n")
        
        history = {'loss': [], 'accuracy': []}
        best_acc = 0
        
        for epoch in range(num_epochs):
            metrics = self.train_epoch(epoch)
            history['loss'].append(metrics['loss'])
            history['accuracy'].append(metrics['accuracy'])
            
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                # Save best model
                self.save('lait_adapter_best.pt')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Acc={metrics['accuracy']:.4f}, "
                      f"LR={metrics['lr']:.6f}")
        
        print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")
        return history
    
    def save(self, path: str):
        """Save adapter weights."""
        torch.save({
            'config': self.adapter.config,
            'state_dict': self.adapter.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load adapter weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.adapter.load_state_dict(checkpoint['state_dict'])


# ==========================================
# 3. MCP TOOL SERVER
# ==========================================

class LAITMCPServer:
    """
    MCP (Model Context Protocol) server for LAIT compression.
    Exposes compression/decompression as tools for Ollama models.
    """
    
    def __init__(self, adapter: LAITAdapter, device: str = 'cpu'):
        self.adapter = adapter.to(device)
        self.adapter.eval()
        self.device = device
        self.compressed_cache = {}  # Store compressed representations
        
    def compress_tool(self, text: str) -> Dict:
        """MCP Tool: Compress text to latent representation."""
        # Generate cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        if cache_key in self.compressed_cache:
            return self.compressed_cache[cache_key]
        
        # Compress
        result = self.adapter.compress(text)
        result['cache_key'] = cache_key
        
        # Store in cache
        self.compressed_cache[cache_key] = result
        
        return result
    
    def decompress_tool(self, cache_key: str) -> str:
        """MCP Tool: Decompress cached representation back to text."""
        if cache_key not in self.compressed_cache:
            return f"Error: No compressed representation found for key {cache_key}"
        
        cached = self.compressed_cache[cache_key]
        latent = cached['latent']
        target_len = cached['original_length']
        
        text = self.adapter.decompress(latent, target_len)
        return text
    
    def list_compressed_tool(self) -> List[Dict]:
        """MCP Tool: List all compressed representations in cache."""
        results = []
        for key, data in self.compressed_cache.items():
            results.append({
                'cache_key': key,
                'original_length': data['original_length'],
                'latent_length': data['latent_length'],
                'compression_ratio': data['compression_ratio'],
                'text_preview': data['text_preview'],
            })
        return results
    
    def clear_cache_tool(self) -> str:
        """MCP Tool: Clear all cached compressed representations."""
        self.compressed_cache.clear()
        return "Cache cleared"
    
    def get_stats_tool(self) -> Dict:
        """MCP Tool: Get compression statistics."""
        total_original = sum(d['original_length'] for d in self.compressed_cache.values())
        total_latent = sum(d['latent_length'] for d in self.compressed_cache.values())
        
        return {
            'num_compressed': len(self.compressed_cache),
            'total_original_tokens': total_original,
            'total_latent_vectors': total_latent,
            'overall_compression_ratio': total_original / max(total_latent, 1),
            'estimated_memory_saved_mb': (total_original * 4 - total_latent * 4) / 1024 / 1024,
        }


# ==========================================
# 4. FASTAPI MCP SERVER
# ==========================================

def create_mcp_app(adapter: LAITAdapter, device: str = 'cpu'):
    """Create FastAPI app for MCP server."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    app = FastAPI(title="LAIT MCP Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    mcp = LAITMCPServer(adapter, device)
    
    class CompressRequest(BaseModel):
        text: str
    
    class DecompressRequest(BaseModel):
        cache_key: str
    
    @app.post("/compress")
    async def compress(request: CompressRequest):
        try:
            result = mcp.compress_tool(request.text)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/decompress")
    async def decompress(request: DecompressRequest):
        try:
            text = mcp.decompress_tool(request.cache_key)
            return {"text": text, "cache_key": request.cache_key}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/list")
    async def list_compressed():
        return mcp.list_compressed_tool()
    
    @app.post("/clear")
    async def clear_cache():
        return mcp.clear_cache_tool()
    
    @app.get("/stats")
    async def get_stats():
        return mcp.get_stats_tool()
    
    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "lait-adapter"}
    
    return app


# ==========================================
# 5. MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LAIT MCP Adapter")
    parser.add_argument("--train", action="store_true", help="Train the adapter")
    parser.add_argument("--serve", action="store_true", help="Start MCP server")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    # Create adapter
    adapter = LAITAdapter(
        vocab_size=256,
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=2,
        n_heads=4,
        compression_ratio=0.125,
        max_seq_len=2048,
    )
    
    if args.train:
        # Train the adapter
        trainer = ReconstructionTrainer(adapter, args.device)
        
        # Load existing weights if available
        if os.path.exists('lait_adapter_best.pt'):
            print("Loading existing adapter weights...")
            trainer.load('lait_adapter_best.pt')
        
        history = trainer.train(num_epochs=args.epochs)
        
        # Save final model
        trainer.save('lait_adapter_final.pt')
        print(f"Adapter saved to lait_adapter_final.pt")
        
    elif args.serve:
        # Load trained adapter
        if os.path.exists('lait_adapter_best.pt'):
            print("Loading trained adapter...")
            checkpoint = torch.load('lait_adapter_best.pt', map_location=args.device)
            adapter.load_state_dict(checkpoint['state_dict'])
        else:
            print("WARNING: No trained adapter found. Using untrained model.")
        
        # Start MCP server
        import uvicorn
        app = create_mcp_app(adapter, args.device)
        print(f"\nStarting LAIT MCP Server on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    else:
        # Demo mode
        print("\n" + "="*60)
        print("LAIT MCP ADAPTER DEMO")
        print("="*60)
        
        # Compress
        text = "The quick brown fox jumps over the lazy dog. This is a test of the LAIT compression system."
        print(f"\nOriginal text ({len(text)} chars):")
        print(f"  {text}")
        
        result = adapter.compress(text)
        print(f"\nCompressed:")
        print(f"  Original length: {result['original_length']} tokens")
        print(f"  Latent length: {result['latent_length']} vectors")
        print(f"  Compression ratio: {result['compression_ratio']:.1f}x")
        
        # Decompress and compare at byte level
        latent_tensor = torch.tensor([result['latent']], dtype=torch.float32)
        logits = adapter.decode(latent_tensor, result['original_length'])
        pred_tokens = logits.argmax(dim=-1).squeeze(0).tolist()
        recon_bytes = bytes(pred_tokens)
        orig_bytes = text.encode('utf-8')
        
        # Byte similarity
        min_len = min(len(orig_bytes), len(recon_bytes))
        if min_len > 0:
            similarity = sum(a == b for a, b in zip(orig_bytes[:min_len], recon_bytes[:min_len])) / min_len
            print(f"\nByte similarity: {similarity:.1%}")
            print(f"  Original bytes: {len(orig_bytes)}")
            print(f"  Reconstructed bytes: {len(recon_bytes)}")
        
        # Show first 40 bytes comparison
        print(f"\nByte comparison (first 40):")
        for i in range(min(40, min(len(orig_bytes), len(recon_bytes)))):
            ob = orig_bytes[i]
            rb = recon_bytes[i]
            oc = chr(ob) if 32 <= ob < 127 else '.'
            rc = chr(rb) if 32 <= rb < 127 else '.'
            match = '=' if ob == rb else 'X'
            print(f"  {i:3d}: 0x{ob:02x}='{oc}' -> 0x{rb:02x}='{rc}' {match}")
        
        print("\n" + "="*60)
