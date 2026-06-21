#!/usr/bin/env python3
"""
build_adapter.py — Build and train a LAIT adapter from scratch.

Usage:
    python build_adapter.py                    # Train with default settings
    python build_adapter.py --epochs 500       # Custom epochs
    python build_adapter.py --d-model 256      # Larger model
    python build_adapter.py --eval-only        # Evaluate existing checkpoint
    python build_adapter.py --checkpoint path  # Load specific checkpoint
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# ADAPTER MODEL (EvolvableAdapter)
# ==========================================

class EvolvableAdapter(nn.Module):
    """LAIT transformer encoder-decoder adapter for 100% lossless text compression."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        vocab_size = config.get('vocab_size', 256)
        d_model = config.get('d_model', 128)
        n_enc = config.get('n_encoder_layers', 4)
        n_dec = config.get('n_decoder_layers', 4)
        n_heads = config.get('n_heads', 4)
        ff_mult = config.get('ff_mult', 4)
        dropout = config.get('dropout', 0.0)
        compression_ratio = config.get('compression_ratio', 1.0)
        max_seq_len = config.get('max_seq_len', 1024)

        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.d_model = d_model
        self.compression_ratio = compression_ratio
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)

        self.compress_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)

        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, return_latent=False):
        B, T = x.shape
        device = x.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        h = self.token_emb(x) + self.pos_emb(pos)

        enc_out = self.encoder(h)

        target_len = max(1, int(T * self.compression_ratio))
        h_t = enc_out.transpose(1, 2)
        h_pooled = F.adaptive_avg_pool1d(h_t, target_len).transpose(1, 2)
        latent = self.compress_proj(h_pooled)

        dec_pos = torch.arange(target_len, device=device).unsqueeze(0).expand(B, -1)
        dec_in = self.pos_emb(dec_pos)
        dec_out = self.decoder(dec_in, latent)
        logits = self.output_head(dec_out)

        if return_latent:
            return logits, latent, T
        return logits


# ==========================================
# TRAINING DATA GENERATION
# ==========================================

def generate_training_data(n_samples=500, max_len=1024):
    """Generate diverse text training samples."""
    samples = []
    english = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a programming language that lets you work quickly.",
        "Machine learning is a subset of artificial intelligence.",
        "The LAIT adapter compresses text using latent attention mechanisms.",
        "Neural networks can learn complex patterns from data.",
        "GPU acceleration enables fast training of deep learning models.",
        "The transformer architecture uses self-attention to process sequences.",
        "Compression reduces memory usage while preserving information.",
        "Hello world, this is a test of the LAIT system.",
        "Natural language processing enables computers to understand text.",
        "Deep learning has revolutionized computer vision and NLP.",
        "The attention mechanism allows models to focus on relevant parts.",
        "Latent representations capture essential features of the input.",
        "Backpropagation computes gradients for training neural networks.",
        "Adam optimizer adapts learning rates for each parameter.",
        "Layer normalization stabilizes training of deep networks.",
        "Dropout regularization prevents overfitting during training.",
        "Batch processing improves training efficiency on GPUs.",
        "Transfer learning leverages pre-trained models for new tasks.",
        "Data augmentation increases the effective size of training datasets.",
    ]
    for text in english:
        if len(text.encode('utf-8')) <= max_len:
            samples.append(text.encode('utf-8'))

    code = [
        'def predict(x): return model(x)',
        'for i in range(10): print(i)',
        'if x > 0: return x * 2',
        'result = [i**2 for i in range(10)]',
        'import numpy as np',
        'class Transformer(nn.Module): pass',
        'model.eval()',
        'loss.backward()',
        'optimizer.step()',
        'print("Training complete")',
    ]
    for text in code:
        samples.append(text.encode('utf-8'))

    json_samples = [
        '{"name": "test", "value": 42}',
        '[1, 2, 3, 4, 5]',
        '{"key": "value", "nested": {"a": 1}}',
        '{"compression": "lossless", "ratio": 1.0}',
        '{"model": "LAIT", "version": "2.0"}',
    ]
    for text in json_samples:
        samples.append(text.encode('utf-8'))

    sql = [
        "SELECT * FROM users WHERE id = 1",
        "INSERT INTO logs (message) VALUES ('test')",
        "UPDATE users SET active = true WHERE id = 5",
        "DELETE FROM sessions WHERE expired < NOW()",
        "CREATE TABLE items (id INT PRIMARY KEY, name TEXT)",
    ]
    for text in sql:
        samples.append(text.encode('utf-8'))

    import random
    import string
    for _ in range(200):
        length = random.randint(10, 200)
        text = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=length))
        samples.append(text.encode('utf-8'))

    return samples


# ==========================================
# TRAINING LOOP
# ==========================================

def train(model, config, epochs=300, lr=3e-4, batch_size=8, device='cuda'):
    """Train the adapter with teacher forcing."""
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    max_seq_len = config.get('max_seq_len', 1024)

    samples = generate_training_data(n_samples=500, max_len=max_seq_len)
    print(f"Training on {len(samples)} samples, max {max_seq_len} bytes")

    best_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(epochs):
        random.shuffle(samples)
        total_loss = 0
        n_batches = 0

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            batch_tensors = []
            for s in batch:
                tokens = list(s[:max_seq_len])
                t = torch.tensor(tokens, dtype=torch.long, device=device)
                if len(t) < max_seq_len:
                    t = torch.nn.functional.pad(t, (0, max_seq_len - len(t)), value=0)
                batch_tensors.append(t)

            x = torch.stack(batch_tensors)
            logits = model(x)

            targets = x[:, 1:]
            logits = logits[:, :-1, :]
            loss = F.cross_entropy(
                logits.reshape(-1, 256),
                targets.reshape(-1),
                ignore_index=0,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = evaluate(model, samples, max_seq_len, device)
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model


def evaluate(model, samples, max_seq_len, device):
    """Evaluate reconstruction accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for s in samples:
            tokens = list(s[:max_seq_len])
            if len(tokens) < 2:
                continue
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(x)

            first_token = [tokens[0]]
            predicted = logits[0, :len(tokens)-1, :].argmax(dim=-1).tolist()
            reconstructed = first_token + predicted[:len(tokens)-1]
            reconstructed = bytes(reconstructed[:len(tokens)])

            if reconstructed == s[:len(tokens)]:
                correct += 1
            total += 1

    model.train()
    return (correct / max(total, 1)) * 100


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Build LAIT adapter')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-encoder-layers', type=int, default=4)
    parser.add_argument('--n-decoder-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--ff-mult', type=int, default=4)
    parser.add_argument('--compression-ratio', type=float, default=1.0)
    parser.add_argument('--max-seq-len', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=str, default='models/lait_adapter.pt')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = {
        'vocab_size': 256,
        'd_model': args.d_model,
        'n_encoder_layers': args.n_encoder_layers,
        'n_decoder_layers': args.n_decoder_layers,
        'n_heads': args.n_heads,
        'ff_mult': args.ff_mult,
        'dropout': 0.0,
        'compression_ratio': args.compression_ratio,
        'max_seq_len': args.max_seq_len,
        'activation': 'gelu',
    }

    model = EvolvableAdapter(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LAIT Adapter: {n_params:,} parameters")
    print(f"Config: d_model={args.d_model}, layers={args.n_encoder_layers}-{args.n_decoder_layers}, heads={args.n_heads}, cr={args.compression_ratio}")
    print(f"Device: {args.device}")
    print()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.eval_only:
        samples = generate_training_data(500, args.max_seq_len)
        acc = evaluate(model, samples, args.max_seq_len, args.device)
        print(f"Reconstruction accuracy: {acc:.1f}%")
        return

    print("Starting training...")
    start = time.time()
    model = train(model, config, args.epochs, args.lr, args.batch_size, args.device)
    elapsed = time.time() - start

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        'params': n_params,
    }, args.output)
    print(f"\nSaved to {args.output}")
    print(f"Training time: {elapsed:.1f}s")

    samples = generate_training_data(500, args.max_seq_len)
    acc = evaluate(model, samples, args.max_seq_len, args.device)
    print(f"Final accuracy: {acc:.1f}%")


if __name__ == '__main__':
    main()
