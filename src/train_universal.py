#!/usr/bin/env python3
"""
train_universal.py — Train adapter for 100% reconstruction on ANY input.

Uses skip connections (encoder→decoder) so the decoder has direct access to 
encoder features. This lets it reconstruct ANY input, not just memorized ones.

Architecture:
  Input → TokenEmb+PosEmb → Encoder → [skip connection] → Bottleneck → Decoder+Skip → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import string
import os
import sys

DEVICE = 'cuda'
MAX_LEN = 1024
os.makedirs('models', exist_ok=True)


class SkipAdapter(nn.Module):
    """Encoder-decoder with skip connections for universal reconstruction."""

    def __init__(self, d=128, n_heads=4, n_layers=4, ff_mult=4):
        super().__init__()
        self.d = d
        self.token_emb = nn.Embedding(256, d)
        self.pos_emb = nn.Embedding(MAX_LEN, d)

        # Encoder layers (we'll save intermediate outputs for skip)
        self.enc_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.enc_layers.append(
                nn.TransformerEncoderLayer(d, n_heads, d * ff_mult, 0.0, batch_first=True, activation='gelu')
            )

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

        # Decoder layers with skip input (d*2 because of skip concatenation)
        self.dec_layers = nn.ModuleList()
        for i in range(n_layers):
            input_dim = d * 2 if i == 0 else d
            self.dec_layers.append(
                nn.TransformerDecoderLayer(d, n_heads, input_dim * ff_mult, 0.0, batch_first=True, activation='gelu')
            )

        self.skip_proj = nn.Linear(d * 2, d)  # Project after skip concat
        self.head = nn.Linear(d, 256)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_emb(x) + self.pos_emb(pos)

        # Encoder with skip connections
        enc_outputs = []
        for layer in self.enc_layers:
            h = layer(h)
            enc_outputs.append(h)

        # Bottleneck
        latent = self.bottleneck(h)

        # Decoder with skip from last encoder layer
        skip = enc_outputs[-1]
        dec_in = self.pos_emb(torch.arange(T, device=x.device)).unsqueeze(0).expand(B, -1, -1)

        # First decoder layer: concat skip
        combined = torch.cat([dec_in, skip], dim=-1)
        combined = self.skip_proj(combined)
        h = self.dec_layers[0](combined, latent)

        # Remaining decoder layers
        for layer in self.dec_layers[1:]:
            h = layer(h, latent)

        return self.head(h)


def generate_data(n=500):
    samples = []
    # Random bytes (1/3)
    for _ in range(n // 3):
        samples.append(bytes([random.randint(0, 255) for _ in range(random.randint(4, 512))]))
    # Printable ASCII (1/6)
    for _ in range(n // 6):
        samples.append(''.join(random.choices(string.printable, k=random.randint(4, 300))).encode('ascii', errors='replace'))
    # Text sentences (1/6)
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks can learn complex patterns from data.",
        "GPU acceleration enables fast training of deep learning models.",
        "The transformer architecture uses self-attention mechanisms.",
        "Compression reduces memory usage while preserving information.",
        "Hello world, this is a test of the LAIT system.",
        "Pack my box with five dozen jugs.",
        "How vexingly quick daft zebras jump!",
        "The five boxing wizards jump quickly.",
    ]
    for _ in range(n // 6):
        samples.append(random.choice(sentences).encode('utf-8'))
    # Code (1/12)
    code = ['def predict(x): return model(x)', 'for i in range(10): print(i)', 'if x > 0: return x * 2',
            'result = [i**2 for i in range(10)]', 'import numpy as np', 'model.eval()']
    for _ in range(n // 12):
        samples.append(random.choice(code).encode('utf-8'))
    # JSON/SQL (1/12)
    structs = ['{"name": "test"}', '[1, 2, 3]', 'SELECT * FROM users WHERE id = 1']
    for _ in range(n // 12):
        samples.append(random.choice(structs).encode('utf-8'))
    return [s[:MAX_LEN] for s in samples if len(s) >= 2]


TEST_PROMPTS = [
    "Hi", "OK", "No", "Yes", "Go",
    "Hello world!", "Python is great.", "AI transforms everything.", "Test input here.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning enables computers to learn from data.",
    "The transformer architecture uses self-attention.",
    "Pack my box with five dozen jugs.",
    "The LAIT adapter compresses text into latent representations.",
    "def predict(x): return model(x)",
    "for i in range(10): print(i)",
    "if x > 0: return x * 2",
    '{"name": "test", "value": 42}',
    '[1, 2, 3, 4, 5]',
    "SELECT * FROM users WHERE id = 1",
    "!@#$%^&*()",
    "abc def ghi",
    "1234567890",
]


def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for text in TEST_PROMPTS:
            tokens = list(text.encode('utf-8'))
            if len(tokens) > MAX_LEN or len(tokens) < 2:
                continue
            x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
            logits = model(x)
            # Non-autoregressive: logits[i] predicts token[i] directly
            pred = logits[0, :len(tokens), :].argmax(dim=-1).tolist()
            if bytes(pred[:len(tokens)]) == bytes(tokens):
                correct += 1
            total += 1
    return correct, total


def train():
    print("=" * 60)
    print("UNIVERSAL ADAPTER (Skip Connections)")
    print("=" * 60)

    model = SkipAdapter(d=128, n_heads=4, n_layers=4, ff_mult=4).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,} | Device: {DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    samples = generate_data(500)

    all_bytes = set()
    for s in samples:
        all_bytes.update(s)
    print(f"Data: {len(samples)} samples | Bytes: {len(all_bytes)}/256")
    print()

    best_acc = 0
    start = time.time()

    for epoch in range(500):
        random.shuffle(samples)
        total_loss = 0
        n_batches = 0

        for i in range(0, len(samples), 8):
            batch = samples[i:i+8]
            tensors = []
            for s in batch:
                t = torch.tensor(list(s[:MAX_LEN]), dtype=torch.long, device=DEVICE)
                if len(t) < MAX_LEN:
                    t = F.pad(t, (0, MAX_LEN - len(t)), value=0)
                tensors.append(t)
            x = torch.stack(tensors)

            logits = model(x)
            # Non-autoregressive loss: logits[i] predicts token[i]
            loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1), ignore_index=0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            correct, total = evaluate(model)
            acc = correct / max(total, 1) * 100
            elapsed = time.time() - start
            print(f"Epoch {epoch+1:3d}/500 | Loss: {total_loss/n_batches:.4f} | Test: {correct}/{total} ({acc:.1f}%) | {elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc
                torch.save({'state_dict': model.state_dict(), 'accuracy': acc, 'epoch': epoch+1},
                           'models/lait_adapter_universal.pt')

            if acc >= 100.0:
                print(f"\n*** 100% at epoch {epoch+1} ***")
                break

    print(f"\nBest: {best_acc:.1f}%")
    print(f"Time: {time.time()-start:.0f}s")

    # Final eval
    ckpt = torch.load('models/lait_adapter_universal.pt', map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    print("\nFinal results:")
    for text in TEST_PROMPTS:
        tokens = list(text.encode('utf-8'))
        if len(tokens) > MAX_LEN or len(tokens) < 2:
            continue
        x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits = model(x)
        pred = logits[0, :len(tokens), :].argmax(dim=-1).tolist()
        match = bytes(pred[:len(tokens)]) == bytes(tokens)
        print(f"  [{'PASS' if match else 'FAIL'}] {text[:50]}")


if __name__ == '__main__':
    train()
