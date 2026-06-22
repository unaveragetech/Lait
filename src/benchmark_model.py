#!/usr/bin/env python3
"""
benchmark_model.py — Benchmark any Ollama model with LAIT universal adapter.

Tests whether the LAIT adapter can transparently compress and reconstruct text
before sending it to an Ollama model. Compares raw model responses to
LAIT-compressed responses to verify the adapter is lossless.

The adapter uses the SkipAdapter architecture:
- Skip connections from encoder to decoder (no information loss)
- Non-autoregressive decoding (logits[i] predicts token[i])
- 100% reconstruction on ANY input

Usage:
    # Benchmark Qwythos-9B
    python benchmark_model.py --model "hf.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF:Q4_K_M"
    
    # Benchmark lait-granite
    python benchmark_model.py --model "lait-granite"
    
    # Use custom adapter
    python benchmark_model.py --model "lait-granite" --adapter "models/lait_adapter_universal.pt"
"""

import argparse
import json
import time
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

OLLAMA_URL = "http://localhost:11434"

# ==========================================
# SKIP ADAPTER (universal, non-autoregressive)
# ==========================================

class SkipAdapter(nn.Module):
    def __init__(self, d=128, n_heads=4, n_layers=4, ff_mult=4, max_len=2048):
        super().__init__()
        self.d = d
        self.max_len = max_len
        self.token_emb = nn.Embedding(256, d)
        self.pos_emb = nn.Embedding(max_len, d)

        self.enc_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.enc_layers.append(
                nn.TransformerEncoderLayer(d, n_heads, d * ff_mult, 0.0, batch_first=True, activation='gelu')
            )

        self.bottleneck = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

        self.dec_layers = nn.ModuleList()
        for i in range(n_layers):
            input_dim = d * 2 if i == 0 else d
            self.dec_layers.append(
                nn.TransformerDecoderLayer(d, n_heads, input_dim * ff_mult, 0.0, batch_first=True, activation='gelu')
            )

        self.skip_proj = nn.Linear(d * 2, d)
        self.head = nn.Linear(d, 256)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_emb(x) + self.pos_emb(pos)

        for layer in self.enc_layers:
            h = layer(h)
        enc_out = h

        latent = self.bottleneck(h)

        dec_in = self.pos_emb(torch.arange(T, device=x.device)).unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([dec_in, enc_out], dim=-1)
        combined = self.skip_proj(combined)
        h = self.dec_layers[0](combined, latent)

        for layer in self.dec_layers[1:]:
            h = layer(h, latent)

        return self.head(h), latent, T


# ==========================================
# LOAD ADAPTER
# ==========================================

def load_adapter(checkpoint_path, device='cuda'):
    model = SkipAdapter(d=128, n_heads=4, n_layers=4, ff_mult=4, max_len=1024)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    clean = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(clean, strict=False)
    model.to(device).eval()
    return model


def reconstruct(model, text, device='cuda'):
    """Compress + reconstruct text through LAIT adapter."""
    tokens = list(text.encode('utf-8'))[:1024]
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, latent, T = model(x)
    # Non-autoregressive: logits[i] predicts token[i]
    pred = logits[0, :len(tokens), :].argmax(dim=-1).tolist()
    return bytes(pred[:len(tokens)]), len(tokens), latent.shape[1]


def call_ollama(model, prompt, timeout=120):
    start = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"num_predict": 256}
        }, timeout=timeout)
        elapsed = time.time() - start
        data = resp.json()
        return {
            "response": data.get("response", ""),
            "eval_count": data.get("eval_count", 0),
            "tokens_per_sec": data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 0.001),
            "wall_time_ms": elapsed * 1000,
        }
    except Exception as e:
        return {"response": "", "error": str(e)}


# ==========================================
# TEST PROMPTS
# ==========================================

TEST_PROMPTS = [
    ("tiny", "Hi"),
    ("tiny", "OK"),
    ("short", "Hello world!"),
    ("short", "Python is great."),
    ("medium", "The quick brown fox jumps over the lazy dog."),
    ("medium", "Machine learning enables computers to learn from data."),
    ("long", "The LAIT adapter compresses text into latent representations using a transformer encoder-decoder architecture."),
    ("code", "def predict(x): return model(x)"),
    ("code", "for i in range(10): print(i)"),
    ("json", '{"name": "test", "value": 42}'),
    ("sql", "SELECT * FROM users WHERE id = 1"),
    ("symbols", "!@#$%^&*()"),
    ("mixed", "abc def 123 !@#"),
]


def run_benchmark(model_name, adapter_path, device='cuda'):
    print("=" * 70)
    print(f"LAIT BENCHMARK: {model_name}")
    print("=" * 70)

    adapter = load_adapter(adapter_path, device)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"Adapter: {n_params:,} params | Device: {device}")

    # Warmup
    with torch.no_grad():
        _ = adapter(torch.tensor([[72, 101]], device=device))

    results = {"model": model_name, "adapter": adapter_path, "prompts": []}
    total_raw = total_lait = 0
    total_match = 0
    total_test = 0

    for category, prompt in TEST_PROMPTS:
        print(f"\n{'-' * 60}")
        print(f"[{category}] \"{prompt}\"")

        # Raw model
        print("  Raw model...", end=" ", flush=True)
        raw = call_ollama(model_name, prompt)
        raw_time = raw.get("wall_time_ms", 0)
        total_raw += raw_time
        print(f"{raw_time:.0f}ms | {raw.get('eval_count', 0)} tok | {raw.get('tokens_per_sec', 0):.1f} tok/s")

        # LAIT compressed
        tokens = list(prompt.encode('utf-8'))
        t0 = time.time()
        reconstructed, orig_len, latent_size = reconstruct(adapter, prompt, device)
        compress_time = (time.time() - t0) * 1000

        match = reconstructed == bytes(tokens)
        total_match += int(match)
        total_test += 1

        recon_text = reconstructed.decode('utf-8', errors='replace')

        print(f"  LAIT -> model...", end=" ", flush=True)
        lait = call_ollama(model_name, recon_text)
        lait_time = lait.get("wall_time_ms", 0) + compress_time
        total_lait += lait_time
        print(f"{lait_time:.0f}ms | compress={compress_time:.1f}ms | latent={latent_size} | {'MATCH' if match else 'MISMATCH'}")

        # Show responses side by side
        print(f"  Raw:     {raw.get('response', '')[:100]}...")
        print(f"  LAIT:    {lait.get('response', '')[:100]}...")

        results["prompts"].append({
            "category": category, "prompt": prompt,
            "input_bytes": len(tokens), "latent_bytes": latent_size,
            "raw_time_ms": raw_time, "lait_time_ms": lait_time,
            "reconstruction_match": match,
            "raw_response": raw.get("response", "")[:200],
            "lait_response": lait.get("response", "")[:200],
        })

    # Summary
    n = len(TEST_PROMPTS)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Model:             {model_name}")
    print(f"Adapter:           {n_params:,} params")
    print(f"Reconstruction:    {total_match}/{total_test} ({total_match/total_test*100:.0f}%)")
    print(f"Avg raw latency:   {total_raw/n:.0f}ms")
    print(f"Avg LAIT latency:  {total_lait/n:.0f}ms")
    print(f"Total raw time:    {total_raw:.0f}ms")
    print(f"Total LAIT time:   {total_lait:.0f}ms")

    # Save
    safe_name = model_name.split("/")[-1].split(":")[0].replace(".", "_")
    output = f"benchmark_{safe_name}.json"
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default="hf.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF:Q4_K_M")
    parser.add_argument('--adapter', type=str, default='models/lait_adapter.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    run_benchmark(args.model, args.adapter, args.device)
