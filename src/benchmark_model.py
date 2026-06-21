#!/usr/bin/env python3
"""
benchmark_model.py — Benchmark an Ollama model with LAIT compression integration.

Tests:
  1. Raw model (no compression)
  2. LAIT compressed input → model → reconstruct
  3. Comparison of quality, latency, tokens/sec

Usage:
    python benchmark_model.py --model "hf.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF:Q4_K_M"
"""

import argparse
import json
import time
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ==========================================
# LAIT ADAPTER (inline for self-contained benchmark)
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvolvableAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.get('vocab_size', 256)
        d_model = config.get('d_model', 128)
        n_enc = config.get('n_encoder_layers', 4)
        n_dec = config.get('n_decoder_layers', 4)
        n_heads = config.get('n_heads', 4)
        ff_mult = config.get('ff_mult', 4)
        dropout = config.get('dropout', 0.0)
        self.compression_ratio = config.get('compression_ratio', 1.0)
        self.max_seq_len = config.get('max_seq_len', 1024)
        self.d_model = d_model
        use_relu = config.get('activation', 'gelu') == 'relu'

        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True,
            activation='relu' if use_relu else 'gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)

        self.compress_proj = nn.Linear(d_model, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True,
            activation='relu' if use_relu else 'gelu',
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)
        self.output_head = nn.Linear(d_model, vocab_size)

    def encode(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.encoder(h)
        target_size = max(1, int(T * self.compression_ratio))
        h = h.transpose(1, 2)
        h = F.adaptive_avg_pool1d(h, target_size)
        h = h.transpose(1, 2)
        h = self.compress_proj(h)
        return h

    def decode(self, latent, target_len):
        B, L, C = latent.shape
        positions = torch.arange(target_len, device=latent.device).unsqueeze(0).expand(B, -1)
        target_emb = self.pos_emb(positions)
        h = self.decoder(target_emb, latent)
        return self.output_head(h)

    def forward(self, x):
        original_len = x.shape[1]
        latent = self.encode(x)
        logits = self.decode(latent, original_len)
        return logits, latent, original_len


# ==========================================
# BENCHMARK
# ==========================================

OLLAMA_URL = "http://localhost:11434"

TEST_PROMPTS = [
    ("tiny", "Hi"),
    ("short", "Hello world!"),
    ("medium", "The quick brown fox jumps over the lazy dog."),
    ("long", "The LAIT adapter compresses text into latent representations using a transformer encoder-decoder architecture with adaptive pooling bottleneck."),
    ("code", "def predict(x): return model(x)"),
    ("json", '{"name": "test", "value": 42}'),
    ("sentence", "This is a sentence used to test the compression adapter."),
]


def load_adapter(checkpoint_path, device='cuda'):
    """Load trained LAIT adapter."""
    config = {
        'vocab_size': 256, 'd_model': 128, 'n_encoder_layers': 4,
        'n_decoder_layers': 4, 'n_heads': 4, 'ff_mult': 4, 'dropout': 0.0,
        'compression_ratio': 1.0, 'max_seq_len': 1024, 'activation': 'gelu',
    }
    model = EvolvableAdapter(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    # Remove _orig_mod prefix if present
    clean_state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()
    return model, config


def compress_text(adapter, text, device='cuda'):
    """Compress text through LAIT adapter, return latent tokens."""
    tokens = list(text.encode('utf-8'))
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, latent, orig_len = adapter(x, return_latent=False) if hasattr(adapter(x), 'shape') else (adapter(x), None, len(tokens))
    # Actually get latent
    with torch.no_grad():
        B, T = x.shape
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        h = adapter.token_emb(x) + adapter.pos_emb(pos)
        enc_out = adapter.encoder(h)
        target_len = max(1, int(T * adapter.compression_ratio))
        h_t = enc_out.transpose(1, 2)
        h_pooled = F.adaptive_avg_pool1d(h_t, target_len).transpose(1, 2)
        latent = adapter.compress_proj(h_pooled)
    return latent, len(tokens)


def decompress_latent(adapter, logits, orig_len, device='cuda'):
    """Reconstruct text from model output logits."""
    first_token = logits[0, 0, :].argmax().item()
    predicted = logits[0, :orig_len-1, :].argmax(dim=-1).tolist()
    reconstructed = bytes([first_token] + predicted[:orig_len-1])
    return reconstructed


def call_ollama(model, prompt, timeout=120):
    """Call Ollama API and return response + timing."""
    start = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 256}
        }, timeout=timeout)
        elapsed = time.time() - start
        data = resp.json()
        return {
            "response": data.get("response", ""),
            "total_duration_ms": data.get("total_duration", 0) / 1e6,
            "eval_count": data.get("eval_count", 0),
            "eval_duration_ms": data.get("eval_duration", 0) / 1e6,
            "tokens_per_sec": data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 0.001),
            "wall_time_ms": elapsed * 1000,
        }
    except Exception as e:
        return {"response": "", "error": str(e)}


def run_benchmark(model_name, adapter_path, device='cuda'):
    """Run full benchmark comparing raw vs LAIT-compressed input."""
    print(f"=" * 70)
    print(f"LAIT BENCHMARK: {model_name}")
    print(f"=" * 70)

    # Load adapter
    print("\nLoading LAIT adapter...")
    adapter, config = load_adapter(adapter_path, device)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"Adapter: {n_params:,} params, device={device}")

    # Warmup
    print("Warming up adapter...")
    with torch.no_grad():
        dummy = torch.tensor([[72, 101, 108]], dtype=torch.long, device=device)
        _ = adapter(dummy)

    results = {"model": model_name, "adapter": adapter_path, "prompts": []}
    total_raw_time = 0
    total_lait_time = 0
    total_matches = 0

    for category, prompt in TEST_PROMPTS:
        print(f"\n{'-' * 50}")
        print(f"[{category}] \"{prompt}\"")

        # 1. Raw model
        print("  Raw model...", end=" ", flush=True)
        raw = call_ollama(model_name, prompt)
        raw_time = raw.get("wall_time_ms", 0)
        total_raw_time += raw_time
        print(f"{raw_time:.0f}ms | {raw.get('eval_count', 0)} tokens | {raw.get('tokens_per_sec', 0):.1f} tok/s")

        # 2. LAIT compressed
        tokens = list(prompt.encode('utf-8'))
        latent_size = 0
        lait_time = 0
        reconstructed_text = ""

        if len(tokens) <= 1024:
            # Compress + reconstruct
            t0 = time.time()
            with torch.no_grad():
                x = torch.tensor([tokens], dtype=torch.long, device=device)
                logits, latent, orig_len = adapter(x)
                # Teacher-forced: logits[i] predicts token[i+1]
                # So reconstructed = [tokens[0]] + logits[0, :T-1].argmax()
                first_token = [tokens[0]]
                predicted = logits[0, :len(tokens)-1, :].argmax(dim=-1).tolist()
                reconstructed = bytes(first_token + predicted[:len(tokens)-1])
                latent_size = latent.shape[1]
            compress_time = (time.time() - t0) * 1000

            # Reconstruction match check
            match = reconstructed == bytes(tokens)
            total_matches += int(match)

            # Send reconstructed text to model for quality comparison
            reconstructed_text = reconstructed.decode('utf-8', errors='replace')
            print(f"  LAIT -> model...", end=" ", flush=True)
            lait = call_ollama(model_name, reconstructed_text)
            lait_time = lait.get("wall_time_ms", 0) + compress_time
            total_lait_time += lait_time
            print(f"{lait_time:.0f}ms | compress={compress_time:.1f}ms | latent={latent_size}B | {'MATCH' if match else 'MISMATCH'}")
            print(f"  Response: {lait.get('response', '')[:120]}...")
        else:
            print(f"  Skipping LAIT (input {len(tokens)}B > 1024 max)")

        results["prompts"].append({
            "category": category,
            "prompt": prompt,
            "input_bytes": len(tokens),
            "raw_time_ms": raw_time,
            "lait_time_ms": lait_time,
            "latent_bytes": latent_size,
            "raw_response": raw.get("response", "")[:200],
            "lait_response": reconstructed_text[:200] if reconstructed_text else "",
        })

    # Summary
    n = len(TEST_PROMPTS)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Model:            {model_name}")
    print(f"Prompts tested:   {n}")
    print(f"Reconstruction:   {total_matches}/{n} ({total_matches/n*100:.0f}%)")
    print(f"Avg raw latency:  {total_raw_time/n:.0f}ms")
    print(f"Avg LAIT latency: {total_lait_time/n:.0f}ms")
    print(f"Total raw time:   {total_raw_time:.0f}ms")
    print(f"Total LAIT time:  {total_lait_time:.0f}ms")

    # Save results
    output_path = f"benchmark_{model_name.split('/')[-1].split(':')[0].replace('.', '_')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Ollama model with LAIT')
    parser.add_argument('--model', type=str,
                        default="hf.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF:Q4_K_M",
                        help='Ollama model name')
    parser.add_argument('--adapter', type=str, default='models/lait_adapter.pt',
                        help='Path to LAIT adapter checkpoint')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_benchmark(args.model, args.adapter, args.device)


if __name__ == '__main__':
    main()
