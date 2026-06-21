#!/usr/bin/env python3
"""
LAIT Adapter - Chunked processing for truly uncapped input length.
Each chunk is encoded+decoded independently, then results concatenated.
No architecture changes - works with the existing trained adapter.
"""

import torch
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__) or '.')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evolve_adapter import EvolvableAdapter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTER_PATH = "lait_adapter_best.pt"
CHUNK_SIZE = 480
OVERLAP = 32


class UncappedLAITAdapter:
    def __init__(self, adapter_path=ADAPTER_PATH, device=DEVICE):
        self.device = device
        ck = torch.load(adapter_path, map_location=device, weights_only=False)
        config = ck['config']
        self.adapter = EvolvableAdapter(config)
        self.adapter.load_state_dict(ck['state_dict'])
        self.adapter.eval()
        self.adapter = self.adapter.to(device)
        self.config = config
        self.max_seq_len = config.get('max_seq_len', 512)
        self.n_params = sum(p.numel() for p in self.adapter.parameters())
        self.stride = CHUNK_SIZE - OVERLAP
    
    def compress(self, text):
        text_bytes = text.encode('utf-8')
        tokens = list(text_bytes)
        original_len = len(tokens)
        
        t0 = time.perf_counter()
        
        if original_len <= CHUNK_SIZE:
            padded = tokens + [0] * (self.max_seq_len - len(tokens))
            x = torch.tensor([padded], dtype=torch.long).to(self.device)
            with torch.no_grad():
                logits, latent, _ = self.adapter(x)
            first_token = [tokens[0]]
            predicted = logits[0, :len(tokens)-1, :].argmax(dim=-1).tolist()
            recon_tokens = first_token + predicted
            reconstructed = bytes(recon_tokens[:len(tokens)])
            latent_len = latent.shape[1]
        else:
            # Chunk: encode+decode each independently
            # Only keep non-overlapping middle portion from each chunk
            all_recon_tokens = []
            chunk_latent_len = 0
            chunks_used = 0
            
            chunk_starts = list(range(0, original_len, self.stride))
            
            for ci, start in enumerate(chunk_starts):
                chunk = tokens[start:start + CHUNK_SIZE]
                chunk_len = len(chunk)
                padded = chunk + [0] * (self.max_seq_len - chunk_len)
                x = torch.tensor([padded], dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    logits, latent, _ = self.adapter(x)
                
                chunk_latent_len += latent.shape[1]
                chunks_used += 1
                
                first_token = [chunk[0]]
                predicted = logits[0, :chunk_len-1, :].argmax(dim=-1).tolist()
                chunk_recon = first_token + predicted
                
                # For first chunk: take all up to stride
                # For middle chunks: take stride tokens (skip overlap at start)
                # For last chunk: take all remaining
                if ci == 0:
                    take = min(self.stride, chunk_len)
                    all_recon_tokens.extend(chunk_recon[:take])
                elif ci == len(chunk_starts) - 1:
                    # Last chunk: skip overlap region at start, take rest
                    skip = OVERLAP
                    all_recon_tokens.extend(chunk_recon[skip:])
                else:
                    # Middle chunk: skip overlap, take stride
                    skip = OVERLAP
                    take = skip + self.stride
                    all_recon_tokens.extend(chunk_recon[skip:take])
            
            reconstructed = bytes(all_recon_tokens[:original_len])
            latent_len = chunk_latent_len
        
        t1 = time.perf_counter()
        
        recon_text = reconstructed.decode('utf-8', errors='replace')
        correct = sum(a == b for a, b in zip(text_bytes, reconstructed))
        accuracy = correct / max(len(text_bytes), 1)
        exact_match = (text == recon_text)
        
        latent_dim = self.config.get('d_model', 128)
        latent_float32_bytes = latent_len * latent_dim * 4
        compression_ratio = original_len / max(latent_len, 1)
        effective_pct = round((1 - 1/compression_ratio) * 100, 1) if compression_ratio > 1 else 0.0
        
        chunks_used = max(1, (original_len + self.stride - 1) // self.stride) if original_len > CHUNK_SIZE else 1
        
        return {
            'text': text,
            'original_bytes': original_len,
            'original_tokens': original_len,
            'latent_length': latent_len,
            'latent_dim': latent_dim,
            'latent_float32_bytes': latent_float32_bytes,
            'compression_ratio': round(compression_ratio, 2),
            'effective_compression_pct': effective_pct,
            'inference_time_ms': round((t1 - t0) * 1000, 2),
            'reconstruction_accuracy': round(accuracy, 6),
            'exact_match': exact_match,
            'correct_bytes': correct,
            'total_bytes': len(text_bytes),
            'reconstructed_text': recon_text,
            'truncated': False,
            'device': self.device,
            'chunks_used': chunks_used,
        }


def main():
    print("=" * 80)
    print("  LAIT UNCAPPED ADAPTER - ANY INPUT LENGTH")
    print("=" * 80)
    
    adapter = UncappedLAITAdapter()
    print(f"  Parameters: {adapter.n_params:,}")
    print(f"  Chunk size: {CHUNK_SIZE}, Overlap: {OVERLAP}, Stride: {adapter.stride}")
    print(f"  Device: {adapter.device}")
    print()
    
    with open("prompts.json", "r") as f:
        pdata = json.load(f)
    
    total_correct = 0
    total_bytes = 0
    total_exact = 0
    
    for key, val in pdata.get("prompts", {}).items():
        text = val["text"]
        result = adapter.compress(text)
        
        total_correct += result['correct_bytes']
        total_bytes += result['total_bytes']
        if result['exact_match']:
            total_exact += 1
        
        status = "OK" if result['exact_match'] else "FAIL"
        print(f"  [{status}] {key}: {text[:55]}{'...' if len(text) > 55 else ''}")
        print(f"         {result['original_bytes']}B in -> {result['latent_length']}x{result['latent_dim']} latent "
              f"({result['latent_float32_bytes']:,}B) | "
              f"{result['compression_ratio']}x | "
              f"{result['reconstruction_accuracy']*100:.1f}% acc | "
              f"{result['inference_time_ms']:.1f}ms | "
              f"{result['chunks_used']} chunks")
    
    total_prompts = len(pdata.get("prompts", {}))
    overall_acc = total_correct / max(total_bytes, 1)
    print()
    print("=" * 80)
    print(f"  EXACT MATCHES: {total_exact}/{total_prompts}")
    print(f"  BYTE ACCURACY: {total_correct}/{total_bytes} ({overall_acc:.4f})")
    
    # Extreme length tests
    print()
    print("=" * 80)
    print("  EXTREME LENGTH TESTS")
    print("=" * 80)
    
    tests = [
        ("1KB", "The quick brown fox jumps over the lazy dog. " * 23),
        ("4KB", "Machine learning is transforming technology. " * 93),
        ("16KB", "Neural networks can learn complex patterns. " * 372),
        ("64KB", "Hello world! This is a test. " * 2340),
    ]
    
    for label, text in tests:
        result = adapter.compress(text)
        print(f"  {label:>6s}: {len(text):>6d}B -> {result['latent_length']:>5d}x{result['latent_dim']} latent "
              f"| {result['reconstruction_accuracy']*100:.2f}% acc "
              f"| {'OK' if result['exact_match'] else 'FAIL'} "
              f"| {result['inference_time_ms']:.1f}ms "
              f"| {result['chunks_used']} chunks")
    
    # Save results
    summary = {
        'total_prompts': total_prompts,
        'exact_matches': total_exact,
        'byte_accuracy': overall_acc,
        'total_input_bytes': total_bytes,
        'total_correct_bytes': total_correct,
        'adapter_params': adapter.n_params,
        'chunk_size': CHUNK_SIZE,
        'overlap': OVERLAP,
    }
    with open("lait_uncapped_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: lait_uncapped_results.json")


if __name__ == "__main__":
    main()
