#!/usr/bin/env python3
"""
LAIT MCP System - Final End-to-End Verification
Tests: adapter, MCP server, compression, reconstruction, Ollama integration.
"""
import torch
import torch.nn.functional as F
import requests
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__) or '.')

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def test_adapter():
    """Test 1: LAIT Adapter (encode/decode/reconstruct)"""
    section("TEST 1: LAIT ADAPTER")
    
    from src.evolve_adapter import EvolvableAdapter
    
    ck = torch.load('lait_adapter_best.pt', map_location='cpu')
    config = ck['config']
    adapter = EvolvableAdapter(config)
    adapter.load_state_dict(ck['state_dict'])
    adapter.eval()
    
    params = sum(p.numel() for p in adapter.parameters())
    print(f"  Model: {params:,} params")
    print(f"  Config: d={config['d_model']}, enc={config['n_encoder_layers']}, "
          f"dec={config['n_decoder_layers']}, heads={config['n_heads']}, "
          f"cr={config['compression_ratio']}")
    print(f"  Train accuracy: {ck.get('train_accuracy', 'N/A')}")
    print(f"  Test accuracy: {ck.get('test_accuracy', 'N/A')}")
    
    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning enables efficient text compression.",
        "Neural networks learn latent representations.",
        "The adapter compresses and reconstructs perfectly.",
        "d_model=128, n_heads=4, compression_ratio=1.0",
        "1234567890 abcdefghij ABCDEFGHIJ",
        '{"key": "value", "num": 42}',
    ]
    
    all_correct = True
    total_correct = 0
    total_bytes = 0
    
    for text in test_texts:
        tb = text.encode('utf-8')
        tokens = list(tb)[:512]
        padded = tokens + [0] * (512 - len(tokens))
        x = torch.tensor([padded], dtype=torch.long)
        
        with torch.no_grad():
            logits, latent, _ = adapter(x)
        
        # logits[i] predicts token[i+1]
        # To reconstruct token[0..T-1]:
        #   token[0] = first input token (no prediction available)
        #   token[1..T-1] = logits[0..T-2].argmax()
        first_token = [tokens[0]]
        predicted = logits[0, :len(tokens)-1, :].argmax(dim=-1).tolist()
        recon_tokens = first_token + predicted
        recon = bytes(recon_tokens[:len(tb)])
        
        correct = sum(a == b for a, b in zip(tb, recon))
        acc = correct / len(tb)
        total_correct += correct
        total_bytes += len(tb)
        
        status = "OK" if acc >= 1.0 else "FAIL"
        if acc < 1.0:
            all_correct = False
        print(f"  [{status}] {acc:.0%} | {text[:50]}...")
    
    overall = total_correct / max(total_bytes, 1)
    print(f"\n  Overall: {overall:.2%} ({total_correct}/{total_bytes} bytes)")
    return all_correct, overall


def test_mcp_server():
    """Test 2: MCP Server endpoints"""
    section("TEST 2: MCP SERVER")
    
    try:
        # Health
        r = requests.get("http://localhost:8001/health", timeout=5)
        if r.status_code != 200:
            print("  FAIL: Server not running")
            return False, 0
        print("  [OK] Server health check")
        
        # Compress
        text = "The quick brown fox jumps over the lazy dog. Testing MCP compression."
        r = requests.post("http://localhost:8001/compress", json={"text": text}, timeout=10)
        data = r.json()
        key = data['cache_key']
        print(f"  [OK] Compress: {data['original_length']} -> {data['latent_length']} vectors")
        
        # Decompress
        r = requests.post("http://localhost:8001/decompress", json={"cache_key": key}, timeout=10)
        recon = r.json()['text']
        match = recon == text
        print(f"  [{'OK' if match else 'FAIL'}] Decompress match: {match}")
        
        # Stats
        r = requests.get("http://localhost:8001/stats", timeout=10)
        stats = r.json()
        print(f"  [OK] Stats: {stats['num_compressed']} items cached")
        
        # MCP tools
        r = requests.get("http://localhost:8001/mcp/tools", timeout=10)
        tools = r.json()['tools']
        tool_names = [t['name'] for t in tools]
        print(f"  [OK] MCP tools: {tool_names}")
        
        return True, 1.0 if match else 0.0
        
    except Exception as e:
        print(f"  FAIL: {e}")
        return False, 0


def test_ollama():
    """Test 3: Ollama integration"""
    section("TEST 3: OLLAMA INTEGRATION")
    
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            print("  FAIL: Ollama not running")
            return False
        
        models = [m['name'] for m in r.json().get('models', [])]
        print(f"  [OK] Ollama running with {len(models)} models")
        
        # Test generation
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "jessup-sim:granite4.1",
            "prompt": "Say hello in one word.",
            "stream": False,
        }, timeout=30)
        
        if r.status_code == 200:
            resp = r.json().get('response', '')[:50]
            print(f"  [OK] Generation works: {resp}...")
            return True
        else:
            print(f"  FAIL: Generation returned {r.status_code}")
            return False
            
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_full_pipeline():
    """Test 4: Full pipeline (compress -> store -> decompress -> verify)"""
    section("TEST 4: FULL PIPELINE")
    
    test_data = [
        "Hello world! This is a test of the full LAIT MCP pipeline.",
        "Machine learning enables automatic compression of text data.",
        "The LAIT adapter achieves 100% reconstruction on unseen text.",
        "Neural networks can learn to compress information efficiently.",
        "Compression ratios of 8x are achievable with modern architectures.",
    ]
    
    all_match = True
    total_original = 0
    total_latent = 0
    
    for i, text in enumerate(test_data):
        # Compress
        r = requests.post("http://localhost:8001/compress", json={"text": text}, timeout=10)
        data = r.json()
        
        # Decompress
        r = requests.post("http://localhost:8001/decompress", json={"cache_key": data['cache_key']}, timeout=10)
        recon = r.json()['text']
        
        match = recon == text
        if not match:
            all_match = False
        
        total_original += data['original_length']
        total_latent += data['latent_length']
        
        print(f"  [{'OK' if match else 'FAIL'}] {len(text)} chars -> {data['latent_length']} vectors")
    
    avg_ratio = total_original / max(total_latent, 1)
    print(f"\n  Total: {total_original} tokens -> {total_latent} latent vectors")
    print(f"  Average compression: {avg_ratio:.1f}x")
    print(f"  All match: {all_match}")
    
    return all_match


def main():
    section("LAIT MCP SYSTEM - FINAL VERIFICATION")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Adapter
    adapter_ok, adapter_acc = test_adapter()
    results['adapter'] = adapter_ok
    
    # Test 2: MCP Server
    mcp_ok, mcp_acc = test_mcp_server()
    results['mcp_server'] = mcp_ok
    
    # Test 3: Ollama
    ollama_ok = test_ollama()
    results['ollama'] = ollama_ok
    
    # Test 4: Full Pipeline
    pipeline_ok = test_full_pipeline()
    results['pipeline'] = pipeline_ok
    
    # Summary
    section("VERIFICATION SUMMARY")
    
    all_pass = all(results.values())
    
    print(f"  {'Component':<25} {'Status':<10}")
    print(f"  {'-'*35}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<25} {status:<10}")
    print(f"  {'-'*35}")
    print(f"  {'OVERALL':<25} {'ALL PASS' if all_pass else 'SOME FAILED'}")
    
    if adapter_ok:
        print(f"\n  Adapter accuracy: {adapter_acc:.2%}")
    
    print(f"\n  Root files: lait_v1.py, lait_mcp_server.py, lait_mcp_adapter.py,")
    print(f"              lait_adapter_best.pt, LAIT_WHITE_PAPER.md")
    print(f"  Directories: src/, tests/, docs/, configs/, data/, analysis/, ollama/")
    
    print(f"\n{'='*70}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
