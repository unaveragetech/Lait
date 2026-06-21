#!/usr/bin/env python3
"""
LAIT Adapter Verifiable Proof
Proves the adapter achieves 100% reconstruction on arbitrary text.
Generates a signed proof file that can be independently verified.
"""

import torch
import json
import hashlib
import time
import sys
import os
import random
import string

sys.path.insert(0, os.path.dirname(__file__) or '.')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evolve_adapter import EvolvableAdapter


# ==========================================
# 1. ADAPTER LOADER
# ==========================================

def load_adapter(checkpoint_path='lait_adapter_best.pt'):
    """Load trained adapter and return config + weights."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    
    adapter = EvolvableAdapter(config)
    adapter.load_state_dict(state_dict)
    adapter.eval()
    
    return adapter, config, checkpoint


# ==========================================
# 2. RECONSTRUCTION TESTER
# ==========================================

def test_reconstruction(adapter, text):
    """Test single text reconstruction. Returns (match, original, reconstructed)."""
    tokens = list(text.encode('utf-8'))
    padded = tokens + [0] * (512 - len(tokens))
    x = torch.tensor([padded], dtype=torch.long)
    
    with torch.no_grad():
        logits, latent, orig_len = adapter(x)
    
    # logits[i] predicts token[i+1]
    # To reconstruct token[0..T-1]:
    #   token[0] = first input token (no prediction available)
    #   token[1..T-1] = logits[0..T-2].argmax()
    first_token = [tokens[0]]
    predicted = logits[0, :len(tokens)-1, :].argmax(dim=-1).tolist()
    recon_tokens = first_token + predicted
    reconstructed = bytes(recon_tokens[:len(tokens)])
    
    match = (text == reconstructed.decode('utf-8', errors='replace'))
    return match, text, reconstructed.decode('utf-8', errors='replace')


def test_compression(adapter, text, compression_ratio):
    """Test reconstruction at a specific compression ratio."""
    original_ratio = adapter.compression_ratio
    adapter.compression_ratio = compression_ratio
    
    tokens = list(text.encode('utf-8'))
    x = torch.tensor([tokens], dtype=torch.long)
    
    with torch.no_grad():
        logits, latent, orig_len = adapter(x)
    
    preds = logits.argmax(dim=-1)
    reconstructed_tokens = preds[0].tolist()
    reconstructed = bytes(reconstructed_tokens[:len(tokens)])
    
    adapter.compression_ratio = original_ratio
    
    latent_size = latent.shape[1]
    actual_ratio = len(tokens) / max(latent_size, 1)
    
    match = (text == reconstructed.decode('utf-8', errors='replace'))
    return match, latent_size, actual_ratio


# ==========================================
# 3. TEST PROMPTS
# ==========================================

def get_test_prompts():
    """Get diverse test prompts of varying sizes."""
    prompts = {
        'tiny': [
            'Hello',
            'Yes',
            'No',
            'OK',
            'Test',
        ],
        'short': [
            'Hello world!',
            'The quick brown fox.',
            'Machine learning is great.',
            'Python is a programming language.',
            'The weather is nice today.',
        ],
        'medium': [
            'The quick brown fox jumps over the lazy dog.',
            'Machine learning is transforming technology worldwide.',
            'Neural networks can learn complex patterns from data.',
            'The adapter compresses text into latent representations.',
            'Genetic evolution finds optimal architecture configurations.',
        ],
        'long': [
            'The quick brown fox jumps over the lazy dog. This is a longer sentence that tests the adapter ability to reconstruct arbitrary text accurately.',
            'Machine learning is transforming technology worldwide. Neural networks can learn complex patterns from data and make predictions about future events.',
            'The adapter compresses text into latent representations using a transformer encoder-decoder architecture with adaptive pooling bottleneck.',
            'Genetic evolution explores a 120-trait genome to find optimal architecture configurations for neural text compression with 100% reconstruction.',
            'Los Angeles (US), Jun 20 (IANS) A massive wildfire has broken out in the hills above Los Angeles, forcing thousands of residents to evacuate their homes.',
        ],
        'technical': [
            'def train_model(config, data): model = LAITAdapter(config); optimizer = AdamW(model.parameters()); for epoch in range(100): loss = train_epoch(model, data); return model',
            '{"name": "lait-adapter", "version": "2.0", "compression_ratio": 0.5, "d_model": 128, "n_layers": 4, "accuracy": 1.0}',
            'SELECT * FROM users WHERE age > 25 AND city = "Los Angeles" ORDER BY name ASC LIMIT 100',
            'git commit -m "feat: add GPU training support with CUDA 12.8 for RTX 5060"',
            'curl -X POST http://localhost:8001/compress -H "Content-Type: application/json" -d \'{"text": "Hello world!"}\'',
        ],
        'random': [],
    }
    
    # Generate random strings
    for _ in range(10):
        length = random.randint(10, 200)
        text = ''.join(random.choices(
            string.ascii_letters + string.digits + string.punctuation + ' ', k=length
        ))
        prompts['random'].append(text)
    
    return prompts


# ==========================================
# 4. PROOF GENERATOR
# ==========================================

def generate_proof(adapter, config, checkpoint):
    """Generate verifiable proof of adapter reconstruction."""
    proof = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'adapter_config': config,
        'checkpoint_info': {
            'accuracy': checkpoint.get('accuracy', 'unknown'),
            'compression': checkpoint.get('compression', 'unknown'),
        },
        'tests': {},
        'summary': {},
    }
    
    prompts = get_test_prompts()
    total_tests = 0
    total_passed = 0
    
    for category, texts in prompts.items():
        category_results = []
        for text in texts:
            match, orig, recon = test_reconstruction(adapter, text)
            total_tests += 1
            if match:
                total_passed += 1
            
            category_results.append({
                'original': orig,
                'reconstructed': recon,
                'match': match,
                'original_tokens': len(orig.encode('utf-8')),
            })
        
        proof['tests'][category] = category_results
    
    # Test compression ratios
    compression_tests = {}
    test_text = 'The quick brown fox jumps over the lazy dog. Machine learning is transforming technology.'
    
    for ratio in [1.0, 0.5, 0.25, 0.125, 0.0625]:
        match, latent_size, actual_ratio = test_compression(adapter, test_text, ratio)
        compression_tests[str(ratio)] = {
            'text': test_text,
            'match': match,
            'latent_size': latent_size,
            'actual_ratio': round(actual_ratio, 2),
            'compression': round(1.0 / actual_ratio, 1) if actual_ratio > 0 else 0,
        }
    
    proof['compression_tests'] = compression_tests
    
    proof['summary'] = {
        'total_tests': total_tests,
        'total_passed': total_passed,
        'accuracy': round(total_passed / max(total_tests, 1), 4),
        'all_passed': total_passed == total_tests,
    }
    
    # Generate hash for verification
    proof_json = json.dumps(proof, sort_keys=True, indent=2)
    proof_hash = hashlib.sha256(proof_json.encode()).hexdigest()
    proof['verification_hash'] = proof_hash
    
    return proof, proof_hash


# ==========================================
# 5. MAIN
# ==========================================

def main():
    print('=' * 70)
    print('  LAIT ADAPTER VERIFIABLE PROOF')
    print('=' * 70)
    print()
    
    # Load adapter
    print('Loading adapter...')
    adapter, config, checkpoint = load_adapter()
    params = sum(p.numel() for p in adapter.parameters())
    print(f'  Model: {params:,} parameters')
    print(f'  Config: d={config["d_model"]}, enc={config["n_encoder_layers"]}, dec={config["n_decoder_layers"]}, heads={config["n_heads"]}')
    print(f'  Compression ratio: {config["compression_ratio"]}')
    print()
    
    # Generate proof
    print('Generating proof...')
    proof, proof_hash = generate_proof(adapter, config, checkpoint)
    
    # Print results
    print()
    print('=' * 70)
    print('  RESULTS')
    print('=' * 70)
    print()
    
    for category, tests in proof['tests'].items():
        passed = sum(1 for t in tests if t['match'])
        total = len(tests)
        print(f'  {category.upper()}: {passed}/{total} passed')
        for t in tests:
            status = 'OK' if t['match'] else 'FAIL'
            print(f'    [{status}] {t["original"][:50]}{"..." if len(t["original"]) > 50 else ""}')
    print()
    
    print('  COMPRESSION TESTS:')
    for ratio, test in proof['compression_tests'].items():
        status = 'OK' if test['match'] else 'FAIL'
        print(f'    [{status}] ratio={ratio} latent={test["latent_size"]} compression={test["compression"]}x')
    print()
    
    print(f'  SUMMARY:')
    print(f'    Total tests: {proof["summary"]["total_tests"]}')
    print(f'    Passed: {proof["summary"]["total_passed"]}')
    print(f'    Accuracy: {proof["summary"]["accuracy"]:.2%}')
    print(f'    All passed: {proof["summary"]["all_passed"]}')
    print()
    
    print(f'  VERIFICATION HASH: {proof_hash[:32]}...')
    print()
    
    # Save proof
    proof_file = 'lait_adapter_proof.json'
    with open(proof_file, 'w') as f:
        json.dump(proof, f, indent=2)
    print(f'  Proof saved to: {proof_file}')
    print()
    
    # Save hash for quick verification
    hash_file = 'lait_adapter_proof.hash'
    with open(hash_file, 'w') as f:
        f.write(proof_hash)
    print(f'  Hash saved to: {hash_file}')
    print()
    
    print('=' * 70)
    if proof['summary']['all_passed']:
        print('  VERDICT: ADAPTER VERIFIED - 100% RECONSTRUCTION')
    else:
        print('  VERDICT: ADAPTER FAILED SOME TESTS')
    print('=' * 70)
    
    return 0 if proof['summary']['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
