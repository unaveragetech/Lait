#!/usr/bin/env python3
"""
LAIT GPU Engine (Fixed Version)
External GPU computation engine that receives tasks via JSON and returns results.
Can be called as a subprocess by the main evolution script.

Usage:
    python gpu_engine.py --task train --input task.json --output result.json
    python gpu_engine.py --task benchmark --input task.json --output result.json
    python gpu_engine.py --task info
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import sys
import os
import argparse
import traceback
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__) or '.')

# ==========================================
# 1. ADAPTER MODEL
# ==========================================

class EvolvableAdapter(nn.Module):
    """Adapter that can be configured with different architectures."""
    
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
        max_seq_len = config.get('max_seq_len', 512)
        use_relu = config.get('activation', 'gelu') == 'relu'
        
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        
        self.d_model = d_model
        self.compression_ratio = compression_ratio
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult, dropout=dropout,
            batch_first=True, activation='relu' if use_relu else 'gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)
        
        self.compress_proj = nn.Linear(d_model, d_model)
        
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult, dropout=dropout,
            batch_first=True, activation='relu' if use_relu else 'gelu',
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
# 2. TRAINING DATA GENERATOR
# ==========================================

def generate_training_data(num_samples: int = 2000) -> List[bytes]:
    """Generate extremely diverse training samples."""
    import random
    import string
    
    samples = []
    
    # English sentences (many variations)
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Neural networks can compress and reconstruct text.",
        "Latent attention mechanisms enable efficient processing.",
        "The adapter learns to encode and decode sequences.",
        "Each token maps to a vector in latent space.",
        "The decoder reconstructs the original input.",
        "Training minimizes the reconstruction loss.",
        "Backpropagation updates the model weights.",
        "The encoder compresses input to latent representation.",
        "Batch processing enables efficient training.",
        "Gradient descent optimizes the parameters.",
        "Cross entropy loss measures prediction accuracy.",
        "The vocabulary size is 256 for byte-level encoding.",
        "Positional encoding adds sequence order information.",
        "Self attention captures long range dependencies.",
        "Multi head attention diversifies representations.",
        "Layer normalization stabilizes training dynamics.",
        "Dropout prevents overfitting during training.",
        "The learning rate controls update magnitude.",
        "Adam optimizer adapts learning rates per parameter.",
        "The loss function measures reconstruction quality.",
        "The bottleneck layer compresses information.",
        "Reconstruction quality improves with more capacity.",
    ]
    
    for sent in base_sentences:
        samples.append(sent.encode('utf-8'))
        samples.append(sent.lower().encode('utf-8'))
        samples.append(sent.upper().encode('utf-8'))
        for prefix in ["Note:", "Point:", "Summary:", "Context:", "Input:", "Text:", "Data:"]:
            samples.append(f"{prefix} {sent}".encode('utf-8'))
        samples.append((sent + " " + sent).encode('utf-8'))
        words = sent.split()
        samples.append(" ".join(reversed(words)).encode('utf-8'))
    
    # Technical patterns
    for d in [64, 128, 256, 512]:
        for h in [1, 2, 4, 8]:
            for cr in [0.125, 0.25, 0.5, 0.75, 1.0]:
                for enc in [1, 2, 4, 6, 8]:
                    samples.append(f"d_model={d}, n_heads={h}, compression={cr}, layers={enc}".encode('utf-8'))
                    samples.append(f"loss=0.{random.randint(1000,9999)}, acc=0.{random.randint(1000,9999)}".encode('utf-8'))
    
    # Random strings
    for _ in range(200):
        length = random.randint(10, 200)
        text = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=length))
        samples.append(text.encode('utf-8'))
    
    # Numbers and math
    for _ in range(100):
        a, b = random.randint(0, 999), random.randint(0, 999)
        op = random.choice(['+', '-', '*', '/'])
        try:
            result = eval(f"{a}{op}{b}")
            samples.append(f"{a} {op} {b} = {result}".encode('utf-8'))
        except:
            pass
    
    # JSON-like
    for _ in range(100):
        key = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        val = random.randint(0, 1000)
        samples.append(f'{{"{key}": {val}}}'.encode('utf-8'))
    
    # Code templates
    code_templates = [
        "def hello(): print('hello')",
        "for i in range(10): print(i)",
        "x = [i**2 for i in range(10)]",
        "if x > 0: print('positive')",
        "class MyClass: def __init__(self): pass",
        "import torch; model = torch.nn.Linear(128, 256)",
        "result = model(input_tensor)",
        "loss = criterion(output, target)",
        "optimizer.step()",
        "print(f'Epoch {epoch}: loss={loss:.4f}')",
    ]
    for code in code_templates:
        samples.append(code.encode('utf-8'))
    
    # Pad with variations
    while len(samples) < num_samples:
        idx = random.randint(0, len(samples) - 1)
        original = samples[idx]
        if random.random() < 0.5:
            pos = random.randint(0, len(original))
            char = random.choice(string.ascii_letters).encode()
            samples.append(original[:pos] + char + original[pos:])
        else:
            if len(original) > 5:
                pos = random.randint(0, len(original) - 1)
                samples.append(original[:pos] + original[pos+1:])
    
    return samples[:num_samples]


# ==========================================
# 3. GPU ENGINE TASKS
# ==========================================

def task_train(config: dict, train_samples: List[bytes], epochs: int = 50, device: str = 'cpu') -> Dict:
    """Train an adapter and return metrics."""
    try:
        import random as rng
        print(f"[GPU Engine] Creating adapter with config...", file=sys.stderr)
        adapter = EvolvableAdapter(config)
        
        # Validate model can be created
        param_count = sum(p.numel() for p in adapter.parameters())
        print(f"[GPU Engine] Model created: {param_count:,} parameters", file=sys.stderr)
        
        # Move to device
        adapter = adapter.to(device)
        print(f"[GPU Engine] Model moved to {device}", file=sys.stderr)
        
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_acc = 0
        history = []
        start_time = time.time()
        
        print(f"[GPU Engine] Starting training for {epochs} epochs...", file=sys.stderr)
        
        adapter.train()
        for epoch in range(epochs):
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle training data each epoch
            rng.shuffle(train_samples)
            
            for i in range(0, len(train_samples), 16):
                batch = train_samples[i:i+16]
                max_len = min(max(len(s) for s in batch), 512)
                
                padded = []
                for s in batch:
                    tokens = list(s[:max_len])
                    tokens = tokens + [0] * (max_len - len(tokens))
                    padded.append(tokens)
                
                x = torch.tensor(padded, dtype=torch.long).to(device)
                
                optimizer.zero_grad()
                logits, latent, orig_len = adapter(x)
                
                targets = x[:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()
                
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=0,
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                
                preds = logits.argmax(dim=-1)
                mask = targets != 0
                correct = (preds[mask] == targets[mask]).sum().item()
                total = mask.sum().item()
                
                epoch_correct += correct
                epoch_total += total
                epoch_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            
            epoch_acc = epoch_correct / max(epoch_total, 1)
            epoch_loss = epoch_loss / max(num_batches, 1)
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            
            history.append({
                'epoch': epoch + 1,
                'accuracy': epoch_acc,
                'loss': epoch_loss,
            })
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"[GPU Engine] Epoch {epoch+1}/{epochs}: acc={epoch_acc:.4f} loss={epoch_loss:.4f}", file=sys.stderr)
        
        print(f"[GPU Engine] Training complete. Best accuracy: {best_acc:.4f}", file=sys.stderr)
        
        # Final evaluation
        adapter.eval()
        total_bytes = 0
        correct_bytes = 0
        
        with torch.no_grad():
            for i in range(0, len(train_samples), 16):
                batch = train_samples[i:i+16]
                max_len = min(max(len(s) for s in batch), 512)
                padded = []
                for s in batch:
                    tokens = list(s[:max_len])
                    tokens = tokens + [0] * (max_len - len(tokens))
                    padded.append(tokens)
                
                x = torch.tensor(padded, dtype=torch.long).to(device)
                logits, latent, orig_len = adapter(x)
                
                targets = x[:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()
                preds = logits.argmax(dim=-1)
                
                mask = targets != 0
                correct = (preds[mask] == targets[mask]).sum().item()
                total = mask.sum().item()
                
                correct_bytes += correct
                total_bytes += total
        
        accuracy = correct_bytes / max(total_bytes, 1)
        
        # Compression metrics
        latent_size = max(1, int(512 * adapter.compression_ratio))
        compression_ratio = 512 / max(latent_size, 1)
        latent_memory = latent_size * adapter.d_model * 4
        attention_memory = 512 * 512 * 4
        memory_savings = attention_memory / max(latent_memory, 1)
        
        elapsed = time.time() - start_time
        
        print(f"[GPU Engine] Final accuracy: {accuracy:.4f}, Compression: {compression_ratio:.1f}x", file=sys.stderr)
        
        return {
            'success': True,
            'accuracy': accuracy,
            'best_accuracy': best_acc,
            'compression_ratio': compression_ratio,
            'memory_savings': memory_savings,
            'param_count': param_count,
            'latent_size': latent_size,
            'training_time': elapsed,
            'epochs': epochs,
            'history': history[-10:],
        }
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
        print(f"[GPU Engine] ERROR: {error_msg}", file=sys.stderr)
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'accuracy': 0,
            'best_accuracy': 0,
            'compression_ratio': 0,
            'memory_savings': 0,
            'param_count': 0,
            'latent_size': 0,
            'training_time': 0,
        }


def task_benchmark(config: dict, device: str = 'cpu') -> Dict:
    """Benchmark adapter performance."""
    try:
        adapter = EvolvableAdapter(config).to(device)
        param_count = sum(p.numel() for p in adapter.parameters())
        model_size_mb = param_count * 4 / 1024 / 1024
        
        tokens = torch.randint(0, 256, (1, 512)).to(device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = adapter.encode(tokens)
        
        # Benchmark encode
        times = []
        for _ in range(100):
            start = time.time()
            with torch.no_grad():
                latent = adapter.encode(tokens)
            times.append((time.time() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        throughput = tokens.shape[1] / (avg_time / 1000)
        
        latent_size = latent.shape[1]
        latent_memory = latent.element_size() * latent.nelement() / 1024 / 1024
        attention_memory = 512 * 512 * 4 / 1024 / 1024
        memory_savings = attention_memory / max(latent_memory, 0.001)
        
        return {
            'success': True,
            'param_count': param_count,
            'model_size_mb': model_size_mb,
            'input_tokens': tokens.shape[1],
            'latent_vectors': latent_size,
            'compression_ratio': tokens.shape[1] / max(latent_size, 1),
            'encode_time_ms': avg_time,
            'throughput_tokens_per_sec': throughput,
            'latent_memory_mb': latent_memory,
            'attention_memory_mb': attention_memory,
            'memory_savings': memory_savings,
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def task_info() -> Dict:
    """Get GPU and system info."""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    
    info = {
        'success': True,
        'torch_version': torch.__version__,
        'cuda_available': cuda_available,
        'cuda_device_count': device_count,
        'cpu_count': os.cpu_count(),
    }
    
    if cuda_available:
        try:
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_memory_total_mb'] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            info['cuda_memory_allocated_mb'] = torch.cuda.memory_allocated(0) / 1024 / 1024
        except Exception as e:
            info['cuda_error'] = str(e)
    else:
        info['cuda_error'] = 'CUDA not available'
    
    return info


# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="LAIT GPU Engine")
    parser.add_argument("--task", type=str, required=True, 
                       choices=["train", "benchmark", "info"],
                       help="Task to perform")
    parser.add_argument("--input", type=str, help="Input JSON file")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: cpu, cuda, auto")
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        # Validate requested device
        if args.device == "cuda" and not torch.cuda.is_available():
            print(f"[GPU Engine] WARNING: CUDA requested but not available, falling back to CPU", file=sys.stderr)
            device = "cpu"
        else:
            device = args.device
    
    print(f"[GPU Engine] Using device: {device}", file=sys.stderr)
    
    # Load input
    input_data = {}
    if args.input and os.path.exists(args.input):
        try:
            with open(args.input, 'r') as f:
                input_data = json.load(f)
            print(f"[GPU Engine] Loaded input from {args.input}", file=sys.stderr)
        except Exception as e:
            print(f"[GPU Engine] Error loading input: {e}", file=sys.stderr)
    
    # Execute task
    if args.task == "info":
        result = task_info()
    
    elif args.task == "train":
        config = input_data.get('config', {})
        epochs = input_data.get('epochs', 50)
        
        print(f"[GPU Engine] Generating training data...", file=sys.stderr)
        train_samples = generate_training_data(500)
        print(f"[GPU Engine] Generated {len(train_samples)} samples", file=sys.stderr)
        
        print(f"[GPU Engine] Training on {device} for {epochs} epochs...", file=sys.stderr)
        result = task_train(config, train_samples, epochs, device)
    
    elif args.task == "benchmark":
        config = input_data.get('config', {})
        result = task_benchmark(config, device)
    
    else:
        result = {'success': False, 'error': f'Unknown task: {args.task}'}
    
    # Add device info
    result['device'] = device
    
    # Save output
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"[GPU Engine] Result saved to {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"[GPU Engine] Error saving output: {e}", file=sys.stderr)
    
    # Always print result to stdout
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
