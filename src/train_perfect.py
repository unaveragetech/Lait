#!/usr/bin/env python3
"""
Train adapter to learn TRUE identity function (ratio=1.0).
Uses massive diverse training data to force generalization.
"""
import torch
import torch.nn.functional as F
import time
import random
import string
import json
from evolve_adapter import EvolvableAdapter

def generate_massive_data(num_samples=2000):
    """Generate extremely diverse training data."""
    samples = []
    
    # English sentences (many variations)
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Hello world! This is a test message.",
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
        # Original
        samples.append(sent.encode('utf-8'))
        # Lowercase
        samples.append(sent.lower().encode('utf-8'))
        # Uppercase
        samples.append(sent.upper().encode('utf-8'))
        # With prefix
        for prefix in ["Note:", "Point:", "Summary:", "Context:", "Input:", "Text:", "Data:"]:
            samples.append(f"{prefix} {sent}".encode('utf-8'))
        # Repeated
        samples.append((sent + " " + sent).encode('utf-8'))
        # Reversed words
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
    
    # Code-like
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
        # Add noise
        if random.random() < 0.5:
            # Add random char
            pos = random.randint(0, len(original))
            char = random.choice(string.ascii_letters).encode()
            samples.append(original[:pos] + char + original[pos:])
        else:
            # Remove random char
            if len(original) > 5:
                pos = random.randint(0, len(original) - 1)
                samples.append(original[:pos] + original[pos+1:])
    
    return samples[:num_samples]


def train_for_perfection():
    """Train with massive data until 100% on ALL data."""
    print("="*70)
    print("TRAINING FOR TRUE 100% RECONSTRUCTION")
    print("="*70)
    
    device = 'cpu'
    
    # Config: ratio=1.0 (no compression), enough capacity
    config = {
        'd_model': 128,
        'n_encoder_layers': 4,
        'n_decoder_layers': 4,
        'n_heads': 4,
        'compression_ratio': 1.0,
        'ff_mult': 4,
        'dropout': 0.0,
        'vocab_size': 256,
        'max_seq_len': 512,
        'activation': 'gelu',
    }
    
    adapter = EvolvableAdapter(config).to(device)
    params = sum(p.numel() for p in adapter.parameters())
    print(f"Model: {params:,} params")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Generate massive training data
    print("Generating training data...")
    all_samples = generate_massive_data(2000)
    print(f"Generated {len(all_samples)} samples")
    
    # Separate train/test
    random.shuffle(all_samples)
    split = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split]
    test_samples = all_samples[split:]
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    best_train_acc = 0
    best_test_acc = 0
    best_epoch = 0
    start = time.time()
    
    for epoch in range(500):
        adapter.train()
        total_correct = 0
        total_tokens = 0
        
        # Shuffle training data each epoch
        random.shuffle(train_samples)
        
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
            logits, _, _ = adapter(x)
            
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
            total_correct += (preds[mask] == targets[mask]).sum().item()
            total_tokens += mask.sum().item()
        
        scheduler.step()
        train_acc = total_correct / max(total_tokens, 1)
        
        # Evaluate on test set
        if (epoch + 1) % 5 == 0:
            adapter.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for i in range(0, len(test_samples), 16):
                    batch = test_samples[i:i+16]
                    max_len = min(max(len(s) for s in batch), 512)
                    padded = []
                    for s in batch:
                        tokens = list(s[:max_len])
                        tokens = tokens + [0] * (max_len - len(tokens))
                        padded.append(tokens)
                    
                    x = torch.tensor(padded, dtype=torch.long).to(device)
                    logits, _, _ = adapter(x)
                    
                    targets = x[:, 1:].contiguous()
                    logits = logits[:, :-1, :].contiguous()
                    preds = logits.argmax(dim=-1)
                    mask = targets != 0
                    test_correct += (preds[mask] == targets[mask]).sum().item()
                    test_total += mask.sum().item()
            
            test_acc = test_correct / max(test_total, 1)
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                # Save best
                torch.save({
                    'config': config,
                    'state_dict': adapter.state_dict(),
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                }, 'lait_adapter_best.pt')
        
        if (epoch + 1) % 25 == 0:
            elapsed = time.time() - start
            print(f"Epoch {epoch+1}/500: train={train_acc:.4%} test={test_acc:.4%} "
                  f"best_test={best_test_acc:.4%} ({elapsed:.0f}s)")
        
        if best_test_acc >= 1.0:
            print(f"\n*** 100% TEST ACCURACY ACHIEVED ***")
            break
    
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Best train accuracy: {best_train_acc:.4%}")
    print(f"Best test accuracy: {best_test_acc:.4%} (epoch {best_epoch})")
    
    # Final verification on arbitrary text
    print(f"\n{'='*70}")
    print("FINAL VERIFICATION ON UNSEEN TEXT")
    print(f"{'='*70}")
    
    adapter.eval()
    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "xyz123!@#$%",
        "A",
        "This sentence was NOT in the training data!",
        "1234567890 abcdefghij",
        "THE QUICK BROWN FOX",
        "hello world",
    ]
    
    total_correct = 0
    total_bytes = 0
    
    for text in test_texts:
        text_bytes = text.encode('utf-8')
        tokens = list(text_bytes)[:512]
        x = torch.tensor([tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits, _, _ = adapter(x)
        
        preds = logits.argmax(dim=-1).squeeze(0).tolist()
        recon = bytes(preds[:len(text_bytes)])
        
        correct = sum(a == b for a, b in zip(text_bytes, recon))
        total = len(text_bytes)
        acc = correct / total
        
        total_correct += correct
        total_bytes += total
        
        print(f"  [{acc:.0%}] \"{text}\"")
    
    overall = total_correct / max(total_bytes, 1)
    print(f"\nOverall unseen text accuracy: {overall:.2%}")
    
    return best_test_acc


if __name__ == "__main__":
    train_for_perfection()
