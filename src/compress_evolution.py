#!/usr/bin/env python3
"""
LAIT Genetic Compression Search (Verbose Version)
Finds the best compression ratio that maintains 100% reconstruction accuracy.
Traces lineage to identify optimal architectural patterns.

Supports GPU acceleration via external engine script.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import json
import os
import sys
import subprocess
import hashlib
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__) or '.')

# ==========================================
# 1. EVOLVABLE ADAPTER
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

def generate_diverse_data(num_samples: int = 2000) -> List[bytes]:
    """Generate extremely diverse training samples for compression testing."""
    import string
    samples = []
    
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
    
    for d in [64, 128, 256, 512]:
        for h in [1, 2, 4, 8]:
            for cr in [0.125, 0.25, 0.5, 0.75, 1.0]:
                for enc in [1, 2, 4, 6, 8]:
                    samples.append(f"d_model={d}, n_heads={h}, compression={cr}, layers={enc}".encode('utf-8'))
                    samples.append(f"loss=0.{random.randint(1000,9999)}, acc=0.{random.randint(1000,9999)}".encode('utf-8'))
    
    for _ in range(200):
        length = random.randint(10, 200)
        text = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=length))
        samples.append(text.encode('utf-8'))
    
    for _ in range(100):
        a, b = random.randint(0, 999), random.randint(0, 999)
        op = random.choice(['+', '-', '*', '/'])
        try:
            result = eval(f"{a}{op}{b}")
            samples.append(f"{a} {op} {b} = {result}".encode('utf-8'))
        except:
            pass
    
    for _ in range(100):
        key = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        val = random.randint(0, 1000)
        samples.append(f'{{"{key}": {val}}}'.encode('utf-8'))
    
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
# 3. GPU ENGINE INTERFACE
# ==========================================

class GPUEngine:
    """Interface to external GPU engine script."""
    
    def __init__(self, engine_path: str = None, device: str = 'auto'):
        if engine_path is None:
            engine_path = os.path.join(os.path.dirname(__file__), 'gpu_engine.py')
        self.engine_path = engine_path
        self.device = device
        self.available = os.path.exists(engine_path)
        
    def _run_engine(self, task: str, input_data: dict = None, timeout: int = 600) -> dict:
        """Run GPU engine with specified task."""
        input_file = f'temp_input_{task}.json'
        output_file = f'temp_output_{task}.json'
        
        try:
            # Write input
            if input_data:
                with open(input_file, 'w') as f:
                    json.dump(input_data, f)
            
            # Build command
            cmd = [
                sys.executable, self.engine_path,
                '--task', task,
                '--device', self.device,
                '--output', output_file,
            ]
            if input_data:
                cmd.extend(['--input', input_file])
            
            # Run engine with timeout
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            
            # Check for errors in stderr
            if result.stderr:
                # Print stderr for debugging (it contains progress info)
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        print(f'\n      [GPU Engine] {line}', end='', flush=True)
            
            # Read output
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    output = json.load(f)
                
                # Validate output has required fields
                if not output.get('success'):
                    return {
                        'success': False,
                        'error': output.get('error', 'GPU engine returned success=False'),
                        'accuracy': 0,
                        'best_accuracy': 0,
                        'compression_ratio': 0,
                        'memory_savings': 0,
                        'param_count': 0,
                        'latent_size': 0,
                        'training_time': 0,
                    }
                
                return output
            else:
                return {
                    'success': False,
                    'error': 'No output file generated',
                    'accuracy': 0,
                    'best_accuracy': 0,
                    'compression_ratio': 0,
                    'memory_savings': 0,
                    'param_count': 0,
                    'latent_size': 0,
                    'training_time': 0,
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Engine timed out after {timeout}s',
                'accuracy': 0,
                'best_accuracy': 0,
                'compression_ratio': 0,
                'memory_savings': 0,
                'param_count': 0,
                'latent_size': 0,
                'training_time': 0,
            }
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'Invalid JSON in output: {str(e)}',
                'accuracy': 0,
                'best_accuracy': 0,
                'compression_ratio': 0,
                'memory_savings': 0,
                'param_count': 0,
                'latent_size': 0,
                'training_time': 0,
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'accuracy': 0,
                'best_accuracy': 0,
                'compression_ratio': 0,
                'memory_savings': 0,
                'param_count': 0,
                'latent_size': 0,
                'training_time': 0,
            }
        finally:
            # Cleanup temp files
            for f in [input_file, output_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
    
    def get_info(self) -> dict:
        """Get GPU and system info."""
        return self._run_engine('info')
    
    def train(self, config: dict, epochs: int = 50) -> dict:
        """Train adapter using GPU engine."""
        return self._run_engine('train', {
            'config': config,
            'epochs': epochs,
        })
    
    def benchmark(self, config: dict) -> dict:
        """Benchmark adapter using GPU engine."""
        return self._run_engine('benchmark', {
            'config': config,
        })


# ==========================================
# 4. FITNESS FUNCTION (CPU Version)
# ==========================================

def evaluate_fitness_cpu(
    adapter: EvolvableAdapter,
    train_samples: List[bytes],
    device: str = 'cpu',
    max_len: int = 512,
) -> Dict:
    """Evaluate adapter fitness on reconstruction accuracy (CPU)."""
    adapter.eval()
    
    total_bytes = 0
    correct_bytes = 0
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, min(len(train_samples), 200), 8):
            batch = train_samples[i:i+8]
            max_batch_len = min(max(len(s) for s in batch), max_len)
            padded = []
            for s in batch:
                tokens = list(s[:max_batch_len])
                tokens = tokens + [0] * (max_batch_len - len(tokens))
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
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
            total_loss += loss.item()
            num_batches += 1
    
    accuracy = correct_bytes / max(total_bytes, 1)
    avg_loss = total_loss / max(num_batches, 1)
    
    latent_size = max(1, int(max_len * adapter.compression_ratio))
    compression_ratio = max_len / max(latent_size, 1)
    latent_memory = latent_size * adapter.d_model * 4
    attention_memory = max_len * max_len * 4
    memory_savings = attention_memory / max(latent_memory, 1)
    param_count = sum(p.numel() for p in adapter.parameters())
    
    fitness = (
        accuracy * 100.0 * 0.70 +
        min(compression_ratio / 10.0, 10.0) * 5.0 * 0.20 +
        min(memory_savings / 100.0, 10.0) * 5.0 * 0.10
    )
    
    return {
        'fitness': fitness,
        'accuracy': accuracy,
        'avg_loss': avg_loss,
        'compression_ratio': compression_ratio,
        'memory_savings': memory_savings,
        'param_count': param_count,
        'latent_size': latent_size,
    }


# ==========================================
# 5. GENETIC OPERATORS
# ==========================================

def random_config() -> dict:
    """Generate a random adapter configuration."""
    d_model = random.choice([64, 96, 128, 192, 256])
    n_heads = random.choice([1, 2, 4, 8])
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads -= 1
    
    return {
        'vocab_size': 256,
        'd_model': d_model,
        'n_encoder_layers': random.randint(2, 8),
        'n_decoder_layers': random.randint(2, 8),
        'n_heads': n_heads,
        'ff_mult': random.choice([2, 4, 8]),
        'dropout': 0.0,
        'compression_ratio': random.choice([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]),
        'max_seq_len': 512,
        'activation': 'gelu',
    }


def mutate_config(config: dict, rate: float = 0.3) -> dict:
    """Mutate a configuration."""
    new_config = dict(config)
    
    if random.random() < rate:
        new_config['d_model'] = random.choice([64, 96, 128, 192, 256])
    if random.random() < rate:
        new_config['n_encoder_layers'] = random.randint(2, 8)
    if random.random() < rate:
        new_config['n_decoder_layers'] = random.randint(2, 8)
    if random.random() < rate:
        new_config['n_heads'] = random.choice([1, 2, 4, 8])
    if random.random() < rate:
        new_config['ff_mult'] = random.choice([2, 4, 8])
    if random.random() < rate:
        new_config['compression_ratio'] = random.choice([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    
    while new_config['d_model'] % new_config['n_heads'] != 0 and new_config['n_heads'] > 1:
        new_config['n_heads'] -= 1
    
    return new_config


def crossover_configs(c1: dict, c2: dict) -> dict:
    """Crossover two configurations."""
    child = {}
    for key in c1:
        if random.random() < 0.5:
            child[key] = c1[key]
        else:
            child[key] = c2[key]
    
    while child['d_model'] % child['n_heads'] != 0 and child['n_heads'] > 1:
        child['n_heads'] -= 1
    
    return child


def config_signature(config: dict) -> str:
    """Generate unique signature for a config."""
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


# ==========================================
# 6. VERBOSE PROGRESS DISPLAY
# ==========================================

class ProgressDisplay:
    """Display real-time progress during evolution."""
    
    def __init__(self):
        self.start_time = time.time()
        self.gen_start_time = None
        self.total_evals = 0
        self.best_accuracy = 0
        self.best_compression = 1.0
        
    def print_header(self, pop_size: int, elite_count: int, epochs: int, device: str):
        """Print evolution header."""
        print()
        print('=' * 80)
        print('  LAIT GENETIC COMPRESSION SEARCH - VERBOSE MODE')
        print('=' * 80)
        print(f'  Start Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'  Population: {pop_size}')
        print(f'  Elite Count: {elite_count}')
        print(f'  Epochs per Config: {epochs}')
        print(f'  Device: {device}')
        print(f'  Target: 100% accuracy + Maximum compression')
        print('=' * 80)
        print()
        
    def print_population(self, configs: List[dict]):
        """Print initial population."""
        print('  INITIAL POPULATION:')
        print('  ' + '-' * 60)
        for i, cfg in enumerate(configs):
            cr = cfg['compression_ratio']
            d = cfg['d_model']
            enc = cfg['n_encoder_layers']
            dec = cfg['n_decoder_layers']
            heads = cfg['n_heads']
            print(f'    [{i+1:2d}] cr={cr:.3f} | d={d:3d} enc={enc} dec={dec} heads={heads}')
        print('  ' + '-' * 60)
        print()
        
    def print_generation_start(self, gen: int, total_gens: int):
        """Print generation start."""
        self.gen_start_time = time.time()
        print()
        print('=' * 80)
        print(f'  GENERATION {gen}/{total_gens}')
        print('=' * 80)
        print()
        
    def print_config_start(self, idx: int, total: int, config: dict):
        """Print config training start."""
        cr = config['compression_ratio']
        d = config['d_model']
        enc = config['n_encoder_layers']
        dec = config['n_decoder_layers']
        heads = config['n_heads']
        params = d * d * 4 * (enc + dec)  # Rough estimate
        
        print(f'  [{idx}/{total}] Training config...')
        print(f'    Architecture:')
        print(f'      d_model={d}, n_heads={heads}')
        print(f'      encoder_layers={enc}, decoder_layers={dec}')
        print(f'      compression_ratio={cr}')
        print(f'      estimated_params={params:,}')
        print(f'    Training...', end='', flush=True)
        
    def print_config_result(self, idx: int, metrics: dict):
        """Print config training result."""
        acc = metrics.get('accuracy', 0)
        comp = metrics.get('compression_ratio', 0)
        fit = metrics.get('fitness', 0)
        params = metrics.get('param_count', 0)
        time_s = metrics.get('training_time', 0)
        
        status = 'OK' if acc >= 1.0 else '--'
        print(f' {status} Done ({time_s:.1f}s)')
        print(f'    Results:')
        print(f'      Accuracy:       {acc:.4%}')
        print(f'      Compression:    {comp:.2f}x')
        print(f'      Fitness:        {fit:.2f}')
        print(f'      Parameters:     {params:,}')
        print(f'      Latent size:    {metrics.get("latent_size", 0)}')
        print()
        
    def print_generation_stats(self, gen: int, scored: list, perfect_count: int):
        """Print generation statistics."""
        gen_time = time.time() - self.gen_start_time if self.gen_start_time else 0
        
        fitnesses = [m.get('fitness', 0) for _, m in scored]
        accuracies = [m.get('best_accuracy', 0) for _, m in scored]
        compressions = [m.get('compression_ratio', 0) for _, m in scored]
        
        best_idx = fitnesses.index(max(fitnesses))
        best_metrics = scored[best_idx][1]
        
        self.total_evals += len(scored)
        total_time = time.time() - self.start_time
        
        print('  ' + '-' * 60)
        print(f'  GENERATION {gen} SUMMARY:')
        print(f'    Time:           {gen_time:.1f}s (total: {total_time:.1f}s)')
        print(f'    Best Fitness:   {max(fitnesses):.2f}')
        print(f'    Avg Fitness:    {sum(fitnesses)/len(fitnesses):.2f}')
        print(f'    Best Accuracy:  {max(accuracies):.4%}')
        print(f'    Best Compress:  {max(compressions):.2f}x')
        print(f'    Perfect (100%): {perfect_count}/{len(scored)}')
        print(f'    Total Evals:    {self.total_evals}')
        print('  ' + '-' * 60)
        
        # Update tracking
        if max(accuracies) >= 1.0 and max(compressions) > self.best_compression:
            self.best_accuracy = max(accuracies)
            self.best_compression = max(compressions)
            print(f'    *** NEW BEST: {self.best_compression:.2f}x at {self.best_accuracy:.2%} ***')
        print()
        
    def print_final_results(self, best_config: dict, best_acc: float, best_comp: float):
        """Print final results."""
        total_time = time.time() - self.start_time
        
        print()
        print('=' * 80)
        print('  EVOLUTION COMPLETE')
        print('=' * 80)
        print(f'  Total Time:     {total_time:.1f}s ({total_time/60:.1f} min)')
        print(f'  Total Evals:    {self.total_evals}')
        print(f'  Best Accuracy:  {best_acc:.4%}')
        print(f'  Best Compress:  {best_comp:.2f}x')
        print()
        
        if best_config:
            print('  BEST CONFIGURATION:')
            print('  ' + '-' * 60)
            for k, v in best_config.items():
                print(f'    {k:25s}: {v}')
            print('  ' + '-' * 60)
        print()
        

# ==========================================
# 7. GENETIC EVOLUTION ENGINE
# ==========================================

class CompressionEvolution:
    """Evolves adapter configurations for maximum compression with 100% reconstruction."""
    
    def __init__(
        self,
        pop_size: int = 16,
        elite_count: int = 4,
        mutation_rate: float = 0.3,
        train_epochs: int = 50,
        target_accuracy: float = 1.0,
        device: str = 'cpu',
        use_gpu_engine: bool = False,
    ):
        self.pop_size = pop_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.train_epochs = train_epochs
        self.target_accuracy = target_accuracy
        self.device = device
        self.use_gpu_engine = use_gpu_engine
        
        self.population: List[dict] = []
        self.lineage: List[Dict] = []
        self.fitness_history: List[Dict] = []
        self.best_config = None
        self.best_fitness = 0
        self.best_accuracy = 0
        self.best_compression = 1.0
        self.generation = 0
        
        self.progress = ProgressDisplay()
        
        # Initialize GPU engine if requested
        if use_gpu_engine:
            self.gpu_engine = GPUEngine(device=device)
            info = self.gpu_engine.get_info()
            if info.get('success'):
                cuda_available = info.get('cuda_available', False)
                print(f'  GPU Engine: {info.get("torch_version")}')
                print(f'  CUDA: {cuda_available}')
                if cuda_available:
                    print(f'  GPU: {info.get("cuda_device_name")}')
                    print(f'  VRAM: {info.get("cuda_memory_total_mb", 0):.0f} MB')
                    # Update device to use CUDA
                    self.device = 'cuda'
                else:
                    print(f'  WARNING: CUDA not available, using CPU')
                    self.use_gpu_engine = False
                    self.device = 'cpu'
            else:
                print(f'  GPU Engine not available: {info.get("error", "Unknown error")}')
                print(f'  Falling back to CPU')
                self.use_gpu_engine = False
                self.device = 'cpu'
        
        # Final device check
        if self.device == 'cuda' and not torch.cuda.is_available():
            print(f'  WARNING: CUDA requested but PyTorch CUDA not available')
            print(f'  Falling back to CPU')
            self.device = 'cpu'
    
    def create_initial_population(self):
        """Create diverse initial population with different compression ratios."""
        print('  Creating initial population...')
        self.population = []
        
        seeds = [
            {'d_model': 256, 'n_encoder_layers': 6, 'n_decoder_layers': 6, 'n_heads': 8, 'compression_ratio': 0.125},
            {'d_model': 192, 'n_encoder_layers': 4, 'n_decoder_layers': 4, 'n_heads': 4, 'compression_ratio': 0.25},
            {'d_model': 128, 'n_encoder_layers': 4, 'n_decoder_layers': 4, 'n_heads': 4, 'compression_ratio': 0.375},
            {'d_model': 256, 'n_encoder_layers': 8, 'n_decoder_layers': 8, 'n_heads': 8, 'compression_ratio': 0.5},
            {'d_model': 192, 'n_encoder_layers': 6, 'n_decoder_layers': 6, 'n_heads': 4, 'compression_ratio': 0.625},
            {'d_model': 128, 'n_encoder_layers': 4, 'n_decoder_layers': 4, 'n_heads': 4, 'compression_ratio': 0.75},
            {'d_model': 128, 'n_encoder_layers': 4, 'n_decoder_layers': 4, 'n_heads': 4, 'compression_ratio': 0.875},
            {'d_model': 128, 'n_encoder_layers': 4, 'n_decoder_layers': 4, 'n_heads': 4, 'compression_ratio': 1.0},
        ]
        
        for s in seeds:
            config = {
                'vocab_size': 256,
                'd_model': s['d_model'],
                'n_encoder_layers': s['n_encoder_layers'],
                'n_decoder_layers': s['n_decoder_layers'],
                'n_heads': s['n_heads'],
                'ff_mult': 4,
                'dropout': 0.0,
                'compression_ratio': s['compression_ratio'],
                'max_seq_len': 512,
                'activation': 'gelu',
            }
            self.population.append(config)
        
        while len(self.population) < self.pop_size:
            self.population.append(random_config())
        
        self.progress.print_population(self.population)
    
    def train_and_evaluate(self, config: dict, train_samples: List[bytes]) -> Dict:
        """Train an adapter and evaluate its fitness."""
        # Try GPU engine first
        if self.use_gpu_engine and self.gpu_engine.available:
            print(f' [GPU]', end='', flush=True)
            result = self.gpu_engine.train(config, self.train_epochs)
            if result.get('success') and result.get('accuracy', 0) > 0:
                return result
            else:
                error = result.get('error', 'Unknown error')
                print(f' FAILED: {error}', end='', flush=True)
                print(f' -> Falling back to CPU', end='', flush=True)
        
        # CPU training
        try:
            adapter = EvolvableAdapter(config).to(self.device)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_epochs)
            
            best_acc = 0
            best_fitness = 0
            best_metrics = {}
            start_time = time.time()
            
            adapter.train()
            for epoch in range(self.train_epochs):
                epoch_loss = 0
                epoch_correct = 0
                epoch_total = 0
                num_batches = 0
                
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
                    
                    x = torch.tensor(padded, dtype=torch.long).to(self.device)
                    
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
                    metrics = evaluate_fitness_cpu(adapter, train_samples, self.device)
                    best_fitness = metrics['fitness']
                    best_metrics = metrics
            
            elapsed = time.time() - start_time
            best_metrics['training_time'] = elapsed
            best_metrics['best_accuracy'] = best_acc
            best_metrics['fitness'] = best_fitness
            
            return best_metrics
            
        except Exception as e:
            import traceback
            print(f' CPU ERROR: {str(e)}', end='', flush=True)
            return {
                'fitness': 0,
                'accuracy': 0,
                'best_accuracy': 0,
                'avg_loss': float('inf'),
                'compression_ratio': 0,
                'memory_savings': 0,
                'param_count': 0,
                'latent_size': 0,
                'training_time': 0,
                'error': str(e),
            }
    
    def evolve_generation(self, train_samples: List[bytes], total_gens: int):
        """Run one generation of evolution."""
        self.generation += 1
        self.progress.print_generation_start(self.generation, total_gens)
        
        # Evaluate all configs
        scored = []
        for i, config in enumerate(self.population):
            self.progress.print_config_start(i + 1, len(self.population), config)
            metrics = self.train_and_evaluate(config, train_samples)
            scored.append((config, metrics))
            self.progress.print_config_result(i + 1, metrics)
        
        # Sort by fitness
        scored.sort(key=lambda x: x[1].get('fitness', 0), reverse=True)
        
        # Filter for 100% accuracy configs
        perfect_configs = [(c, m) for c, m in scored if m.get('best_accuracy', 0) >= 1.0]
        
        self.progress.print_generation_stats(self.generation, scored, len(perfect_configs))
        
        # Track lineage
        if scored:
            best_cfg, best_m = scored[0]
            sig = config_signature(best_cfg)
            self.lineage.append({
                'generation': self.generation,
                'signature': sig,
                'config': best_cfg,
                'fitness': best_m.get('fitness', 0),
                'accuracy': best_m.get('best_accuracy', 0),
                'compression': best_m.get('compression_ratio', 0),
            })
        
        # Selection: keep elites
        if perfect_configs:
            elites = [cfg for cfg, _ in perfect_configs[:self.elite_count]]
        else:
            elites = [cfg for cfg, _ in scored[:self.elite_count]]
        
        # Generate next generation
        new_pop = list(elites)
        
        while len(new_pop) < self.pop_size:
            if random.random() < 0.7:
                parent = random.choice(elites)
                child = mutate_config(parent, self.mutation_rate)
            else:
                p1, p2 = random.sample(elites, 2)
                child = crossover_configs(p1, p2)
            new_pop.append(child)
        
        self.population = new_pop
        
        # Update best tracking
        if scored:
            best_m = scored[0][1]
            if best_m.get('best_accuracy', 0) >= 1.0:
                self.best_accuracy = best_m['best_accuracy']
                self.best_config = scored[0][0]
                self.best_fitness = best_m['fitness']
                comp = best_m.get('compression_ratio', 1.0)
                if comp > self.best_compression:
                    self.best_compression = comp
    
    def run(self, num_generations: int = 10):
        """Run full evolution."""
        self.progress.print_header(self.pop_size, self.elite_count, self.train_epochs, self.device)
        
        train_samples = generate_diverse_data(500)
        self.create_initial_population()
        
        for gen in range(num_generations):
            self.evolve_generation(train_samples, num_generations)
        
        self.progress.print_final_results(self.best_config, self.best_accuracy, self.best_compression)
        
        # Save results
        results = {
            'best_config': self.best_config,
            'best_accuracy': self.best_accuracy,
            'best_compression': self.best_compression,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'lineage': self.lineage,
        }
        
        with open('compression_evolution_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Results saved to: compression_evolution_results.json')
        
        if self.best_config:
            adapter = EvolvableAdapter(self.best_config).to(self.device)
            torch.save({
                'config': self.best_config,
                'state_dict': adapter.state_dict(),
                'accuracy': self.best_accuracy,
                'compression': self.best_compression,
            }, 'lait_best_compression.pt')
            print(f'  Model saved to:   lait_best_compression.pt')
        
        return self.best_config, self.best_accuracy, self.best_compression


# ==========================================
# 8. MAIN
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LAIT Genetic Compression Search (Verbose)")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--population", type=int, default=16, help="Population size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per config")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, auto")
    parser.add_argument("--gpu-engine", action="store_true", help="Use external GPU engine")
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    evo = CompressionEvolution(
        pop_size=args.population,
        elite_count=max(2, args.population // 4),
        mutation_rate=0.3,
        train_epochs=args.epochs,
        target_accuracy=1.0,
        device=device,
        use_gpu_engine=args.gpu_engine,
    )
    
    best_config, best_acc, best_comp = evo.run(num_generations=args.generations)
    
    print('=' * 80)
    print('  FINAL RESULTS')
    print('=' * 80)
    print(f'  Best compression with 100% accuracy: {best_comp:.2f}x')
    print(f'  Accuracy: {best_acc:.4%}')
    print()
    print('  To use this model:')
    print('    python -c "from src.evolve_adapter import EvolvableAdapter; ...')
    print('=' * 80)
