#!/usr/bin/env python3
"""
LAIT Demo - Interactive demonstration of LAIT architecture capabilities.
Shows compression, memory savings, and Ollama integration.
"""

import torch
import time
import sys
from lait_export import LAITModel, LAITConfig, get_winning_config

def print_header():
    """Print demo header."""
    print("\n" + "="*70)
    print("LAIT - Latent Attention in Tokens")
    print("Evolved Architecture for Efficient Context Compression")
    print("="*70)

def print_stats(model, seq_len):
    """Print memory statistics for given sequence length."""
    stats = model.get_memory_stats(seq_len)
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 45)
    print(f"{'Sequence Length':<30} {seq_len:>15}")
    print(f"{'Latent Vectors':<30} {stats['latent_size']:>15}")
    print(f"{'Compression Ratio':<30} {stats['compression_ratio']:>14.1f}x")
    print(f"{'Standard KV Cache':<30} {stats['standard_kv_bytes']/1024/1024:>13.2f} MB")
    print(f"{'LAIT Memory':<30} {stats['lait_memory_bytes']/1024:>13.2f} KB")
    print(f"{'Memory Savings':<30} {stats['memory_savings']:>14.1f}x")

def demo_compression(model):
    """Demonstrate compression across different sequence lengths."""
    print("\n" + "="*70)
    print("COMPRESSION DEMONSTRATION")
    print("="*70)
    
    print("\nHow LAIT compresses context at different scales:\n")
    
    for seq_len in [128, 256, 512, 1024, 2048, 4096]:
        stats = model.get_memory_stats(seq_len)
        bar_len = int(stats['compression_ratio'] / 10)
        bar = "#" * min(bar_len, 30)
        
        print(f"{seq_len:>5} tokens -> {stats['latent_size']:>4} latent ({stats['compression_ratio']:>6.1f}x) {bar}")

def demo_forward_pass(model):
    """Demonstrate a forward pass."""
    print("\n" + "="*70)
    print("FORWARD PASS DEMONSTRATION")
    print("="*70)
    
    # Create dummy input
    seq_len = 512
    batch_size = 1
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    
    print(f"\nInput: {seq_len} random tokens")
    print(f"Model: LAIT Hybrid (4-2-1 architecture)")
    
    # Time forward pass
    start = time.time()
    with torch.no_grad():
        outputs = model(input_ids, return_reconstruction=True)
    elapsed = time.time() - start
    
    latent = outputs["latent_memory"]
    recon = outputs["reconstruction_logits"]
    
    print(f"\nOutput:")
    print(f"  Latent memory: {list(latent.shape)}")
    print(f"  Reconstruction: {list(recon.shape)}")
    print(f"  Forward time: {elapsed*1000:.1f}ms")
    print(f"  Compression: {outputs['compression_ratio']:.1f}x")

def demo_architecture(model):
    """Show architecture details."""
    print("\n" + "="*70)
    print("ARCHITECTURE DETAILS")
    print("="*70)
    
    config = model.config
    
    print(f"\nWinning Configuration (Fitness: 119.7):")
    print(f"  Bottleneck: {config.bottleneck_type}")
    print(f"  Initial Encoder: {config.n_initial_encoder_layers} layers")
    print(f"  Compressor: {config.n_compressor_layers} layers")
    print(f"  Decoder: {config.n_decoder_layers} layers")
    print(f"  d_model: {config.d_model}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  Compression ratio: {config.compression_ratio}")
    print(f"  Dynamic resizing: {config.dynamic_resizing}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Memory breakdown
    stats = model.get_memory_stats(2048)
    print(f"\nFor 2048 token context:")
    print(f"  Standard KV cache: {stats['standard_kv_bytes']/1024/1024:.2f} MB")
    print(f"  LAIT latent: {stats['lait_memory_bytes']/1024:.2f} KB")
    print(f"  Savings: {stats['memory_savings']:.0f}x")

def demo_evolution_history():
    """Show evolutionary history."""
    print("\n" + "="*70)
    print("EVOLUTIONARY HISTORY")
    print("="*70)
    
    print("\nThis architecture was evolved through 3 generations of search:")
    
    history = [
        ("300s Run", "14 generations", "105.2", "320x memory, 64x compression"),
        ("600s Run", "8 generations", "119.4", "Hybrid bottleneck discovered"),
        ("900s Run", "13 generations", "119.7", "Final optimization")
    ]
    
    print(f"\n{'Run':<12} {'Generations':<15} {'Fitness':<10} {'Key Achievement'}")
    print("-" * 70)
    for run, gens, fitness, achievement in history:
        print(f"{run:<12} {gens:<15} {fitness:<10} {achievement}")
    
    print("\nLineage:")
    print("  cx_hist_hist_g1 (crossover of historical winners)")
    print("  Parents: hist_leap_cx_cx_cx + hist_mut_mnn_g14")
    print("  Architecture: hybrid bottleneck with pooling + attention")

def demo_ollama_integration():
    """Show Ollama integration."""
    print("\n" + "="*70)
    print("OLLAMA INTEGRATION")
    print("="*70)
    
    print("\nTo use LAIT with Ollama:")
    print("\n1. Direct Ollama usage:")
    print("   ollama create lait -f lait_model/Modelfile")
    print("   ollama run lait")
    
    print("\n2. Python server (recommended):")
    print("   pip install fastapi uvicorn")
    print("   python lait_ollama_server.py")
    print("   # Server runs on http://localhost:11434")
    print("   ollama run lait")
    
    print("\n3. API usage:")
    print('   curl -s http://localhost:11434/api/generate \\')
    print('     -d \'{"model": "lait", "prompt": "Hello, LAIT!"}\'')
    
    print("\n4. Python client:")
    print("   import requests")
    print('   response = requests.post(')
    print('       "http://localhost:11434/api/generate",')
    print('       json={"model": "lait", "prompt": "Hello, LAIT!"}')
    print('   )')
    print('   print(response.json()["response"])')

def main():
    """Run the interactive demo."""
    print_header()
    
    # Load model
    print("\nLoading LAIT model...")
    config = get_winning_config()
    model = LAITModel(config)
    model.eval()
    
    print(f"Loaded: {config.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    while True:
        print("\n" + "="*70)
        print("DEMO MENU")
        print("="*70)
        print("1. Compression demonstration")
        print("2. Forward pass demonstration")
        print("3. Architecture details")
        print("4. Evolution history")
        print("5. Ollama integration")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            demo_compression(model)
        elif choice == "2":
            demo_forward_pass(model)
        elif choice == "3":
            demo_architecture(model)
        elif choice == "4":
            demo_evolution_history()
        elif choice == "5":
            demo_ollama_integration()
        elif choice == "6":
            print("\nThank you for exploring LAIT!")
            print("For more information, see README.md")
            sys.exit(0)
        else:
            print("Invalid option. Please select 1-6.")


if __name__ == "__main__":
    main()
