#!/usr/bin/env python3
"""
LAIT MCP Chat Client
Demonstrates full pipeline: text → compress → Ollama → decompress → output

Usage:
  python lait_mcp_chat.py
  
Prerequisites:
  1. Train adapter: python lait_mcp_adapter.py --train --epochs 100
  2. Start MCP server: python lait_mcp_server.py --port 8001
  3. Run this chat: python lait_mcp_chat.py
"""

import requests
import json
import sys
import time
from typing import Optional

# Configuration
MCP_SERVER_URL = "http://localhost:8001"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "jessup-sim:granite4.1"

class LAITMCPChat:
    """Chat client with LAIT compression via MCP."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.conversation_history = []
        self.compressed_cache = {}
        
    def check_services(self) -> bool:
        """Check if MCP server and Ollama are running."""
        try:
            # Check MCP server
            mcp_health = requests.get(f"{MCP_SERVER_URL}/health", timeout=5)
            if mcp_health.status_code != 200:
                print("ERROR: MCP server not responding")
                return False
            print("✓ MCP server is running")
            
            # Check Ollama
            ollama_health = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if ollama_health.status_code != 200:
                print("ERROR: Ollama not responding")
                return False
            print("✓ Ollama is running")
            
            return True
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    
    def compress_text(self, text: str) -> dict:
        """Compress text using LAIT MCP server."""
        try:
            response = requests.post(
                f"{MCP_SERVER_URL}/compress",
                json={"text": text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Compression failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"Compression error: {e}")
            return None
    
    def decompress_text(self, cache_key: str) -> str:
        """Decompress text using LAIT MCP server."""
        try:
            response = requests.post(
                f"{MCP_SERVER_URL}/decompress",
                json={"cache_key": cache_key},
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                print(f"Decompression failed: {response.status_code}")
                return ""
        except Exception as e:
            print(f"Decompression error: {e}")
            return ""
    
    def get_stats(self) -> dict:
        """Get compression statistics."""
        try:
            response = requests.get(f"{MCP_SERVER_URL}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def send_to_ollama(self, messages: list, compress: bool = True) -> str:
        """Send messages to Ollama with optional compression."""
        try:
            if compress and len(messages) > 1:
                # Compress older messages
                compressed_messages = []
                for i, msg in enumerate(messages):
                    if msg['role'] == 'system':
                        compressed_messages.append(msg)
                    elif i == len(messages) - 1:
                        # Keep last message uncompressed
                        compressed_messages.append(msg)
                    else:
                        # Compress this message
                        result = self.compress_text(msg['content'])
                        if result:
                            compressed_msg = {
                                'role': msg['role'],
                                'content': f"[LAIT Compressed {result['compression_ratio']:.1f}x, Key: {result['cache_key'][:8]}...] {msg['content'][:100]}..."
                            }
                            compressed_messages.append(compressed_msg)
                            self.compressed_cache[result['cache_key']] = msg['content']
                        else:
                            compressed_messages.append(msg)
                messages = compressed_messages
            
            # Send to Ollama
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                return f"Error: Ollama returned {response.status_code}"
                
        except Exception as e:
            return f"Error: {e}"
    
    def chat(self, user_input: str) -> str:
        """Process user input and return response."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Check for special commands
        if user_input.lower().startswith("/compress "):
            text_to_compress = user_input[10:]
            result = self.compress_text(text_to_compress)
            if result:
                self.compressed_cache[result['cache_key']] = text_to_compress
                return (f"✓ Compressed successfully!\n"
                       f"  Original: {result['original_length']} tokens\n"
                       f"  Latent: {result['latent_length']} vectors\n"
                       f"  Ratio: {result['compression_ratio']:.1f}x\n"
                       f"  Cache key: {result['cache_key']}")
            return "Compression failed"
        
        elif user_input.lower().startswith("/decompress "):
            cache_key = user_input[12:]
            text = self.decompress_text(cache_key)
            if text:
                return f"✓ Decompressed:\n{text}"
            return "Decompression failed"
        
        elif user_input.lower() == "/stats":
            stats = self.get_stats()
            if stats:
                return (f"📊 Compression Statistics:\n"
                       f"  Cached items: {stats.get('num_compressed', 0)}\n"
                       f"  Total original tokens: {stats.get('total_original_tokens', 0)}\n"
                       f"  Total latent vectors: {stats.get('total_latent_vectors', 0)}\n"
                       f"  Overall compression: {stats.get('overall_compression_ratio', 0):.1f}x\n"
                       f"  Memory saved: {stats.get('estimated_memory_saved_mb', 0):.2f} MB")
            return "Could not get stats"
        
        elif user_input.lower() == "/clear":
            try:
                requests.post(f"{MCP_SERVER_URL}/clear", timeout=10)
                self.compressed_cache.clear()
                return "✓ Cache cleared"
            except:
                return "Failed to clear cache"
        
        elif user_input.lower() == "/help":
            return ("LAIT MCP Chat Commands:\n"
                   "  /compress <text>  - Compress text to latent representation\n"
                   "  /decompress <key> - Decompress text by cache key\n"
                   "  /stats           - Show compression statistics\n"
                   "  /clear           - Clear compression cache\n"
                   "  /help            - Show this help\n"
                   "  /quit            - Exit chat\n\n"
                   "Just type normally to chat with LAIT compression!")
        
        elif user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            return "Goodbye!"
        
        # Send to Ollama with compression
        response = self.send_to_ollama(self.conversation_history, compress=True)
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def run(self):
        """Run interactive chat."""
        print("\n" + "="*60)
        print("LAIT MCP CHAT")
        print("="*60)
        print("Type /help for commands, /quit to exit")
        print("Context will be automatically compressed for efficiency")
        print("="*60 + "\n")
        
        if not self.check_services():
            print("\nPlease start the required services:")
            print("  1. MCP Server: python lait_mcp_server.py --port 8001")
            print("  2. Ollama: ollama serve")
            return
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                
                response = self.chat(user_input)
                print(f"\nAssistant: {response}")
                
                if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                    break
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LAIT MCP Chat Client")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--mcp-url", type=str, default=MCP_SERVER_URL, help="MCP server URL")
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_URL, help="Ollama URL")
    args = parser.parse_args()
    
    MCP_SERVER_URL = args.mcp_url
    OLLAMA_URL = args.ollama_url
    
    chat = LAITMCPChat(model=args.model)
    chat.run()
