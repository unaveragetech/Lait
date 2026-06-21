#!/usr/bin/env python3
"""
LAIT Ollama Full Yield Demo
Runs all prompts from prompts.json through LAIT compression + Ollama model.
Produces full yield results with every stat a user would need to validate the system.

Usage:
  python lait_ollama_demo.py
  python lait_ollama_demo.py --model lait-granite
  python lait_ollama_demo.py --no-ollama
  python lait_ollama_demo.py --prompt "custom text"
"""

import torch
import json
import time
import sys
import os
import argparse
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__) or '.')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evolve_adapter import EvolvableAdapter

# ==========================================
# 1. CONFIGURATION
# ==========================================

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "lait-granite"
ADAPTER_PATH = "lait_adapter_best.pt"
PROMPTS_PATH = "prompts.json"
RESULTS_PATH = "lait_ollama_demo_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 2. ADAPTER LOADER
# ==========================================

def load_adapter(path=ADAPTER_PATH):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']
    adapter = EvolvableAdapter(config)
    adapter.load_state_dict(checkpoint['state_dict'])
    adapter.eval()
    adapter = adapter.to(DEVICE)
    return adapter, config, checkpoint


# ==========================================
# 3. COMPRESSION ENGINE
# ==========================================

def compress_text(adapter, text):
    text_bytes = text.encode('utf-8')
    tokens = list(text_bytes)
    original_len = len(tokens)

    max_seq_len = adapter.config.get('max_seq_len', 512)
    truncated = len(tokens) > max_seq_len
    if truncated:
        tokens = tokens[:max_seq_len]

    padded = tokens + [0] * (max_seq_len - len(tokens))
    x = torch.tensor([padded], dtype=torch.long).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits, latent, orig_len = adapter(x)
    t1 = time.perf_counter()

    latent_len = latent.shape[1]
    latent_dim = latent.shape[2]
    compression_ratio = original_len / max(latent_len, 1)

    first_token = [tokens[0]]
    predicted = logits[0, :len(tokens)-1, :].argmax(dim=-1).tolist()
    recon_tokens = first_token + predicted
    reconstructed = bytes(recon_tokens[:len(tokens)])
    recon_text = reconstructed.decode('utf-8', errors='replace')

    correct = sum(a == b for a, b in zip(text_bytes, reconstructed))
    accuracy = correct / max(len(text_bytes), 1)
    exact_match = (text == recon_text)

    latent_float32_bytes = latent_len * latent_dim * 4
    latent_float64_bytes = latent_len * latent_dim * 8
    original_bytes = original_len * 1

    latent_np = latent.squeeze(0).cpu().numpy()
    latent_json = json.dumps(latent_np.tolist())
    latent_json_bytes = len(latent_json.encode('utf-8'))

    effective_pct = round((1 - 1/compression_ratio) * 100, 1) if compression_ratio > 1 else 0.0

    return {
        'text': text,
        'original_bytes': original_bytes,
        'original_tokens': original_len,
        'latent_length': latent_len,
        'latent_dim': latent_dim,
        'latent_float32_bytes': latent_float32_bytes,
        'latent_float64_bytes': latent_float64_bytes,
        'latent_json_bytes': latent_json_bytes,
        'compression_ratio': round(compression_ratio, 2),
        'effective_compression_pct': effective_pct,
        'inference_time_ms': round((t1 - t0) * 1000, 2),
        'reconstruction_accuracy': round(accuracy, 6),
        'exact_match': exact_match,
        'correct_bytes': correct,
        'total_bytes': len(text_bytes),
        'reconstructed_text': recon_text,
        'truncated': truncated,
        'device': DEVICE,
    }


# ==========================================
# 4. OLLAMA CLIENT
# ==========================================

def check_ollama():
    try:
        r = requests.get(OLLAMA_URL + "/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def list_ollama_models():
    try:
        r = requests.get(OLLAMA_URL + "/api/tags", timeout=5)
        if r.status_code == 200:
            return [m['name'] for m in r.json().get('models', [])]
    except Exception:
        pass
    return []


def query_ollama(model, prompt, system=None, temperature=0.7):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(OLLAMA_URL + "/api/chat", json=payload, timeout=120)
        t1 = time.perf_counter()
        if r.status_code == 200:
            result = r.json()
            content = result.get("message", {}).get("content", "")
            tokens_eval = result.get("eval_count", 0)
            tokens_prompt = result.get("prompt_eval_count", 0)
            duration_ns = result.get("eval_duration", 0)
            tokens_per_sec = tokens_eval / (duration_ns / 1e9) if duration_ns > 0 else 0
            return {
                'success': True,
                'response': content,
                'response_bytes': len(content.encode('utf-8')),
                'response_words': len(content.split()),
                'prompt_eval_count': tokens_prompt,
                'eval_count': tokens_eval,
                'eval_duration_ms': round(duration_ns / 1e6, 2),
                'tokens_per_sec': round(tokens_per_sec, 2),
                'latency_ms': round((t1 - t0) * 1000, 2),
                'model': model,
            }
        else:
            return {'success': False, 'error': "HTTP " + str(r.status_code),
                    'latency_ms': round((t1 - t0) * 1000, 2)}
    except Exception as e:
        t1 = time.perf_counter()
        return {'success': False, 'error': str(e),
                'latency_ms': round((t1 - t0) * 1000, 2)}


# ==========================================
# 5. PROMPT LOADER
# ==========================================

def load_prompts(path=PROMPTS_PATH):
    with open(path, 'r') as f:
        data = json.load(f)
    prompts = []
    for key, val in data.get('prompts', {}).items():
        prompts.append({
            'id': key,
            'text': val['text'],
            'category': val.get('category', 'unknown'),
            'description': val.get('description', ''),
            'declared_tokens': val.get('tokens', 0),
        })
    return prompts


# ==========================================
# 6. MAIN DEMO
# ==========================================

def run_demo(model=DEFAULT_MODEL, use_ollama=True, custom_prompt=None):
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': DEVICE,
        'model': model,
        'adapter_path': ADAPTER_PATH,
        'adapter_config': {},
        'adapter_params': 0,
        'ollama_available': False,
        'ollama_models': [],
        'prompts_total': 0,
        'results': [],
        'summary': {},
    }

    sep = '=' * 80
    sep2 = '-' * 80

    print(sep)
    print('  LAIT OLLAMA FULL YIELD DEMO')
    print(sep)
    print('  Timestamp : ' + results['timestamp'])
    print('  Device    : ' + DEVICE)
    print('  Model     : ' + model)
    print(sep)
    print()

    # Load adapter
    print('[1/4] Loading LAIT adapter...')
    try:
        adapter, config, ckpt = load_adapter()
        n_params = sum(p.numel() for p in adapter.parameters())
        results['adapter_config'] = config
        results['adapter_params'] = n_params
        print('  OK   ' + str(n_params) + ' parameters')
        print('       d_model=' + str(config['d_model'])
              + ', enc=' + str(config['n_encoder_layers'])
              + ', dec=' + str(config['n_decoder_layers'])
              + ', heads=' + str(config['n_heads']))
        print('       compression_ratio=' + str(config['compression_ratio'])
              + ', max_seq_len=' + str(config['max_seq_len']))
        print('       train_acc=' + str(ckpt.get('train_accuracy', 'N/A'))
              + ', test_acc=' + str(ckpt.get('test_accuracy', 'N/A')))
    except Exception as e:
        print('  FAIL Could not load adapter: ' + str(e))
        return results
    print()

    # Check Ollama
    if use_ollama:
        print('[2/4] Checking Ollama...')
        ollama_ok = check_ollama()
        results['ollama_available'] = ollama_ok
        if ollama_ok:
            models = list_ollama_models()
            results['ollama_models'] = models
            print('  OK   Ollama is running')
            print('       Models: ' + ', '.join(models))
            model_found = any(model in m for m in models)
            if not model_found:
                print('  WARN Model "' + model + '" not found. Will try anyway.')
        else:
            print('  WARN Ollama not available. Compression-only mode.')
        print()
    else:
        print('[2/4] Ollama check skipped (--no-ollama)')
        print()

    # Load prompts
    print('[3/4] Loading prompts...')
    if custom_prompt:
        prompts = [{
            'id': 'custom_1',
            'text': custom_prompt,
            'category': 'custom',
            'description': 'User-provided',
            'declared_tokens': len(custom_prompt.encode('utf-8')),
        }]
    else:
        prompts = load_prompts()
    results['prompts_total'] = len(prompts)
    print('  OK   ' + str(len(prompts)) + ' prompts loaded')
    cats = {}
    for p in prompts:
        c = p['category']
        cats[c] = cats.get(c, 0) + 1
    for c in sorted(cats.keys()):
        print('       ' + c + ': ' + str(cats[c]))
    print()

    # Process prompts
    print('[4/4] Processing prompts...')
    print(sep2)

    total_in = 0
    total_out_json = 0
    total_out_f32 = 0
    total_match = 0
    total_tests = 0
    total_inference_ms = 0.0
    total_ollama_latency_ms = 0.0
    total_ollama_tokens = 0
    total_ollama_speed = 0.0
    total_ollama_count = 0
    cat_stats = {}

    for i, prompt in enumerate(prompts):
        pid = prompt['id']
        text = prompt['text']
        cat = prompt['category']
        text_bytes_len = len(text.encode('utf-8'))

        print()
        print('  [' + str(i+1).zfill(2) + '/' + str(len(prompts)) + '] '
              + pid + ' (' + cat + ')')
        preview = text[:72] + ('...' if len(text) > 72 else '')
        print('  Text: ' + preview)
        print('  Size: ' + str(text_bytes_len) + ' bytes | '
              + str(len(text)) + ' chars')

        # Compress
        comp = compress_text(adapter, text)
        total_in += comp['original_bytes']
        total_out_json += comp['latent_json_bytes']
        total_out_f32 += comp['latent_float32_bytes']
        total_tests += 1
        total_inference_ms += comp['inference_time_ms']
        if comp['exact_match']:
            total_match += 1

        # Category stats
        if cat not in cat_stats:
            cat_stats[cat] = {'count': 0, 'match': 0, 'total_bytes': 0,
                              'total_latent_json': 0, 'total_inference_ms': 0.0}
        cat_stats[cat]['count'] += 1
        cat_stats[cat]['total_bytes'] += comp['original_bytes']
        cat_stats[cat]['total_latent_json'] += comp['latent_json_bytes']
        cat_stats[cat]['total_inference_ms'] += comp['inference_time_ms']
        if comp['exact_match']:
            cat_stats[cat]['match'] += 1

        print()
        print('  --- LAIT Compression ---')
        print('  Input tokens   : ' + str(comp['original_tokens']))
        print('  Latent length  : ' + str(comp['latent_length'])
              + ' x ' + str(comp['latent_dim']) + ' dim')
        print('  Latent size    : ' + '{:,}'.format(comp['latent_json_bytes'])
              + ' bytes (JSON) | '
              + '{:,}'.format(comp['latent_float32_bytes'])
              + ' bytes (float32)')
        print('  Compression    : ' + str(comp['compression_ratio']) + 'x ('
              + str(comp['effective_compression_pct']) + '% saved)')
        print('  Inference      : ' + str(round(comp['inference_time_ms'], 2))
              + ' ms on ' + comp['device'])
        print('  Recon accuracy : '
              + str(round(comp['reconstruction_accuracy'] * 100, 2)) + '% ('
              + str(comp['correct_bytes']) + '/' + str(comp['total_bytes'])
              + ' bytes)')
        print('  Exact match    : ' + ('YES' if comp['exact_match'] else 'NO'))

        if comp['truncated']:
            print('  WARNING: Input truncated from '
                  + str(text_bytes_len) + ' to '
                  + str(comp['original_tokens']) + ' tokens')

        # Ollama query
        ollama_result = None
        if use_ollama and results['ollama_available']:
            system_msg = ("You are LAIT Granite, a language model. "
                          "Respond concisely and accurately.")
            ollama_result = query_ollama(model, text, system=system_msg)
            total_ollama_latency_ms += ollama_result.get('latency_ms', 0)
            if ollama_result.get('success'):
                total_ollama_tokens += ollama_result.get('eval_count', 0)
                total_ollama_speed += ollama_result.get('tokens_per_sec', 0)
                total_ollama_count += 1

            print()
            print('  --- Ollama Response ---')
            if ollama_result['success']:
                resp = ollama_result['response']
                resp_preview = resp[:200] + ('...' if len(resp) > 200 else '')
                print('  Response       : ' + resp_preview)
                print('  Response size  : '
                      + str(ollama_result['response_bytes']) + ' bytes')
                print('  Response words : '
                      + str(ollama_result['response_words']))
                print('  Prompt tokens  : '
                      + str(ollama_result['prompt_eval_count']))
                print('  Eval tokens    : '
                      + str(ollama_result['eval_count']))
                print('  Speed          : '
                      + str(round(ollama_result['tokens_per_sec'], 1))
                      + ' tok/s')
                print('  Latency        : '
                      + str(round(ollama_result['latency_ms'], 0))
                      + ' ms')
            else:
                print('  ERROR          : ' + ollama_result['error'])
        elif use_ollama:
            print()
            print('  --- Ollama ---')
            print('  SKIPPED: Ollama not available')
        else:
            print()
            print('  --- Ollama ---')
            print('  SKIPPED: --no-ollama flag')

        # Decompress verification
        print()
        print('  --- Decompress Verify ---')
        re_tokens = list(text.encode('utf-8'))
        re_padded = re_tokens + [0] * (512 - len(re_tokens))
        re_x = torch.tensor([re_padded], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            re_logits, re_latent, re_orig = adapter(re_x)
        re_first = [re_tokens[0]]
        re_pred = re_logits[0, :len(re_tokens)-1, :].argmax(dim=-1).tolist()
        re_recon = re_first + re_pred
        re_bytes = bytes(re_recon[:len(re_tokens)])
        re_text = re_bytes.decode('utf-8', errors='replace')
        re_match = (text == re_text)
        print('  Decompressed   : '
              + re_text[:72] + ('...' if len(re_text) > 72 else ''))
        print('  Decompress OK  : ' + ('YES' if re_match else 'NO'))

        # Save per-prompt result
        entry = {
            'id': pid,
            'category': cat,
            'description': prompt['description'],
            'input': {
                'text': text,
                'bytes': text_bytes_len,
                'chars': len(text),
                'declared_tokens': prompt['declared_tokens'],
            },
            'lait': {
                'original_tokens': comp['original_tokens'],
                'latent_length': comp['latent_length'],
                'latent_dim': comp['latent_dim'],
                'latent_json_bytes': comp['latent_json_bytes'],
                'latent_float32_bytes': comp['latent_float32_bytes'],
                'compression_ratio': comp['compression_ratio'],
                'effective_compression_pct': comp['effective_compression_pct'],
                'inference_time_ms': comp['inference_time_ms'],
                'reconstruction_accuracy': comp['reconstruction_accuracy'],
                'exact_match': comp['exact_match'],
                'correct_bytes': comp['correct_bytes'],
                'total_bytes': comp['total_bytes'],
                'truncated': comp['truncated'],
            },
            'decompress': {
                'exact_match': re_match,
            },
        }

        if ollama_result:
            entry['ollama'] = ollama_result

        results['results'].append(entry)

        print()
        print(sep2)

    # Final summary
    print()
    print(sep)
    print('  SUMMARY')
    print(sep)
    print()
    print('  Adapter:')
    print('    Parameters    : ' + '{:,}'.format(results['adapter_params']))
    print('    Config        : d=' + str(config['d_model'])
          + ' enc=' + str(config['n_encoder_layers'])
          + ' dec=' + str(config['n_decoder_layers'])
          + ' heads=' + str(config['n_heads']))
    print('    Device        : ' + DEVICE)
    print()

    print('  Compression:')
    print('    Total inputs  : ' + str(total_tests) + ' prompts')
    print('    Total in      : ' + '{:,}'.format(total_in) + ' bytes')
    print('    Total out     : ' + '{:,}'.format(total_out_json)
          + ' bytes (JSON)')
    print('    Total out     : ' + '{:,}'.format(total_out_f32)
          + ' bytes (float32)')
    if total_in > 0:
        overall_ratio = total_in / max(total_out_json, 1)
        overall_pct = round((1 - total_out_json / max(total_in, 1)) * 100, 1)
        print('    Overall ratio : ' + str(round(overall_ratio, 2)) + 'x ('
              + str(overall_pct) + '% saved)')
    print('    Exact matches : ' + str(total_match) + '/' + str(total_tests)
          + ' (' + str(round(total_match / max(total_tests, 1) * 100, 1))
          + '%)')
    print('    Total infer   : ' + str(round(total_inference_ms, 1)) + ' ms')
    if total_tests > 0:
        avg_ms = round(total_inference_ms / total_tests, 2)
        print('    Avg infer     : ' + str(avg_ms) + ' ms/prompt')
    print()

    if use_ollama and results['ollama_available']:
        print('  Ollama:')
        print('    Model         : ' + model)
        print('    Prompts sent  : ' + str(total_ollama_count))
        print('    Total latency : '
              + str(round(total_ollama_latency_ms, 0)) + ' ms')
        if total_ollama_count > 0:
            avg_lat = round(total_ollama_latency_ms / total_ollama_count, 0)
            print('    Avg latency   : ' + str(avg_lat) + ' ms/prompt')
        print('    Total tokens  : ' + str(total_ollama_tokens))
        if total_ollama_count > 0:
            avg_speed = round(total_ollama_speed / total_ollama_count, 1)
            print('    Avg speed     : '
                  + str(avg_speed) + ' tok/s')
        print()

    print('  Per-Category Breakdown:')
    for cat in sorted(cat_stats.keys()):
        cs = cat_stats[cat]
        cat_ratio = 'N/A'
        if cs['total_latent_json'] > 0:
            r = cs['total_bytes'] / max(cs['total_latent_json'], 1)
            cat_ratio = str(round(r, 2)) + 'x'
        print('    ' + cat.ljust(12)
              + ' : ' + str(cs['match']) + '/' + str(cs['count'])
              + ' match | '
              + '{:,}'.format(cs['total_bytes']) + ' bytes in | '
              + '{:,}'.format(cs['total_latent_json']) + ' bytes out | '
              + cat_ratio)
    print()

    # Save results
    results['summary'] = {
        'total_prompts': total_tests,
        'exact_matches': total_match,
        'accuracy_pct': round(total_match / max(total_tests, 1) * 100, 1),
        'total_input_bytes': total_in,
        'total_output_json_bytes': total_out_json,
        'total_output_f32_bytes': total_out_f32,
        'overall_compression_ratio': round(
            total_in / max(total_out_json, 1), 2) if total_out_json > 0 else 0,
        'overall_space_saved_pct': round(
            (1 - total_out_json / max(total_in, 1)) * 100, 1) if total_in > 0 else 0,
        'total_inference_ms': round(total_inference_ms, 1),
        'avg_inference_ms': round(
            total_inference_ms / max(total_tests, 1), 2),
        'total_ollama_latency_ms': round(total_ollama_latency_ms, 0),
        'avg_ollama_latency_ms': round(
            total_ollama_latency_ms / max(total_ollama_count, 1), 0),
        'total_ollama_eval_tokens': total_ollama_tokens,
        'avg_ollama_tokens_per_sec': round(
            total_ollama_speed / max(total_ollama_count, 1), 1),
    }

    results_json = json.dumps(results, indent=2, ensure_ascii=False)
    with open(RESULTS_PATH, 'w') as f:
        f.write(results_json)
    print('  Results saved to: ' + RESULTS_PATH)
    print('  Results size    : '
          + str(len(results_json.encode('utf-8'))) + ' bytes')

    print()
    print(sep)
    print('  VERDICT: ' + str(total_match) + '/' + str(total_tests)
          + ' EXACT MATCHES (' + str(round(
              total_match / max(total_tests, 1) * 100, 1)) + '%)')
    print(sep)

    return results


# ==========================================
# 7. CLI
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LAIT Ollama Full Yield Demo')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='Ollama model name')
    parser.add_argument('--no-ollama', action='store_true',
                        help='Skip Ollama queries')
    parser.add_argument('--prompt', default=None,
                        help='Test a single custom prompt')
    args = parser.parse_args()

    run_demo(model=args.model, use_ollama=not args.no_ollama,
             custom_prompt=args.prompt)
