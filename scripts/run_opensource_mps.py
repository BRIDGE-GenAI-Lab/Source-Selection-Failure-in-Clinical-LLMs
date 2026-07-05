#!/usr/bin/env python3
"""
Open-Source Model Experiment Runner (MPS Optimized for M1 Mac)
Reduced context length to fit within MPS memory limits.
"""

import json
import os
import random
import time
from pathlib import Path
from datetime import datetime
import re

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "<YOUR_HF_TOKEN>")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "google/gemma-3-270m",
    "Qwen/Qwen3-0.6B",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3-4B-Thinking-2507",
    "microsoft/Phi-4-mini-instruct",
]

OUTPUT_DIR = Path("results_opensource")
OUTPUT_DIR.mkdir(exist_ok=True)


def sanitize_for_case(case, rng):
    """Sanitize tools for a case."""
    tool_t = case["tools"]["T"]
    tool_s = case["tools"]["S"]
    
    if rng.random() < 0.5:
        tool_a, tool_b = tool_t, tool_s
        mapping = {"A": "T", "B": "S"}
    else:
        tool_a, tool_b = tool_s, tool_t
        mapping = {"A": "S", "B": "T"}
    
    def sanitize(tool, tool_id):
        meta = tool.get("metadata", {})
        return {
            "tool_id": tool_id,
            "organization": meta.get("organization", "Unknown")[:50],
            "title": meta.get("guideline_title", "Guideline")[:80],
            "version": meta.get("version_label", "1.0")[:20],
            "excerpt": tool.get("excerpt_text", "")[:600],  # Reduced for MPS
        }
    
    return sanitize(tool_a, "A"), sanitize(tool_b, "B"), mapping


def format_prompt(case, tool_a, tool_b):
    """Compact prompt for MPS memory limits."""
    return f"""Case: {case.get('chief_complaint', '')[:100]}

Guideline A [{tool_a['organization']}]: {tool_a['excerpt'][:400]}

Guideline B [{tool_b['organization']}]: {tool_b['excerpt'][:400]}

Which is more trustworthy, A or B? Reply JSON: {{"selected_tool": "A" or "B"}}
Answer:"""


def extract_decision(response_text):
    """Extract decision from response."""
    json_match = re.search(r'\{[^}]+\}', response_text)
    if json_match:
        try:
            d = json.loads(json_match.group())
            s = str(d.get('selected_tool', 'A')).upper()
            if s in ('A', 'B'):
                return {'selected_tool': s, 'confidence': 0.5}
        except:
            pass
    
    text = response_text.upper()
    if 'A' in text and 'B' not in text:
        return {'selected_tool': 'A', 'confidence': 0.5}
    elif 'B' in text and 'A' not in text:
        return {'selected_tool': 'B', 'confidence': 0.5}
    return {'selected_tool': 'A', 'confidence': 0.5}


def run_experiment(model_name, cases, max_cases=500, seed=42):
    """Run experiment with MPS GPU."""
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    
    rng = random.Random(seed)
    model_short = model_name.split('/')[-1]
    output_dir = OUTPUT_DIR / f"run_{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model on MPS
    print("Loading model on MPS...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Use MPS with float32 (float16 causes nan issues)
        device = torch.device("mps")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
        
        print(f"Loaded on MPS (float32)")
    except Exception as e:
        print(f"Failed: {e}")
        return None
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({"model": model_name, "max_cases": max_cases, "device": "mps"}, f)
    
    results = []
    for i, case in enumerate(cases[:max_cases]):
        case_id = case["case_id"]
        print(f"  [{i+1}/{max_cases}] {case_id}...", end=" ", flush=True)
        
        try:
            tool_a, tool_b, mapping = sanitize_for_case(case, rng)
            prompt = format_prompt(case, tool_a, tool_b)
            
            start = time.time()
            
            # Tokenize with strict limit
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            latency = (time.time() - start) * 1000
            
            decision = extract_decision(response)
            correct = mapping.get(decision['selected_tool']) == "T"
            
            result = {
                "case_id": case_id,
                "mapping": mapping,
                "sham_trap_type": case.get("eval", {}).get("sham_trap_type", "unknown"),
                "model_decision": decision,
                "selected_tool_correct": correct,
                "latency_ms": latency,
            }
            
            print(f"{'✓' if correct else '✗'} ({decision['selected_tool']}) {latency:.0f}ms")
            
        except Exception as e:
            result = {"case_id": case_id, "errors": [str(e)], "selected_tool_correct": None}
            print(f"ERROR: {str(e)[:50]}")
        
        results.append(result)
        with open(output_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
    
    # Summary
    correct = sum(1 for r in results if r.get('selected_tool_correct') == True)
    print(f"\n{model_short}: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    
    del model
    torch.mps.empty_cache()
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="500cases_with_real_tools.json")
    parser.add_argument("--max_cases", type=int, default=500)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Open-Source Model Evaluation (MPS GPU)")
    print("=" * 50)
    
    with open(args.input) as f:
        data = json.load(f)
    cases = data["cases_with_real_tools"]["cases"]
    print(f"Loaded {len(cases)} cases")
    
    models = args.models if args.models else MODELS
    
    for model_name in models:
        try:
            run_experiment(model_name, cases, max_cases=args.max_cases)
        except Exception as e:
            print(f"Failed {model_name}: {e}")


if __name__ == "__main__":
    main()
