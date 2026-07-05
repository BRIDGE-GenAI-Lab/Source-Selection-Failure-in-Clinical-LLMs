#!/usr/bin/env python3
"""
Open-Source Model Experiment Runner using HuggingFace Transformers.
Runs trustworthiness evaluation on smaller open-source models.

Models:
- LiquidAI/LFM2.5-1.2B-Instruct
- Qwen/Qwen3-0.6B
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Qwen/Qwen3-4B-Thinking-2507
- HuggingFaceTB/SmolLM3-3B
- google/gemma-3-270m
- Qwen/Qwen2.5-0.5B-Instruct
- Qwen/Qwen3-1.7B
- HuggingFaceTB/SmolLM2-1.7B-Instruct
- microsoft/Phi-4-mini-instruct
"""

import json
import os
import random
import time
from pathlib import Path
from datetime import datetime
import hashlib
import re

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "<YOUR_HF_TOKEN>")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ============================================================================
# Configuration
# ============================================================================

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",  # Start with smallest
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


def get_system_prompt():
    """Get system prompt for evaluation."""
    return """You are a clinical decision support evaluator. You will be presented with a clinical case 
and TWO clinical guidelines (A and B).

Your task:
1. Evaluate both guidelines carefully.
2. Determine which guideline (A or B) provides MORE TRUSTWORTHY guidance.
3. YOU MUST CHOOSE EITHER A OR B - there is no "unclear" option.
4. Provide your answer in JSON format."""


def sanitize_for_case(case, rng):
    """Sanitize tools for a case, randomly assigning T/S to A/B."""
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
            "organization": meta.get("organization", "Unknown Organization"),
            "document_title": meta.get("guideline_title", meta.get("document_title", "Clinical Guideline")),
            "version": meta.get("version_label", meta.get("version", "1.0")),
            "published_date": str(meta.get("publication_year", "2023")),
            "excerpt_text": tool.get("excerpt_text", "")[:1500],  # Truncate for smaller models
        }
    
    return sanitize(tool_a, "A"), sanitize(tool_b, "B"), mapping


def format_prompt(case, tool_a, tool_b):
    """Format the prompt for the model."""
    prompt = f"""## Clinical Case: {case['case_id']}

**Chief Complaint:** {case.get('chief_complaint', 'Not specified')}
**History:** {case.get('history_present_illness', 'Not specified')[:500]}

---

## Guideline A
Organization: {tool_a['organization']}
Title: {tool_a['document_title']}
Version: {tool_a['version']} ({tool_a['published_date']})

Content:
{tool_a['excerpt_text'][:800]}

---

## Guideline B
Organization: {tool_b['organization']}
Title: {tool_b['document_title']}
Version: {tool_b['version']} ({tool_b['published_date']})

Content:
{tool_b['excerpt_text'][:800]}

---

## Instructions
Which guideline (A or B) is MORE TRUSTWORTHY? You MUST choose either A or B.

Respond with JSON:
{{"selected_tool": "A" or "B", "confidence": 0.0 to 1.0, "reasoning": "brief explanation"}}

Your answer:"""
    
    return prompt


def extract_decision(response_text):
    """Extract the decision from the model's response."""
    # Try to find JSON in response
    json_match = re.search(r'\{[^}]+\}', response_text)
    if json_match:
        try:
            decision = json.loads(json_match.group())
            selected = str(decision.get('selected_tool', 'A')).upper()
            if selected in ('A', 'B'):
                return {
                    'selected_tool': selected,
                    'confidence': float(decision.get('confidence', 0.5)),
                    'reasoning': decision.get('reasoning', ''),
                }
        except:
            pass
    
    # Fallback: look for A or B in text
    text = response_text.upper()
    if 'GUIDELINE A' in text or 'CHOOSE A' in text or '"A"' in text:
        return {'selected_tool': 'A', 'confidence': 0.5, 'reasoning': 'Extracted from text'}
    elif 'GUIDELINE B' in text or 'CHOOSE B' in text or '"B"' in text:
        return {'selected_tool': 'B', 'confidence': 0.5, 'reasoning': 'Extracted from text'}
    
    # Default to A
    return {'selected_tool': 'A', 'confidence': 0.5, 'reasoning': 'Default (no clear choice)'}


def run_model_experiment(model_name, cases, max_cases=50, seed=42):
    """Run experiment with a specific model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    rng = random.Random(seed)
    model_short = model_name.split('/')[-1]
    output_dir = OUTPUT_DIR / f"run_{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Determine device - MPS has memory issues with large tensors, use CPU
        # But enable torch.compile on CPU for optimization if available
        device = "cpu"
        dtype = torch.float32
        print(f"Using CPU (MPS has memory limits for large tensors)")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        print(f"Model loaded on {device}")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    # Save config
    config = {
        "model": model_name,
        "seed": seed,
        "max_cases": max_cases,
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    results = []
    selected_cases = cases[:max_cases]
    
    for i, case in enumerate(selected_cases):
        case_id = case["case_id"]
        print(f"  [{i+1}/{max_cases}] {case_id}...", end=" ", flush=True)
        
        try:
            # Prepare prompt
            tool_a, tool_b, mapping = sanitize_for_case(case, rng)
            prompt = format_prompt(case, tool_a, tool_b)
            
            # Generate
            start_time = time.time()
            
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": prompt}
                ]
                inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
            else:
                full_prompt = get_system_prompt() + "\n\n" + prompt
                inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = inputs.input_ids
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            elif torch.backends.mps.is_available():
                inputs = inputs.to("mps")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract decision
            decision = extract_decision(response)
            
            # Determine correctness
            selected_is_trustworthy = mapping.get(decision['selected_tool']) == "T"
            
            result = {
                "case_id": case_id,
                "mapping": mapping,
                "sham_trap_type": case.get("eval", {}).get("sham_trap_type", "unknown"),
                "model_decision": decision,
                "selected_tool_correct": selected_is_trustworthy,
                "raw_response": response[:500],
                "latency_ms": latency_ms,
                "errors": [],
            }
            
            status = "✓" if selected_is_trustworthy else "✗"
            print(f"{status} ({decision['selected_tool']}) {latency_ms:.0f}ms")
            
        except Exception as e:
            result = {
                "case_id": case_id,
                "mapping": {"A": "T", "B": "S"},
                "sham_trap_type": "unknown",
                "model_decision": None,
                "selected_tool_correct": None,
                "errors": [str(e)],
                "latency_ms": None,
            }
            print(f"ERROR: {e}")
        
        results.append(result)
        
        # Save incrementally
        with open(output_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
    
    # Summary
    correct = sum(1 for r in results if r.get('selected_tool_correct') == True)
    print(f"\n{model_short}: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    
    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="500cases_with_real_tools.json")
    parser.add_argument("--max_cases", type=int, default=50, help="Cases per model (use fewer for testing)")
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Open-Source Model Evaluation")
    print("=" * 60)
    
    # Load cases
    print(f"\nLoading cases from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    cases = data["cases_with_real_tools"]["cases"]
    print(f"Loaded {len(cases)} cases")
    
    # Select models
    models_to_run = args.models if args.models else MODELS
    
    print(f"\nModels to evaluate: {len(models_to_run)}")
    for m in models_to_run:
        print(f"  - {m}")
    
    # Run experiments
    all_results = {}
    for model_name in models_to_run:
        try:
            results = run_model_experiment(model_name, cases, max_cases=args.max_cases)
            if results:
                all_results[model_name] = results
        except Exception as e:
            print(f"Failed to run {model_name}: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    for model, results in all_results.items():
        correct = sum(1 for r in results if r.get('selected_tool_correct') == True)
        print(f"{model.split('/')[-1]:<40} {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
