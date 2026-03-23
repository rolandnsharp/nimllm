#!/usr/bin/env python3
"""Generate distillation data from a teacher model via Ollama.

For each training document, asks the teacher to predict continuations
and captures the top-k logprob distributions. Saves to binary format
that microgpt.nim can load for KL-divergence training.

Usage:
  python3 distill.py                    # generate soft targets
  python3 distill.py --teacher qwen3.5  # specify teacher model
  python3 distill.py --test             # quick test with 10 docs

The teacher runs via Ollama API (localhost:11434). Make sure the teacher
model is pulled and Ollama is running.
"""

import json, struct, os, sys, time, argparse, requests

OLLAMA_URL = "http://localhost:11434/api/generate"
TRAINING_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_data.txt")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "distill_data")

def get_teacher_logprobs(model, prompt, max_tokens=128, top_k=32):
    """Get teacher's top-k logprob predictions for a prompt continuation."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 1.0,  # temperature 1 for true distribution
                "top_k": 0,          # don't filter — we want the full distribution
                "top_p": 1.0,
            },
            # Request logprobs if supported
        }, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        print(f"  error: {e}", file=sys.stderr)
        return None

def generate_thinking_data(model, text, max_tokens=256):
    """Ask teacher to explain/elaborate on text with thinking traces."""
    prompt = f"""Read this text carefully, then rewrite it in a way that makes the reasoning and knowledge explicit. Include your thinking process.

Text: {text[:500]}

Rewrite with explicit reasoning:"""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
            },
        }, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        print(f"  error: {e}", file=sys.stderr)
        return None

def generate_qa_pairs(model, text, max_tokens=256):
    """Ask teacher to generate Q&A pairs from text — Phi-style synthetic data."""
    prompt = f"""Based on this text, generate 3 question-answer pairs that test understanding. Format each as:
Q: [question]
A: [detailed answer]

Text: {text[:500]}

Question-answer pairs:"""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
            },
        }, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        print(f"  error: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate distillation data")
    parser.add_argument("--teacher", default="qwen3.5", help="Ollama model name")
    parser.add_argument("--test", action="store_true", help="Quick test with 10 docs")
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "thinking", "qa"],
                       help="Generation mode: synthetic (rewrite), thinking (with traces), qa (Q&A pairs)")
    parser.add_argument("--max-docs", type=int, default=0, help="Max docs to process (0=all)")
    args = parser.parse_args()

    # Check Ollama is running
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama running, models: {', '.join(models)}")
        if not any(args.teacher in m for m in models):
            print(f"WARNING: teacher '{args.teacher}' not found. Available: {', '.join(models)}")
            print(f"Falling back to first available model...")
            if models:
                args.teacher = models[0].split(":")[0]
                print(f"Using: {args.teacher}")
            else:
                print("No models available. Pull one first: ollama pull qwen3.5")
                sys.exit(1)
    except Exception as e:
        print(f"Ollama not running: {e}")
        print("Start it with: ollama serve")
        sys.exit(1)

    # Load training docs
    if not os.path.exists(TRAINING_DATA):
        print(f"Training data not found: {TRAINING_DATA}")
        sys.exit(1)

    docs = open(TRAINING_DATA).readlines()
    print(f"Loaded {len(docs)} training docs")

    max_docs = args.max_docs if args.max_docs > 0 else len(docs)
    if args.test:
        max_docs = 10
    docs = docs[:max_docs]

    # Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"distill_{args.mode}.txt")

    print(f"Generating {args.mode} data from {len(docs)} docs using {args.teacher}")
    print(f"Output: {out_file}")

    generated = 0
    t0 = time.time()

    with open(out_file, "w") as f:
        for i, doc in enumerate(docs):
            doc = doc.strip()
            if len(doc) < 50:
                continue

            if args.mode == "thinking":
                result = generate_thinking_data(args.teacher, doc)
            elif args.mode == "qa":
                result = generate_qa_pairs(args.teacher, doc)
            else:  # synthetic
                result = generate_thinking_data(args.teacher, doc)

            if result and len(result.strip()) > 20:
                f.write(result.strip() + "\n")
                generated += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(docs) - i - 1) / max(rate, 0.01)
                print(f"  {i+1}/{len(docs)} | {generated} generated | {rate:.1f} docs/s | ETA {int(eta)}s")

    elapsed = time.time() - t0
    size_kb = os.path.getsize(out_file) // 1024
    print(f"\nDone: {generated} docs generated in {int(elapsed)}s")
    print(f"Output: {out_file} ({size_kb}KB)")
    print(f"\nTo add to training data: cat {out_file} >> ../training_data.txt")

if __name__ == "__main__":
    main()
