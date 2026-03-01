import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Baseline single-request inference")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated(0) / (1024**3) if device == "cuda" else 0

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
    mem_after = torch.cuda.memory_allocated(0) / (1024**3) if device == "cuda" else 0
    mem_peak = torch.cuda.max_memory_allocated(0) / (1024**3) if device == "cuda" else 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"baseline_{timestamp}.json"

    result = {
        "experiment": "baseline",
        "model": args.model,
        "device": device,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "latency_sec": round(elapsed, 4),
        "tokens_generated": tokens_generated,
        "tokens_per_sec": round(tokens_generated / elapsed, 2) if elapsed > 0 else None,
        "gpu_memory_gb": round(mem_peak, 2) if device == "cuda" else None,
        "timestamp": timestamp,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Latency: {elapsed:.2f}s | Tokens: {tokens_generated} | Tok/s: {tokens_generated/elapsed:.1f} | GPU mem: {mem_peak:.2f} GB")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
