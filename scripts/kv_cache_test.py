import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_one(device, model, tokenizer, prompt, max_new_tokens, use_cache):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    return elapsed, tokens


def main():
    parser = argparse.ArgumentParser(description="KV cache: use_cache=True vs False, multiple token lengths")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--token-lengths", type=int, nargs="+", default=[32, 64, 128], help="max_new_tokens to test")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Warm-up run...")
    run_one(device, model, tokenizer, args.prompt, args.token_lengths[0], True)

    by_tokens = {}
    for max_tok in args.token_lengths:
        print(f"--- max_new_tokens={max_tok} ---")
        results = {}
        for use_cache, label in [(True, "use_cache_true"), (False, "use_cache_false")]:
            print(f"  use_cache={use_cache}...")
            elapsed, tokens = run_one(
                device, model, tokenizer, args.prompt, max_tok, use_cache
            )
            tok_per_sec = tokens / elapsed if elapsed > 0 else 0
            results[label] = {
                "use_cache": use_cache,
                "latency_sec": round(elapsed, 4),
                "tokens_generated": tokens,
                "tokens_per_sec": round(tok_per_sec, 2),
            }
            print(f"    latency: {elapsed:.2f}s  tok/s: {tok_per_sec:.1f}")
        by_tokens[str(max_tok)] = results

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"kv_cache_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"prompt": args.prompt, "token_lengths": args.token_lengths, "by_tokens": by_tokens}, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
