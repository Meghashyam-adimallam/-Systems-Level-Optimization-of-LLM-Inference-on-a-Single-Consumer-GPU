"""
Run load tests for multiple configs (light, medium, heavy).
Start the server first (baseline, batched, or dynamic), then run:
  python scripts/run_benchmark_suite.py --url http://127.0.0.1:8000 --strategy baseline
  python scripts/run_benchmark_suite.py --url http://127.0.0.1:8000 --strategy batched
  python scripts/run_benchmark_suite.py --url http://127.0.0.1:8000 --strategy dynamic
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root))
from benchmark.load_generator import run_load_test

BENCHMARK_LOADS = [
    {"name": "light", "num_requests": 10, "concurrency": 2, "max_new_tokens": 32},
    {"name": "medium", "num_requests": 30, "concurrency": 5, "max_new_tokens": 64},
    {"name": "heavy", "num_requests": 30, "concurrency": 8, "max_new_tokens": 128},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--strategy", type=str, required=True, choices=["baseline", "batched", "dynamic"])
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for cfg in BENCHMARK_LOADS:
        label = f"{args.strategy}_{cfg['name']}"
        print(f"Running {label} ({cfg['num_requests']} req, {cfg['concurrency']} conc, {cfg['max_new_tokens']} tok)...")
        result = asyncio.run(
            run_load_test(
                args.url,
                num_requests=cfg["num_requests"],
                concurrency=cfg["concurrency"],
                max_new_tokens=cfg["max_new_tokens"],
            )
        )
        result["_label"] = label
        result["_config"] = cfg["name"]
        all_results.append(result)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"load_{label}_{ts}.json"
        save = {k: v for k, v in result.items() if not k.startswith("_")}
        with open(out_path, "w") as f:
            json.dump(save, f, indent=2)
        print(f"  req/s: {result['req_per_sec']:.3f}  p50: {result['p50_latency_sec']:.2f}s  saved {out_path.name}")

    print(f"Done. Ran {len(BENCHMARK_LOADS)} configs for {args.strategy}.")


if __name__ == "__main__":
    main()
