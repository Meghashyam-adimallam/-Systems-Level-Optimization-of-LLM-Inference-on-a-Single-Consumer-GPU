import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List

import httpx
import numpy as np


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str = "Hello, world!",
    max_new_tokens: int = 64,
    max_retries: int = 3,
) -> tuple[float, int]:
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens}
    url = f"{url.rstrip('/')}/generate"
    last_err = None
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency = time.perf_counter() - t0
            num_tokens = data.get("num_tokens", 0)
            return latency, num_tokens
        except (httpx.ReadError, httpx.ReadTimeout, httpx.ConnectError) as e:
            last_err = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2.0)
            continue
    raise last_err


async def run_load_test(
    url: str,
    num_requests: int,
    concurrency: int,
    max_new_tokens: int = 64,
) -> dict:
    latencies: List[float] = []
    token_counts: List[int] = []

    timeout = httpx.Timeout(30.0, read=600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        sem = asyncio.Semaphore(concurrency)

        async def one_request():
            async with sem:
                lat, tokens = await send_request(
                    client, url, max_new_tokens=max_new_tokens
                )
                latencies.append(lat)
                token_counts.append(tokens)

        tasks = [one_request() for _ in range(num_requests)]
        t_start = time.perf_counter()
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    total_sec = t_end - t_start
    arr = np.array(latencies)
    return {
        "num_requests": num_requests,
        "concurrency": concurrency,
        "wall_clock_sec": total_sec,
        "req_per_sec": num_requests / total_sec if total_sec > 0 else 0,
        "p50_latency_sec": float(np.percentile(arr, 50)),
        "p95_latency_sec": float(np.percentile(arr, 95)),
        "tokens_per_sec": sum(token_counts) / total_sec if total_sec > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Load generator for LLM API")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--label", type=str, default="", help="e.g. baseline or batched; used in saved filename")
    parser.add_argument("--out-dir", type=str, default="results", help="directory to save JSON results")
    args = parser.parse_args()

    result = asyncio.run(
        run_load_test(
            args.url,
            args.num_requests,
            args.concurrency,
            args.max_new_tokens,
        )
    )
    print("Results:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"load_{args.label}_{ts}.json" if args.label else f"load_{ts}.json"
    out_path = out_dir / name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
