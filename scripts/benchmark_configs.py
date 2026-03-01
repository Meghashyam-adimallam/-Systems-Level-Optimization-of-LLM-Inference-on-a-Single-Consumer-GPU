BENCHMARK_LOADS = [
    {"name": "light", "num_requests": 10, "concurrency": 2, "max_new_tokens": 32},
    {"name": "medium", "num_requests": 30, "concurrency": 5, "max_new_tokens": 64},
    {"name": "heavy", "num_requests": 30, "concurrency": 8, "max_new_tokens": 128},
]

KV_CACHE_TOKEN_LENGTHS = [32, 64, 128]
