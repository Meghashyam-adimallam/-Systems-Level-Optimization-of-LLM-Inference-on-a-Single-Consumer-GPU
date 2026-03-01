import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str = "results", pattern: str = "load_*.json"):
    results = []
    path = Path(results_dir)
    for f in sorted(path.glob(pattern)):
        with open(f) as fp:
            data = json.load(fp)
        raw = f.stem.replace("load_", "").replace(".json", "")
        raw = re.sub(r"_\d{8}_\d{6}$", "", raw)
        label = raw.replace("baseline_full", "Baseline (full)").replace("batched_full", "Batched (full)")
        label = label.replace("dynamic_full", "Dynamic (full)")
        label = label.replace("baseline", "Baseline").replace("batched", "Batched").replace("dynamic", "Dynamic")
        if "req_per_sec" in data:
            data["_label"] = label
            data["_raw"] = raw
            results.append(data)
    return results


def main():
    results = load_results()
    if not results:
        print("No load_*.json files found in results/")
        return

    out_dir = Path("report")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = ["light", "medium", "heavy"]
    strategies = ["baseline", "batched", "dynamic"]
    by_config = defaultdict(dict)
    flat_labels = []
    flat_req_s, flat_p50, flat_p95, flat_tok_s = [], [], [], []

    for r in results:
        raw = r.get("_raw", "")
        matched = False
        for c in configs:
            for s in strategies:
                if raw == f"{s}_{c}" or raw.startswith(f"{s}_{c}_"):
                    by_config[c][s] = r
                    matched = True
                    break
            if matched:
                break
        if not matched:
            flat_labels.append(r["_label"])
            flat_req_s.append(r["req_per_sec"])
            flat_p50.append(r["p50_latency_sec"])
            flat_p95.append(r["p95_latency_sec"])
            flat_tok_s.append(r["tokens_per_sec"])

    has_configs = len(by_config) > 0

    if has_configs and by_config:
        config_order = [c for c in configs if c in by_config]
        n_configs = len(config_order)
        n_strat = len(strategies)
        x = np.arange(n_configs)
        width = 0.25
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("Systems-Level Optimization of LLM Inference on a Single Consumer GPU — Multiple loads (light / medium / heavy)", fontsize=12, fontweight="bold", y=1.02)
        colors = {"baseline": "#2d7d46", "batched": "#1a5fb4", "dynamic": "#c64600"}
        for i, (metric_key, title, ylabel) in enumerate([
            ("req_per_sec", "Throughput", "Requests / sec"),
            ("p50_latency_sec", "p50 Latency", "Seconds"),
            ("p95_latency_sec", "p95 Latency", "Seconds"),
            ("tokens_per_sec", "Token throughput", "Tokens / sec"),
        ]):
            ax = axes[i // 2, i % 2]
            for j, strat in enumerate(strategies):
                vals = [by_config[c].get(strat, {}).get(metric_key, 0) for c in config_order]
                if any(v != 0 for v in vals):
                    ax.bar(x + j * width, vals, width, label=strat.capitalize(), color=colors.get(strat, "gray"))
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(x + width)
            ax.set_xticklabels([c.capitalize() for c in config_order])
            ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        out_path = out_dir / "benchmark_comparison.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path} (grouped by load)")

        # KV cache chart if available
        kv_path = Path("results")
        kv_files = sorted(kv_path.glob("kv_cache_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if kv_files:
            with open(kv_files[0]) as f:
                kv_data = json.load(f)
            by_tokens = kv_data.get("by_tokens") or {}
            if by_tokens:
                tokens_list = [k for k in ["32", "64", "128"] if k in by_tokens]
                x = np.arange(len(tokens_list))
                width = 0.35
                true_vals = [by_tokens[t]["use_cache_true"]["tokens_per_sec"] for t in tokens_list]
                false_vals = [by_tokens[t]["use_cache_false"]["tokens_per_sec"] for t in tokens_list]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(x - width / 2, true_vals, width, label="use_cache=True", color="#2d7d46")
                ax.bar(x + width / 2, false_vals, width, label="use_cache=False", color="#7d2525")
                ax.set_ylabel("Tokens / sec")
                ax.set_title("KV cache: use_cache True vs False")
                ax.set_xticks(x)
                ax.set_xticklabels([f"{t} tok" for t in tokens_list])
                ax.legend()
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                kv_out = out_dir / "kv_cache_comparison.png"
                plt.savefig(kv_out, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved {kv_out}")
        return

    x = np.arange(len(flat_labels))
    width = 0.5
    colors = ["#2d7d46", "#1a5fb4", "#c64600", "#613583"][: len(flat_labels)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].bar(x, flat_req_s, width, color=colors)
    axes[0, 0].set_ylabel("Requests / sec")
    axes[0, 0].set_title("Throughput")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(flat_labels, rotation=45, ha="right")
    axes[0, 1].bar(x, flat_p50, width, color=colors)
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].set_title("p50 Latency")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(flat_labels, rotation=45, ha="right")
    axes[1, 0].bar(x, flat_p95, width, color=colors)
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].set_title("p95 Latency")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(flat_labels, rotation=45, ha="right")
    axes[1, 1].bar(x, flat_tok_s, width, color=colors)
    axes[1, 1].set_ylabel("Tokens / sec")
    axes[1, 1].set_title("Token throughput")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(flat_labels, rotation=45, ha="right")
    plt.tight_layout()
    out_path = out_dir / "benchmark_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # KV cache chart if available
    kv_path = Path("results")
    kv_files = sorted(kv_path.glob("kv_cache_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if kv_files:
        with open(kv_files[0]) as f:
            kv_data = json.load(f)
        by_tokens = kv_data.get("by_tokens") or {}
        if by_tokens:
            tokens_list = [k for k in ["32", "64", "128"] if k in by_tokens]
            x = np.arange(len(tokens_list))
            width = 0.35
            true_vals = [by_tokens[t]["use_cache_true"]["tokens_per_sec"] for t in tokens_list]
            false_vals = [by_tokens[t]["use_cache_false"]["tokens_per_sec"] for t in tokens_list]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - width / 2, true_vals, width, label="use_cache=True", color="#2d7d46")
            ax.bar(x + width / 2, false_vals, width, label="use_cache=False", color="#7d2525")
            ax.set_ylabel("Tokens / sec")
            ax.set_title("KV cache: use_cache True vs False")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{t} tok" for t in tokens_list])
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            kv_out = out_dir / "kv_cache_comparison.png"
            plt.savefig(kv_out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved {kv_out}")


if __name__ == "__main__":
    main()
