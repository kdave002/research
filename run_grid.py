"""
Main 45-cell experiment runner.
3 compressors x 3 datasets x 5 ratios, 100 examples per cell.
Primary evaluator: Claude Haiku (independent from GPT-4o abstractive compressor).

Outputs:
    results/grid_results.jsonl   — one JSON line per completed cell
    results/grid_summary.csv     — pivot table for paper

Designed for resumability: cells already in grid_results.jsonl are skipped.

Usage:
    python run_grid.py
    python run_grid.py --n 100 --concurrency 10
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.compressors import AbstractiveCompressor, LLMLinguaCompressor, TFIDFCompressor
from datasets_loader import load_all
from evaluator import evaluate_cell

RATIOS = [0.9, 0.7, 0.5, 0.3, 0.1]
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "grid_results.jsonl"
SUMMARY_FILE = RESULTS_DIR / "grid_summary.csv"

FIELDNAMES = ["compressor", "dataset", "ratio", "f1_mean", "f1_std", "em_mean", "n", "evaluator", "duration_sec"]


def load_completed() -> set:
    done = set()
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["compressor"], r["dataset"], float(r["ratio"])))
    return done


def compress_examples(compressor_obj, compressor_name: str, examples: list, ratio: float) -> list:
    compressed = []
    for ex in tqdm(examples, desc=f"  compress {compressor_name} r={ratio:.1f}", leave=False):
        ctx = compressor_obj.compress(
            text=ex["context"],
            ratio=ratio,
            question=ex["question"],
        )
        compressed.append({**ex, "context_compressed": ctx, "ratio": ratio})
    return compressed


async def run_cell(
    compressor_obj,
    compressor_name: str,
    dataset_name: str,
    examples: list,
    ratio: float,
    evaluator: str = "haiku",
) -> dict:
    print(f"\n  [{compressor_name}] [{dataset_name}] ratio={ratio:.1f}", flush=True)
    t0 = time.perf_counter()

    compressed = compress_examples(compressor_obj, compressor_name, examples, ratio)
    cell_result = await evaluate_cell(compressed, evaluator=evaluator)

    duration = round(time.perf_counter() - t0, 2)
    record = {
        "compressor": compressor_name,
        "dataset": dataset_name,
        "ratio": ratio,
        "f1_mean": round(cell_result["f1_mean"], 6),
        "f1_std": round(cell_result["f1_std"], 6),
        "em_mean": round(cell_result["em_mean"], 4),
        "n": cell_result["n"],
        "evaluator": evaluator,
        "duration_sec": duration,
    }

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"    -> F1={record['f1_mean']:.3f} (+/-{record['f1_std']:.3f})  EM={record['em_mean']:.3f}  ({duration:.0f}s)")
    return record


def write_summary():
    records = []
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                records.append(json.loads(line))
    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in FIELDNAMES})
    print(f"\nSummary written to {SUMMARY_FILE}")


async def main(n: int = 100, evaluator: str = "haiku"):
    print("=== Stage 4 (Revised): 45-cell grid ===")
    print(f"  n_per_cell={n}  evaluator={evaluator}\n")

    print("Loading datasets...")
    datasets = load_all(n_per_dataset=n, seed=42)

    print("\nInitializing compressors...")
    compressors = {
        "llmlingua2":  LLMLinguaCompressor(),
        "abstractive": AbstractiveCompressor(model="gpt-4o"),
        "tfidf":       TFIDFCompressor(),
    }

    done = load_completed()
    total = len(compressors) * len(datasets) * len(RATIOS)
    print(f"\nProgress: {len(done)}/{total} cells already complete.")

    for compressor_name, compressor_obj in compressors.items():
        for dataset_name, examples in datasets.items():
            for ratio in RATIOS:
                key = (compressor_name, dataset_name, float(ratio))
                if key in done:
                    print(f"  SKIP: {compressor_name} | {dataset_name} | {ratio}")
                    continue
                await run_cell(
                    compressor_obj, compressor_name,
                    dataset_name, examples, ratio, evaluator,
                )

    write_summary()
    print(f"\nAll cells complete. Results in {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Examples per cell")
    parser.add_argument("--evaluator", default="haiku", choices=["haiku", "gpt4o"])
    args = parser.parse_args()
    asyncio.run(main(n=args.n, evaluator=args.evaluator))
