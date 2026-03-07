"""
Decontamination experiment.

QUESTION: Does the GPT-4o circular measurement inflate abstractive F1?

DESIGN:
  - Take the 15 abstractive cells (GPT-4o compressed)
  - Evaluate EACH cell with BOTH Claude Haiku AND GPT-4o
  - Report delta_F1 = GPT-4o_eval - Haiku_eval per cell

INTERPRETATION:
  - |delta| < 0.03: contamination not material -> claim evaluator-robustness in paper
  - |delta| > 0.05: contamination real -> report as methodological finding
  Either outcome is publishable and strengthens the paper.

Outputs:
    results/decontamination.jsonl
    results/decontamination_summary.csv

Usage:
    python run_decontamination.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.compressors import AbstractiveCompressor
from datasets_loader import load_all
from evaluator import evaluate_cell

RATIOS = [0.9, 0.7, 0.5, 0.3, 0.1]
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
DECONTAM_FILE = RESULTS_DIR / "decontamination.jsonl"
DECONTAM_CSV  = RESULTS_DIR / "decontamination_summary.csv"


def load_completed() -> set:
    done = set()
    if DECONTAM_FILE.exists():
        for line in DECONTAM_FILE.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["dataset"], float(r["ratio"])))
    return done


async def main(n: int = 100):
    print("=== Decontamination Experiment ===")
    print("Compressor: abstractive (GPT-4o)")
    print("Evaluators: Claude Haiku (primary) vs GPT-4o (contamination check)\n")

    print("Loading datasets...")
    datasets = load_all(n_per_dataset=n, seed=42)

    print("Initializing abstractive compressor...")
    compressor = AbstractiveCompressor(model="gpt-4o")

    done = load_completed()
    print(f"Already completed: {len(done)}/15 cells\n")

    all_records = []

    for dataset_name, examples in datasets.items():
        for ratio in RATIOS:
            key = (dataset_name, float(ratio))
            if key in done:
                print(f"  SKIP: {dataset_name} | ratio={ratio}")
                continue

            print(f"\n  [abstractive] [{dataset_name}] ratio={ratio}")
            t0 = time.perf_counter()

            # Compress once
            compressed = []
            for ex in examples:
                ctx = compressor.compress(ex["context"], ratio=ratio, question=ex["question"])
                compressed.append({**ex, "context_compressed": ctx})

            # Evaluate with both models
            print("    Evaluating with Claude Haiku...")
            haiku = await evaluate_cell(compressed, evaluator="haiku")

            print("    Evaluating with GPT-4o...")
            gpt4o = await evaluate_cell(compressed, evaluator="gpt4o")

            delta_f1 = gpt4o["f1_mean"] - haiku["f1_mean"]
            duration = round(time.perf_counter() - t0, 2)

            print(f"    Haiku F1={haiku['f1_mean']:.3f}  GPT-4o F1={gpt4o['f1_mean']:.3f}  delta={delta_f1:+.3f}")

            record = {
                "dataset": dataset_name,
                "ratio": ratio,
                "haiku_f1": round(haiku["f1_mean"], 6),
                "haiku_em": round(haiku["em_mean"], 4),
                "haiku_f1_std": round(haiku["f1_std"], 6),
                "gpt4o_f1": round(gpt4o["f1_mean"], 6),
                "gpt4o_em": round(gpt4o["em_mean"], 4),
                "gpt4o_f1_std": round(gpt4o["f1_std"], 6),
                "delta_f1": round(delta_f1, 6),
                "contamination_material": abs(delta_f1) > 0.05,
                "n": haiku["n"],
                "duration_sec": duration,
            }
            all_records.append(record)

            with open(DECONTAM_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    # Summary CSV
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(DECONTAM_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

    # Print summary
    all_lines = []
    if DECONTAM_FILE.exists():
        for line in DECONTAM_FILE.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                all_lines.append(json.loads(line))

    if all_lines:
        deltas = [r["delta_f1"] for r in all_lines]
        material = [r for r in all_lines if r["contamination_material"]]
        mean_delta = sum(deltas) / len(deltas)
        print(f"\n=== DECONTAMINATION SUMMARY ===")
        print(f"  Cells evaluated: {len(all_lines)}/15")
        print(f"  Mean delta(F1) [GPT-4o_eval - Haiku_eval]: {mean_delta:+.3f}")
        print(f"  Max |delta|: {max(abs(d) for d in deltas):.3f}")
        print(f"  Cells where contamination material (>0.05): {len(material)}/15")
        if abs(mean_delta) < 0.03:
            print("  VERDICT: Contamination NOT material -> claim evaluator-robustness.")
        elif mean_delta > 0.05:
            print(f"  VERDICT: Contamination IS material ({mean_delta:+.3f}) -> report as finding.")

    print(f"\nResults saved to {DECONTAM_FILE}")


if __name__ == "__main__":
    asyncio.run(main(n=100))
