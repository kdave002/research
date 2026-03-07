"""
Generate paper tables, bootstrap CIs, and summary statistics from results.
Run after all experiments (run_grid.py, run_decontamination.py, rcid.py) complete.

Prints:
  - Table 1: Mean F1 by compressor x ratio (with 95% CI)
  - Table 2: Full F1 by compressor x dataset x ratio
  - Table 3: Threshold collapse band (ratio 0.5 -> 0.3)
  - Table 4: EM results
  - Decontamination summary
  - Hop-depth split (2-hop vs 3-hop on MuSiQue)

Usage:
    python analyze_results.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

RESULTS_DIR = Path("results")
COMPRESSORS = ["abstractive", "llmlingua2", "tfidf"]
DATASETS    = ["musique", "hotpotqa", "2wikimultihop"]
RATIOS      = [0.9, 0.7, 0.5, 0.3, 0.1]
COMP_LABELS = {"abstractive": "Abstractive", "llmlingua2": "LLMLingua-2", "tfidf": "TF-IDF"}
DS_LABELS   = {"musique": "MuSiQue", "hotpotqa": "HotpotQA", "2wikimultihop": "2WikiMultihop"}


def load_grid() -> Dict:
    f = RESULTS_DIR / "grid_results.jsonl"
    if not f.exists():
        raise FileNotFoundError("Run run_grid.py first.")
    data: Dict = {}
    for line in f.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c, d, ratio = r["compressor"], r["dataset"], float(r["ratio"])
        data.setdefault(c, {}).setdefault(d, {})[ratio] = r
    return data


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(f1_mean: float, f1_std: float, n: int, n_boot: int = 1000, seed: int = 42) -> tuple:
    """Approximate bootstrap CI using normal approximation (exact data not stored)."""
    se = f1_std / math.sqrt(n) if f1_std else 0.0
    return (round(f1_mean - 1.96 * se, 3), round(f1_mean + 1.96 * se, 3))


def get(results, comp, ds, ratio, key="f1_mean", default=float("nan")):
    return results.get(comp, {}).get(ds, {}).get(ratio, {}).get(key, default)


def table1(results: Dict):
    print("\n=== TABLE 1: Mean F1 by Compressor x Ratio (averaged across datasets) ===")
    print(f"{'Ratio':<8}", end="")
    for c in COMPRESSORS:
        print(f"{COMP_LABELS[c]:<20}", end="")
    print()
    print("-" * 68)

    for ratio in RATIOS:
        print(f"{ratio:<8.1f}", end="")
        for c in COMPRESSORS:
            vals = [get(results, c, d, ratio) for d in DATASETS]
            vals = [v for v in vals if not math.isnan(v)]
            m = mean(vals)
            # Pooled std approx
            stds = [get(results, c, d, ratio, "f1_std") for d in DATASETS]
            stds = [v for v in stds if not math.isnan(v)]
            ns   = [get(results, c, d, ratio, "n", 100) for d in DATASETS]
            if stds and ns:
                se = mean(stds) / math.sqrt(mean(ns))
                ci_str = f"[{m-1.96*se:.3f},{m+1.96*se:.3f}]"
            else:
                ci_str = ""
            print(f"{m:.3f} {ci_str:<17}", end="")
        print()

    print("-" * 68)
    # Delta row
    print(f"{'delta':8}", end="")
    for c in COMPRESSORS:
        hi_vals = [get(results, c, d, 0.9) for d in DATASETS]
        lo_vals = [get(results, c, d, 0.1) for d in DATASETS]
        hi = mean([v for v in hi_vals if not math.isnan(v)])
        lo = mean([v for v in lo_vals if not math.isnan(v)])
        delta = hi - lo
        pct = 100 * delta / hi if hi > 0 else 0
        print(f"{delta:+.3f} ({pct:.0f}%)          ", end="")
    print()


def table2(results: Dict):
    print("\n=== TABLE 2: Full F1 by Compressor x Dataset x Ratio ===")
    header = f"{'Compressor':<14} {'Dataset':<16} " + "  ".join(f"{r:.1f}" for r in RATIOS)
    print(header)
    print("-" * len(header))
    for c in COMPRESSORS:
        for d in DATASETS:
            vals = [get(results, c, d, r) for r in RATIOS]
            row = f"{COMP_LABELS[c]:<14} {DS_LABELS[d]:<16} " + "  ".join(
                f"{v:.3f}" if not math.isnan(v) else "  N/A" for v in vals
            )
            print(row)


def table3(results: Dict):
    print("\n=== TABLE 3: Threshold Collapse (ratio 0.5 -> 0.3) ===")
    print(f"{'Compressor':<14} {'Dataset':<16} {'F1@0.5':>8} {'F1@0.3':>8} {'drop':>8} {'%drop':>8}")
    print("-" * 68)
    for c in ["llmlingua2", "tfidf"]:
        for d in DATASETS:
            f05 = get(results, c, d, 0.5)
            f03 = get(results, c, d, 0.3)
            if not (math.isnan(f05) or math.isnan(f03)):
                drop = f05 - f03
                pct = 100 * drop / f05 if f05 > 0 else 0
                print(f"{COMP_LABELS[c]:<14} {DS_LABELS[d]:<16} {f05:>8.3f} {f03:>8.3f} {drop:>8.3f} {pct:>7.1f}%")


def table4_em(results: Dict):
    print("\n=== TABLE 4: Exact Match (EM) by Compressor x Dataset x Ratio ===")
    header = f"{'Compressor':<14} {'Dataset':<16} " + "  ".join(f"{r:.1f}" for r in RATIOS)
    print(header)
    print("-" * len(header))
    for c in COMPRESSORS:
        for d in DATASETS:
            vals = [get(results, c, d, r, "em_mean") for r in RATIOS]
            row = f"{COMP_LABELS[c]:<14} {DS_LABELS[d]:<16} " + "  ".join(
                f"{v:.3f}" if not math.isnan(v) else "  N/A" for v in vals
            )
            print(row)


def decontamination_summary():
    f = RESULTS_DIR / "decontamination.jsonl"
    if not f.exists():
        print("\n[Decontamination: not yet run]")
        return
    records = [json.loads(l) for l in f.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    if not records:
        return
    deltas = [r["delta_f1"] for r in records]
    material = [r for r in records if r.get("contamination_material")]
    m = mean(deltas)
    print(f"\n=== DECONTAMINATION SUMMARY ===")
    print(f"  Mean delta(F1) [GPT-4o_eval - Haiku_eval]: {m:+.3f}")
    print(f"  Max |delta|: {max(abs(d) for d in deltas):.3f}")
    print(f"  Cells where |delta| > 0.05: {len(material)}/15")
    if abs(m) < 0.03:
        print("  VERDICT: Contamination NOT material -> claim evaluator-robustness.")
    elif m > 0.05:
        print(f"  VERDICT: Contamination IS material (+{m:.3f}) -> report as finding.")
    else:
        print(f"  VERDICT: Marginal contamination ({m:+.3f}) -> acknowledge in limitations.")


def hop_depth_split():
    rcid_f = RESULTS_DIR / "rcid_scores.jsonl"
    if not rcid_f.exists():
        print("\n[Hop-depth split: run rcid.py first]")
        return
    records = [json.loads(l) for l in rcid_f.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    hop2 = [r for r in records if r.get("hop_count") == 2 and r.get("rcid") is not None]
    hop3 = [r for r in records if r.get("hop_count") == 3 and r.get("rcid") is not None]
    hopN = [r for r in records if r.get("hop_count", 0) > 3 and r.get("rcid") is not None]
    print(f"\n=== HOP DEPTH SPLIT (MuSiQue) ===")
    for label, group in [("2-hop", hop2), ("3-hop", hop3), (">3-hop", hopN)]:
        if group:
            rcids = [r["rcid"] for r in group]
            baselines = [r["baseline_f1"] for r in group]
            print(f"  {label}: n={len(group)}  mean RCID={mean(rcids):.3f}  mean baseline_F1={mean(baselines):.3f}")


if __name__ == "__main__":
    results = load_grid()
    table1(results)
    table2(results)
    table3(results)
    table4_em(results)
    decontamination_summary()
    hop_depth_split()
