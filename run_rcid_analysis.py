"""
RCID correlation analysis.

After running rcid.py and run_grid.py, this script:
1. Reports RCID distribution statistics
2. Validates RCID against MuSiQue ground-truth supporting facts
3. Validates the r^k phase-transition model against empirical EM values
4. Correlates RCID with per-cell F1 drop

Run after both rcid.py and run_grid.py are complete:
    python run_rcid_analysis.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS_DIR = Path("results")


def load_rcid():
    f = RESULTS_DIR / "rcid_scores.jsonl"
    if not f.exists():
        raise FileNotFoundError("Run rcid.py first.")
    return [json.loads(l) for l in f.read_text(encoding="utf-8").strip().splitlines() if l.strip()]


def load_grid():
    f = RESULTS_DIR / "grid_results.jsonl"
    if not f.exists():
        raise FileNotFoundError("Run run_grid.py first.")
    return [json.loads(l) for l in f.read_text(encoding="utf-8").strip().splitlines() if l.strip()]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def phase_transition_pred(r: float, k: int) -> float:
    return r ** k


def analyze():
    # -----------------------------------------------------------------------
    # 1. RCID distribution
    # -----------------------------------------------------------------------
    rcid_data = load_rcid()
    rcid_vals = [d["rcid"] for d in rcid_data if d.get("rcid") is not None]

    print("=== 1. RCID Distribution (MuSiQue) ===")
    print(f"  n examples:  {len(rcid_vals)}")
    print(f"  Mean RCID:   {mean(rcid_vals):.3f}")
    sorted_rcid = sorted(rcid_vals)
    n = len(sorted_rcid)
    print(f"  Median RCID: {sorted_rcid[n//2]:.3f}")
    print(f"  Std RCID:    {(sum((x - mean(rcid_vals))**2 for x in rcid_vals)/n)**0.5:.3f}")
    print(f"  Low RCID (<0.3):  {sum(1 for r in rcid_vals if r < 0.3)} examples")
    print(f"  Mid RCID (0.3-0.7): {sum(1 for r in rcid_vals if 0.3 <= r <= 0.7)} examples")
    print(f"  High RCID (>0.7): {sum(1 for r in rcid_vals if r > 0.7)} examples")

    # -----------------------------------------------------------------------
    # 2. RCID validation against ground-truth supporting facts
    # -----------------------------------------------------------------------
    validated = [d for d in rcid_data if d.get("validation") and d["validation"]]
    if validated:
        precs = [d["validation"]["precision"] for d in validated]
        recs  = [d["validation"]["recall"]    for d in validated]
        print(f"\n=== 2. RCID Validation vs. Ground-Truth Supporting Facts ===")
        print(f"  n validated examples: {len(validated)}")
        print(f"  Mean Precision: {mean(precs):.3f}")
        print(f"  Mean Recall:    {mean(recs):.3f}")
        f1_val = 2 * mean(precs) * mean(recs) / max(1e-9, mean(precs) + mean(recs))
        print(f"  Macro F1:       {f1_val:.3f}")
        if mean(precs) >= 0.6:
            print("  VERDICT: RCID reliably identifies reasoning-critical sentences.")
        else:
            print("  VERDICT: RCID precision below 0.6 — review epsilon threshold.")

    # -----------------------------------------------------------------------
    # 3. Phase transition model validation
    # -----------------------------------------------------------------------
    print("\n=== 3. Phase Transition Model: r^k vs. Empirical EM ===")
    print(f"  {'Ratio':<8} {'k=2 pred':>10} {'k=3 pred':>10}")
    print(f"  {'-'*30}")
    for r in [0.9, 0.7, 0.5, 0.3, 0.1]:
        print(f"  {r:<8.1f} {phase_transition_pred(r, 2):>10.3f} {phase_transition_pred(r, 3):>10.3f}")

    # Load grid and show observed EM for comparison
    try:
        grid = load_grid()
        musique_ll = {
            round(g["ratio"], 1): g
            for g in grid
            if g["compressor"] == "llmlingua2" and g["dataset"] == "musique"
        }
        hotpot_ll = {
            round(g["ratio"], 1): g
            for g in grid
            if g["compressor"] == "llmlingua2" and g["dataset"] == "hotpotqa"
        }
        if musique_ll or hotpot_ll:
            print(f"\n  Observed LLMLingua-2 EM (compare to r^k predictions):")
            print(f"  {'Ratio':<8} {'MuSiQue EM':>12} {'HotpotQA EM':>13}")
            print(f"  {'-'*36}")
            for r in [0.9, 0.7, 0.5, 0.3, 0.1]:
                ms = musique_ll.get(r, {}).get("em_mean", float("nan"))
                hp = hotpot_ll.get(r, {}).get("em_mean", float("nan"))
                print(f"  {r:<8.1f} {ms:>12.3f} {hp:>13.3f}")
    except FileNotFoundError:
        print("  (Grid results not yet available for empirical comparison.)")

    # -----------------------------------------------------------------------
    # 4. RCID × compression: high-RCID vs low-RCID split
    # -----------------------------------------------------------------------
    if rcid_vals and len(rcid_vals) >= 10:
        threshold = sorted_rcid[n // 2]  # median split
        high_rcid = [d for d in rcid_data if d.get("rcid") is not None and d["rcid"] >= threshold]
        low_rcid  = [d for d in rcid_data if d.get("rcid") is not None and d["rcid"] <  threshold]

        high_baseline = mean([d["baseline_f1"] for d in high_rcid if d.get("baseline_f1") is not None])
        low_baseline  = mean([d["baseline_f1"] for d in low_rcid  if d.get("baseline_f1") is not None])

        print(f"\n=== 4. High-RCID vs Low-RCID (median split at {threshold:.2f}) ===")
        print(f"  High-RCID group (n={len(high_rcid)}): baseline F1 = {high_baseline:.3f}")
        print(f"  Low-RCID  group (n={len(low_rcid)}):  baseline F1 = {low_baseline:.3f}")
        print(f"  Interpretation: high-RCID examples have {'higher' if high_baseline > low_baseline else 'lower'} baseline F1.")
        print(f"  Under aggressive compression, high-RCID examples should collapse faster.")

    # -----------------------------------------------------------------------
    # 5. Pearson correlation: RCID vs baseline F1
    # -----------------------------------------------------------------------
    paired = [(d["rcid"], d["baseline_f1"]) for d in rcid_data
              if d.get("rcid") is not None and d.get("baseline_f1") is not None]
    if paired:
        r_vals, f1s = zip(*paired)
        r_corr = pearson_r(list(r_vals), list(f1s))
        print(f"\n=== 5. Pearson Correlation: RCID vs Baseline F1 ===")
        print(f"  r = {r_corr:.3f}  (n={len(paired)})")
        if r_corr > 0.4:
            print("  Strong positive correlation: high-RCID contexts are harder to answer.")
        elif r_corr > 0.2:
            print("  Moderate positive correlation.")
        else:
            print("  Weak correlation — RCID and F1 are largely orthogonal at baseline.")


if __name__ == "__main__":
    analyze()
