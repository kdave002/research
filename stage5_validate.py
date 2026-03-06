"""
Stage 5 validation: checks outputs/results.csv for correctness.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path("outputs/results.csv")
EXPECTED_COMPRESSORS = {"llmlingua", "abstractive", "tfidf"}
EXPECTED_DATASETS = {"musique", "hotpotqa", "2wikimultihopqa"}
EXPECTED_RATIOS = {0.9, 0.7, 0.5, 0.3, 0.1}
EXPECTED_CELLS = len(EXPECTED_COMPRESSORS) * len(EXPECTED_DATASETS) * len(EXPECTED_RATIOS)
ABSTRACTIVE_VARIANCE_THRESHOLD = 0.05
MONOTONE_COMPRESSORS = {"llmlingua", "tfidf"}


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f": {detail}"
    print(line)


def check_cell_count(rows: list[dict]) -> bool:
    ok = len(rows) == EXPECTED_CELLS
    _fmt(
        f"45 cells present",
        ok,
        f"{len(rows)} rows found (expected {EXPECTED_CELLS})",
    )
    return ok


def check_n_examples(rows: list[dict]) -> bool:
    bad = [
        f"{r['compressor']}|{r['dataset']}|{r['ratio']} -> n={r['n_examples']}"
        for r in rows
        if int(r["n_examples"]) != 100
    ]
    ok = len(bad) == 0
    _fmt(
        "n_examples=100 on every row",
        ok,
        f"{len(bad)} violations: {bad}" if bad else "all rows n=100",
    )
    return ok


def check_f1_range(rows: list[dict]) -> bool:
    bad = []
    for r in rows:
        f1 = float(r["f1"])
        if not (0.0 <= f1 <= 1.0):
            bad.append(f"{r['compressor']}|{r['dataset']}|{r['ratio']} -> f1={f1:.4f}")
    ok = len(bad) == 0
    _fmt(
        "F1 in [0, 1] for all rows",
        ok,
        f"{len(bad)} out-of-range: {bad}" if bad else "all F1 values valid",
    )
    return ok


def check_no_duplicates(rows: list[dict]) -> bool:
    seen: set[tuple] = set()
    dupes = []
    for r in rows:
        key = (r["compressor"], r["dataset"], r["ratio"])
        if key in seen:
            dupes.append(str(key))
        seen.add(key)
    ok = len(dupes) == 0
    _fmt(
        "No duplicate compressor/dataset/ratio combinations",
        ok,
        f"{len(dupes)} duplicates: {dupes}" if dupes else "no duplicates",
    )
    return ok


def check_monotone_decrease(rows: list[dict]) -> bool:
    # Group by compressor x dataset, then check F1 order by ratio descending.
    grouped: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        c = r["compressor"]
        if c not in MONOTONE_COMPRESSORS:
            continue
        grouped[(c, r["dataset"])].append((float(r["ratio"]), float(r["f1"])))

    violations = []
    for (comp, ds), pairs in sorted(grouped.items()):
        ordered = sorted(pairs, key=lambda x: x[0], reverse=True)  # high ratio first
        for i in range(len(ordered) - 1):
            r_hi, f1_hi = ordered[i]
            r_lo, f1_lo = ordered[i + 1]
            if f1_lo > f1_hi:
                violations.append(
                    f"{comp}|{ds}: ratio {r_hi}->{r_lo} f1 {f1_hi:.4f}->{f1_lo:.4f} (went up)"
                )

    ok = len(violations) == 0
    detail = f"{len(violations)} violation(s):\n    " + "\n    ".join(violations) if violations else "monotone for all extractive compressors"
    _fmt(
        "LLMLingua & TF-IDF F1 decreases monotonically as ratio decreases",
        ok,
        detail,
    )
    return ok


def check_abstractive_variance(rows: list[dict]) -> bool:
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["compressor"] != "abstractive":
            continue
        grouped[r["dataset"]].append(float(r["f1"]))

    violations = []
    details = []
    for ds, f1s in sorted(grouped.items()):
        var = statistics.variance(f1s) if len(f1s) > 1 else 0.0
        details.append(f"{ds}: var={var:.5f} (n={len(f1s)}, min={min(f1s):.4f}, max={max(f1s):.4f})")
        if var >= ABSTRACTIVE_VARIANCE_THRESHOLD:
            violations.append(f"{ds} variance={var:.5f} >= {ABSTRACTIVE_VARIANCE_THRESHOLD}")

    ok = len(violations) == 0
    summary = "; ".join(details)
    if violations:
        summary += f" | VIOLATIONS: {violations}"
    _fmt(
        f"Abstractive F1 variance < {ABSTRACTIVE_VARIANCE_THRESHOLD} per dataset",
        ok,
        summary,
    )
    return ok


def main() -> None:
    print(f"=== Stage 5 Validation: {CSV_PATH} ===\n")

    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        return

    rows = _load(CSV_PATH)
    print(f"Loaded {len(rows)} rows.\n")

    results = [
        check_cell_count(rows),
        check_n_examples(rows),
        check_f1_range(rows),
        check_no_duplicates(rows),
        check_monotone_decrease(rows),
        check_abstractive_variance(rows),
    ]

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Result: {passed}/{total} checks passed", "— ALL PASS" if passed == total else "— SOME FAILED")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
