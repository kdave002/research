"""
RCID (Reasoning-Critical Information Density) computation at scale.

For each MuSiQue example:
1. Get baseline F1 with full context (Claude Haiku evaluator)
2. For each sentence, mask it and recompute F1
3. Sentence is reasoning-critical if F1 drop > epsilon
4. RCID = fraction of critical sentences / total sentences

Validation against ground-truth:
  MuSiQue has `is_supporting` paragraph annotations. We check whether
  RCID-identified critical sentences correspond to annotated supporting facts.
  This gives precision/recall validation of the metric itself.

Outputs:
    results/rcid_scores.jsonl   — one line per example

Usage:
    python rcid.py
    python rcid.py --n 200 --epsilon 0.1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluator import token_f1
from token_tracker import get_haiku_client, record_usage

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RCID_FILE = RESULTS_DIR / "rcid_scores.jsonl"


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def mask_sentence(sentences: List[str], idx: int) -> str:
    return " ".join(s for i, s in enumerate(sentences) if i != idx)


async def _qa(context: str, question: str, client, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        try:
            msg = await client.messages.create(
                model="anthropic/claude-haiku-4-5",
                max_tokens=64,
                system="Answer with a short factoid from the provided context only.",
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }],
            )
            record_usage(msg.usage)
            return msg.content[0].text.strip()
        except Exception as e:
            print(f"  [rcid qa error] {e}")
            return ""


async def compute_rcid_example(
    example: Dict[str, Any],
    client,
    semaphore: asyncio.Semaphore,
    epsilon: float = 0.1,
) -> Dict[str, Any]:
    sentences = split_sentences(example["context"])
    n = len(sentences)

    if n < 2:
        return {
            "id": example["id"],
            "dataset": example["dataset"],
            "rcid": None,
            "n_sentences": n,
            "note": "too_short",
        }

    # Baseline F1 with full context
    baseline_pred = await _qa(example["context"], example["question"], client, semaphore)
    baseline_f1 = token_f1(baseline_pred, example["answer"])

    # Mask each sentence individually
    mask_tasks = [
        _qa(mask_sentence(sentences, i), example["question"], client, semaphore)
        for i in range(n)
    ]
    masked_preds = await asyncio.gather(*mask_tasks)

    sentence_results = []
    for idx, pred in enumerate(masked_preds):
        masked_f1 = token_f1(pred, example["answer"])
        delta = baseline_f1 - masked_f1
        sentence_results.append({
            "idx": idx,
            "sentence_snippet": sentences[idx][:80],
            "delta_f1": round(delta, 4),
            "is_critical": delta > epsilon,
        })

    critical_count = sum(1 for s in sentence_results if s["is_critical"])
    rcid = critical_count / n

    # Validate against MuSiQue ground-truth supporting facts
    validation: Optional[Dict] = None
    if example.get("supporting_facts"):
        supporting_idxs = set()
        for _, para_text in example["supporting_facts"]:
            for i, sent in enumerate(sentences):
                if para_text[:60].strip() in sent:
                    supporting_idxs.add(i)

        if supporting_idxs:
            tp = sum(1 for s in sentence_results if s["is_critical"] and s["idx"] in supporting_idxs)
            precision = tp / max(1, critical_count)
            recall = tp / max(1, len(supporting_idxs))
            validation = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "n_supporting": len(supporting_idxs),
                "n_critical": critical_count,
            }

    return {
        "id": example["id"],
        "dataset": example["dataset"],
        "question": example["question"][:100],
        "answer": example["answer"],
        "hop_count": example.get("hop_count", -1),
        "baseline_f1": round(baseline_f1, 4),
        "rcid": round(rcid, 4),
        "n_sentences": n,
        "n_critical": critical_count,
        "epsilon": epsilon,
        "sentence_results": sentence_results,
        "validation": validation,
    }


async def run_rcid(n: int = 200, epsilon: float = 0.1):
    from datasets_loader import _musique

    print(f"=== RCID Computation: {n} MuSiQue examples, epsilon={epsilon} ===\n")

    examples = _musique(n=n, seed=42)

    completed_ids: set = set()
    if RCID_FILE.exists():
        for line in RCID_FILE.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                r = json.loads(line)
                completed_ids.add(r["id"])
    print(f"Already computed: {len(completed_ids)}/{len(examples)}")

    remaining = [ex for ex in examples if ex["id"] not in completed_ids]
    if not remaining:
        print("All examples already computed.")
        return

    client = get_haiku_client()
    semaphore = asyncio.Semaphore(8)

    tasks = [
        compute_rcid_example(ex, client, semaphore, epsilon=epsilon)
        for ex in remaining
    ]

    done_count = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        with open(RCID_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        done_count += 1
        rcid_str = f"{result.get('rcid', 'N/A'):.2f}" if result.get("rcid") is not None else "N/A"
        print(
            f"  [{done_count}/{len(remaining)}] "
            f"RCID={rcid_str}  "
            f"baseline_F1={result.get('baseline_f1', 0):.2f}  "
            f"n_sent={result.get('n_sentences', '?')}  "
            f"hop={result.get('hop_count', '?')}"
        )

    # Print summary stats
    all_results = []
    for line in RCID_FILE.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("rcid") is not None:
                all_results.append(r)

    if all_results:
        rcid_vals = [r["rcid"] for r in all_results]
        validated = [r for r in all_results if r.get("validation")]
        print(f"\n=== RCID Summary ({len(all_results)} examples) ===")
        print(f"  Mean RCID:   {sum(rcid_vals)/len(rcid_vals):.3f}")
        print(f"  Low (<0.3):  {sum(1 for r in rcid_vals if r < 0.3)}")
        print(f"  High (>0.7): {sum(1 for r in rcid_vals if r > 0.7)}")
        if validated:
            prec = sum(r["validation"]["precision"] for r in validated) / len(validated)
            rec  = sum(r["validation"]["recall"]    for r in validated) / len(validated)
            print(f"  Validation vs ground-truth ({len(validated)} examples):")
            print(f"    Mean precision: {prec:.3f}")
            print(f"    Mean recall:    {rec:.3f}")

    print(f"\nResults saved to {RCID_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()
    asyncio.run(run_rcid(n=args.n, epsilon=args.epsilon))
