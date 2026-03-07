"""
Load MuSiQue, HotpotQA, 2WikiMultihopQA from HuggingFace datasets.
Returns standardized dicts: {id, question, context, answer, hop_count, supporting_facts, dataset}
Uses dev/validation splits only (matching original paper design).

Run directly to verify all three datasets load:
    python datasets_loader.py
"""
from __future__ import annotations

import random
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# MuSiQue
# ---------------------------------------------------------------------------

def _musique(n: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("dgslibisey/musique", split="validation")
    rng = random.Random(seed)
    sample = rng.sample(list(ds), min(n, len(ds)))
    out = []
    for ex in sample:
        paragraphs = ex.get("paragraphs", [])
        context = " ".join(p["paragraph_text"] for p in paragraphs)
        out.append({
            "id": ex["id"],
            "question": ex["question"],
            "context": context,
            "answer": ex["answer"],
            "hop_count": len(ex.get("question_decomposition", [])),
            "supporting_facts": [
                (p.get("idx", i), p["paragraph_text"])
                for i, p in enumerate(paragraphs)
                if p.get("is_supporting", False)
            ],
            "dataset": "musique",
        })
    return out


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

def _hotpotqa(n: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    rng = random.Random(seed)
    sample = rng.sample(list(ds), min(n, len(ds)))
    out = []
    for ex in sample:
        sentences = ex["context"]["sentences"]
        context = " ".join(" ".join(sents) for sents in sentences)
        out.append({
            "id": ex["id"],
            "question": ex["question"],
            "context": context,
            "answer": ex["answer"],
            "hop_count": 2,
            "supporting_facts": ex.get("supporting_facts", {}),
            "dataset": "hotpotqa",
        })
    return out


# ---------------------------------------------------------------------------
# 2WikiMultihopQA
# ---------------------------------------------------------------------------

def _2wiki(n: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    # framolfese/2WikiMultihopQA uses parquet format (no custom script) and
    # follows HotpotQA's field layout: context.title + context.sentences
    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    rng = random.Random(seed)
    sample = rng.sample(list(ds), min(n, len(ds)))
    out = []
    for ex in sample:
        # context is a dict with "title" and "sentences" lists (HotpotQA schema)
        sentences_lists = ex["context"]["sentences"]
        context = " ".join(" ".join(sents) for sents in sentences_lists)
        out.append({
            "id": ex["id"],
            "question": ex["question"],
            "context": context,
            "answer": ex["answer"],
            "hop_count": len(ex.get("evidences", [])) or 2,
            "supporting_facts": ex.get("supporting_facts", {}),
            "dataset": "2wikimultihop",
        })
    return out


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_all(n_per_dataset: int = 100, seed: int = 42) -> Dict[str, List[Dict]]:
    print("Loading MuSiQue...")
    musique = _musique(n_per_dataset, seed)
    print(f"  {len(musique)} examples  |  hop_count: {set(e['hop_count'] for e in musique)}")

    print("Loading HotpotQA...")
    hotpot = _hotpotqa(n_per_dataset, seed)
    print(f"  {len(hotpot)} examples  |  hop_count: {set(e['hop_count'] for e in hotpot)}")

    print("Loading 2WikiMultihopQA...")
    wiki2 = _2wiki(n_per_dataset, seed)
    print(f"  {len(wiki2)} examples  |  hop_count: {set(e['hop_count'] for e in wiki2)}")

    return {
        "musique": musique,
        "hotpotqa": hotpot,
        "2wikimultihop": wiki2,
    }


if __name__ == "__main__":
    import sys
    from collections import Counter
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Use n=100 to match the actual grid — n=5 is misleading for hop distribution
    datasets = load_all(n_per_dataset=100)
    for name, examples in datasets.items():
        hop_dist = Counter(e["hop_count"] for e in examples)
        print(f"\n--- {name} (n={len(examples)}) ---")
        print(f"  hop distribution: {dict(sorted(hop_dist.items()))}")
        ex = examples[0]
        print(f"  Q: {ex['question'][:100]}")
        print(f"  A: {ex['answer']}")
        ctx_preview = ex["context"][:100].encode("ascii", errors="replace").decode("ascii")
        print(f"  Context ({len(ex['context'].split())} words): {ctx_preview}...")
