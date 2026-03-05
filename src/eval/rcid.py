"""
Logic for computing RCID via sentence masking.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Awaitable, Callable, Dict, List, Sequence, Tuple

from .metrics import compute_f1

ModelFn = Callable[[str, str], str | Awaitable[str]]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _named_entities(sentence: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence))


def _content_tokens(sentence: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", sentence.lower()))


def _numbers(sentence: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", sentence))


def _is_paraphrase_like(a: str, b: str) -> bool:
    tok_a = _content_tokens(a)
    tok_b = _content_tokens(b)
    if not tok_a or not tok_b:
        return False
    jaccard = len(tok_a & tok_b) / len(tok_a | tok_b)
    nums_match = bool(_numbers(a) & _numbers(b))
    ents_overlap = bool(_named_entities(a) & _named_entities(b))
    return jaccard >= 0.6 or (jaccard >= 0.45 and (nums_match or ents_overlap))


def _mask_indices(sentences: Sequence[str], indices: set[int]) -> str:
    return " ".join("[MASKED]" if i in indices else s for i, s in enumerate(sentences))


async def _call_model(model: ModelFn, context: str, question: str) -> str:
    result = model(context, question)
    if asyncio.iscoroutine(result):
        return str(await result)
    return str(result)


async def compute_rcid_details(
    context: str,
    question: str,
    answer: str | Sequence[str],
    model: ModelFn,
    epsilon: float = 0.1,
) -> Dict[str, Any]:
    """
    RCID protocol:
    1) Baseline prediction and baseline F1.
    2) Single-sentence masking pass.
    3) Pairwise masking for redundant sentence pairs (same entities or paraphrase-like facts).
    """
    sentences = _split_sentences(context)
    if not sentences:
        return {
            "rcid": 0.0,
            "baseline_f1": 0.0,
            "baseline_prediction": "",
            "critical_indices": [],
            "single_deltas": [],
            "pairwise_deltas": [],
        }

    baseline_prediction = await _call_model(model, context, question)
    baseline_f1 = compute_f1(baseline_prediction, answer)

    single_deltas: List[float] = []
    critical = set()
    for i in range(len(sentences)):
        masked_context = _mask_indices(sentences, {i})
        masked_prediction = await _call_model(model, masked_context, question)
        masked_f1 = compute_f1(masked_prediction, answer)
        delta = baseline_f1 - masked_f1
        single_deltas.append(delta)
        if delta > epsilon:
            critical.add(i)

    pairwise_deltas: List[Tuple[int, int, float]] = []
    entities = [_named_entities(s) for s in sentences]
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            same_entities = bool(entities[i]) and entities[i] == entities[j]
            paraphrase_like = _is_paraphrase_like(sentences[i], sentences[j])
            if not (same_entities or paraphrase_like):
                continue
            masked_context = _mask_indices(sentences, {i, j})
            masked_prediction = await _call_model(model, masked_context, question)
            masked_f1 = compute_f1(masked_prediction, answer)
            delta = baseline_f1 - masked_f1
            pairwise_deltas.append((i, j, delta))

            # Redundancy rule: a pair can jointly be critical even if singles are not.
            if delta > epsilon and single_deltas[i] <= epsilon and single_deltas[j] <= epsilon:
                critical.update({i, j})

    rcid = len(critical) / len(sentences)
    return {
        "rcid": rcid,
        "baseline_f1": baseline_f1,
        "baseline_prediction": baseline_prediction,
        "critical_indices": sorted(critical),
        "single_deltas": single_deltas,
        "pairwise_deltas": pairwise_deltas,
    }


def calculate_rcid(
    context: str,
    question: str,
    answer: str | Sequence[str],
    model: ModelFn,
    epsilon: float = 0.1,
) -> float:
    """
    Synchronous wrapper returning RCID scalar.
    """
    details = asyncio.run(
        compute_rcid_details(context=context, question=question, answer=answer, model=model, epsilon=epsilon)
    )
    return float(details["rcid"])
