"""
EM and F1 scoring logic.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List, Sequence


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _tokens(text: str) -> List[str]:
    normalized = _normalize_answer(text)
    if not normalized:
        return []
    return normalized.split()


def _as_answers(ground_truth: str | Sequence[str]) -> Sequence[str]:
    if isinstance(ground_truth, str):
        return [ground_truth]
    return list(ground_truth)


def _f1_single(prediction: str, answer: str) -> float:
    pred_tokens = _tokens(prediction)
    gt_tokens = _tokens(answer)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1(prediction: str, ground_truth: str | Sequence[str]) -> float:
    answers = _as_answers(ground_truth)
    if not answers:
        return 0.0
    return max(_f1_single(prediction, answer) for answer in answers)


def _em_single(prediction: str, answer: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(answer) else 0.0


def compute_em(prediction: str, ground_truth: str | Sequence[str]) -> float:
    answers = _as_answers(ground_truth)
    if not answers:
        return 0.0
    return max(_em_single(prediction, answer) for answer in answers)
