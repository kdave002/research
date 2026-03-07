"""
QA Evaluation using two independent evaluators:

  PRIMARY:    Claude Haiku 4.5 via Vercel AI Gateway — independent from GPT-4o compressor
  SECONDARY:  GPT-4o — for contamination comparison experiment only

Vercel AI Gateway is used for Haiku so that a single API key (VERCEL_AI_GATEWAY_KEY)
covers all Claude calls without needing a direct Anthropic account.
Set VERCEL_AI_GATEWAY_KEY in your environment (or .env file).

This two-evaluator design directly addresses the circular-measurement criticism.
All metrics (F1, EM) are computed locally — no external API calls for scoring.
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import List, Dict, Any

from token_tracker import get_haiku_client, record_usage, HAIKU_MODEL


# ---------------------------------------------------------------------------
# Local metric computation (no API)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_toks = _normalize(prediction).split()
    gold_toks = _normalize(ground_truth).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    prec = sum(min(pred_toks.count(t), gold_toks.count(t)) for t in common) / len(pred_toks)
    rec  = sum(min(pred_toks.count(t), gold_toks.count(t)) for t in common) / len(gold_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize(prediction) == _normalize(ground_truth))


# ---------------------------------------------------------------------------
# Claude Haiku evaluator (PRIMARY — via Vercel AI Gateway)
# ---------------------------------------------------------------------------

async def _haiku_answer(context: str, question: str, client) -> str:
    msg = await client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=64,
        system=(
            "Answer with a short factoid from the provided context only. "
            "If the answer is not in the context, reply: unanswerable"
        ),
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    record_usage(msg.usage)
    return msg.content[0].text.strip()


# ---------------------------------------------------------------------------
# GPT-4o evaluator (SECONDARY — contamination comparison only)
# ---------------------------------------------------------------------------

async def _gpt4o_answer(context: str, question: str, client) -> str:
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer with a short factoid from the provided context only. "
                    "If the answer is not in the context, reply: unanswerable"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.0,
        max_tokens=64,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Batch cell evaluation
# ---------------------------------------------------------------------------

async def evaluate_cell(
    examples: List[Dict[str, Any]],
    evaluator: str = "haiku",
    concurrency: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate one cell (list of compressed examples).
    Each example must have: context_compressed, question, answer.

    Returns:
        f1_mean, em_mean, f1_std, per_example (list), n
    """
    if evaluator == "haiku":
        client = get_haiku_client()
        answer_fn = _haiku_answer
    elif evaluator == "gpt4o":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        answer_fn = _gpt4o_answer
    else:
        raise ValueError(f"Unknown evaluator: {evaluator}. Use 'haiku' or 'gpt4o'.")

    semaphore = asyncio.Semaphore(concurrency)

    async def eval_one(ex: Dict) -> Dict:
        async with semaphore:
            try:
                pred = await answer_fn(ex["context_compressed"], ex["question"], client)
            except Exception as e:
                print(f"  [evaluator error] {e}")
                pred = ""
            f1 = token_f1(pred, ex["answer"])
            em = exact_match(pred, ex["answer"])
            return {
                "id": ex.get("id", ""),
                "prediction": pred,
                "answer": ex["answer"],
                "f1": f1,
                "em": em,
            }

    results = await asyncio.gather(*[eval_one(ex) for ex in examples])

    f1s = [r["f1"] for r in results]
    ems = [r["em"] for r in results]
    n = len(f1s)
    mean_f1 = sum(f1s) / n
    std_f1 = (sum((x - mean_f1) ** 2 for x in f1s) / n) ** 0.5

    return {
        "f1_mean": mean_f1,
        "em_mean": sum(ems) / n,
        "f1_std": std_f1,
        "per_example": list(results),
        "n": n,
    }
