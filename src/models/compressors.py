"""
Implementations of LLMLingua, GPT-4o Abstractive, and TF-IDF compressors.
"""
from __future__ import annotations

import asyncio
import math
import os
import re
from collections import Counter, defaultdict
from typing import Iterable, List, Sequence


def _split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _target_token_count(text: str, ratio: float) -> int:
    raw = max(1, len(text.split()))
    clipped = min(1.0, max(0.01, ratio))
    return max(1, int(raw * clipped))


def _truncate_to_tokens(text: str, token_budget: int) -> str:
    words = text.split()
    return " ".join(words[:token_budget])


class BaseCompressor:
    def compress(self, text: str, ratio: float, question: str | None = None) -> str:
        raise NotImplementedError

    async def acompress(self, text: str, ratio: float, question: str | None = None) -> str:
        return self.compress(text=text, ratio=ratio, question=question)


class LLMLinguaCompressor(BaseCompressor):
    _stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
    }

    def compress(self, text: str, ratio: float, question: str | None = None) -> str:
        sentences = _split_sentences(text)
        if not sentences:
            return text

        token_budget = _target_token_count(text, ratio)
        question_terms = set(_tokenize(question or ""))

        doc_tokens = [_tokenize(s) for s in sentences]
        df = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        n_sent = len(sentences)

        scored = []
        for idx, (sentence, tokens) in enumerate(zip(sentences, doc_tokens)):
            tf = Counter(t for t in tokens if t not in self._stop)
            rarity = sum((1.0 / (df[t] or 1.0)) * c for t, c in tf.items())
            has_number = 1.2 if re.search(r"\d", sentence) else 0.0
            has_entity = 1.0 if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence) else 0.0
            connective = 0.8 if re.search(r"\b(because|therefore|after|before|then|if|when)\b", sentence.lower()) else 0.0
            q_overlap = 1.5 * sum(1 for t in set(tokens) if t in question_terms)
            score = rarity + has_number + has_entity + connective + q_overlap
            scored.append((idx, score, sentence))

        # Select best sentences by score, then restore original order.
        selected = []
        total_tokens = 0
        for idx, _, sentence in sorted(scored, key=lambda x: x[1], reverse=True):
            sent_tokens = len(sentence.split())
            if total_tokens >= token_budget:
                break
            selected.append((idx, sentence))
            total_tokens += sent_tokens

        if not selected:
            return _truncate_to_tokens(text, token_budget)

        ordered = [s for _, s in sorted(selected, key=lambda x: x[0])]
        return _truncate_to_tokens(" ".join(ordered), token_budget)


class AbstractiveCompressor(BaseCompressor):
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for AbstractiveCompressor")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for AbstractiveCompressor") from exc
        self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    def compress(self, text: str, ratio: float, question: str | None = None) -> str:
        return asyncio.run(self.acompress(text=text, ratio=ratio, question=question))

    async def acompress(self, text: str, ratio: float, question: str | None = None) -> str:
        token_budget = _target_token_count(text, ratio)
        client = self._get_client()
        prompt = (
            "Compress the context while preserving facts needed for multi-hop QA. "
            f"Target length: <= {token_budget} words. Keep named entities, numbers, dates, and causal/temporal links. "
            "Do not invent facts."
        )
        if question:
            prompt += f"\nQuestion focus: {question}"
        resp = await client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "You are a precise context compressor."},
                {"role": "user", "content": f"{prompt}\n\nContext:\n{text}"},
            ],
            temperature=0.0,
        )
        compressed = (getattr(resp, "output_text", "") or "").strip()
        if not compressed:
            compressed = text
        return _truncate_to_tokens(compressed, token_budget)


class TFIDFCompressor(BaseCompressor):
    def compress(self, text: str, ratio: float, question: str | None = None) -> str:
        sentences = _split_sentences(text)
        if not sentences:
            return text

        token_budget = _target_token_count(text, ratio)
        tokenized = [_tokenize(s) for s in sentences]
        n = len(sentences)
        df = Counter()
        for toks in tokenized:
            df.update(set(toks))

        question_terms = set(_tokenize(question or ""))
        scored = []
        for idx, toks in enumerate(tokenized):
            tf = Counter(toks)
            score = 0.0
            for t, f in tf.items():
                idf = math.log((n + 1) / (df[t] + 1)) + 1.0
                score += f * idf
            if question_terms:
                score += 1.2 * sum(1 for t in set(toks) if t in question_terms)
            scored.append((idx, score, sentences[idx]))

        chosen = []
        total_tokens = 0
        for idx, _, sentence in sorted(scored, key=lambda x: x[1], reverse=True):
            if total_tokens >= token_budget:
                break
            chosen.append((idx, sentence))
            total_tokens += len(sentence.split())

        if not chosen:
            return _truncate_to_tokens(text, token_budget)
        ordered = " ".join(s for _, s in sorted(chosen, key=lambda x: x[0]))
        return _truncate_to_tokens(ordered, token_budget)
