"""
Compressor implementations:
  - LLMLinguaCompressor  : real LLMLingua-2 (microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank)
  - AbstractiveCompressor: GPT-4o prompted summarizer  (sync + async)
  - TFIDFCompressor      : TF-IDF sentence-ranking baseline

Key fixes vs previous version
-------------------------------
1. LLMLingua-2 compress_prompt() expects `context` as a *list of strings*, not a raw string.
   Passing a raw string causes silent empty-output bugs that silently trigger the fallback.
2. Removed double-truncation: LLMLingua-2 already honours the rate.
   Double-truncation would push actual compression below the target ratio and
   make cross-cell comparisons invalid.
3. AbstractiveCompressor now uses the *sync* OpenAI client in compress() to avoid
   asyncio.run() crashes inside Jupyter / already-running event loops.
4. TFIDFCompressor unchanged — kept as the clean heuristic baseline.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import List, Optional


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _target_word_count(text: str, ratio: float) -> int:
    raw = max(1, len(text.split()))
    clipped = min(1.0, max(0.01, ratio))
    return max(1, int(raw * clipped))


def _truncate_to_words(text: str, budget: int) -> str:
    words = text.split()
    return " ".join(words[:budget])


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseCompressor:
    def compress(self, text: str, ratio: float, question: Optional[str] = None) -> str:
        raise NotImplementedError

    async def acompress(self, text: str, ratio: float, question: Optional[str] = None) -> str:
        return self.compress(text=text, ratio=ratio, question=question)


# ---------------------------------------------------------------------------
# LLMLingua-2 compressor (real implementation)
# ---------------------------------------------------------------------------

class LLMLinguaCompressor(BaseCompressor):
    """
    Uses the published Microsoft LLMLingua-2 model for extractive compression.

    Critical API notes (llmlingua >= 0.2.0)
    ----------------------------------------
    - compress_prompt(context, ...) expects context as a LIST OF STRINGS,
      not a single string. Passing a raw string returns empty/wrong output
      without raising an error (silent failure).
    - `rate` = fraction of tokens to keep (0.3 keeps 30%). Matches our ratio
      convention directly.
    - We do NOT re-truncate the output: LLMLingua-2 already honours rate.
      Double-truncation would push actual compression below target ratio and
      make cross-cell comparisons invalid.
    - `force_tokens` preserves sentence boundaries and question marks.
    """

    MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

    def __init__(self) -> None:
        self._compressor = None
        self._fallback = False
        self._load()

    def _load(self) -> None:
        try:
            from llmlingua import PromptCompressor
            self._compressor = PromptCompressor(
                model_name=self.MODEL_NAME,
                use_llmlingua2=True,
                device_map="cpu",
            )
            print("[LLMLinguaCompressor] Model loaded successfully.")
        except Exception as exc:
            print(
                f"[LLMLinguaCompressor] Could not load LLMLingua-2 ({exc}). "
                "Falling back to TF-IDF scoring. "
                "Install with: pip install llmlingua"
            )
            self._fallback = True

    def compress(self, text: str, ratio: float, question: Optional[str] = None) -> str:
        if self._fallback or self._compressor is None:
            return self._tfidf_fallback(text, ratio, question)
        try:
            # FIX 1: context must be a list of strings, not a bare string
            context_list = _split_sentences(text)
            if not context_list:
                context_list = [text]

            result = self._compressor.compress_prompt(
                context_list,
                rate=float(ratio),
                question=question or "",
                force_tokens=["\n", "?", "."],
                condition_compare=False,
            )

            compressed: str = result.get("compressed_prompt", "").strip()

            if not compressed:
                print("[LLMLinguaCompressor] Empty output; using TF-IDF fallback.")
                return self._tfidf_fallback(text, ratio, question)

            # FIX 2: do NOT re-truncate — LLMLingua-2 already respected rate
            return compressed

        except Exception as exc:
            print(f"[LLMLinguaCompressor] compress_prompt failed ({exc}); using TF-IDF fallback.")
            return self._tfidf_fallback(text, ratio, question)

    def _tfidf_fallback(self, text: str, ratio: float, question: Optional[str]) -> str:
        sentences = _split_sentences(text)
        if not sentences:
            return text
        budget = _target_word_count(text, ratio)
        tokenized = [_tokenize(s) for s in sentences]
        n = len(sentences)
        df: Counter = Counter()
        for toks in tokenized:
            df.update(set(toks))
        question_terms = set(_tokenize(question or ""))
        scored = []
        for idx, toks in enumerate(tokenized):
            tf = Counter(toks)
            score = sum(f * (math.log((n + 1) / (df[t] + 1)) + 1.0) for t, f in tf.items())
            score += 1.2 * sum(1 for t in set(toks) if t in question_terms)
            scored.append((idx, score, sentences[idx]))
        chosen, total = [], 0
        for idx, _, sent in sorted(scored, key=lambda x: x[1], reverse=True):
            if total >= budget:
                break
            chosen.append((idx, sent))
            total += len(sent.split())
        if not chosen:
            return _truncate_to_words(text, budget)
        return _truncate_to_words(
            " ".join(s for _, s in sorted(chosen, key=lambda x: x[0])), budget
        )


# ---------------------------------------------------------------------------
# Abstractive compressor (GPT-4o) — sync + async
# ---------------------------------------------------------------------------

class AbstractiveCompressor(BaseCompressor):
    """
    GPT-4o prompted abstractive compressor.

    FIX 3: Sync path uses openai.OpenAI (not asyncio.run(AsyncOpenAI)),
    which is safe in Jupyter and scripts without a running event loop.
    Async path uses openai.AsyncOpenAI for parallel batch evaluation.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self._sync_client = None
        self._async_client = None

    def _get_sync_client(self):
        if self._sync_client is not None:
            return self._sync_client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("pip install openai") from exc
        self._sync_client = OpenAI(api_key=api_key)
        return self._sync_client

    def _get_async_client(self):
        if self._async_client is not None:
            return self._async_client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("pip install openai") from exc
        self._async_client = AsyncOpenAI(api_key=api_key)
        return self._async_client

    def _build_messages(self, text: str, ratio: float, question: Optional[str]) -> list:
        budget = _target_word_count(text, ratio)
        instruction = (
            "Compress the context below while preserving all facts needed for "
            "multi-hop question answering. "
            f"Target: at most {budget} words. "
            "Keep named entities, numbers, dates, and causal/temporal links. "
            "Do not invent or hallucinate facts."
        )
        if question:
            instruction += f"\nQuestion focus: {question}"
        return [
            {"role": "system", "content": "You are a precise context compressor."},
            {"role": "user", "content": f"{instruction}\n\nContext:\n{text}"},
        ]

    def compress(self, text: str, ratio: float, question: Optional[str] = None) -> str:
        client = self._get_sync_client()
        messages = self._build_messages(text, ratio, question)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        compressed = resp.choices[0].message.content.strip() or text
        budget = _target_word_count(text, ratio)
        return _truncate_to_words(compressed, budget)

    async def acompress(self, text: str, ratio: float, question: Optional[str] = None) -> str:
        client = self._get_async_client()
        messages = self._build_messages(text, ratio, question)
        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        compressed = resp.choices[0].message.content.strip() or text
        budget = _target_word_count(text, ratio)
        return _truncate_to_words(compressed, budget)


# ---------------------------------------------------------------------------
# TF-IDF baseline (clean heuristic reference)
# ---------------------------------------------------------------------------

class TFIDFCompressor(BaseCompressor):
    """
    Classical TF-IDF sentence-ranking baseline with question-term overlap bonus.
    Named clearly to distinguish it from LLMLinguaCompressor.
    """

    def compress(self, text: str, ratio: float, question: Optional[str] = None) -> str:
        sentences = _split_sentences(text)
        if not sentences:
            return text

        budget = _target_word_count(text, ratio)
        tokenized = [_tokenize(s) for s in sentences]
        n = len(sentences)
        df: Counter = Counter()
        for toks in tokenized:
            df.update(set(toks))

        question_terms = set(_tokenize(question or ""))
        scored = []
        for idx, toks in enumerate(tokenized):
            tf = Counter(toks)
            score = sum(
                f * (math.log((n + 1) / (df[t] + 1)) + 1.0)
                for t, f in tf.items()
            )
            if question_terms:
                score += 1.2 * sum(1 for t in set(toks) if t in question_terms)
            scored.append((idx, score, sentences[idx]))

        chosen, total_tokens = [], 0
        for idx, _, sentence in sorted(scored, key=lambda x: x[1], reverse=True):
            if total_tokens >= budget:
                break
            chosen.append((idx, sentence))
            total_tokens += len(sentence.split())

        if not chosen:
            return _truncate_to_words(text, budget)

        ordered = " ".join(s for _, s in sorted(chosen, key=lambda x: x[0]))
        return _truncate_to_words(ordered, budget)


# ---------------------------------------------------------------------------
# Smoke test (python compressors.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE = (
        "Marie Curie was born in Warsaw in 1867. She moved to Paris in 1891 to study physics. "
        "She discovered polonium and radium. In 1903 she shared the Nobel Prize in Physics with "
        "her husband Pierre Curie and Henri Becquerel. She later won the Nobel Prize in Chemistry "
        "in 1911, making her the first person to win Nobel Prizes in two different sciences. "
        "She died in 1934 from aplastic anaemia caused by radiation exposure."
    )
    Q = "Who was the first person to win Nobel Prizes in two different sciences?"

    print("=== TF-IDF ===")
    tfidf = TFIDFCompressor()
    for r in [0.9, 0.5, 0.3, 0.1]:
        out = tfidf.compress(SAMPLE, r, Q)
        print(f"  ratio={r} ({len(out.split())} words): {out[:120]}")

    print("\n=== LLMLingua-2 ===")
    llm = LLMLinguaCompressor()
    for r in [0.9, 0.5, 0.3, 0.1]:
        out = llm.compress(SAMPLE, r, Q)
        print(f"  ratio={r} ({len(out.split())} words): {out[:120]}")
