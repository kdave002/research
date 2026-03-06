from __future__ import annotations

import argparse
import asyncio
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from data.loaders import load_dataset
from eval.metrics import compute_em, compute_f1
from models.compressors import AbstractiveCompressor, LLMLinguaCompressor, TFIDFCompressor


DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
RATIOS = [0.9, 0.7, 0.5, 0.3, 0.1]
PER_QA_CALL_COST_USD = 0.003
RCID_SUBSET_COST_USD = 15.0


@dataclass
class CellConfig:
    compressor_name: str
    dataset_name: str
    ratio: float


class ResultLogger:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()
        self._lock = asyncio.Lock()

    def _ensure_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "compressor",
                    "dataset",
                    "ratio",
                    "n_examples",
                    "em",
                    "f1",
                    "duration_sec",
                    "status",
                    "error",
                ],
            )
            writer.writeheader()

    async def append(self, row: Dict[str, object]) -> None:
        async with self._lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)


class GPT4oQA:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = None
        self._fallback_mode = False

        if not self.api_key:
            self._fallback_mode = True
            return
        try:
            from openai import AsyncOpenAI
        except ImportError:
            self._fallback_mode = True
            return
        self._client = AsyncOpenAI(api_key=self.api_key)

    @property
    def using_fallback(self) -> bool:
        return self._fallback_mode

    async def answer(self, question: str, context: str) -> str:
        if self._fallback_mode:
            # Deterministic fallback for environments without API/key.
            for sentence in context.split("."):
                if sentence.strip():
                    return sentence.strip()
            return context[:120]
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
    "You are a precise question answering system. "
    "Read the context carefully and answer the question with "
    "the exact entity, name, date, or short phrase from the context. "
    "If the answer requires combining facts from multiple sentences, do so. "
    "Answer in 1-5 words only. Do not explain."
)},
{"role": "user", "content": (
    f"Context:\n{context}\n\n"
    f"Question: {question}\n\n"
    f"Answer (1-5 words, exact phrase from context):"
)},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


def _projected_full_run_cost(per_cell_examples: int = 100) -> Tuple[float, float, float]:
    total_cells = 3 * 3 * 5
    total_qa_calls = total_cells * per_cell_examples
    qa_cost = total_qa_calls * PER_QA_CALL_COST_USD
    low = qa_cost + RCID_SUBSET_COST_USD
    high = 35.0
    return qa_cost, low, high


def _build_compressors() -> Dict[str, object]:
    return {
        "llmlingua": LLMLinguaCompressor(),
        "abstractive": AbstractiveCompressor(model="gpt-4o"),
        "tfidf": TFIDFCompressor(),
    }


async def _compress(compressor: object, context: str, ratio: float, question: str) -> str:
    if hasattr(compressor, "acompress"):
        return await compressor.acompress(text=context, ratio=ratio, question=question)
    return compressor.compress(text=context, ratio=ratio, question=question)


async def evaluate_cell(
    cell: CellConfig,
    examples: List[Tuple[str, str, str]],
    compressors: Dict[str, object],
    qa: GPT4oQA,
) -> Dict[str, object]:
    start = time.perf_counter()
    compressor = compressors[cell.compressor_name]
    em_scores: List[float] = []
    f1_scores: List[float] = []

    for context, question, answer in examples:
        compressed = await _compress(compressor, context=context, ratio=cell.ratio, question=question)
        prediction = await qa.answer(question=question, context=compressed)
        em_scores.append(compute_em(prediction, answer))
        f1_scores.append(compute_f1(prediction, answer))

    duration = time.perf_counter() - start
    n = max(1, len(examples))
    return {
        "timestamp": int(time.time()),
        "compressor": cell.compressor_name,
        "dataset": cell.dataset_name,
        "ratio": cell.ratio,
        "n_examples": len(examples),
        "em": sum(em_scores) / n,
        "f1": sum(f1_scores) / n,
        "duration_sec": round(duration, 3),
        "status": "ok",
        "error": "",
    }


async def run_grid(
    cells: Sequence[CellConfig],
    dataset_examples: Dict[str, List[Tuple[str, str, str]]],
    logger: ResultLogger,
    max_concurrency: int = 4,
) -> None:
    compressors = _build_compressors()
    qa = GPT4oQA(model="gpt-4o")
    sem = asyncio.Semaphore(max_concurrency)

    async def runner(cell: CellConfig) -> Dict[str, object]:
        async with sem:
            try:
                result = await evaluate_cell(
                    cell=cell,
                    examples=dataset_examples[cell.dataset_name],
                    compressors=compressors,
                    qa=qa,
                )
                return result
            except Exception as exc:
                return {
                    "timestamp": int(time.time()),
                    "compressor": cell.compressor_name,
                    "dataset": cell.dataset_name,
                    "ratio": cell.ratio,
                    "n_examples": len(dataset_examples.get(cell.dataset_name, [])),
                    "em": 0.0,
                    "f1": 0.0,
                    "duration_sec": 0.0,
                    "status": "error",
                    "error": str(exc),
                }

    tasks = [asyncio.create_task(runner(cell)) for cell in cells]
    for task in asyncio.as_completed(tasks):
        row = await task
        await logger.append(row)
        print(
            f"[completed] {row['compressor']} | {row['dataset']} | ratio={row['ratio']} | "
            f"f1={row['f1']:.4f} | status={row['status']}"
        )

    if qa.using_fallback:
        print("Warning: OPENAI_API_KEY or openai package unavailable; fallback QA mode was used.")


def _confirm_full_run(non_interactive_yes: bool) -> bool:
    if non_interactive_yes:
        return True
    answer = input("Proceed with full 45-cell run? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 evaluation runner")
    parser.add_argument("--dry-run", action="store_true", help="Run one-cell pipeline check only")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt for full grid")
    parser.add_argument("--split", default="dev", help="Dataset split")
    parser.add_argument("--data-dir", default="data", help="Dataset root directory")
    parser.add_argument("--max-examples", type=int, default=100, help="Examples per dataset for each cell")
    parser.add_argument("--output-csv", default="outputs/results.csv", help="CSV path for streaming results")
    parser.add_argument("--max-concurrency", type=int, default=4, help="Concurrent cell workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa_cost, low, high = _projected_full_run_cost(per_cell_examples=100)
    print("=== Stage 4 Dry-Run Precheck ===")
    print(f"Projected full 45-cell QA cost: ${qa_cost:.2f}")
    print(f"Projected full Stage 4 total: ${low:.2f} - ${high:.2f}")

    compressors = ["llmlingua", "abstractive", "tfidf"]
    full_cells = [CellConfig(c, d, r) for c in compressors for d in DATASETS for r in RATIOS]
    if args.dry_run:
        selected_cells = [full_cells[0]]
        print("Dry-run mode: executing exactly 1 cell.")
    else:
        if not _confirm_full_run(args.yes):
            print("Cancelled.")
            return
        selected_cells = full_cells
        print(f"Launching full run: {len(selected_cells)} cells.")

    per_dataset_limit = 1 if args.dry_run else args.max_examples
    dataset_examples: Dict[str, List[Tuple[str, str, str]]] = {}
    for dataset_name in DATASETS:
        rows = load_dataset(
            dataset_name,
            split=args.split,
            data_dir=args.data_dir,
            max_examples=per_dataset_limit,
            allow_fallback=True,
        )
        dataset_examples[dataset_name] = rows
        print(f"Loaded {len(rows)} examples for {dataset_name}.")

    logger = ResultLogger(Path(args.output_csv))
    asyncio.run(
        run_grid(
            cells=selected_cells,
            dataset_examples=dataset_examples,
            logger=logger,
            max_concurrency=args.max_concurrency,
        )
    )
    print(f"Results streamed to {args.output_csv}")


if __name__ == "__main__":
    main()
