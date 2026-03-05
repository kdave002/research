"""
Dataset loaders for MuSiQue, HotpotQA, and 2WikiMultihopQA.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

Example = Tuple[str, str, str]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _coerce_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    payload = _read_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "rows"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    raise ValueError(f"Unsupported dataset payload structure in {path}")


def _resolve_path(dataset_name: str, split: str, data_dir: str | Path | None) -> Path:
    root = Path(data_dir or "data")
    candidates = [
        root / dataset_name / f"{split}.json",
        root / dataset_name / f"{split}.jsonl",
        root / f"{dataset_name}_{split}.json",
        root / f"{dataset_name}_{split}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {dataset_name} split={split}. "
        f"Checked: {', '.join(str(p) for p in candidates)}"
    )


def _musique_context(row: Dict[str, Any]) -> str:
    paragraphs = row.get("paragraphs", [])
    if isinstance(paragraphs, list) and paragraphs:
        parts = []
        for p in paragraphs:
            if isinstance(p, dict):
                title = p.get("title", "")
                text = p.get("paragraph_text") or p.get("paragraph") or p.get("text") or ""
                parts.append(f"{title}: {text}".strip(": "))
            elif isinstance(p, str):
                parts.append(p)
        if parts:
            return "\n".join(parts)
    return row.get("context", "") or row.get("paragraph", "")


def _hotpot_context(row: Dict[str, Any]) -> str:
    context = row.get("context", [])
    if isinstance(context, list) and context:
        parts = []
        for item in context:
            if isinstance(item, list) and len(item) >= 2:
                title = item[0]
                sentences = item[1]
                if isinstance(sentences, list):
                    text = " ".join(str(s) for s in sentences)
                else:
                    text = str(sentences)
                parts.append(f"{title}: {text}".strip(": "))
            elif isinstance(item, dict):
                title = item.get("title", "")
                sentences = item.get("sentences", item.get("text", []))
                if isinstance(sentences, list):
                    text = " ".join(str(s) for s in sentences)
                else:
                    text = str(sentences)
                parts.append(f"{title}: {text}".strip(": "))
        if parts:
            return "\n".join(parts)
    return row.get("paragraph", "") or row.get("context_str", "")


def _wiki2_context(row: Dict[str, Any]) -> str:
    context = row.get("context")
    if isinstance(context, list) and context:
        parts = []
        for item in context:
            if isinstance(item, list) and len(item) >= 2:
                title = item[0]
                sentences = item[1]
                text = " ".join(str(s) for s in sentences) if isinstance(sentences, list) else str(sentences)
                parts.append(f"{title}: {text}".strip(": "))
            elif isinstance(item, dict):
                title = item.get("title", "")
                text = item.get("text", "")
                if isinstance(text, list):
                    text = " ".join(str(s) for s in text)
                parts.append(f"{title}: {text}".strip(": "))
        if parts:
            return "\n".join(parts)
    return row.get("paragraph", "") or row.get("context_str", "")


def _extract_answer(row: Dict[str, Any]) -> str:
    answer = row.get("answer", "")
    if isinstance(answer, list):
        return str(answer[0]) if answer else ""
    return str(answer)


def _limit(rows: List[Example], max_examples: int | None) -> List[Example]:
    if max_examples is None:
        return rows
    return rows[: max(0, max_examples)]


def _fallback_dataset(dataset_name: str) -> List[Example]:
    return [
        (
            "Paris is the capital of France. France is in Europe.",
            f"Fallback sample for {dataset_name}: What is the capital of France?",
            "Paris",
        )
    ]


def load_musique(split: str = "dev", data_dir: str | Path | None = None, max_examples: int | None = None) -> List[Example]:
    path = _resolve_path("musique", split, data_dir)
    rows = _coerce_rows(path)
    out: List[Example] = []
    for row in rows:
        context = _musique_context(row)
        question = row.get("question", "")
        answer = _extract_answer(row)
        if context and question and answer:
            out.append((context, str(question), answer))
    return _limit(out, max_examples)


def load_hotpotqa(split: str = "dev", data_dir: str | Path | None = None, max_examples: int | None = None) -> List[Example]:
    path = _resolve_path("hotpotqa", split, data_dir)
    rows = _coerce_rows(path)
    out: List[Example] = []
    for row in rows:
        context = _hotpot_context(row)
        question = row.get("question", "")
        answer = _extract_answer(row)
        if context and question and answer:
            out.append((context, str(question), answer))
    return _limit(out, max_examples)


def load_2wikimultihopqa(
    split: str = "dev", data_dir: str | Path | None = None, max_examples: int | None = None
) -> List[Example]:
    path = _resolve_path("2wikimultihopqa", split, data_dir)
    rows = _coerce_rows(path)
    out: List[Example] = []
    for row in rows:
        context = _wiki2_context(row)
        question = row.get("question", "")
        answer = _extract_answer(row)
        if context and question and answer:
            out.append((context, str(question), answer))
    return _limit(out, max_examples)


def load_dataset(
    name: str,
    split: str = "dev",
    data_dir: str | Path | None = None,
    max_examples: int | None = None,
    allow_fallback: bool = True,
) -> List[Example]:
    dataset = name.strip().lower()
    try:
        if dataset in {"musique"}:
            return load_musique(split=split, data_dir=data_dir, max_examples=max_examples)
        if dataset in {"hotpotqa", "hotpot"}:
            return load_hotpotqa(split=split, data_dir=data_dir, max_examples=max_examples)
        if dataset in {"2wikimultihopqa", "2wiki", "wikimultihopqa"}:
            return load_2wikimultihopqa(split=split, data_dir=data_dir, max_examples=max_examples)
    except FileNotFoundError:
        if allow_fallback:
            return _limit(_fallback_dataset(name), max_examples)
        raise
    raise ValueError(f"Unknown dataset: {name}")
