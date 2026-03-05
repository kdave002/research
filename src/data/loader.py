"""
Data loaders for MuSiQue, HotpotQA, and 2WikiMultihopQA.
"""
from .loaders import load_2wikimultihopqa, load_dataset, load_hotpotqa, load_musique

__all__ = [
    "load_dataset",
    "load_musique",
    "load_hotpotqa",
    "load_2wikimultihopqa",
]
