"""Tokenisation and BM25 sparse encoding.

BM25 is stored inside Qdrant as a *sparse vector* whose collection uses the IDF modifier.
Document side: saturated term-frequency with length normalisation (computed here).
Query side: weight 1 per unique term. Qdrant multiplies by IDF at query time from the live
corpus, so scores stay correct after upserts and deletes, and filters apply to this leg too.
"""
from __future__ import annotations

import functools
import re
import zlib
from collections import defaultdict

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")
_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "she", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "will", "with", "would", "you", "your"]
)


@functools.lru_cache(maxsize=500_000)
def _stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in STOPWORDS]


def token_id(token: str) -> int:
    return zlib.crc32(token.encode()) & 0x7FFFFFFF


class SparseEncoder:
    def __init__(self, k: float = 1.2, b: float = 0.75, avg_len: float = 30.0):
        self.k, self.b, self.avg_len = k, b, avg_len

    @staticmethod
    def measure_avg_len(texts) -> float:
        lens = [len(tokens(t)) for t in texts]
        return sum(lens) / max(len(lens), 1)

    def doc(self, text: str) -> tuple[list[int], list[float]]:
        toks = tokens(text)
        length = max(len(toks), 1)
        counts: dict[str, int] = defaultdict(int)
        for t in toks:
            counts[t] += 1
        weights: dict[int, float] = defaultdict(float)
        norm = self.k * (1 - self.b + self.b * length / self.avg_len)
        for t, c in counts.items():
            weights[token_id(t)] += c * (self.k + 1) / (c + norm)
        idx = sorted(weights)
        return idx, [weights[i] for i in idx]

    def query(self, text: str) -> tuple[list[int], list[float]]:
        idx = sorted({token_id(t) for t in tokens(text)})
        return idx, [1.0] * len(idx)
