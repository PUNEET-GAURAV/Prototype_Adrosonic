"""Corpus builders.

- build_demo_corpus(): real text from NLTK corpora (fetched from GitHub), chunked into passages, with
  natural category / source metadata. Used for the offline prototype.
- iter_msmarco(): the real challenge dataset (HF `microsoft/ms_marco`, config v2.1).
  NOTE: not executed in the build sandbox (no HuggingFace access). Verify on first run:
  `python -m verity.datasets --peek`.
"""
from __future__ import annotations

import hashlib
import random
import re
from urllib.parse import urlparse

_SENT = re.compile(r"(?<=[.!?])\s+")


def _norm_hash(text: str) -> str:
    return hashlib.sha1(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()


def _pack(text: str, min_w: int = 30, max_w: int = 65):
    cur, n = [], 0
    for s in _SENT.split(re.sub(r"\s+", " ", text).strip()):
        w = len(s.split())
        if cur and n + w > max_w and n >= min_w:
            yield " ".join(cur)
            cur, n = [], 0
        cur.append(s)
        n += w
    if n >= min_w:
        yield " ".join(cur)


def _detok(t: str) -> str:
    t = t.replace("`` ", '"').replace(" ''", '"')
    t = re.sub(r"\s+([,.;:!?%)])", r"\1", t)
    return re.sub(r"([($])\s+", r"\1", t)


def build_demo_corpus(target: int = 100_000, seed: int = 42) -> list[dict]:
    import nltk
    for rsrc in ["brown", "gutenberg", "inaugural", "movie_reviews", "reuters", "state_union", "webtext", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{rsrc}")
        except LookupError:
            try:
                nltk.data.find(f"tokenizers/{rsrc}")
            except LookupError:
                nltk.download(rsrc)
    from nltk.corpus import (
        brown,
        gutenberg,
        inaugural,
        movie_reviews,
        reuters,
        state_union,
        webtext,
    )
    sources = [("literature", gutenberg), ("news", reuters), ("reviews", movie_reviews), ("web", webtext),
               ("speeches", inaugural), ("speeches", state_union), ("mixed", brown)]
    seen, out = set(), []
    for category, corpus in sources:
        for fid in corpus.fileids():
            raw = _detok(" ".join(corpus.words(fid))) if corpus is brown else corpus.raw(fid)
            for text in _pack(raw):
                h = _norm_hash(text)
                if h not in seen:
                    seen.add(h)
                    out.append({"text": text, "category": category, "source": fid})
    random.Random(seed).shuffle(out)
    out = out[:target]
    for i, r in enumerate(out):
        r["passage_id"] = f"P{i:06d}"
    return out


def iter_msmarco(n_passages: int, seed: int = 42, split: str = "validation") -> list[dict]:
    """Passages from MS MARCO v2.1: category = query_type, source = URL domain."""
    from datasets import load_dataset
    ds = load_dataset("microsoft/ms_marco", "v2.1", split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    seen, out = set(), []
    for row in ds:
        for text, url in zip(row["passages"]["passage_text"], row["passages"]["url"]):
            h = _norm_hash(text)
            if h in seen:
                continue
            seen.add(h)
            out.append({"text": text, "category": row["query_type"], "source": urlparse(url).netloc or "unknown"})
        if len(out) >= n_passages:
            break
    for i, r in enumerate(out[:n_passages]):
        r["passage_id"] = f"M{i:07d}"
    return out[:n_passages]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", action="store_true")
    ap.add_argument("--dataset", default="demo")
    a = ap.parse_args()
    rows = build_demo_corpus(20) if a.dataset == "demo" else iter_msmarco(20)
    for r in rows[:5]:
        print(r)
