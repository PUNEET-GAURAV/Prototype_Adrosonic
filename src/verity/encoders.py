"""Dense encoders behind one interface.

- LSAEncoder: TF-IDF + truncated SVD, fitted on the corpus. Fully offline; used for the sandbox
  prototype. It is a genuine dense retriever but much weaker than a neural model.
- BGEEncoder: BAAI/bge-small-en-v1.5 via sentence-transformers. The production choice.
  NOTE: not executed in the build sandbox (no HuggingFace access); verify on first run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class DenseEncoder(Protocol):
    name: str
    dim: int

    def encode_docs(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


class LSAEncoder:
    def __init__(self, dim: int = 128, min_df: int = 3, max_df: float = 0.4, max_features: int = 80_000):
        self.dim, self.name = dim, f"LSA-{dim} (offline fallback)"
        self.vec = TfidfVectorizer(sublinear_tf=True, min_df=min_df, max_df=max_df, stop_words="english",
                                   max_features=max_features, dtype=np.float32)
        self.svd = TruncatedSVD(dim, n_iter=4, random_state=42)

    def fit(self, texts: list[str]) -> LSAEncoder:
        matrix = self.vec.fit_transform(texts)
        self.dim = min(self.dim, matrix.shape[0], matrix.shape[1])
        self.name = f"LSA-{self.dim} (offline fallback)"
        self.svd = TruncatedSVD(self.dim, n_iter=4, random_state=42)
        self.svd.fit(matrix)
        return self

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return _l2(self.svd.transform(self.vec.transform(texts))).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_docs([text])[0]

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> LSAEncoder:
        return joblib.load(path)


class BGEEncoder:
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy
        self.model, self.batch_size, self.name = SentenceTransformer(model), batch_size, model
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.QUERY_PREFIX + text], normalize_embeddings=True)[0].astype(np.float32)


def make_encoder(cfg: dict, lsa_path: Path):
    if cfg["encoder"]["dense"] == "bge":
        return BGEEncoder(cfg["encoder"]["bge_model"])
    return LSAEncoder.load(lsa_path)
