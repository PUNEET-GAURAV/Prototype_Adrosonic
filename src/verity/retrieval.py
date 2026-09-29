"""Retrieval pipeline: dense, sparse (BM25) and hybrid modes with per-stage timings.

Both hybrid legs run in parallel with the SAME database-level filter, then are fused client-side
(weighted RRF) so the fusion method and weights are explicit, documented and configurable.
"""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

from .fusion import weighted_rrf
from .store import Store


@dataclass
class Hit:
    id: str
    passage_id: str
    text: str
    category: str
    source: str
    origin: str
    score: float
    ranks: dict = field(default_factory=dict)    # rank per leg (1-based)
    scores: dict = field(default_factory=dict)   # raw score per leg

    def to_dict(self) -> dict:
        return asdict(self)


def _ms(t0: int) -> float:
    return (time.perf_counter_ns() - t0) / 1e6


class Pipeline:
    def __init__(self, store: Store, dense, sparse, cfg: dict):
        self.store, self.dense, self.sparse, self.cfg = store, dense, sparse, cfg["retrieval"]
        self._pool = ThreadPoolExecutor(max_workers=2)

    def _timed(self, fn, *a):
        t0 = time.perf_counter_ns()
        return fn(*a), _ms(t0)

    @staticmethod
    def _hit(p, score, ranks, scores) -> Hit:
        pl = p.payload
        return Hit(str(p.id), pl["passage_id"], pl["text"], pl.get("category", ""), pl.get("source", ""),
                   pl.get("origin", ""), score, ranks, scores)

    def search(self, query: str, mode: str = "hybrid", k: int | None = None, flt=None, depth: int | None = None,
               w_dense: float | None = None, w_sparse: float | None = None, rrf_k: int | None = None):
        c = self.cfg
        k, depth = k or c["k"], depth or c["depth"]
        wd = c["w_dense"] if w_dense is None else w_dense
        ws = c["w_sparse"] if w_sparse is None else w_sparse
        rrf_k = rrf_k or c["rrf_k"]
        T0, tm = time.perf_counter_ns(), {}

        qv = si = sv = None
        if mode in ("dense", "hybrid"):
            t = time.perf_counter_ns(); qv = self.dense.encode_query(query); tm["embed_ms"] = _ms(t)
        if mode in ("sparse", "hybrid"):
            t = time.perf_counter_ns(); si, sv = self.sparse.query(query); tm["tokenize_ms"] = _ms(t)

        if mode == "dense":
            pts, tm["dense_ms"] = self._timed(self.store.query_dense, qv, k, flt)
            hits = [self._hit(p, p.score, {"dense": i + 1}, {"dense": p.score}) for i, p in enumerate(pts)]
        elif mode == "sparse":
            pts, tm["sparse_ms"] = self._timed(self.store.query_sparse, si, sv, k, flt)
            hits = [self._hit(p, p.score, {"sparse": i + 1}, {"sparse": p.score}) for i, p in enumerate(pts)]
        elif mode == "hybrid":
            fd = self._pool.submit(self._timed, self.store.query_dense, qv, depth, flt)
            fs = self._pool.submit(self._timed, self.store.query_sparse, si, sv, depth, flt)
            (dp, tm["dense_ms"]), (sp, tm["sparse_ms"]) = fd.result(), fs.result()
            t = time.perf_counter_ns()
            fused = weighted_rrf({"dense": [str(p.id) for p in dp], "sparse": [str(p.id) for p in sp]},
                                 {"dense": wd, "sparse": ws}, rrf_k)[:k]
            pts = {str(p.id): p for p in [*sp, *dp]}
            rd = {str(p.id): (i + 1, p.score) for i, p in enumerate(dp)}
            rs = {str(p.id): (i + 1, p.score) for i, p in enumerate(sp)}
            hits = []
            for pid, sc in fused:
                ranks = {n: r[pid][0] for n, r in (("dense", rd), ("sparse", rs)) if pid in r}
                scores = {n: r[pid][1] for n, r in (("dense", rd), ("sparse", rs)) if pid in r}
                hits.append(self._hit(pts[pid], sc, ranks, scores))
            tm["fuse_ms"] = _ms(t)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        tm["total_ms"] = _ms(T0)
        return hits, tm
