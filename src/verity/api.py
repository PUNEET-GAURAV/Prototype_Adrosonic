"""FastAPI service: /search /compare /upsert /passages/{id} /stats /bench, plus the web UI at /."""
from __future__ import annotations
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import ROOT, load_config
from .encoders import make_encoder
from .metrics import latency_summary
from .retrieval import Pipeline
from .store import Store, build_filter
from .text import SparseEncoder

app = FastAPI(title="Verity-RAG prototype")
S: dict = {}


@app.on_event("startup")
def _startup() -> None:
    cfg = load_config()
    meta = json.loads((ROOT / "data" / "index_meta.json").read_text())
    store = Store(
        url=cfg["qdrant"].get("url"), 
        collection=cfg["qdrant"].get("collection", "verity"), 
        prefer_grpc=cfg["qdrant"].get("prefer_grpc", False), 
        grpc_port=cfg["qdrant"].get("grpc_port", 6334),
        path=cfg["qdrant"].get("path")
    )
    dense = make_encoder(cfg, ROOT / "data" / "lsa.pkl")
    sparse = SparseEncoder(meta["k"], meta["b"], meta["avg_len"])
    S.update(cfg=cfg, meta=meta, store=store, dense=dense, sparse=sparse, pipe=Pipeline(store, dense, sparse, cfg))
    S["pipe"].search("warm up", "hybrid")  # load models / open connections before the first real query


class SearchReq(BaseModel):
    query: str
    mode: str = "hybrid"
    k: int = 5
    categories: list[str] | None = None
    source: str | None = None
    origin: str | None = None
    date_from: int | None = None
    date_to: int | None = None
    w_dense: float | None = None
    w_sparse: float | None = None
    rrf_k: int | None = None
    depth: int | None = None


class UpsertReq(BaseModel):
    text: str
    category: str = "live"
    source: str = "live-demo"
    passage_id: str | None = None


def _run(r: SearchReq, mode: str | None = None):
    flt = build_filter(r.categories, r.source, r.origin, r.date_from, r.date_to)
    return S["pipe"].search(r.query, mode or r.mode, r.k, flt, r.depth, r.w_dense, r.w_sparse, r.rrf_k)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/stats")
def stats():
    return {**S["store"].info(), "encoder": S["dense"].name, "dim": S["dense"].dim, "avg_len": S["meta"]["avg_len"],
            "dataset": S["meta"]["dataset"], "retrieval": S["cfg"]["retrieval"]}


@app.post("/search")
def search(r: SearchReq):
    if r.mode not in ("dense", "sparse", "hybrid"):
        raise HTTPException(400, "mode must be dense | sparse | hybrid")
    hits, tm = _run(r)
    return {"mode": r.mode, "timings": tm, "hits": [h.to_dict() for h in hits]}


@app.post("/compare")
def compare(r: SearchReq):
    dh, dt = _run(r, "dense")
    hh, ht = _run(r, "hybrid")
    drank = {h.id: i for i, h in enumerate(dh)}
    out = []
    for i, h in enumerate(hh):
        d = h.to_dict()
        d["move"] = "new" if h.id not in drank else ("up" if drank[h.id] > i else "down" if drank[h.id] < i else "same")
        out.append(d)
    return {"dense": {"timings": dt, "hits": [h.to_dict() for h in dh]}, "hybrid": {"timings": ht, "hits": out}}


@app.post("/upsert")
def upsert(u: UpsertReq):
    pid = u.passage_id or f"LIVE-{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter_ns()
    si, sv = S["sparse"].doc(u.text)
    S["store"].upsert([{"dense": S["dense"].encode_docs([u.text])[0].tolist(), "sp_idx": si, "sp_val": sv,
                        "payload": {"passage_id": pid, "text": u.text, "category": u.category, "source": u.source,
                                    "origin": "live_upsert", "ingested_at": int(time.time())}}], wait=True)
    ms = (time.perf_counter_ns() - t0) / 1e6
    hits, _ = S["pipe"].search(u.text, "hybrid", 5)
    rank = next((i + 1 for i, h in enumerate(hits) if h.passage_id == pid), None)
    return {"passage_id": pid, "upsert_ms": ms, "rank_when_queried_by_own_text": rank, "points": S["store"].count()}


@app.delete("/passages/{passage_id}")
def delete(passage_id: str):
    S["store"].delete([passage_id])
    return {"deleted": passage_id, "points": S["store"].count()}


@app.post("/bench")
def bench(mode: str = "hybrid", n: int = 100, warmup: int = 20):
    qs = json.loads((ROOT / "data" / "bench_queries.json").read_text())
    for q in qs[:warmup]:
        S["pipe"].search(q, mode)
    ms = []
    for q in qs[warmup:warmup + n]:
        _, tm = S["pipe"].search(q, mode)
        ms.append(tm["total_ms"])
    return {"mode": mode, "points": S["store"].count(), "summary": latency_summary(ms), "latencies_ms": ms}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "web" / "index.html")
