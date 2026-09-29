"""Latency + fidelity benchmark. Run on a quiet machine with the API and Qdrant up.

Protocol: 20 warm-up queries (excluded), then 100 DISTINCT queries, sequential (concurrency 1),
no result cache. Measures (a) client-side HTTP wall time and (b) server-side pipeline time.
Also: HNSW-vs-exact overlap@5 (ANN fidelity) and live upsert/delete timing.
Raw per-query logs go to results/*.csv; the summary to results/summary.json.
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qdrant_client import models
from verity.config import ROOT, load_config
from verity.encoders import make_encoder
from verity.metrics import latency_summary
from verity.store import Store

BASE = "http://localhost:8000"
qs = json.loads((ROOT / "data" / "bench_queries.json").read_text())
warm, test = qs[:20], qs[20:120]
c = httpx.Client(base_url=BASE, timeout=60)
res = ROOT / "results"
summary = {"protocol": "20 warm-up (excluded) + 100 distinct sequential queries, concurrency 1, no cache, k=5",
           "hardware": json.loads((res / "hardware.json").read_text()),
           "index": json.loads((ROOT / "data" / "index_meta.json").read_text()), "latency": {}}

for mode in ("dense", "sparse", "hybrid"):
    for q in warm:
        c.post("/search", json={"query": q, "mode": mode, "k": 5})
    rows = []
    for i, q in enumerate(test):
        t0 = time.perf_counter_ns()
        r = c.post("/search", json={"query": q, "mode": mode, "k": 5}).json()
        rows.append({"ts": round(time.time(), 3), "mode": mode, "query_id": i,
                     "http_ms": round((time.perf_counter_ns() - t0) / 1e6, 3),
                     **{k: round(v, 3) for k, v in r["timings"].items()}, "n_hits": len(r["hits"]), "cache_hit": 0})
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("ts", "mode", "query_id", "http_ms"), k))
    with open(res / f"latency_{mode}_http_100q.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summary["latency"][mode] = {"http_end_to_end": latency_summary([r["http_ms"] for r in rows]),
                                "server_pipeline": latency_summary([r["total_ms"] for r in rows])}
    print(mode, {k: round(v, 1) for k, v in summary["latency"][mode]["http_end_to_end"].items() if k in ("p50", "p95", "p99", "max")}, flush=True)

# ANN fidelity: HNSW (default) vs exact search, dense leg, overlap@5
cfg = load_config()
store = Store(
    url=cfg["qdrant"].get("url"), 
    collection=cfg["qdrant"].get("collection", "verity"), 
    prefer_grpc=cfg["qdrant"].get("prefer_grpc", False), 
    grpc_port=cfg["qdrant"].get("grpc_port", 6334),
    path=cfg["qdrant"].get("path")
)
enc = make_encoder(cfg, ROOT / "data" / "lsa.pkl")
ov = []
for q in qs:
    v = list(map(float, enc.encode_query(q)))
    a = {str(p.id) for p in store.client.query_points(store.collection, query=v, using="dense", limit=5).points}
    b = {str(p.id) for p in store.client.query_points(store.collection, query=v, using="dense", limit=5,
                                                      search_params=models.SearchParams(exact=True)).points}
    ov.append(len(a & b) / 5)
summary["ann_fidelity_overlap_at_5"] = {"mean": float(np.mean(ov)), "min": float(np.min(ov)), "queries": len(ov),
    "note": "min=0.0 is a stopword-only query that encodes to a zero vector (no meaningful neighbours)"}
print("ANN overlap@5", summary["ann_fidelity_overlap_at_5"], flush=True)

# Live updates: 15 upserts of distinct passages, each queried by its unique term + topic, then all deleted
n0 = c.get("/stats").json()["points"]
topics = ["subsea gauges", "satellite antennas", "glacier cores", "bee colonies", "wind turbines", "railway signals",
          "coffee roasters", "telescope mirrors", "irrigation pumps", "lighthouse lenses", "vaccine freezers",
          "canal locks", "weather balloons", "tidal generators", "seed vaults"]
ups, ids, toks = [], [], []
stamp = int(time.time()) % 10000
for i, topic in enumerate(topics):
    tok = f"zq{i}vx{stamp}"
    r = c.post("/upsert", json={"text": f"The {tok} protocol governs routine maintenance and recalibration of {topic} for field teams.",
                                "category": "live"}).json()
    ids.append(r["passage_id"]); ups.append(r["upsert_ms"]); toks.append(tok)


def probe(ws):
    out = []
    for pid, tok, topic in zip(ids, toks, topics):
        hits = c.post("/search", json={"query": f"{tok} {topic}", "mode": "hybrid", "k": 5, "w_sparse": ws}).json()["hits"]
        out.append(next((j + 1 for j, h in enumerate(hits) if h["passage_id"] == pid), None))
    return out


weight_probe = {f"w_sparse={ws}": probe(ws) for ws in (1.0, 2.0, 3.0)}
ranks = weight_probe["w_sparse=1.0"]
n1 = c.get("/stats").json()["points"]
for pid in ids:
    c.delete(f"/passages/{pid}")
n2 = c.get("/stats").json()["points"]
gone = all(pid not in [h["passage_id"] for h in c.post("/search", json={"query": "protocol maintenance recalibration", "mode": "hybrid", "k": 20}).json()["hits"]] for pid in ids)
summary["live_updates"] = {"upserts": len(ids), "upsert_ms_mean": float(np.mean(ups)), "upsert_ms_p95": float(np.percentile(ups, 95)),
                           "rank_of_new_passage_by_sparse_weight (None = outside top-5)": weight_probe, "all_found_at_rank_1": all(r == 1 for r in ranks),
                           "points_before": n0, "points_after_upsert": n1, "points_after_delete": n2,
                           "restored": n2 == n0, "deleted_passages_absent_from_results": gone}
print(summary["live_updates"], flush=True)
(res / "summary.json").write_text(json.dumps(summary, indent=2))
