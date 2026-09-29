"""Build the corpus, fit/load the encoder, and index everything into Qdrant. Timed and logged.

    python scripts/build_index.py --dataset demo --n 100000 --recreate
    python scripts/build_index.py --dataset msmarco --encoder bge --n 100000 --recreate   (needs HF access)
"""
from __future__ import annotations
import argparse, json, platform, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verity.config import ROOT, load_config
from verity.datasets import build_demo_corpus, iter_msmarco
from verity.encoders import BGEEncoder, LSAEncoder
from verity.store import Store
from verity.text import SparseEncoder, tokens

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="demo", choices=["demo", "msmarco"])
ap.add_argument("--encoder", default=None, choices=["lsa", "bge"])
ap.add_argument("--n", type=int, default=100_000)
ap.add_argument("--recreate", action="store_true")
ap.add_argument("--batch", type=int, default=500)
a = ap.parse_args()

cfg = load_config()
if a.encoder:
    cfg["encoder"]["dense"] = a.encoder
elif a.dataset == "demo":
    cfg["encoder"]["dense"] = "lsa"
timings, t_all = {}, time.time()


def lap(name, t0):
    timings[name] = round(time.time() - t0, 2)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {timings[name]}s", flush=True)


t = time.time()
rows = build_demo_corpus(a.n) if a.dataset == "demo" else iter_msmarco(a.n)
lap("corpus", t)
texts = [r["text"] for r in rows]
print(f"passages: {len(rows):,}", flush=True)

t = time.time()
avg_len = SparseEncoder.measure_avg_len(texts)
sparse = SparseEncoder(cfg["sparse"]["k"], cfg["sparse"]["b"], avg_len)
lap("bm25_avg_len", t)
print(f"measured BM25 avg_len = {avg_len:.1f} stemmed tokens (library default would be 256)", flush=True)

data = ROOT / "data"
data.mkdir(exist_ok=True)
t = time.time()
if cfg["encoder"]["dense"] == "bge":
    enc = BGEEncoder(cfg["encoder"]["bge_model"])
else:
    enc = LSAEncoder(cfg["encoder"]["lsa_dim"]).fit(texts)
    enc.save(data / "lsa.pkl")
lap("encoder_fit_or_load", t)

store = Store(
    url=cfg["qdrant"].get("url"), 
    collection=cfg["qdrant"].get("collection", "verity"), 
    prefer_grpc=cfg["qdrant"].get("prefer_grpc", False), 
    grpc_port=cfg["qdrant"].get("grpc_port", 6334),
    path=cfg["qdrant"].get("path")
)
store.create(enc.dim, recreate=a.recreate)

t0 = time.time(); now = int(time.time())
t_enc = t_up = 0.0
for s in range(0, len(rows), a.batch):
    chunk = rows[s:s + a.batch]
    t = time.time(); dense = enc.encode_docs([r["text"] for r in chunk]); t_enc += time.time() - t
    items = []
    for r, dv in zip(chunk, dense):
        si, sv = sparse.doc(r["text"])
        items.append({"dense": dv.tolist(), "sp_idx": si, "sp_val": sv, "payload": {
            "passage_id": r["passage_id"], "text": r["text"], "category": r["category"],
            "source": r["source"], "origin": a.dataset, "ingested_at": now}})
    t = time.time(); store.upsert(items, wait=True); t_up += time.time() - t
    if (s // a.batch) % 20 == 0:
        print(f"  {s + len(chunk):,}/{len(rows):,}", flush=True)
timings.update(encode_docs=round(t_enc, 2), upsert=round(t_up, 2))
t = time.time(); status = store.wait_green(); lap("wait_index_green", t)
lap("total_ingest", t0)

info = store.info()
meta = {"dataset": a.dataset, "n": len(rows), "encoder": enc.name, "dim": enc.dim, "avg_len": avg_len,
        "k": cfg["sparse"]["k"], "b": cfg["sparse"]["b"], "collection": cfg["qdrant"]["collection"],
        "timings_s": timings, "qdrant": info, "built_at": int(time.time())}
(data / "index_meta.json").write_text(json.dumps(meta, indent=2))

# Benchmark queries (latency only, NOT a quality set): 3-5 content words sampled from random passages.
rng = random.Random(7)
qs = []
while len(qs) < 120:
    w = [x for x in tokens(rng.choice(rows)["text"]) if len(x) > 3]
    raw = rng.choice(rows)["text"].lower().split()
    words = [x.strip(".,;:!?\"'()") for x in raw if len(x) > 4 and x.isalpha()]
    if len(words) >= 5:
        qs.append(" ".join(rng.sample(words, rng.randint(3, 5))))
(data / "bench_queries.json").write_text(json.dumps(qs, indent=1))

hw = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}
try:
    hw["cpu"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    hw["cores"] = sum(1 for l in open("/proc/cpuinfo") if l.startswith("processor"))
    hw["ram_gb"] = round(int(open("/proc/meminfo").readline().split()[1]) / 1e6, 1)
except Exception:
    pass
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "hardware.json").write_text(json.dumps(hw, indent=2))
(ROOT / "results" / "index_build.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
