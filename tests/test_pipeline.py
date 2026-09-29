"""End-to-end on an in-memory Qdrant: filters are pre-retrieval, live upsert/delete work, hybrid returns per-leg ranks."""
import pytest
from verity.config import load_config
from verity.encoders import LSAEncoder
from verity.retrieval import Pipeline
from verity.store import Store, build_filter
from verity.text import SparseEncoder

CORPUS = []
for i in range(40):
    cat = "space" if i % 2 == 0 else "cooking"
    base = ("rocket orbit telescope planet launch satellite mission crater galaxy" if cat == "space"
            else "recipe oven flour butter sugar simmer sauce garlic pasta bake")
    CORPUS.append({"passage_id": f"T{i:03d}", "category": cat, "source": f"src{i % 3}",
                   "text": f"{base} item{i} " + " ".join(base.split()[i % 5:] + base.split()[:i % 5])})


@pytest.fixture(scope="module")
def pipe():
    cfg = load_config()
    texts = [r["text"] for r in CORPUS]
    dense = LSAEncoder(dim=6, min_df=1, max_df=1.0).fit(texts)
    sparse = SparseEncoder(avg_len=SparseEncoder.measure_avg_len(texts))
    store = Store(location=":memory:", collection="t")
    store.create(dense.dim, recreate=True)
    vecs = dense.encode_docs(texts)
    items = []
    for r, v in zip(CORPUS, vecs):
        si, sv = sparse.doc(r["text"])
        items.append({"dense": v.tolist(), "sp_idx": si, "sp_val": sv,
                      "payload": {**r, "origin": "test", "ingested_at": 1}})
    store.upsert(items)
    return Pipeline(store, dense, sparse, cfg), dense, sparse, store


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_modes_return_k(pipe, mode):
    p = pipe[0]
    hits, tm = p.search("rocket telescope orbit", mode, k=5)
    assert len(hits) == 5 and tm["total_ms"] > 0


def test_hybrid_has_per_leg_ranks(pipe):
    hits, _ = pipe[0].search("rocket orbit", "hybrid", k=5)
    assert all(h.ranks for h in hits)


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_filter_is_applied_in_database_and_k_is_still_returned(pipe, mode):
    hits, _ = pipe[0].search("rocket orbit oven", mode, k=5, flt=build_filter(categories=["cooking"]))
    assert len(hits) == 5 and all(h.category == "cooking" for h in hits)


def test_live_upsert_then_delete(pipe):
    p, dense, sparse, store = pipe
    text = "zzyzx quasar unique-token passage"
    si, sv = sparse.doc(text)
    store.upsert([{"dense": dense.encode_docs([text])[0].tolist(), "sp_idx": si, "sp_val": sv,
                   "payload": {"passage_id": "LIVE-1", "text": text, "category": "live", "source": "x",
                               "origin": "live_upsert", "ingested_at": 2}}])
    assert p.search("zzyzx", "sparse", k=1)[0][0].passage_id == "LIVE-1"
    store.delete(["LIVE-1"])
    assert all(h.passage_id != "LIVE-1" for h in p.search("zzyzx", "hybrid", k=5)[0])
