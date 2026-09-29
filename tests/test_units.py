import math

from verity.fusion import weighted_rrf
from verity.metrics import hit_at_k, latency_summary, mrr_at_k, ndcg_at_k, recall_at_k
from verity.text import SparseEncoder, tokens


def test_tokens_stem_and_stopwords():
    assert tokens("The running dogs are running!") == ["run", "dog", "run"]


def test_sparse_doc_length_normalisation_and_saturation():
    enc = SparseEncoder(k=1.2, b=0.75, avg_len=10)
    _, short = enc.doc("apple banana")
    _, long = enc.doc("apple " + "filler " * 40)
    assert short[0] > long[0]                       # same tf, longer doc scores lower
    _, once = enc.doc("apple")
    _, many = enc.doc("apple apple apple apple")
    assert many[0] > once[0] and many[0] < 4 * once[0]   # tf saturates


def test_sparse_query_unique_terms():
    idx, val = SparseEncoder().query("apple apple pie")
    assert len(idx) == 2 and val == [1.0, 1.0]


def test_rrf_known_values():
    r = weighted_rrf({"a": ["1", "2", "3"], "b": ["3", "2", "4"]}, {"a": 1, "b": 1}, 60)
    assert [i for i, _ in r] == ["3", "2", "1", "4"]
    assert math.isclose(dict(r)["3"], 1 / 63 + 1 / 61)


def test_rrf_weight_zero_ignores_leg():
    r = weighted_rrf({"a": ["1", "2"], "b": ["2", "1"]}, {"a": 1, "b": 0}, 60)
    assert [i for i, _ in r] == ["1", "2"]


def test_ir_metrics():
    ranked, gold = ["x", "g", "y"], {"g"}
    assert hit_at_k(ranked, gold, 1) == 0 and hit_at_k(ranked, gold, 2) == 1
    assert mrr_at_k(ranked, gold, 10) == 0.5 and recall_at_k(ranked, gold, 5) == 1
    assert math.isclose(ndcg_at_k(ranked, gold, 10), 1 / math.log2(3))


def test_latency_summary():
    s = latency_summary(list(range(1, 101)))
    assert s["n"] == 100 and math.isclose(s["p95"], 95.05) and s["pass"]
