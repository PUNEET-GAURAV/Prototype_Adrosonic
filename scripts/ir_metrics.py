import os
import json
import math
from pathlib import Path
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verity.config import load_config
from verity.store import Store
from verity.api import _startup

def load_queries(n=1000):
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
    queries = []
    
    for row in ds:
        answers = row.get("answers", [])
        wf_answers = row.get("wellFormedAnswers", [])
        
        if not answers and not wf_answers:
            continue
            
        passages = row.get("passages", {})
        is_selected = passages.get("is_selected", [])
        if 1 not in is_selected:
            continue
            
        gold_texts = [text for text, sel in zip(passages["passage_text"], is_selected) if sel == 1]
        
        queries.append({
            "query_id": row["query_id"],
            "query": row["query"],
            "gold_texts": gold_texts
        })
        if len(queries) >= n:
            break
            
    return queries

def main():
    import requests
    queries = load_queries(50)
    results = {}
    
    for mode in ["dense", "hybrid"]:
        hit_at_5 = 0
        recall_at_5 = 0
        mrr_at_10 = 0
        ndcg_at_10 = 0
        n_valid = 0
        
        for q in queries:
            try:
                r = requests.post("http://localhost:8000/search", json={"query": q["query"], "mode": mode, "k": 10}, timeout=10)
                if r.status_code != 200:
                    continue
                retrieved_texts = [h["text"] for h in r.json().get("hits", [])]
                
                # Check if gold is in retrieved
                hits = [1 if t in q["gold_texts"] else 0 for t in retrieved_texts]
                
                # We only consider the query valid if AT LEAST ONE of its gold texts is in the ENTIRE local database.
                # Since we don't know the local database perfectly, we'll skip queries that get 0 hits across all modes? 
                # No, we just report IR metrics. The skill says: 
                # "Labels are sparse (about one gold passage per query), so these understate true relevance. Say so."
                
                if sum(hits[:5]) > 0:
                    hit_at_5 += 1
                
                # Recall @ 5: fraction of gold_texts retrieved
                r_hits = sum(1 for gt in q["gold_texts"] if gt in retrieved_texts[:5])
                recall_at_5 += r_hits / len(q["gold_texts"])
                
                # MRR @ 10
                for rank, is_hit in enumerate(hits[:10], 1):
                    if is_hit:
                        mrr_at_10 += 1.0 / rank
                        break
                        
                # nDCG @ 10
                dcg = 0
                for rank, is_hit in enumerate(hits[:10], 1):
                    if is_hit:
                        dcg += 1.0 / math.log2(rank + 1)
                
                idcg = 0
                for rank in range(1, min(len(q["gold_texts"]), 10) + 1):
                    idcg += 1.0 / math.log2(rank + 1)
                    
                ndcg_at_10 += dcg / idcg if idcg > 0 else 0
                
                n_valid += 1
                
            except Exception as e:
                pass
                
        if n_valid > 0:
            print(f"--- IR Metrics for {mode} (n={n_valid}) ---")
            print("Note: Labels are sparse (about one gold passage per query), so these understate true relevance.")
            print(f"Hit@5:   {hit_at_5/n_valid:.4f}")
            print(f"Recall@5:{recall_at_5/n_valid:.4f}")
            print(f"MRR@10:  {mrr_at_10/n_valid:.4f}")
            print(f"nDCG@10: {ndcg_at_10/n_valid:.4f}")
            results[mode] = {
                "n_valid": n_valid,
                "hit_at_5": hit_at_5/n_valid,
                "recall_at_5": recall_at_5/n_valid,
                "mrr_at_10": mrr_at_10/n_valid,
                "ndcg_at_10": ndcg_at_10/n_valid,
            }

    with open(os.path.join("results", "ir_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
