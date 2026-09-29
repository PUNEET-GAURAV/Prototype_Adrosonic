import os
import json
import time
import httpx
from pathlib import Path
from datasets import load_dataset
from langchain_groq import ChatGroq
from langchain_core.language_models.llms import BaseLLM
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper

ROOT = Path(__file__).resolve().parents[1]

def build_eval_set(n=30):
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
    queries = []
    
    # Ragas test query conditions:
    # 1. Answer is not empty and not "No Answer Present."
    # 2. At least one passage has is_selected == 1
    # 3. Reference answer exists
    for row in ds:
        answers = row.get("answers", [])
        wf_answers = row.get("wellFormedAnswers", [])
        
        # Check answer validity
        if not answers and not wf_answers:
            continue
        
        ans = wf_answers[0] if (wf_answers and wf_answers[0] and wf_answers[0] != "[]") else answers[0] if answers else None
        if not ans or ans == "No Answer Present.":
            continue
            
        passages = row.get("passages", {})
        is_selected = passages.get("is_selected", [])
        if 1 not in is_selected:
            continue
            
        queries.append({
            "query_id": row["query_id"],
            "query": row["query"],
            "query_type": row["query_type"],
            "reference": ans
        })
        if len(queries) >= n:
            break
            
    return queries

def run_evaluation(mode="dense"):
    # Load Groq model
    os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    ragas_llm = LangchainLLMWrapper(llm)
    
    queries = build_eval_set(30)
    
    precision_metric = LLMContextPrecisionWithReference()
    precision_metric.llm = ragas_llm
    
    recall_metric = LLMContextRecall()
    recall_metric.llm = ragas_llm
    
    client = httpx.Client(base_url="http://localhost:8000", timeout=60)
    
    precision_scores = []
    recall_scores = []
    nans = 0
    
    for q in queries:
        try:
            r = client.post("/search", json={"query": q["query"], "mode": mode, "k": 5}).json()
            retrieved = [h["text"] for h in r.get("hits", [])]
            
            sample = SingleTurnSample(
                user_input=q["query"],
                retrieved_contexts=retrieved,
                reference=q["reference"]
            )
            
            p_score = precision_metric.single_turn_ascore(sample)
            r_score = recall_metric.single_turn_ascore(sample)
            
            import asyncio
            p_score = asyncio.run(p_score)
            r_score = asyncio.run(r_score)
            
            if p_score is None or r_score is None:
                nans += 1
                continue
                
            precision_scores.append(p_score)
            recall_scores.append(r_score)
            
        except Exception as e:
            print(f"Error on query {q['query_id']}: {e}")
            nans += 1
            
    import numpy as np
    print(f"--- Evaluation for {mode} ---")
    print(f"Valid queries: {len(precision_scores)}")
    print(f"NaN rate: {nans / len(queries):.2%}")
    if precision_scores:
        print(f"LLMContextPrecisionWithReference: {np.mean(precision_scores):.4f}")
        print(f"LLMContextRecall: {np.mean(recall_scores):.4f}")

if __name__ == "__main__":
    print("Evaluating Dense Retrieval...")
    run_evaluation("dense")
    print("\nEvaluating Hybrid Retrieval...")
    run_evaluation("hybrid")
