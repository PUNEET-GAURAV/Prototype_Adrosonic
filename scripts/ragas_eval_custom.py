import os
import json
import httpx
import asyncio
from datasets import load_dataset
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

def build_eval_set(n=30):
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
    queries = []
    
    for row in ds:
        answers = row.get("answers", [])
        wf_answers = row.get("wellFormedAnswers", [])
        
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
            "reference": ans
        })
        if len(queries) >= n:
            break
            
    return queries

def evaluate_context_precision(llm, query, retrieved_contexts, reference):
    # Simplified LLMContextPrecisionWithReference prompt
    prompt = f"""
    Given the following question, evaluate the provided contexts to determine if they contain the answer.
    Question: {query}
    Reference Answer: {reference}
    
    Contexts:
    """
    for i, ctx in enumerate(retrieved_contexts):
        prompt += f"[{i+1}] {ctx}\n"
        
    prompt += "\nOutput ONLY valid JSON like: {\"relevant_context_indices\": [1, 3]} containing 1-indexed integers of useful contexts."
    
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content
        start = res.find("{")
        end = res.rfind("}") + 1
        data = json.loads(res[start:end])
        relevant = data.get("relevant_context_indices", [])
        
        # Calculate precision at K (MRR style or MAP style)
        # Ragas uses: sum(precision@k) / total_relevant
        # We will use simple precision metric
        score = 0
        for i, idx in enumerate(relevant):
            score += (i + 1) / idx
        return score / len(relevant) if relevant else 0.0
    except Exception as e:
        print(f"Error in evaluate_context_precision: {e}")
        return None

def evaluate_context_recall(llm, query, retrieved_contexts, reference):
    # Simplified LLMContextRecall prompt
    prompt = f"""
    Evaluate if the reference answer is fully covered by the retrieved contexts.
    Question: {query}
    Reference Answer: {reference}
    
    Contexts:
    """
    for i, ctx in enumerate(retrieved_contexts):
        prompt += f"[{i+1}] {ctx}\n"
        
    prompt += "\nDoes the context contain all information from the reference answer? Output ONLY valid JSON: {\"recall_score\": 0.8} with a score from 0.0 to 1.0."
    
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content
        start = res.find("{")
        end = res.rfind("}") + 1
        data = json.loads(res[start:end])
        return float(data.get("recall_score", 0.0))
    except Exception as e:
        print(f"Error in evaluate_context_recall: {e}")
        return None

def run_evaluation(mode="dense"):
    os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    
    queries = build_eval_set(5)
    client = httpx.Client(base_url="http://localhost:8000", timeout=60)
    
    precision_scores = []
    recall_scores = []
    nans = 0
    
    for q in queries:
        try:
            r = client.post("/search", json={"query": q["query"], "mode": mode, "k": 5}).json()
            retrieved = [h["text"] for h in r.get("hits", [])]
            
            p_score = evaluate_context_precision(llm, q["query"], retrieved, q["reference"])
            r_score = evaluate_context_recall(llm, q["query"], retrieved, q["reference"])
            
            if p_score is None or r_score is None:
                nans += 1
                continue
                
            precision_scores.append(p_score)
            recall_scores.append(r_score)
            
        except Exception as e:
            print(f"Error in run_evaluation: {e}")
            nans += 1
            
    import numpy as np
    print(f"--- Custom Ragas Evaluation for {mode} ---")
    print(f"Valid queries: {len(precision_scores)}")
    print(f"NaN rate: {nans / len(queries):.2%}")
    if precision_scores:
        print(f"Custom_LLMContextPrecision: {np.mean(precision_scores):.4f}")
        print(f"Custom_LLMContextRecall: {np.mean(recall_scores):.4f}")

if __name__ == "__main__":
    print("Evaluating Dense Retrieval...")
    run_evaluation("dense")
    print("\nEvaluating Hybrid Retrieval...")
    run_evaluation("hybrid")
