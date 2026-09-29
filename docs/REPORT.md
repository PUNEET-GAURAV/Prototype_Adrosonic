# Verity-RAG Benchmarking Report

## Hardware
- CPU: N/A
- Cores: N/A
- RAM (GB): N/A

## IR Metrics
### Dense Mode
- Valid Queries: 50
- Hit@5: 0.4000
- Recall@5: 0.3800
- MRR@10: 0.3500
- nDCG@10: 0.3600

### Hybrid Mode
- Valid Queries: 50
- Hit@5: 0.4500
- Recall@5: 0.4200
- MRR@10: 0.4000
- nDCG@10: 0.4100

## RAGAS Evaluation
### Dense Mode
- Custom_LLMContextPrecision: 0.2000
- Custom_LLMContextRecall: 0.2200
### Hybrid Mode
- Custom_LLMContextPrecision: 0.1000
- Custom_LLMContextRecall: 0.2000

## Latency Benchmark (100 sequential queries)
### Dense Mode
- Mean: 22.40 ms
- p50: 22.26 ms
- p95: 27.45 ms
- Max: 33.40 ms

### Sparse Mode
- Mean: 2.98 ms
- p50: 2.68 ms
- p95: 4.18 ms
- Max: 4.82 ms

### Hybrid Mode
- Mean: 30.43 ms
- p50: 28.68 ms
- p95: 44.36 ms
- Max: 45.51 ms

