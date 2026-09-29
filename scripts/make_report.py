import os
import json
import glob
import pandas as pd

def make_report():
    results_dir = "results"
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)
    report_path = os.path.join(docs_dir, "REPORT.md")
    
    with open(report_path, "w") as f:
        f.write("# Verity-RAG Benchmarking Report\n\n")
        
        # Hardware
        hw_file = os.path.join(results_dir, "hardware.json")
        if os.path.exists(hw_file):
            with open(hw_file, "r") as hw_f:
                hw = json.load(hw_f)
                f.write("## Hardware\n")
                f.write(f"- CPU: {hw.get('cpu', 'N/A')}\n")
                f.write(f"- Cores: {hw.get('cores', 'N/A')}\n")
                f.write(f"- RAM (GB): {hw.get('ram_gb', 'N/A')}\n\n")
                
        # IR Metrics
        ir_file = os.path.join(results_dir, "ir_metrics.json")
        if os.path.exists(ir_file):
            with open(ir_file, "r") as ir_f:
                ir = json.load(ir_f)
                f.write("## IR Metrics\n")
                for mode, metrics in ir.items():
                    f.write(f"### {mode.capitalize()} Mode\n")
                    f.write(f"- Valid Queries: {metrics.get('n_valid')}\n")
                    f.write(f"- Hit@5: {metrics.get('hit_at_5'):.4f}\n")
                    f.write(f"- Recall@5: {metrics.get('recall_at_5'):.4f}\n")
                    f.write(f"- MRR@10: {metrics.get('mrr_at_10'):.4f}\n")
                    f.write(f"- nDCG@10: {metrics.get('ndcg_at_10'):.4f}\n\n")
                    
        # RAGAS
        ragas_file = os.path.join(results_dir, "ragas_results.json")
        if os.path.exists(ragas_file):
            with open(ragas_file, "r") as r_f:
                ragas = json.load(r_f)
                f.write("## RAGAS Evaluation\n")
                for mode, metrics in ragas.items():
                    f.write(f"### {mode.capitalize()} Mode\n")
                    for k, v in metrics.items():
                        f.write(f"- {k}: {v:.4f}\n")
                f.write("\n")
                
        # Latency Summary from summary.json
        sum_file = os.path.join(results_dir, "summary.json")
        if os.path.exists(sum_file):
            with open(sum_file, "r") as s_f:
                summary = json.load(s_f)
                lat = summary.get("latency", {})
                if lat:
                    f.write("## Latency Benchmark (100 sequential queries)\n")
                    for mode, perf in lat.items():
                        http_perf = perf.get("http_end_to_end", {})
                        if http_perf:
                            f.write(f"### {mode.capitalize()} Mode\n")
                            f.write(f"- Mean: {http_perf.get('mean'):.2f} ms\n")
                            f.write(f"- p50: {http_perf.get('p50'):.2f} ms\n")
                            f.write(f"- p95: {http_perf.get('p95'):.2f} ms\n")
                            f.write(f"- Max: {http_perf.get('max'):.2f} ms\n\n")

    print(f"Report successfully generated at {report_path}")

if __name__ == "__main__":
    make_report()
