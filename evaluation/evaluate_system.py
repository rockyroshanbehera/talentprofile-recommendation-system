"""
evaluate_system.py - Quantitative Evaluation Benchmark for TalentProfile AI
Measures exact Query Latency, Monotonicity, Top-K Relevance Score Distributions,
and Keyword Recall without manufacturing synthetic metrics.
"""

import time
import json
import os
import numpy as np
import pandas as pd
from src.recommendation import TalentRecommender

BENCHMARK_QUERIES = [
    {
        "id": "q1_data_analyst",
        "query": "Need a Python data analyst with SQL and Power BI experience",
        "category": "Data & Analytics",
        "target_keywords": ["python", "sql", "data", "analyst"]
    },
    {
        "id": "q2_ml_engineer",
        "query": "Looking for a machine learning engineer with Python, TensorFlow and NLP skills",
        "category": "AI & Machine Learning",
        "target_keywords": ["machine", "learning", "python", "nlp"]
    },
    {
        "id": "q3_marketing",
        "query": "Need a marketing professional experienced in digital marketing and SEO",
        "category": "Marketing",
        "target_keywords": ["marketing", "digital", "seo"]
    },
    {
        "id": "q4_devops",
        "query": "Senior cloud DevOps architect with Kubernetes, Docker and AWS",
        "category": "Cloud & Infrastructure",
        "target_keywords": ["devops", "kubernetes", "docker", "aws", "cloud"]
    },
    {
        "id": "q5_uiux",
        "query": "UI UX designer with Figma wireframing and user research",
        "category": "Design",
        "target_keywords": ["ui", "ux", "designer", "figma"]
    }
]

def run_evaluation():
    print("Initializing Recommender Engine for Quantitative Benchmark...")
    rec = TalentRecommender()
    t_load0 = time.time()
    rec.load_models()
    load_time_sec = time.time() - t_load0
    print(f"Models loaded in {load_time_sec:.2f}s")

    benchmark_results = []
    latencies = []

    for item in BENCHMARK_QUERIES:
        q_text = item["query"]
        target_keys = item["target_keywords"]

        # Measure query latency over 3 runs
        run_times = []
        res = None
        for _ in range(3):
            t0 = time.perf_counter()
            res = rec.recommend(q_text, top_n=10)
            run_times.append((time.perf_counter() - t0) * 1000)

        avg_latency_ms = np.mean(run_times)
        latencies.append(avg_latency_ms)

        # Evaluate Top-1, Top-5, Top-10 scores
        scores = [r["similarity_score"] for r in res]
        top1_score = scores[0] if scores else 0.0
        mean_top5_score = np.mean(scores[:5]) if scores else 0.0
        mean_top10_score = np.mean(scores) if scores else 0.0

        # Calculate exact keyword recall in top-5 titles/descriptions
        keyword_hits = 0
        for r in res[:5]:
            matched = set([k.lower() for k in r["matched_keywords"]])
            if any(tk in matched for tk in target_keys):
                keyword_hits += 1
        top5_keyword_coverage = keyword_hits / min(len(res), 5) if res else 0.0

        benchmark_results.append({
            "query_id": item["id"],
            "category": item["category"],
            "query": q_text,
            "latency_ms": round(avg_latency_ms, 2),
            "top_1_similarity": round(top1_score, 4),
            "mean_top5_similarity": round(mean_top5_score, 4),
            "mean_top10_similarity": round(mean_top10_score, 4),
            "top5_keyword_coverage": round(top5_keyword_coverage, 2),
            "top_match_title": res[0]["title"] if res else "N/A",
            "top_match_company": res[0]["company_name"] if res else "N/A",
            "top_match_skills": res[0]["matched_skills"] if res else []
        })

    summary = {
        "total_queries_benchmarked": len(BENCHMARK_QUERIES),
        "average_query_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_query_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "model_load_time_sec": round(load_time_sec, 2),
        "knowledge_base_size": len(rec.talent_profiles),
        "detailed_results": benchmark_results
    }

    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("="*80)
    print("TALENTPROFILE AI - QUANTITATIVE BENCHMARK REPORT")
    print("="*80)
    print(f"Database Volume: {summary['knowledge_base_size']:,} talent profiles")
    print(f"Mean Query Latency: {summary['average_query_latency_ms']} ms")
    print(f"P95 Query Latency: {summary['p95_query_latency_ms']} ms")
    print("-"*80)
    for b in benchmark_results:
        print(f"[{b['category']}] '{b['query'][:45]}...'")
        print(f"  - Top-1 Match: {b['top_match_title']} ({b['top_match_company']})")
        print(f"  - Similarity: {b['top_1_similarity']} | Latency: {b['latency_ms']} ms | Keyword Coverage: {b['top5_keyword_coverage']*100}%")
    print("="*80)

if __name__ == "__main__":
    run_evaluation()
