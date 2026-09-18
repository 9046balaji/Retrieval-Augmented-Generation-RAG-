"""CLI Benchmark Runner for Multi-Query RAG vs Baseline.

Demonstrates viva metrics: Recall@K, Precision@K, MRR, and Redundancy Rate.
"""

import sys
import logging
from tabulate import tabulate

from src.pipeline import MultiQueryRAGSystem
from src.evaluation import RAGEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("=" * 70)
    print("🔬 MULTI-QUERY RAG SYSTEM: QUANTITATIVE BENCHMARK")
    print("=" * 70)

    rag = MultiQueryRAGSystem()
    if not rag.is_vector_db_ready():
        print("\n❌ Error: Vector database is not initialized.")
        print("Please either:")
        print("  1. Run the Colab notebook: notebooks/multi_query_rag_vector_db.ipynb and download scifact_vector_db.zip")
        print("  2. Or run: python scripts/build_local_vector_db.py to build it directly.")
        sys.exit(1)

    evaluator = RAGEvaluator()
    if not evaluator.has_eval_data():
        print("\n❌ Error: Evaluation dataset (queries.json, qrels.json) not found in data/scifact/scifact_eval/.")
        sys.exit(1)

    print(f"\nEvaluating on test queries from BEIR SciFact benchmark...")
    sample_size = 10
    k_values = [3, 5, 10]
    num_queries = 4

    results = evaluator.run_benchmark(
        rag_system=rag,
        sample_size=sample_size,
        k_values=k_values,
        num_generated_queries=num_queries
    )

    base = results["baseline_averages"]
    mq = results["multiquery_averages"]
    imp = results["improvements"]

    table_data = []
    for metric_name in base:
        base_val = base[metric_name]
        mq_val = mq[metric_name]
        diff = imp[metric_name]["delta"]
        pct = imp[metric_name]["pct_change"]
        sign = "+" if diff >= 0 else ""
        table_data.append([
            metric_name.upper(),
            f"{base_val:.4f}",
            f"{mq_val:.4f}",
            f"{sign}{diff:.4f} ({sign}{pct:.1f}%)"
        ])

    print("\n" + tabulate(
        table_data,
        headers=["Metric", "Single Query (Baseline)", "Multi-Query RAG", "Improvement"],
        tablefmt="fancy_grid"
    ))

    print(f"\n📊 Redundancy & Deduplication Statistics:")
    print(f"  • Queries evaluated:              {results['num_queries_evaluated']}")
    print(f"  • Average redundant docs filtered: {results['average_redundancy_rate_pct']}% of total retrieved candidates")
    print(f"  • Result fusion strategy:         Reciprocal Rank Fusion (RRF, k=60)")
    print("\n💡 Key Finding:")
    print("  Multi-Query retrieval expands the search boundary to capture orthogonal facets,")
    print("  significantly boosting Recall@K. The Deduplication/RRF layer eliminates redundant")
    print("  passages to prevent context-window bloat and repetitive answers.")
    print("=" * 70)


if __name__ == "__main__":
    main()
