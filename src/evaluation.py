"""Quantitative Evaluation Module for BEIR SciFact Benchmarks.

Calculates Recall@K, Precision@K, and MRR@K against ground-truth qrels
to empirically demonstrate the recall advantages of Multi-Query RAG.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.pipeline import MultiQueryRAGSystem
from src.config import EVAL_DIR

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates retrieval quality against ground truth relevance labels (qrels)."""

    def __init__(self, eval_data_dir: Path = EVAL_DIR):
        self.eval_data_dir = Path(eval_data_dir)
        self.queries = {}
        self.qrels = {}
        self._load_eval_data()

    def _load_eval_data(self):
        """Load test queries and qrels from JSON files if available."""
        queries_file = self.eval_data_dir / "queries.json"
        qrels_file = self.eval_data_dir / "qrels.json"

        if queries_file.exists() and qrels_file.exists():
            try:
                with open(queries_file, "r", encoding="utf-8") as f:
                    self.queries = json.load(f)
                with open(qrels_file, "r", encoding="utf-8") as f:
                    self.qrels = json.load(f)
                logger.info(f"Loaded {len(self.queries)} queries and {len(self.qrels)} qrel entries for evaluation.")
            except Exception as e:
                logger.warning(f"Error reading evaluation files ({e}).")

    def has_eval_data(self) -> bool:
        """Check if evaluation queries and qrels are loaded."""
        return bool(self.queries and self.qrels)

    def calculate_metrics_for_query(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """Compute Recall@K, Precision@K, and MRR for a single query."""
        relevant_set = set(str(did) for did in relevant_doc_ids)
        retrieved_list = [str(did) for did in retrieved_doc_ids]

        if not relevant_set:
            return {}

        metrics = {}
        # Calculate Recall@K and Precision@K
        for k in k_values:
            top_k_retrieved = retrieved_list[:k]
            hits = sum(1 for did in top_k_retrieved if did in relevant_set)
            metrics[f"recall@{k}"] = round(hits / len(relevant_set), 4)
            metrics[f"precision@{k}"] = round(hits / k, 4)

        # Calculate MRR (Mean Reciprocal Rank)
        rr = 0.0
        for rank, did in enumerate(retrieved_list, start=1):
            if did in relevant_set:
                rr = 1.0 / rank
                break
        metrics["mrr"] = round(rr, 4)

        return metrics

    def run_benchmark(
        self,
        rag_system: MultiQueryRAGSystem,
        sample_size: int = 15,
        k_values: List[int] = [3, 5, 10],
        num_generated_queries: int = 4
    ) -> Dict[str, Any]:
        """Run a side-by-side benchmark comparing Baseline vs Multi-Query RAG."""
        if not self.has_eval_data():
            return {"error": "Evaluation dataset not loaded. Ensure data/scifact/scifact_eval/ has queries.json and qrels.json."}

        # Filter queries that have at least one relevant document in qrels
        eval_query_ids = [qid for qid in self.queries if qid in self.qrels and any(rel > 0 for rel in self.qrels[qid].values())]
        test_qids = eval_query_ids[:sample_size]

        baseline_results = []
        multiquery_results = []
        redundancy_rates = []

        logger.info(f"Running benchmark across {len(test_qids)} evaluation queries...")

        for qid in test_qids:
            query_text = self.queries[qid]
            relevant_docs = [did for did, score in self.qrels[qid].items() if score > 0]

            # 1. Baseline Single-Query
            base_run = rag_system.run_baseline(query_text, top_k=max(k_values), generate_answer=False)
            base_doc_ids = [d["id"] for d in base_run["retrieved_documents"]]
            base_metrics = self.calculate_metrics_for_query(base_doc_ids, relevant_docs, k_values)
            baseline_results.append(base_metrics)

            # 2. Multi-Query RAG
            mq_run = rag_system.run_multi_query(
                query_text,
                num_queries=num_generated_queries,
                top_k_per_query=max(k_values),
                final_top_k=max(k_values),
                fusion_strategy="rrf",
                generate_answer=False
            )
            mq_doc_ids = [d["id"] for d in mq_run["fused_contexts"]]
            mq_metrics = self.calculate_metrics_for_query(mq_doc_ids, relevant_docs, k_values)
            multiquery_results.append(mq_metrics)

            red_rate = mq_run["redundancy_metrics"].get("redundancy_rate_pct", 0.0)
            redundancy_rates.append(red_rate)

        # Compute aggregate averages
        def aggregate(metric_list):
            if not metric_list:
                return {}
            keys = metric_list[0].keys()
            return {k: round(sum(m[k] for m in metric_list) / len(metric_list), 4) for k in keys}

        avg_baseline = aggregate(baseline_results)
        avg_multiquery = aggregate(multiquery_results)

        # Calculate improvement percentages
        improvements = {}
        for k in avg_baseline:
            base_val = avg_baseline[k]
            mq_val = avg_multiquery[k]
            delta = mq_val - base_val
            pct = round((delta / base_val * 100.0), 1) if base_val > 0 else 0.0
            improvements[k] = {"delta": round(delta, 4), "pct_change": pct}

        avg_redundancy = round(sum(redundancy_rates) / len(redundancy_rates), 2) if redundancy_rates else 0.0

        return {
            "num_queries_evaluated": len(test_qids),
            "baseline_averages": avg_baseline,
            "multiquery_averages": avg_multiquery,
            "improvements": improvements,
            "average_redundancy_rate_pct": avg_redundancy,
            "k_values": k_values
        }
