"""End-to-End Multi-Query RAG Pipeline.

Coordinates the complete workflow:
1. Query Transformation (LLM)
2. Multi-Query Vector Retrieval (ChromaDB)
3. Result Fusion & Deduplication (RRF)
4. Redundancy & Overlap Tracking
5. Grounded Answer Synthesis (MedGemma)
"""

import time
import logging
from typing import Dict, Any, Optional

from src.query_generator import MultiQueryGenerator
from src.retriever import DenseRetriever
from src.deduplication import ResultFusionDeduplicator
from src.generator import GroundedAnswerGenerator
from src.judge import LLMJudge
from src.config import (
    DEFAULT_NUM_QUERIES,
    DEFAULT_TOP_K_PER_QUERY,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_RRF_K
)

logger = logging.getLogger(__name__)


class MultiQueryRAGSystem:
    """Complete Multi-Query RAG system supporting baseline and multi-query flows."""

    def __init__(
        self,
        retriever: Optional[DenseRetriever] = None,
        query_generator: Optional[MultiQueryGenerator] = None,
        answer_generator: Optional[GroundedAnswerGenerator] = None,
        deduplicator: Optional[ResultFusionDeduplicator] = None,
        judge: Optional[LLMJudge] = None
    ):
        self.retriever = retriever or DenseRetriever()
        self.query_generator = query_generator or MultiQueryGenerator()
        self.answer_generator = answer_generator or GroundedAnswerGenerator()
        self.deduplicator = deduplicator or ResultFusionDeduplicator(rrf_k=DEFAULT_RRF_K)
        self.judge = judge or LLMJudge()

    def is_vector_db_ready(self) -> bool:
        """Check if vector database is loaded and ready."""
        return self.retriever.is_ready()

    def run_baseline(
        self,
        query: str,
        top_k: int = DEFAULT_FINAL_TOP_K,
        generate_answer: bool = True
    ) -> Dict[str, Any]:
        """Execute standard single-query baseline RAG."""
        start_time = time.time()

        # Step 1: Single query retrieval
        retrieval_start = time.time()
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        retrieval_time = time.time() - retrieval_start

        # Step 2: Generate answer
        answer_dict = {}
        if generate_answer:
            answer_dict = self.answer_generator.generate_answer(query, retrieved_docs)

        total_time = time.time() - start_time

        return {
            "mode": "baseline_single_query",
            "query": query,
            "retrieved_documents": retrieved_docs,
            "document_count": len(retrieved_docs),
            "answer": answer_dict.get("answer", ""),
            "citations": answer_dict.get("citations", []),
            "timings": {
                "retrieval_sec": round(retrieval_time, 3),
                "generation_sec": answer_dict.get("elapsed_sec", 0.0),
                "total_sec": round(total_time, 3)
            }
        }

    def run_multi_query(
        self,
        user_query: str,
        num_queries: int = DEFAULT_NUM_QUERIES,
        top_k_per_query: int = DEFAULT_TOP_K_PER_QUERY,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        fusion_strategy: str = "rrf",
        generate_answer: bool = True
    ) -> Dict[str, Any]:
        """Execute the proposed Multi-Query RAG pipeline.

        Workflow:
            User Query -> MedGemma Query Transformation -> Multi-Query Dense Search
            -> Result Fusion & Deduplication -> Grounded Answer Synthesis
        """
        start_time = time.time()

        # Step 1: Query Transformation
        qgen_res = self.query_generator.generate_queries(
            user_query=user_query,
            num_queries=num_queries,
            include_original=True
        )
        queries = qgen_res["queries"]
        qgen_time = qgen_res["elapsed_sec"]

        # Step 2: Multi-Query Dense Retrieval
        retrieval_start = time.time()
        results_by_query = self.retriever.retrieve_batch(
            queries=queries,
            top_k=top_k_per_query
        )
        retrieval_time = time.time() - retrieval_start

        # Step 3: Deduplication & Fusion
        fusion_start = time.time()
        fused_contexts, redundancy_metrics = self.deduplicator.fuse_and_deduplicate(
            results_by_query=results_by_query,
            final_top_k=final_top_k,
            fusion_strategy=fusion_strategy
        )
        fusion_time = time.time() - fusion_start

        # Step 4: Grounded Answer Synthesis
        answer_dict = {}
        if generate_answer:
            answer_dict = self.answer_generator.generate_answer(user_query, fused_contexts)

        total_time = time.time() - start_time

        return {
            "mode": "multi_query_rag",
            "original_query": user_query,
            "generated_queries": queries,
            "raw_results_by_query": results_by_query,
            "fused_contexts": fused_contexts,
            "redundancy_metrics": redundancy_metrics,
            "answer": answer_dict.get("answer", ""),
            "citations": answer_dict.get("citations", []),
            "timings": {
                "query_gen_sec": round(qgen_time, 3),
                "retrieval_sec": round(retrieval_time, 3),
                "fusion_sec": round(fusion_time, 3),
                "generation_sec": answer_dict.get("elapsed_sec", 0.0),
                "total_sec": round(total_time, 3)
            }
        }

    def compare(
        self,
        user_query: str,
        top_k: int = DEFAULT_FINAL_TOP_K,
        num_queries: int = DEFAULT_NUM_QUERIES,
        run_judge: bool = True
    ) -> Dict[str, Any]:
        """Execute both Single-Query Baseline and Multi-Query RAG side-by-side."""
        baseline_result = self.run_baseline(user_query, top_k=top_k, generate_answer=True)
        multi_query_result = self.run_multi_query(
            user_query,
            num_queries=num_queries,
            top_k_per_query=top_k,
            final_top_k=top_k,
            fusion_strategy="rrf",
            generate_answer=True
        )

        # Calculate comparative delta
        baseline_ids = {str(d["id"]) for d in baseline_result["retrieved_documents"]}
        mq_ids = {str(d["id"]) for d in multi_query_result["fused_contexts"]}
        new_docs_discovered = mq_ids - baseline_ids

        # LLM-as-a-Judge Evaluation
        judge_data = {}
        if run_judge and baseline_result.get("answer") and multi_query_result.get("answer"):
            try:
                judge_data = self.judge.compare_answers(
                    question=user_query,
                    baseline_answer=baseline_result["answer"],
                    multiquery_answer=multi_query_result["answer"],
                    contexts_baseline=baseline_result["retrieved_documents"],
                    contexts_multiquery=multi_query_result["fused_contexts"]
                )
                judge_data["faithfulness_eval"] = self.judge.evaluate_groundedness(
                    user_query,
                    multi_query_result["answer"],
                    multi_query_result["fused_contexts"]
                )
            except Exception as e:
                logger.warning(f"Judge evaluation failed: {e}")
                judge_data = {"error": str(e)}

        return {
            "query": user_query,
            "baseline": baseline_result,
            "multi_query": multi_query_result,
            "comparison": {
                "baseline_doc_count": len(baseline_ids),
                "multi_query_unique_doc_count": len(mq_ids),
                "overlap_doc_count": len(baseline_ids.intersection(mq_ids)),
                "new_documents_discovered_by_multi_query": list(new_docs_discovered),
                "novelty_ratio": round(len(new_docs_discovered) / len(mq_ids), 2) if mq_ids else 0.0
            },
            "judge": judge_data
        }
