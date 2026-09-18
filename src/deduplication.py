"""Deduplication and Fusion Engine for Multi-Query Retrieval.

Implements Reciprocal Rank Fusion (RRF), score-based aggregation,
document deduplication, and calculates redundancy and overlap metrics
crucial for viva demonstration.
"""

from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from src.config import DEFAULT_RRF_K, DEFAULT_FINAL_TOP_K


class ResultFusionDeduplicator:
    """Fuses multi-query retrieved candidates, eliminates duplicates, and tracks redundancy."""

    def __init__(self, rrf_k: int = DEFAULT_RRF_K):
        self.rrf_k = rrf_k

    def fuse_and_deduplicate(
        self,
        results_by_query: Dict[str, List[Dict[str, Any]]],
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        fusion_strategy: str = "rrf",
        rrf_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Fuse and deduplicate document lists retrieved across multiple queries.

        Args:
            results_by_query: Mapping of {query_str: [doc_dicts]}.
            final_top_k: Number of unique documents to return for final context.
            fusion_strategy: 'rrf' (Reciprocal Rank Fusion), 'max_score', or 'sum_score'.

        Returns:
            Tuple of:
                1. List of top-K unique document dicts with fusion scores and audit trail.
                2. Redundancy & overlap metrics dictionary.
        """
        doc_store: Dict[str, Dict[str, Any]] = {}
        query_appearances: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        rrf_scores: Dict[str, float] = defaultdict(float)
        score_sums: Dict[str, float] = defaultdict(float)
        max_scores: Dict[str, float] = defaultdict(float)

        total_candidates_count = 0
        all_queries = list(results_by_query.keys())

        # Traverse query results
        for q, docs in results_by_query.items():
            total_candidates_count += len(docs)
            for rank, doc in enumerate(docs, start=1):
                did = str(doc["id"])
                
                # Store passage info if first seen
                if did not in doc_store:
                    doc_store[did] = {
                        "id": did,
                        "title": doc.get("title", ""),
                        "text": doc.get("text", ""),
                    }
                
                # Audit entry: track which query found this document at what rank/score
                appearance = {
                    "query": q,
                    "rank": rank,
                    "score": doc.get("score", 0.0)
                }
                query_appearances[did].append(appearance)

                # Update RRF: RRF(d) = sum(1 / (k + rank))
                k_val = rrf_k if rrf_k is not None else self.rrf_k
                rrf_scores[did] += 1.0 / (k_val + rank)

                # Update score aggregations
                score_sums[did] += doc.get("score", 0.0)
                if doc.get("score", 0.0) > max_scores[did]:
                    max_scores[did] = doc.get("score", 0.0)

        unique_count = len(doc_store)
        redundant_count = max(0, total_candidates_count - unique_count)
        redundancy_rate = (redundant_count / total_candidates_count * 100.0) if total_candidates_count > 0 else 0.0

        # Rank unique documents according to chosen strategy
        scored_docs = []
        for did, base_info in doc_store.items():
            apps = query_appearances[did]
            item = dict(base_info)
            item["retrieved_by_queries"] = [a["query"] for a in apps]
            item["query_count"] = len(apps)
            item["is_duplicate"] = len(apps) > 1
            item["appearances"] = apps
            item["rrf_score"] = round(rrf_scores[did], 6)
            item["max_score"] = round(max_scores[did], 4)
            item["sum_score"] = round(score_sums[did], 4)

            if fusion_strategy == "rrf":
                item["final_score"] = item["rrf_score"]
            elif fusion_strategy == "max_score":
                item["final_score"] = item["max_score"]
            else:
                item["final_score"] = item["sum_score"]

            scored_docs.append(item)

        # Sort descending by final fusion score
        scored_docs.sort(key=lambda x: x["final_score"], reverse=True)
        top_k_docs = scored_docs[:final_top_k]

        # Calculate pairwise query overlap (Jaccard similarity matrix)
        query_doc_sets = {
            q: {str(d["id"]) for d in docs}
            for q, docs in results_by_query.items()
        }
        overlap_matrix = {}
        for i, q1 in enumerate(all_queries):
            overlap_matrix[q1] = {}
            for j, q2 in enumerate(all_queries):
                s1 = query_doc_sets[q1]
                s2 = query_doc_sets[q2]
                intersection = len(s1.intersection(s2))
                union = len(s1.union(s2))
                jaccard = round(intersection / union, 3) if union > 0 else 0.0
                overlap_matrix[q1][q2] = {
                    "shared_count": intersection,
                    "jaccard_sim": jaccard
                }

        # Multi-query metrics summary
        single_query_count = len(results_by_query.get(all_queries[0], [])) if all_queries else 0
        expansion_ratio = round(unique_count / single_query_count, 2) if single_query_count > 0 else 1.0

        metrics = {
            "total_candidates_retrieved": total_candidates_count,
            "unique_documents_found": unique_count,
            "redundant_documents_filtered": redundant_count,
            "redundancy_rate_pct": round(redundancy_rate, 2),
            "search_expansion_ratio": expansion_ratio,
            "num_queries_executed": len(all_queries),
            "fusion_strategy": fusion_strategy,
            "multi_query_hits": sum(1 for d in scored_docs if d["is_duplicate"]),
            "overlap_matrix": overlap_matrix
        }

        return top_k_docs, metrics
