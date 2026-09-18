"""Dense Vector Retrieval Module using ChromaDB.

Executes semantic similarity searches exclusively on ChromaDB using
dense embeddings from BAAI/bge-base-en-v1.5.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
# SentenceTransformer must be imported before chromadb on Windows to avoid OpenMP/ONNX runtime conflict
from sentence_transformers import SentenceTransformer
import chromadb

from src.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    QUERY_INSTRUCTION,
    DEFAULT_TOP_K_PER_QUERY
)

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Handles semantic retrieval exclusively from ChromaDB vector store."""

    def __init__(
        self,
        chroma_path: Path = CHROMA_DIR,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        device: Optional[str] = None
    ):
        self.chroma_path = Path(chroma_path)
        if device:
            self.device = device
        elif EMBEDDING_DEVICE in ("cuda", "cpu"):
            self.device = EMBEDDING_DEVICE
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backend = "chromadb"

        # Load embedding model
        logger.info(f"Loading embedding model '{embedding_model_name}' on {self.device}...")
        self.embed_model = SentenceTransformer(embedding_model_name, device=self.device)

        # Initialize ChromaDB client & collection
        self.chroma_client = None
        self.collection = None
        self._init_chromadb()

    def _init_chromadb(self):
        """Connect directly to persistent ChromaDB collection."""
        if not self.chroma_path.exists() or not any(self.chroma_path.iterdir()):
            logger.error(f"ChromaDB directory not found at '{self.chroma_path}'.")
            return

        try:
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
            existing_cols = [c.name for c in self.chroma_client.list_collections()]
            col_name = (
                CHROMA_COLLECTION_NAME
                if CHROMA_COLLECTION_NAME in existing_cols
                else (existing_cols[0] if existing_cols else None)
            )
            if col_name:
                self.collection = self.chroma_client.get_collection(name=col_name)
                logger.info(f"Connected to ChromaDB collection '{col_name}' ({self.collection.count()} items).")
            else:
                logger.error(f"No collection found in ChromaDB at '{self.chroma_path}'.")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB ({e}).")

    def is_ready(self) -> bool:
        """Check if ChromaDB collection is loaded and ready for queries."""
        return self.collection is not None

    def encode_queries(self, queries: List[str]) -> List[List[float]]:
        """Encode queries with BGE instruction prefix into normalized unit vectors."""
        prefixed = [QUERY_INSTRUCTION + q for q in queries]
        embeddings = self.embed_model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K_PER_QUERY) -> List[Dict[str, Any]]:
        """Retrieve top-K passages for a single query from ChromaDB."""
        res = self.retrieve_batch([query], top_k=top_k)
        return res.get(query, [])

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = DEFAULT_TOP_K_PER_QUERY
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve top-K passages for each query in the list from ChromaDB."""
        if not self.is_ready():
            logger.error("ChromaDB is not initialized.")
            return {q: [] for q in queries}

        query_embeddings = self.encode_queries(queries)
        results_by_query = {}

        chroma_res = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        for i, q in enumerate(queries):
            docs = []
            doc_ids = chroma_res["ids"][i]
            texts = chroma_res["documents"][i]
            metas = chroma_res["metadatas"][i]
            distances = chroma_res["distances"][i]

            for rank, (did, text, meta, dist) in enumerate(zip(doc_ids, texts, metas, distances), start=1):
                # ChromaDB cosine distance: dist = 1 - cosine_sim
                cosine_sim = 1.0 - dist if dist is not None else 0.0
                title = meta.get("title", "") if meta else ""
                docs.append({
                    "id": str(did),
                    "title": title,
                    "text": text,
                    "score": round(float(cosine_sim), 4),
                    "query": q,
                    "rank": rank
                })
            results_by_query[q] = docs

        return results_by_query
