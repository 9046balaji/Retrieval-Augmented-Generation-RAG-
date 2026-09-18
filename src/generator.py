"""Answer Generation Module using MedGemma via Ollama.

Synthesizes a grounded scientific answer citing the retrieved,
deduplicated context passages with bracketed citations.
"""

import re
import time
import logging
from typing import List, Dict, Any
import ollama

from src.config import OLLAMA_MODEL, OLLAMA_NUM_CTX, GROUNDED_GEN_TEMPERATURE

logger = logging.getLogger(__name__)


class GroundedAnswerGenerator:
    """Generates cited answers from deduplicated retrieved contexts."""

    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name

    def generate_answer(
        self,
        question: str,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a grounded answer for the question using the provided context passages.

        Args:
            question: The user's original question.
            contexts: List of unique document dictionaries.

        Returns:
            Dict containing:
                - 'answer': Clean grounded answer with citations.
                - 'citations': List of document IDs cited.
                - 'elapsed_sec': Generation time in seconds.
                - 'model': Model name.
        """
        if not contexts:
            return {
                "answer": "No relevant documents were retrieved from the scientific literature database.",
                "citations": [],
                "elapsed_sec": 0.0,
                "model": self.model_name
            }

        # Build context block
        context_blocks = []
        for i, ctx in enumerate(contexts, start=1):
            did = ctx.get("id", f"Doc-{i}")
            title = ctx.get("title", "Untitled")
            text = ctx.get("text", "").strip()
            context_blocks.append(f"[Document {did}] Title: {title}\n{text}")

        context_str = "\n\n".join(context_blocks)

        prompt = (
            f"You are an evidence-grounded scientific research assistant. Your task is to answer the user question based EXCLUSIVELY on the provided literature below.\n\n"
            f"STRICT GROUNDING RULES:\n"
            f"1. Rely ONLY on the facts stated in the provided documents. Do NOT answer from your internal pre-training memory or make assumptions.\n"
            f"2. Every factual claim or finding you write MUST be immediately followed by its source citation: [Document <id>].\n"
            f"3. If the provided literature does not contain information to answer the question, state: \"The retrieved literature does not contain evidence to answer this question.\" Do not attempt to guess or use outside memory.\n"
            f"4. Integrate complementary details across the multiple retrieved documents into a coherent answer.\n\n"
            f"Provided Scientific Literature:\n"
            f"----------------------------------------\n"
            f"{context_str}\n"
            f"----------------------------------------\n\n"
            f"User Question: {question}\n\n"
            f"Grounded Answer (citing [Document <id>]):"
        )

        start_time = time.time()
        try:
            res = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": OLLAMA_NUM_CTX,
                    "temperature": GROUNDED_GEN_TEMPERATURE
                }
            )
            raw_answer = res["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to generate answer via Ollama ({e}).")
            raw_answer = f"Error during generation: {e}"

        elapsed_sec = time.time() - start_time
        cleaned_answer = self._clean_output(raw_answer)

        # Detect cited document IDs in output
        found_citations = list(set(re.findall(r"\[Document\s+([^\]]+)\]", cleaned_answer, re.IGNORECASE)))

        return {
            "answer": cleaned_answer,
            "citations": found_citations,
            "elapsed_sec": round(elapsed_sec, 3),
            "model": self.model_name,
            "context_count": len(contexts)
        }

    def _clean_output(self, text: str) -> str:
        """Strip internal reasoning tags or thinking steps."""
        # Strip <thought>...</thought> tags
        cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
        
        # If output begins with 'thought\n', find where the answer actually starts
        if cleaned.strip().lower().startswith("thought"):
            # Split lines and look for non-thought response
            parts = cleaned.split("\n\n", 2)
            if len(parts) > 1 and not parts[-1].strip().lower().startswith("thought"):
                cleaned = parts[-1]

        return cleaned.strip()
