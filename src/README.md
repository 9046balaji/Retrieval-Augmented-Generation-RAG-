# Exact Techniques & Parameters Used in This Project

This document lists the exact techniques, parameters, and strategies implemented across the `src/` modules in this project.

---

## 1. Quick Summary Table

| Component | Exact Technique Used | Specific Value / Setting | Code File |
| :--- | :--- | :--- | :--- |
| **Dataset Corpus** | BEIR SciFact Research Papers | 5,183 papers (abstracts + claims) | `notebooks/`, `src/config.py` |
| **Chunking Strategy** | Title-Prefixed Passage Chunking | `Title + "\n" + Abstract` (< 512 tokens) | `notebooks/`, `src/retriever.py` |
| **Embedding Model** | Dense Transformer Encoder | `BAAI/bge-base-en-v1.5` (768-dim, offline) | `src/config.py`, `models/` |
| **Vector Database** | ChromaDB Persistent Store | Collection: `scifact_collection` | `src/retriever.py` |
| **Vector Index & Space** | HNSW Graph Index | Cosine Similarity (`hnsw:space: "cosine"`) | `src/retriever.py` |
| **Query Transformation** | Multi-Query 4-Angle Expansion | MedGemma 1.5 4B (4 orthogonal queries) | `src/query_generator.py` |
| **Retrieval Strategy** | Multi-Branch Dense Search | Top-5 per query (4 × 5 = 20 candidates) | `src/retriever.py` |
| **Fusion Algorithm** | Reciprocal Rank Fusion (RRF) | Smoothing factor $k = 60$ | `src/deduplication.py` |
| **Deduplication** | Unique ID Consensus Filter | Removes ~30% duplicate overlap | `src/deduplication.py` |
| **Context Bounding** | Strict Evidence Prompting | Mandates inline `[Document <id>]` citations | `src/generator.py` |
| **LLM Model** | MedGemma 1.5 4B IT (Ollama) | 16,384 tokens context (`num_ctx: 16384`) | `src/config.py` |
| **Evaluation Method** | LLM-as-a-Judge | Faithfulness (1–5), Utilization, Relevance | `src/judge.py`, `run_evaluation.py` |

---

## 2. Chunking Technique Used

* **Technique**: **Passage-Level Semantic Chunking with Title Prefix**
* **How it works in our code**:
  Instead of slicing texts at arbitrary character counts (which cuts sentences in half), each research paper is formatted as:
  ```
  [Title of Paper]
  [Complete Abstract Text]
  ```
* **Chunk Size**: Natural abstract length (average 150–250 words, capped at 512 tokens).
* **Why we used this**:
  Scientific abstracts contain self-contained claims, methods, and results. Keeping the title attached ensures that topic keywords are always encoded alongside the evidence.
* **Metadata stored with each chunk**:
  * `id`: Original document ID from SciFact (e.g. `25523969`).
  * `title`: Paper title string.

---

## 3. Query Transformation Technique Used

* **Technique**: **4-Perspective Orthogonal Query Decomposition**
* **Model**: `medgemma-1.5-4b-it`
* **Temperature**: `0.4` (slight diversity for distinct angles)
* **How it works**:
  When you ask a question, MedGemma decomposes it into 4 distinct search branches:
  1. **Branch 1 (Direct Mechanism)**: Core keywords and medical synonyms.
  2. **Branch 2 (Molecular Pathway)**: Biochemical pathways, signaling cascades, oncogenesis.
  3. **Branch 3 (Clinical Phenotype)**: Physical manifestation, invasion, tumor metastasis, EMT.
  4. **Branch 4 (Genetic Targets)**: Associated genes, miRNAs, proteins, and biomarkers (e.g. miR-21, miR-200).

---

## 4. Vector Embedding & Retrieval Technique Used

* **Embedding Model**: `BAAI/bge-base-en-v1.5` (running locally from `models/bge-base-en-v1.5/`)
* **Output Dimension**: `768` dimensions
* **Normalization**: L2 normalization (`normalize_embeddings=True`)
* **Instruction Prefix**: `"Represent this sentence for searching relevant passages: "`
* **Vector Store**: **ChromaDB** with HNSW (Hierarchical Navigable Small World) index
* **Distance Space**: `cosine`
* **Retrieval Multiplier**:
  * Queries: `4`
  * Top-K per query: `5`
  * Candidate Pool: `20` passages total

---

## 5. Fusion & Deduplication Technique Used

* **Algorithm**: **Reciprocal Rank Fusion (RRF)** with smoothing constant $k = 60$
* **Formula Used**:
  $$RRF(d) = \sum_{q \in Q} \frac{1}{60 + \text{rank}_q(d)}$$
* **Consensus Boosting**:
  If a document is found by 2 or 3 of the search queries, its reciprocal rank scores add up, pushing it above documents found by only one query.
* **Deduplication**:
  Documents are grouped by their unique `id`. Duplicates are pruned, tracking a redundancy rate of **25% to 35%**.
* **Final Selection**: The Top `5` highest-scoring unique documents are sent to the generator.

---

## 6. Generation Technique Used

* **Model**: Google `MedGemma 1.5 4B IT` (quantized 4-bit GGUF via Ollama)
* **Context Window**: `16384` tokens (16k)
* **Generation Temperature**: `0.0` (deterministic greedy decoding to eliminate creative guessing)
* **Prompt Bounding Technique**:
  * Strict evidence bounding: Instructs model to answer **only** from the retrieved context passages.
  * If the answer is not present in the passages, the model must output: *"The provided documents do not contain sufficient evidence to answer this question."*
  * Mandatory citations: Every statement must cite its source document: `[Document <id>]`.

---

## 7. Self-Evaluation Technique Used (LLM-as-a-Judge)

* **Evaluator**: MedGemma running an independent evaluation prompt.
* **Evaluation Temperature**: `0.0`
* **Scores Computed**:
  1. **Faithfulness (1.0 to 5.0)**: Checks whether every sentence is grounded in the retrieved passages.
  2. **Context Utilization**: Measures how thoroughly the retrieved facts were used.
  3. **Answer Relevance (1.0 to 5.0)**: Checks if the response directly answers the user's question.
