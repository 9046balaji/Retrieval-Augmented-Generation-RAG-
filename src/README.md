# RAG Techniques & Architecture Guide (Simple Explanation)

This guide explains all the techniques and strategies used in our Multi-Query RAG pipeline in plain, simple language.

---

## 1. Chunking Strategy (How We Prepare Documents)

### What is Chunking?
Scientific papers are too long to fit into a search model all at once. Chunking means cutting large documents into smaller, meaningful pieces called **passages**.

### How We Do It:
* **Passage-Level Chunking**: Instead of cutting text randomly in the middle of a sentence, we chunk by paragraphs or natural scientific sections (abstracts, claims, findings).
* **Metadata Attachment**: Every single chunk keeps its original `Document ID` and `Title` attached to it.
* **Token Length (Max 512 Tokens)**: Matches the sweet spot of our embedding model so no sentences get cut off or lost.
* **Why this helps**: When the AI answers your question, it can tell you the exact document number where it found the fact!

---

## 2. Multi-Query Transformation (4 Search Angles)

### The Problem:
If you ask: *"How does microRNA affect cancer metastasis?"*, a standard search only looks for those exact words. But important medical papers might use terms like *"miR-21 regulation of epithelial-mesenchymal transition"* or *"oncogenic cell migration"*. A single search misses those papers completely!

### Our Technique:
We give your question to **MedGemma 1.5 4B** first, which decomposes your question into **4 different search angles**:
1. **Direct Mechanism**: Uses everyday keywords and synonyms.
2. **Molecular Pathway**: Searches for biochemical processes (e.g., signaling cascades, oncogenesis).
3. **Clinical Phenotype**: Searches for physical results (e.g., cell invasion, metastasis, tumor spread).
4. **Genetic Targets**: Searches for specific genes and markers (e.g., miR-21, miR-155, E-cadherin).

Now, instead of searching once, we search **4 times across 4 different perspectives**!

---

## 3. Dense Vector Retrieval (ChromaDB + BGE)

### How We Search:
1. **Embedding Model (`BAAI/bge-base-en-v1.5`)**: 
   Converts each of the 4 text queries into a list of 768 numbers (a mathematical coordinate in meaning space).
2. **ChromaDB Vector Store**:
   Stores 5,183 scientific research papers indexed using **HNSW (Hierarchical Navigable Small World)** graphs.
3. **Cosine Similarity**:
   Measures the angle between your question's vector and each document's vector. Closer angle = closer meaning.
4. **Branch Retrieval**:
   Each of the 4 queries retrieves its **Top-5 closest papers**.
   $$\text{Total Candidates} = 4 \text{ queries} \times 5 \text{ papers} = 20 \text{ passages}$$

---

## 4. Reciprocal Rank Fusion (RRF) & Deduplication

### Why We Need This:
Because we ran 4 searches, some great papers were found by multiple queries, while some irrelevant papers were only found once. Also, there are duplicate papers across the lists.

### How RRF Works:
Every document gets a score based on where it ranked in each query list:

$$RRF(d) = \sum \frac{1}{60 + \text{rank}}$$

* **Consensus Boost**: If a paper was ranked #1 in Query 1 and #2 in Query 3, its scores add together. A paper found by multiple search angles easily beats a paper found only once!
* **Deduplication**: We delete repeat copies of the same document (~30% redundancy is removed).
* **Final Top-5**: We select the top 5 highest-scoring unique papers to send to the generator.

---

## 5. Strict Grounded Answer Synthesis (Zero Hallucination)

### The Problem with Normal AI:
Normal AI models often make things up ("hallucinate") using their pre-training memory when they don't know the exact answer.

### Our Solution:
We put the AI into a strict "open-book test" container:
1. **Context Bounding**: The prompt explicitly tells MedGemma: *"Answer ONLY using the provided research passages. If the answer is not in the text, honestly reply that the evidence is insufficient."*
2. **Mandatory Citations**: Every sentence or claim must end with an inline citation tag like `[Document 25523969]`.
3. **No Guessing**: The model is forbidden from using outside assumptions.

---

## 6. LLM-as-a-Judge (Automated Self-Verification)

After the answer is generated, a second instance of MedGemma acts as an impartial judge to score the answer:

| Metric | What it Checks | Target Score |
| :--- | :--- | :--- |
| **Faithfulness** | Does the answer contain any facts NOT in the retrieved documents? | `5.0 / 5.0` (100% grounded) |
| **Context Utilization** | Did the model actually use the scientific evidence provided? | `High` |
| **Answer Relevance** | Does the response directly answer what the user asked? | `5.0 / 5.0` |

If an answer tries to make things up, the judge detects it and flags the response.

---

## 7. Memory & Hardware Optimization

* **16k Context Window (`OLLAMA_NUM_CTX = 16384`)**:
  Feeding 20 scientific passages into an LLM requires lots of memory. We configured Ollama with a 16k token context so no passages ever get truncated.
* **Offline Local Execution**:
  All embeddings run locally using PyTorch, and MedGemma runs through Ollama on your own machine without sending data to any paid third-party cloud.
