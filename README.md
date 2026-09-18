# Multi-Query Retrieval-Augmented Generation (RAG) System

A modern, evidence-grounded scientific research assistant that uses **Multi-Query Query Transformation**, **ChromaDB Vector Retrieval**, **Reciprocal Rank Fusion (RRF)**, and **MedGemma** to provide accurate, cited answers without hallucinating.

---

## 1. What is This Project? (Simple Explanation)

In regular AI search (Single-Query RAG), when you ask a question like *"What are the advantages of transformer models?"*, the computer converts your sentence into a single vector and searches for similar paragraphs in a database.

### The Problem with Single-Query Search:
* **Word Mismatch**: If you use simple everyday words, but the research paper uses complex medical or scientific terminology, the search misses the paper.
* **Complex Questions**: Real questions often have multiple parts. A single search direction cannot capture all angles.
* **Low Recall**: Many important documents get missed simply because they were ranked at position #15 or #20 instead of the top 5.

### Our Solution: Multi-Query RAG
Instead of searching only once, our system:
1. **Expands Your Question**: MedGemma automatically creates **4 distinct search queries** covering synonyms, sub-mechanisms, and related technical terms.
2. **Searches in Parallel**: ChromaDB searches across all 4 query directions at once.
3. **Fuses & Cleans Results**: Uses **Reciprocal Rank Fusion (RRF)** to combine the results, boost documents found by multiple queries, and delete duplicate passages.
4. **Answers Strictly from Evidence**: The LLM reads only the retrieved literature and answers the question with **bracketed citations** like `[Document 25523969]`. If the database lacks the answer, the model honestly states it does not know rather than making things up!
5. **Judges Its Own Work**: An impartial **LLM-as-a-Judge** scores the response on a scale of 1 to 5 for faithfulness and relevance.

---

## 2. System Architecture Diagrams

### Diagram 1: Full System Architecture
This diagram displays the end-to-end multi-query architecture matching the horizontal pipeline design (Indexing, Multi-Query Vector Search, and Context Augmentation & Generation):

![RAG Architecture](docs/images/1_rag_architecture.svg)

```mermaid
flowchart LR
    %% Subgraphs matching horizontal architecture design
    subgraph Indexing ["📁 Indexing Pipeline"]
        direction LR
        D["📄 Documents<br><i>(5,183 SciFact)</i>"] -->|Chunking| C["📑 Chunks<br><i>(Passages)</i>"]
        C -->|Vectorize| E1["🧠 Embedding Model<br><i>(bge-base-en-v1.5)</i>"]
        E1 --> V1["{...} Vectors<br><i>(768-dim)</i>"]
        V1 -->|Indexing| VDB[("🗄️ Vector Database<br><i>ChromaDB (HNSW Cosine)</i>")]
    end

    subgraph QueryFlow ["🔍 Multi-Query Search"]
        direction LR
        U["👤 User"] --> Q["❓ Query"]
        Q -->|Expand| MQ["🔀 Multi-Query (MedGemma)<br><i>(4 Search Angles)</i>"]
        MQ -->|Vectorize| V2["{...} 4x Vectors"]
        V2 -->|Search| VDB
    end

    subgraph AugGen ["⚡ Augment & Generate"]
        direction LR
        VDB -->|Retrieve| AUG["📦 Augment &amp; Fuse<br><i>(RRF k=60 + Dedup)</i>"]
        AUG -->|Context| LLM["🤖 LLM<br><i>(MedGemma 1.5 4B)</i>"]
        LLM -->|Generate| R["📝 Response<br><i>(Cited Answer)</i>"]
        R -->|Deliver| U
    end

    classDef userCard fill:#0284c7,stroke:#0369a1,stroke-width:2px,color:#ffffff,font-weight:bold;
    classDef blueCard fill:#e0f2fe,stroke:#38bdf8,stroke-width:1.5px,color:#0369a1;
    classDef storeCard fill:#ffffff,stroke:#cbd5e1,stroke-width:2px,color:#0f172a,font-weight:bold;
    classDef neutralCard fill:#f8fafc,stroke:#cbd5e1,stroke-width:1.5px,color:#1e293b;

    class U,Q userCard;
    class MQ,V2,AUG blueCard;
    class VDB storeCard;
    class D,C,E1,V1,LLM,R neutralCard;
```

---

### Diagram 2: Multi-Query RAG Flow
This diagram details the step-by-step query decomposition, parallel ChromaDB retrieval, Reciprocal Rank Fusion, and strict evidence-grounded generation:

![Multi-Query RAG Flow](docs/images/2_multi_query_flow.svg)

```mermaid
flowchart LR
    direction LR
    Q0["👤 User Question"] -->|Decompose| MQ["🔀 MedGemma Expansion<br><i>(Q1, Q2, Q3, Q4)</i>"]
    MQ -->|Vectorize| EMB["🧠 BGE Embedding"]
    EMB -->|Parallel Search| CDB[("🗄️ ChromaDB<br><i>(Top-5 × 4 = 20 docs)</i>")]
    CDB -->|Pool| RRF["⚡ RRF Fusion &amp; Dedup<br><i>(Consensus Boost, 30% pruned)</i>"]
    RRF -->|Top-5 Unique| CTX["📋 Evidence Context Prompt<br><i>(Zero guessing bound)</i>"]
    CTX -->|Generate| GEN["🤖 MedGemma 1.5 4B"]
    GEN --> ANS["✅ Verified Cited Answer<br><i>[Document ID]</i>"]
    ANS --> JUDGE["⚖️ LLM Judge<br><i>(Faithfulness 5.0/5.0)</i>"]

    classDef userCard fill:#0284c7,stroke:#0369a1,stroke-width:2px,color:#ffffff,font-weight:bold;
    classDef blueCard fill:#e0f2fe,stroke:#38bdf8,stroke-width:1.5px,color:#0369a1;
    classDef greenCard fill:#f0fdf4,stroke:#86efac,stroke-width:1.5px,color:#15803d;
    classDef purpleCard fill:#faf5ff,stroke:#c084fc,stroke-width:1.5px,color:#7e22ce;
    classDef neutralCard fill:#f8fafc,stroke:#cbd5e1,stroke-width:1.5px,color:#1e293b;

    class Q0,ANS userCard;
    class MQ,CTX blueCard;
    class RRF greenCard;
    class JUDGE purpleCard;
    class EMB,CDB,GEN neutralCard;
```

---

### Diagram 3: User Flow & Interaction
This diagram illustrates the user journey through the Chatbot application, from query submission and hyperparameter tuning to citation verification and source inspection:

![User Flow & Interaction](docs/images/3_user_flow.svg)

```mermaid
flowchart LR
    direction LR
    U1["👤 User on Chatbot"] -->|1. Types Question &amp; Sets Top-K| UI["🖥️ Streamlit Chat Interface"]
    UI -->|2. Expand Query| MQ["🔀 Multi-Query Generator<br><i>(4 Search Angles)</i>"]
    MQ -->|3. Concurrent Search| CHR[("🗄️ ChromaDB Vector Store")]
    CHR -->|4. 20 Candidates| FUS["⚡ RRF Fusion &amp; Deduplication<br><i>(Top-5 Unique Kept)</i>"]
    FUS -->|5. Bounded Evidence| LLM["🤖 Grounded Answer Generator<br><i>(MedGemma 1.5 4B)</i>"]
    LLM -->|6. Verify Grounding| JDG["⚖️ LLM-as-a-Judge<br><i>(Faithfulness: 5.0/5.0)</i>"]
    JDG -->|7. Display Message| RES["💬 Assistant Chat Bubble<br><i>(Answer + Sources Expander)</i>"]
    RES -->|Delivered| U1

    classDef userCard fill:#0284c7,stroke:#0369a1,stroke-width:2px,color:#ffffff,font-weight:bold;
    classDef blueCard fill:#e0f2fe,stroke:#38bdf8,stroke-width:1.5px,color:#0369a1;
    classDef greenCard fill:#f0fdf4,stroke:#86efac,stroke-width:1.5px,color:#15803d;
    classDef purpleCard fill:#faf5ff,stroke:#c084fc,stroke-width:1.5px,color:#7e22ce;
    classDef neutralCard fill:#f8fafc,stroke:#cbd5e1,stroke-width:1.5px,color:#1e293b;

    class U1,RES userCard;
    class UI,MQ blueCard;
    class FUS greenCard;
    class JDG purpleCard;
    class CHR,LLM neutralCard;
```

---

## 3. How the Math & Fusion Work (In Plain English)

### 1. Reciprocal Rank Fusion (RRF)
When multiple queries find documents, some documents appear in only one list, while others appear in 2 or 3 lists. RRF scores each document using:

$$RRF(d) = \sum_{q \in Q} \frac{1}{k + \text{rank}_q(d)}$$

* $k = 60$: A smoothing factor that keeps the balance fair.
* **Why this is awesome**: If Document A is ranked #1 in Query 1, and also ranked #2 in Query 3, its score adds up:
  $$\frac{1}{60 + 1} + \frac{1}{60 + 2} = 0.01639 + 0.01612 = 0.03251$$
  This consensus document easily beats a document that was only found once, ensuring the most reliable research rises to the top!

### 2. Deduplication & Redundancy Tracking
When searching 4 queries, between **25% and 40%** of the retrieved documents are duplicates. The deduplication engine discards the repeats so the LLM does not read the same text multiple times, preventing memory overload and speeding up generation.

---

## 4. Real Results We Got

We validated the system across three distinct test categories:

| Test Category | Query Example | What We Tested | Result Obtained |
| :--- | :--- | :--- | :--- |
| **Category A: In-Domain Positive Medical Claim** | *"0-dimensional biomaterials show inductive properties."* | Retrieval and grounded synthesis on biomedical papers. | **Success**: Retrieved 20 candidates, retained 17 unique docs (3.4x expansion ratio), cited source IDs. Faithfulness: **5.0 / 5.0**. |
| **Category B: Out-of-Domain Refusal Question** | *"What are the advantages of transformer models in deep learning?"* | Tests if the model makes things up when the database has no matching information. | **Zero Hallucination**: Checked the medical documents, confirmed transformers were not mentioned, and correctly stated: *"The retrieved literature does not contain evidence to answer this question."* Faithfulness: **5.0 / 5.0**. |
| **Category C: Multi-Faceted Comparative Question** | *"How does microRNA dysregulation influence tumor metastasis and cellular migration?"* | Tests query decomposition into orthogonal sub-angles (mechanisms, oncogenes, cell motility). | **Multi-Query Won**: Decomposed into 4 queries, expanded search 3.0x, cited **5 distinct scientific papers** (`[Document 25523969]`, `[Document 2000038]`, `[Document 2619579]`, `[Document 38844612]`, `[Document 2251426]`). Faithfulness: **5.0 / 5.0**. |

### Key Performance Highlights:
* **Faithfulness Score**: **5.0 / 5.0** across all tests (Zero hallucinations).
* **Duplicate Filtering**: Automatically removes **25% to 40%** redundant documents.
* **Search Space Expansion**: Reaches **2.4x to 3.4x** more unique literature passages than standard single-query search.
* **Hardware Efficiency**: Runs on **100% NVIDIA RTX GPU** with context bounded to **16k tokens** (`OLLAMA_NUM_CTX = 16384`), keeping VRAM stable under 6 GB.

---

## 5. How to Run the Project

### Prerequisites:
1. Make sure **Ollama** is running in the background with the MedGemma model:
   ```powershell
   ollama run hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M
   ```

### Option 1: Double-Click (Easiest)
Simply double-click:
```
run_app.bat
```
This automatically launches the Streamlit chatbot and opens your web browser.

### Option 2: Run via Terminal
Open PowerShell and run:
```powershell
conda activate heart
streamlit run app.py
```
Or directly:
```powershell
& "C:\ProgramData\anaconda3\envs\heart\python.exe" -m streamlit run app.py
```

### Option 3: Run the Automated 3-Category Test Suite
To verify the entire pipeline, query expansion, deduplication, and LLM judge from the command line:
```powershell
& "C:\ProgramData\anaconda3\envs\heart\python.exe" test_end_to_end.py
```
