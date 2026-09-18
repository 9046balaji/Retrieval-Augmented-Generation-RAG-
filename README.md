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
This diagram shows how all the components—the Chatbot UI, Ollama LLM, ChromaDB vector store, and offline embedding model—connect together.

```mermaid
graph TD
    subgraph Frontend["🖥️ User Interface"]
        UI["Streamlit Chatbot UI<br><i>(Chat conversation, Parameter sliders, Source expander)</i>"]
    end

    subgraph BackendEngine["⚙️ Multi-Query RAG Pipeline"]
        Pipeline["RAG Pipeline Coordinator<br><i>(src/pipeline.py)</i>"]
        QGen["Query Transformation Engine<br><i>(src/query_generator.py)</i>"]
        Fusion["RRF Fusion & Deduplication<br><i>(src/deduplication.py)</i>"]
        Gen["Grounded Generator<br><i>(src/generator.py)</i>"]
        Judge["LLM-as-a-Judge Engine<br><i>(src/judge.py)</i>"]
    end

    subgraph Storage["💾 Storage & Models"]
        Chroma["ChromaDB Vector Database<br><i>(5,183 SciFact Papers, HNSW Cosine Index)</i>"]
        BGE["BAAI/bge-base-en-v1.5<br><i>(768-dim Offline Embedding Model)</i>"]
        Ollama["Ollama Local Server<br><i>(MedGemma 1.5 4B IT, 16k Context Window)</i>"]
    end

    UI -->|"User query & parameters (N, K, rrf_k)"| Pipeline
    Pipeline -->|"1. Generate query angles"| QGen
    QGen <-->|"Prompt & receive 4 queries"| Ollama
    Pipeline -->|"2. Encode queries"| BGE
    Pipeline -->|"3. Vector search across all queries"| Chroma
    Pipeline -->|"4. Merge & filter candidates"| Fusion
    Pipeline -->|"5. Generate cited answer"| Gen
    Gen <-->|"Prompt with retrieved passages"| Ollama
    Pipeline -->|"6. Score faithfulness (Optional)"| Judge
    Judge <-->|"Evaluate groundedness"| Ollama
    Pipeline -->|"Final answer + citations + metrics"| UI
```

---

### Diagram 2: Multi-Query RAG Flow
This diagram details the step-by-step mathematical and data flow from your initial question to the final answer.

```mermaid
flowchart TD
    UserQ["👤 User Asks Question<br><i>'How does microRNA dysregulation influence tumor metastasis?'</i>"]
    
    subgraph Step1["Step 1: Query Transformation (MedGemma)"]
        UserQ --> LLMQ["MedGemma decomposes into 4 search angles:"]
        LLMQ --> Q1["Q1: microRNA dysregulation in tumor metastasis"]
        LLMQ --> Q2["Q2: microRNA regulation mechanisms"]
        LLMQ --> Q3["Q3: epithelial-mesenchymal transition EMT invasion"]
        LLMQ --> Q4["Q4: miR-21 miR-155 miR-200 target genes"]
    end

    subgraph Step2["Step 2: Vector Embedding & Dense Retrieval"]
        Q1 & Q2 & Q3 & Q4 --> Embed["BGE-base-en-v1.5<br><i>(Add instruction prefix & normalize to 768-dim)</i>"]
        Embed --> ChromaSearch["ChromaDB HNSW Cosine Search<br><i>Retrieve Top-5 per query branch</i>"]
        ChromaSearch --> Pool["Candidate Pool<br><i>4 queries × 5 = 20 total retrieved passages</i>"]
    end

    subgraph Step3["Step 3: Reciprocal Rank Fusion & Deduplication"]
        Pool --> RRF["Reciprocal Rank Fusion Formula:<br><b>RRF(d) = Σ 1 / (60 + rank_q(d))</b>"]
        RRF --> Dedup["Deduplication Engine:<br>• Remove redundant documents (25% - 40% duplicate rate)<br>• Consensus documents get highest ranks"]
        Dedup --> TopDocs["Top-5 Final Unique Passages Selected"]
    end

    subgraph Step4["Step 4: Strict Evidence-Grounded Generation"]
        TopDocs --> Prompt["Strict Context Bounding Prompt:<br>• Rely ONLY on provided documents<br>• Cite every fact with [Document id]<br>• Refuse to guess from memory if context lacks answer"]
        Prompt --> MedGemma["MedGemma 1.5 4B Generation"]
        MedGemma --> FinalAnswer["Final Answer with Verified Citations<br><i>[Document 25523969], [Document 2619579]...</i>"]
    end

    subgraph Step5["Step 5: LLM-as-a-Judge Evaluation"]
        FinalAnswer --> JudgeModel["LLM-as-a-Judge Evaluation:"]
        JudgeModel --> S1["Faithfulness / Groundedness: 5.0 / 5.0"]
        JudgeModel --> S2["Context Utilization: High"]
        JudgeModel --> S3["Answer Relevance: High"]
    end
```

---

### Diagram 3: User Flow & Interaction
This diagram shows what happens when a user interacts with the Chatbot application.

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 User
    participant UI as 🖥️ Streamlit Chatbot
    participant Pipe as ⚙️ Pipeline
    participant DB as 🗄️ ChromaDB
    participant LLM as 🧠 MedGemma (Ollama)

    User->>UI: Adjust settings (Top-K=5, Queries=4, RRF k=60)
    User->>UI: Types question: "How does microRNA affect cancer metastasis?"
    UI->>Pipe: Execute Multi-Query RAG
    
    Note over Pipe,LLM: Step 1: Query Expansion
    Pipe->>LLM: Generate 4 orthogonal search angles
    LLM-->>Pipe: Returns Q1, Q2, Q3, Q4

    Note over Pipe,DB: Step 2: Parallel Retrieval
    Pipe->>DB: Query ChromaDB for all 4 queries simultaneously
    DB-->>Pipe: Returns 20 candidates

    Note over Pipe: Step 3: RRF Fusion & Deduplication
    Pipe->>Pipe: Apply RRF(k=60), boost consensus, remove duplicates (15 unique kept)

    Note over Pipe,LLM: Step 4: Grounded Synthesis
    Pipe->>LLM: Generate answer using ONLY top 5 passages with citations
    LLM-->>Pipe: Grounded text + citations: [Document 25523969]

    opt If Judge Evaluation is Enabled
        Pipe->>LLM: Grade answer faithfulness (1 to 5)
        LLM-->>Pipe: Faithfulness: 5/5, Verdict: Verified
    end

    Pipe-->>UI: Deliver response package
    UI-->>User: Display Assistant Bubble with Answer & Citations
    User->>UI: Click "🔍 Multi-Query Insights & Retrieved Sources"
    UI-->>User: Expands generated queries, redundancy stats, & source passages
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
