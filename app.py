"""Multi-Query RAG Conversational Chatbot Interface.

A clean, modern chat interface allowing users to:
- Chat naturally with the Multi-Query RAG assistant.
- Adjust retrieval parameters (Top-K per query, Final Top-K, Num Queries, RRF k, Context window).
- Inspect multi-query expansions, deduplicated sources, and LLM-as-a-Judge scores in an expander.
"""

import os
import sys
import subprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
HEART_PYTHON = r"C:\ProgramData\anaconda3\envs\heart\python.exe"

def _ensure_streamlit_runner():
    """Ensure app is executed via 'streamlit run' in the 'heart' conda environment."""
    if os.environ.get("STREAMLIT_RUNNING_APP") == "1":
        return

    try:
        import streamlit.runtime as rt
        if rt.exists():
            return  # Successfully running inside Streamlit runner!
    except Exception:
        pass

    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return
    except Exception:
        pass

    python_exe = HEART_PYTHON if os.path.exists(HEART_PYTHON) else sys.executable
    print(f"[*] Starting Multi-Query RAG in 'heart' environment: {python_exe} -m streamlit run {__file__}")
    cmd = [python_exe, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
    env = os.environ.copy()
    env["STREAMLIT_RUNNING_APP"] = "1"
    sys.exit(subprocess.call(cmd, env=env))

_ensure_streamlit_runner()

import time
import streamlit as st

from src.pipeline import MultiQueryRAGSystem
from src.config import (
    OLLAMA_MODEL,
    EMBEDDING_MODEL_NAME,
    DEFAULT_NUM_QUERIES,
    DEFAULT_TOP_K_PER_QUERY,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_RRF_K,
    OLLAMA_NUM_CTX
)

# Page configuration
st.set_page_config(
    page_title="Multi-Query RAG Assistant",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern chatbot UI
st.markdown("""
<style>
    .chat-header {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1976D2 0%, #7B1FA2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .chat-sub {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .source-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-left: 4px solid #1976D2;
        font-size: 0.9rem;
    }
    .query-tag {
        display: inline-block;
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stat-pill {
        display: inline-block;
        background-color: #EDE7F6;
        color: #4A148C;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_system():
    return MultiQueryRAGSystem()


def main():
    rag = get_rag_system()
    db_ready = rag.is_vector_db_ready()

    # Sidebar: Configurations & Controls
    with st.sidebar:
        st.markdown("### ⚙️ System Settings")
        
        # Connection status
        if db_ready:
            st.success(f"🟢 ChromaDB Connected ({rag.retriever.collection.count():,} docs)")
        else:
            st.error("🔴 Vector Database not found!")

        st.caption(f"**LLM:** `{OLLAMA_MODEL.split('/')[-1]}`")
        st.caption(f"**Embedder:** `{EMBEDDING_MODEL_NAME.split('/')[-1]}`")

        st.markdown("---")
        st.markdown("### 🔍 Retrieval Parameters")
        num_queries = st.slider("Generated Search Queries (N)", min_value=2, max_value=6, value=DEFAULT_NUM_QUERIES)
        top_k_per_query = st.slider("Candidates per Query (K)", min_value=2, max_value=10, value=DEFAULT_TOP_K_PER_QUERY)
        final_top_k = st.slider("Final Passages for Answering", min_value=2, max_value=10, value=DEFAULT_FINAL_TOP_K)
        rrf_k = st.slider("RRF Smoothing Factor (k)", min_value=10, max_value=100, value=DEFAULT_RRF_K, step=5)

        st.markdown("---")
        st.markdown("### 🛠️ Options")
        run_judge = st.checkbox("Grade with LLM-as-a-Judge", value=False, help="Evaluates faithfulness, context utilization, and relevance.")
        show_sources = st.checkbox("Show Multi-Query Details & Sources", value=True, help="Expands query angles and retrieved literature under each answer.")

        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("### 💡 Try an Example")
        sample_queries = [
            "0-dimensional biomaterials show inductive properties.",
            "How does microRNA dysregulation influence tumor metastasis?",
            "What are the advantages of transformer models in deep learning?"
        ]
        for sq in sample_queries:
            if st.button(f"📌 {sq[:38]}...", help=sq, use_container_width=True):
                st.session_state.pending_query = sq
                st.rerun()

    # Main Chat View
    st.markdown('<div class="chat-header">🧬 Multi-Query RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-sub">Ask any research question. The assistant generates multiple orthogonal queries, searches ChromaDB, fuses results with RRF, and synthesizes a strictly grounded answer.</div>', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your **Multi-Query RAG Assistant**. Ask me any scientific or technical question. I will search across multiple query angles in ChromaDB and synthesize an evidence-grounded answer with source citations."
            }
        ]

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧬"):
            st.markdown(msg["content"])
            
            # If there are retrieval details attached to assistant message, show them in expander
            if "details" in msg and show_sources:
                d = msg["details"]
                with st.expander("🔍 Multi-Query Insights & Retrieved Sources", expanded=False):
                    st.markdown("**Generated Search Angles:**")
                    for q in d.get("generated_queries", []):
                        st.markdown(f'<span class="query-tag">🔎 {q}</span>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    red = d.get("redundancy_metrics", {})
                    if red:
                        st.markdown(
                            f'<span class="stat-pill">Pooled: {red.get("total_candidates_retrieved")}</span>'
                            f'<span class="stat-pill">Unique: {red.get("unique_documents_found")}</span>'
                            f'<span class="stat-pill">Duplicates Filtered: {red.get("redundant_documents_filtered")} ({red.get("redundancy_rate_pct")}%)</span>'
                            f'<span class="stat-pill">Expansion: {red.get("search_expansion_ratio")}x</span>',
                            unsafe_allow_html=True
                        )

                    st.markdown("**Retrieved Passages:**")
                    for p in d.get("retrieved_documents", [])[:final_top_k]:
                        doc_id = p.get("id", "N/A")
                        title = p.get("title", "Untitled Document")
                        snippet = p.get("text", "")[:280]
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>[Document {doc_id}] {title}</strong><br>
                            <span style="color: #444;">{snippet}...</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # If judge scores exist
                    if "judge" in d and d["judge"]:
                        j = d["judge"]
                        st.markdown("---")
                        st.markdown("**⚖️ LLM-as-a-Judge Evaluation:**")
                        if "faithfulness_eval" in j:
                            fe = j["faithfulness_eval"]
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Faithfulness", f"{fe.get('faithfulness_score', 'N/A')}/5.0")
                            c2.metric("Utilization", f"{fe.get('utilization_score', 'N/A')}/5.0")
                            c3.metric("Relevance", f"{fe.get('relevance_score', 'N/A')}/5.0")
                            if "verdict" in fe:
                                st.caption(f"**Verdict:** {fe['verdict']}")

    # Handle incoming query (from chat input or clicked sample button)
    user_input = st.chat_input("Ask a scientific question...")
    if "pending_query" in st.session_state and st.session_state.pending_query:
        user_input = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Generate assistant response
        with st.chat_message("assistant", avatar="🧬"):
            with st.spinner("Transforming query & retrieving from ChromaDB..."):
                start_t = time.time()
                
                # Update pipeline deduplicator RRF k dynamically if changed
                rag.deduplicator.rrf_k = rrf_k

                # Run Multi-Query retrieval and grounded answer generation
                res = rag.run_multi_query(
                    user_query=user_input,
                    num_queries=num_queries,
                    top_k_per_query=top_k_per_query,
                    final_top_k=final_top_k,
                    fusion_strategy="rrf",
                    generate_answer=True
                )

                answer_text = res.get("answer", "No response generated.")
                st.markdown(answer_text)

                # Optional LLM Judge evaluation
                judge_res = {}
                if run_judge:
                    with st.spinner("Running LLM-as-a-Judge quality evaluation..."):
                        judge_res = {
                            "faithfulness_eval": rag.judge.evaluate_groundedness(
                                question=user_input,
                                answer=answer_text,
                                contexts=res.get("retrieved_documents", [])
                            )
                        }

                # Attach details
                details_data = {
                    "generated_queries": res.get("generated_queries", []),
                    "redundancy_metrics": res.get("redundancy_metrics", {}),
                    "retrieved_documents": res.get("retrieved_documents", []),
                    "citations": res.get("citations", []),
                    "judge": judge_res
                }

                # Render details in expander
                if show_sources:
                    with st.expander("🔍 Multi-Query Insights & Retrieved Sources", expanded=False):
                        st.markdown("**Generated Search Angles:**")
                        for q in res.get("generated_queries", []):
                            st.markdown(f'<span class="query-tag">🔎 {q}</span>', unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        red = res.get("redundancy_metrics", {})
                        if red:
                            st.markdown(
                                f'<span class="stat-pill">Pooled: {red.get("total_candidates_retrieved")}</span>'
                                f'<span class="stat-pill">Unique: {red.get("unique_documents_found")}</span>'
                                f'<span class="stat-pill">Duplicates Filtered: {red.get("redundant_documents_filtered")} ({red.get("redundancy_rate_pct")}%)</span>'
                                f'<span class="stat-pill">Expansion: {red.get("search_expansion_ratio")}x</span>',
                                unsafe_allow_html=True
                            )

                        st.markdown("**Retrieved Passages:**")
                        for p in res.get("retrieved_documents", [])[:final_top_k]:
                            doc_id = p.get("id", "N/A")
                            title = p.get("title", "Untitled Document")
                            snippet = p.get("text", "")[:280]
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>[Document {doc_id}] {title}</strong><br>
                                <span style="color: #444;">{snippet}...</span>
                            </div>
                            """, unsafe_allow_html=True)

                        if judge_res and "faithfulness_eval" in judge_res:
                            fe = judge_res["faithfulness_eval"]
                            st.markdown("---")
                            st.markdown("**⚖️ LLM-as-a-Judge Evaluation:**")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Faithfulness", f"{fe.get('faithfulness_score', 'N/A')}/5.0")
                            c2.metric("Utilization", f"{fe.get('utilization_score', 'N/A')}/5.0")
                            c3.metric("Relevance", f"{fe.get('relevance_score', 'N/A')}/5.0")
                            if "verdict" in fe:
                                st.caption(f"**Verdict:** {fe['verdict']}")

                # Save to session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "details": details_data
                })


if __name__ == "__main__":
    main()
