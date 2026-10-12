import time
import streamlit as st
import httpx
from typing import List, Dict, Any

# API settings
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="InsightEngine | AI Document Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling follows DESIGN.md
st.markdown("""
    <style>
    :root {
        --canvas: #0a0b0d; --surface-1: #101215; --surface-2: #15181c;
        --hairline: #22262c; --hairline-strong: #323740;
        --ink: #eef0f2; --ink-muted: #b4bac3; --ink-subtle: #8a919c;
        --accent: #3ecf8e; --on-accent: #04130c; --warn: #f2b95c;
        --mono: "Geist Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
    }
    html, body, [class*="css"], .stApp, .main {
        font-family: "Geist", system-ui, -apple-system, "Segoe UI", sans-serif;
        background-color: var(--canvas);
        color: var(--ink);
    }
    [data-testid="stSidebar"] { background-color: var(--surface-1); border-right: 1px solid var(--hairline); }
    .title-container {
        padding: 1.75rem 0 1.5rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--hairline);
    }
    .app-title {
        font-size: 2rem; font-weight: 600; color: var(--ink);
        letter-spacing: -0.03em; line-height: 1.15; margin: 0;
    }
    .app-subtitle { font-size: 0.95rem; color: var(--ink-subtle); margin-top: 0.4rem; max-width: 62ch; }
    .tag-web, .tag-local {
        padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 500;
        display: inline-block; margin-bottom: 12px; border: 1px solid var(--hairline);
    }
    .tag-web { background: #231c0e; color: var(--warn); }
    .tag-local { background: #0e2019; color: var(--accent); }
    .passage-card {
        background: var(--surface-1); border: 1px solid var(--hairline);
        border-left: 2px solid var(--accent); padding: 14px; border-radius: 12px; margin-bottom: 12px;
    }
    .passage-header { font-family: var(--mono); font-weight: 500; color: var(--ink); font-size: 0.8rem; margin-bottom: 6px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid var(--hairline); }
    .stTabs [data-baseweb="tab"] { height: 44px; background: transparent; color: var(--ink-subtle); font-size: 0.95rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: var(--ink) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 8px; border: 1px solid var(--hairline-strong);
        background: var(--surface-1); color: var(--ink); font-weight: 500;
        transition: transform 150ms ease, background-color 150ms ease, border-color 150ms ease;
    }
    .stButton > button:hover { background: var(--surface-2); border-color: var(--ink-subtle); color: var(--ink); }
    .stButton > button:active { transform: translateY(1px); }
    .stButton > button[kind="primary"] { background: var(--accent); color: var(--on-accent); border-color: var(--accent); }
    .stButton > button[kind="primary"]:hover { background: #5fdca5; border-color: #5fdca5; color: var(--on-accent); }
    .stButton > button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    [data-testid="stChatMessage"] { background: var(--surface-1); border: 1px solid var(--hairline); border-radius: 12px; }
    [data-testid="stFileUploaderDropzone"] { background: var(--surface-1); border: 1px dashed var(--hairline-strong); border-radius: 12px; }
    @media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
    </style>
""", unsafe_allow_html=True)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "registered_docs" not in st.session_state:
    st.session_state.registered_docs = []

# Fetch active documents from the backend
def refresh_documents():
    try:
        resp = httpx.get(f"{API_BASE_URL}/api/documents")
        if resp.status_code == 200:
            st.session_state.registered_docs = resp.json()
    except Exception:
        st.session_state.registered_docs = []

# Sidebar control panel
with st.sidebar:
    st.title("Configuration")
    st.caption("Customize engine characteristics.")

    selected_model = st.selectbox(
        "Embedding Generator",
        options=["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
        index=0,
        help="Model used to convert text and queries into vectors."
    )

    search_strategy = st.selectbox(
        "Retrieval Mechanism",
        options=["dense", "mmr", "hybrid", "multi_query", "reranker"],
        format_func=lambda x: {
            "dense": "Dense Semantic Vector",
            "mmr": "MMR Diversity Search",
            "hybrid": "Lexical + Dense Hybrid",
            "multi_query": "Multi-Query Expansion",
            "reranker": "Cross-Encoder Rerank"
        }[x],
        index=0,
        help="Algorithm used to query and rank document fragments."
    )

    passage_limit = st.slider(
        "Context Chunk Limit",
        min_value=1,
        max_value=20,
        value=6,
        help="Maximum chunks fed to the LLM context window."
    )

    use_web_fallback = st.toggle(
        "Corrective Web Fallback (CRAG)",
        value=True,
        help="Fallback to DuckDuckGo search if document match confidence is low."
    )

    verbosity = st.selectbox(
        "Answer Target Length",
        options=["Detailed Analysis", "Concise Summary"],
        index=0
    )

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")
        st.rerun()

    if st.button("Wipe database", use_container_width=True):
        try:
            resp = httpx.delete(f"{API_BASE_URL}/api/database/wipe?embedding_model={selected_model}", timeout=30)
            if resp.status_code == 200:
                st.session_state.chat_history = []
                st.session_state.registered_docs = []
                st.success("Successfully reset database and logs.")
                st.rerun()
            else:
                st.error("Failed to delete database indexes.")
        except Exception as e:
            st.error(f"Failed to communicate with backend: {e}")

# Header title block
st.markdown("""
    <div class="title-container">
        <div class="app-title">InsightEngine</div>
        <div class="app-subtitle">Ask questions across your PDFs and see which retrieval strategy finds the best passages.</div>
    </div>
""", unsafe_allow_html=True)

# Run initial fetch
refresh_documents()

# Primary Navigation Tabs
tab_chat, tab_ingest, tab_metrics = st.tabs(["Chat", "Upload and ingest", "Strategy evaluator"])

# Tab 1: Chat UI
with tab_chat:
    st.subheader("Interactive QA Session")
    
    # Draw existing messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "passages" in msg and msg["passages"]:
                with st.expander("Supporting context"):
                    for idx, passage in enumerate(msg["passages"], 1):
                        st.markdown(
                            f"**Passage {idx}** — *{passage['doc_id']}* (Page {passage['page_num']}) | Score: `{passage['relevance_score']:.4f}`"
                        )
                        st.info(passage["content"])

    # Accept user prompt
    if prompt := st.chat_input("Ask a question about the uploaded documents..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Executing RAG Pipeline..."):
                payload = {
                    "question": prompt,
                    "strategy": search_strategy,
                    "top_k": passage_limit,
                    "embedding_model": selected_model,
                    "chat_history": st.session_state.chat_history[:-1],
                    "enable_crag": use_web_fallback,
                    "answer_length": verbosity
                }
                
                try:
                    resp = httpx.post(f"{API_BASE_URL}/api/query", json=payload, timeout=60)
                    if resp.status_code == 200:
                        result = resp.json()
                        
                        # Display tags based on retrieval origin
                        if result.get("web_fallback_used", False):
                            st.markdown('<span class="tag-web">Web fallback used</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="tag-local">Local retrieval ({search_strategy.upper()})</span>', unsafe_allow_html=True)
                        
                        answer_text = result.get("answer", "")
                        st.markdown(answer_text)
                        
                        passages = result.get("passages", [])
                        if passages:
                            with st.expander(f"Retrieved passages ({len(passages)})"):
                                for idx, p in enumerate(passages, 1):
                                    st.markdown(
                                        f"**[{idx}] {p['doc_id']} — Page {p['page_num']}** (Score: `{p['relevance_score']:.4f}`)"
                                    )
                                    st.caption(p["content"])
                                    st.divider()
                                    
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer_text,
                            "passages": passages
                        })
                    else:
                        st.error(f"Error from QA pipeline: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with RAG service: {e}")

# Tab 2: Ingest Panel
with tab_ingest:
    st.subheader("Document Ingestion Panel")
    st.caption("Upload PDFs to parse, partition, and commit to the Qdrant instance.")

    files = st.file_uploader("Select PDF documents to process", type=["pdf"], accept_multiple_files=True)

    if files:
        if st.button("Run ingestion", type="primary"):
            for file_obj in files:
                st.write(f"### Processing: `{file_obj.name}`")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                # Send file upload request
                files_payload = {"file": (file_obj.name, file_obj.getvalue(), "application/pdf")}
                data_payload = {"embedding_model": selected_model}
                
                try:
                    resp = httpx.post(
                        f"{API_BASE_URL}/api/documents/ingest",
                        files=files_payload,
                        data=data_payload,
                        timeout=30
                    )
                    
                    if resp.status_code == 200:
                        task = resp.json()
                        task_id = task["task_id"]
                        
                        # Poll status endpoint until completed/failed
                        while True:
                            status_resp = httpx.get(f"{API_BASE_URL}/api/documents/tasks/{task_id}")
                            if status_resp.status_code == 200:
                                current_task = status_resp.json()
                                status = current_task["status"]
                                progress = current_task["progress"]
                                msg = current_task["message"]
                                
                                progress_bar.progress(progress / 100.0)
                                status_text.info(f"**Status:** {status.upper()} | {msg}")
                                
                                if status == "completed":
                                    st.success(f"Successfully processed `{file_obj.name}`!")
                                    break
                                elif status == "failed":
                                    st.error(f"Ingestion failed for `{file_obj.name}`: {current_task.get('error')}")
                                    break
                            else:
                                st.error("Failed to query task status.")
                                break
                            time.sleep(1.0)
                    else:
                        st.error(f"Failed to trigger ingestion: {resp.text}")
                except Exception as e:
                    st.error(f"Error submitting file: {e}")
            
            # Refresh doc list
            refresh_documents()
            st.rerun()

    st.divider()
    st.subheader("Registered Document Directory")
    if st.session_state.registered_docs:
        for doc in st.session_state.registered_docs:
            st.write(f"- `{doc}`")
    else:
        st.info("No active documents loaded.")

# Tab 3: Evaluator Metrics
with tab_metrics:
    st.subheader("Retrieval Strategy Analysis")
    st.caption("Run side-by-side queries to compare multiple search strategies.")

    test_input = st.text_input("Test query for inspection:", value="What are the main findings?")
    if st.button("Compare strategies"):
        with st.spinner("Fetching comparisons in parallel..."):
            strategies = [
                ("Dense Vector", "dense"),
                ("MMR Diversity", "mmr"),
                ("BM25 Hybrid", "hybrid"),
                ("Cross-Encoder Reranked", "reranker"),
            ]
            
            columns = st.columns(4)
            for idx, (label, strategy_code) in enumerate(strategies):
                with columns[idx]:
                    st.markdown(f"### {label}")
                    
                    payload = {
                        "question": test_input,
                        "strategy": strategy_code,
                        "top_k": 3,
                        "embedding_model": selected_model,
                        "chat_history": [],
                        "enable_crag": False,
                        "answer_length": "Concise Summary"
                    }
                    
                    try:
                        resp = httpx.post(f"{API_BASE_URL}/api/query", json=payload, timeout=30)
                        if resp.status_code == 200:
                            result = resp.json()
                            passages = result.get("passages", [])
                            if passages:
                                for p_idx, p in enumerate(passages, 1):
                                    st.markdown(
                                        f"<div class='passage-card'>"
                                        f"<div class='passage-header'>{p_idx}. Page {p['page_num']} (Score: {p['relevance_score']:.3f})</div>"
                                        f"<div style='font-size:0.85rem; color:#D1D5DB;'>{p['content'][:180]}...</div>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.info("No passages found.")
                        else:
                            st.error(f"Error: {resp.status_code}")
                    except Exception as e:
                        st.error(f"Error calling API: {e}")
