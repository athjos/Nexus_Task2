"""Smart Document Knowledge Assistant - Streamlit web app (RAG over uploaded PDFs / TXT files).

Run:  streamlit run app.py
"""
import os

import streamlit as st

from rag import (
    NO_ANSWER,
    KnowledgeBase,
    generate_answer,
    retrieval_query,
)

st.set_page_config(page_title="Document Knowledge Assistant", page_icon="📚", layout="wide")


def get_secret(name: str) -> str:
    """Read a key from Streamlit secrets (deployed) or environment variables (local)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml locally
        pass
    return os.getenv(name, "")


# provider label -> (backend, secret name, default model, base_url)
PROVIDERS = {
    "Groq (free tier)": ("openai", "GROQ_API_KEY", "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1"),
    "Google Gemini (free tier)": ("openai", "GEMINI_API_KEY", "gemini-2.5-flash",
                                  "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "OpenRouter (free models)": ("openai", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free",
                                 "https://openrouter.ai/api/v1"),
    "Anthropic (Claude)": ("anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001", ""),
    "OpenAI": ("openai", "OPENAI_API_KEY", "gpt-4o-mini", ""),
    "No LLM (extractive demo)": ("none", "", "", ""),
}


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
if "kb" not in st.session_state:
    # Each browser session gets its own knowledge base (the embedding model is loaded once and shared).
    with st.spinner("Loading embedding model (first run only)..."):
        st.session_state.kb = KnowledgeBase()
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "sources"?}
if "seen_files" not in st.session_state:
    st.session_state.seen_files = set()

kb: KnowledgeBase = st.session_state.kb

# --------------------------------------------------------------------------- #
# Sidebar: settings + documents
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📚 Knowledge Assistant")
    st.caption("Ask questions answered strictly from your own files.")

    st.subheader("1. Upload documents")
    uploads = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True, key="uploader"
    )
    for f in uploads or []:
        fid = f"{f.name}-{f.size}"
        if fid in st.session_state.seen_files:
            continue
        st.session_state.seen_files.add(fid)
        try:
            with st.spinner(f"Processing {f.name}..."):
                n = kb.add_document(f.name, f.getvalue())
            st.toast(f"{f.name}: {n} chunks indexed", icon="✅")
        except Exception as e:  # unreadable / empty / scanned file
            st.error(f"Could not process {f.name}: {e}")

    st.subheader("2. Your documents")
    if not kb.docs:
        st.info("No documents yet.")
    for name, d in list(kb.docs.items()):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{name}**  \n{len(d['chunks'])} chunks")
        if c2.button("🗑", key=f"rm-{name}", help=f"Remove {name}"):
            kb.remove_document(name)
            st.rerun()

    st.subheader("3. Settings")
    top_k = st.slider("Snippets to retrieve (top-k)", 3, 5, 4)
    labels = list(PROVIDERS)
    default_idx = next((i for i, l in enumerate(labels) if PROVIDERS[l][1] and get_secret(PROVIDERS[l][1])), len(labels) - 1)
    provider_label = st.selectbox("Answer generator", labels, index=default_idx)
    provider, secret_name, default_model, default_url = PROVIDERS[provider_label]
    api_key, model, base_url = "", default_model, default_url
    if provider != "none":
        api_key = get_secret(secret_name)
        if api_key:
            st.caption(f"🔑 {secret_name} loaded from secrets")
        else:
            api_key = st.text_input(f"{provider_label.split(' (')[0]} API key", type="password")
        model = st.text_input("Model", value=default_model)
        if provider == "openai":
            base_url = st.text_input("Base URL", value=default_url, help="Leave as is for the selected provider")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Retrieval: {'semantic embeddings' if kb.kind == 'sbert' else 'TF-IDF (install sentence-transformers for semantic search)'}")

# --------------------------------------------------------------------------- #
# Main: chat
# --------------------------------------------------------------------------- #
st.header("Ask your documents")


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"📎 Sources ({len(sources)} snippets)"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['label']}** · similarity {s['score']:.2f}")
            st.markdown(f"> {s['text']}")


if not kb.docs:
    st.info("👈 Upload one or more PDF/TXT files in the sidebar to get started.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

question = st.chat_input("Ask a question about your documents...", disabled=not kb.docs)
if question:
    history = list(st.session_state.messages)  # before adding the new question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            hits = kb.search(retrieval_query(question, history), k=top_k)

        if not kb.is_relevant(hits):
            answer, sources = NO_ANSWER, []
            st.markdown(answer)
        else:
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(question, hits, history, provider, api_key, model, base_url)
                except Exception as e:
                    st.warning(f"LLM call failed ({type(e).__name__}: {e}). Showing extractive answer instead.")
                    answer = generate_answer(question, hits, history, "none")
            sources = [{"label": c.label, "score": s, "text": c.text} for c, s in hits]
            st.markdown(answer)
            show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
