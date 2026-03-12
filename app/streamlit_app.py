import os
import tempfile

import streamlit as st

from rag import RAGSystem

st.set_page_config(page_title="Simple RAG", page_icon="🔍", layout="wide")

st.title("📄 Simple RAG — Document Q&A")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder=os.getenv("OPENAI_API_KEY", "placeholder"),
    )

    st.divider()
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported: PDF, TXT, DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if st.button("🗑️ Clear All Documents", use_container_width=True):
        if "rag" in st.session_state:
            st.session_state.rag.clear()
        st.session_state.processed_files = []
        st.session_state.messages = []
        st.success("All documents cleared.")

# ── Session state init ────────────────────────────────────────────────────────
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Init / reinit RAG when key changes ───────────────────────────────────────
if api_key:
    if "rag" not in st.session_state or st.session_state.get("_api_key") != api_key:
        st.session_state.rag = RAGSystem(api_key)
        st.session_state._api_key = api_key
        st.session_state.processed_files = []
        st.session_state.messages = []

# ── Process newly uploaded files ─────────────────────────────────────────────
if uploaded_files and api_key:
    for file in uploaded_files:
        if file.name not in st.session_state.processed_files:
            ext = file.name.rsplit(".", 1)[-1].lower()
            with st.sidebar:
                with st.spinner(f"Processing {file.name}…"):
                    suffix = f".{ext}"
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    try:
                        docs = st.session_state.rag.load_document(tmp_path, ext)
                        st.session_state.rag.process_documents(docs)
                        st.session_state.processed_files.append(file.name)
                        st.success(f"✅ {file.name}")
                    except Exception as exc:
                        st.error(f"❌ {file.name}: {exc}")
                    finally:
                        os.unlink(tmp_path)

# ── Show processed file list ──────────────────────────────────────────────────
if st.session_state.processed_files:
    with st.sidebar:
        st.divider()
        st.header("📋 Loaded Documents")
        for name in st.session_state.processed_files:
            st.write(f"• {name}")

# ── Chat UI ───────────────────────────────────────────────────────────────────
st.header("💬 Ask Questions")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something about your documents…"):
    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar.")
    elif not st.session_state.processed_files:
        st.warning("Upload and process at least one document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                try:
                    result = st.session_state.rag.query(question)
                    answer = result["answer"]
                    st.markdown(answer)

                    if result["sources"]:
                        with st.expander("📎 Source excerpts"):
                            for i, doc in enumerate(result["sources"], 1):
                                src = doc.metadata.get("source", "unknown")
                                page = doc.metadata.get("page", "")
                                label = f"**[{i}] {os.path.basename(src)}"
                                if page != "":
                                    label += f" — page {page + 1}"
                                label += "**"
                                st.markdown(label)
                                st.caption(doc.page_content[:400] + "…")
                                st.divider()

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as exc:
                    err = f"Error: {exc}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
