import os

import streamlit as st

from retrieval.ingest import process_pdfs, DATA_PATH
from retrieval.embed import embed_documents
from retrieval.query_rewriter import rewrite_query
from retrieval.rag import (
    infer_metadata_filter,
    retrieve_bilingual,
    build_prompt,
    generate_typhoon_answer,
    generate_gemini_answer,
    is_thai,
)
from retrieval.rerank import rerank


st.set_page_config(page_title="UniGraph RAG", layout="wide")


# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("UniGraph")
section = st.sidebar.radio(
    "Navigate",
    ["💬 Chat", "📄 Documents"],
    index=0,
)

st.title("UniGraph RAG")
st.caption("Multilingual academic assistant for KMITL")


def show_documents_page() -> None:
    """Documents page – upload PDFs and build the index."""

    st.subheader("Documents")
    st.write(
        "Upload one or more PDF files. They will be saved into "
        "`data/raw_pdfs/`, chunked, and indexed into ChromaDB."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Ingest & Index", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
            return

        os.makedirs(DATA_PATH, exist_ok=True)

        with st.spinner("Saving PDFs..."):
            for f in uploaded_files:
                out_path = os.path.join(DATA_PATH, f.name)
                with open(out_path, "wb") as out_f:
                    out_f.write(f.read())

        st.success(f"Saved {len(uploaded_files)} PDF(s) to data/raw_pdfs/.")

        try:
            with st.spinner(
                "Scanning PDFs, extracting chunks, and preparing pending changes..."
            ):
                process_pdfs()

            with st.spinner("Embedding pending changes into ChromaDB..."):
                embed_documents()

            st.success(
                "Index updated successfully. You can now chat with the AI in the Chat view."
            )
        except Exception as e:
            st.error(f"Ingestion/indexing failed: {e}")


def show_chat_page() -> None:
    """Chat page – conversational interface backed by the RAG pipeline."""

    st.subheader("Chat")
    st.write(
        "Ask questions about the ingested documents. "
        "Thai questions are routed to Typhoon; English to Gemini."
    )

    # Simple chat-like UI using session_state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for role, content in st.session_state["messages"]:
        if role == "user":
            st.chat_message("user").markdown(content)
        else:
            st.chat_message("assistant").markdown(content)

    user_input = st.chat_input("Type your question...")

    if user_input:
        q = user_input.strip()
        if not q:
            return

        st.session_state["messages"].append(("user", q))
        st.chat_message("user").markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                try:
                    # Agent 1 – bilingual rewrite (EN + TH search queries)
                    en_q, th_q = rewrite_query(q)

                    # Metadata filter based on inferred document_type (never language)
                    where_filter = infer_metadata_filter(q)

                    # Bilingual retrieval + merge
                    docs, sources = retrieve_bilingual(
                        en_q, th_q, where_filter, top_k=10
                    )

                    # Multilingual reranking with the original query
                    docs, sources = rerank(q, docs, sources, top_k=3)

                    if not docs:
                        answer = (
                            "No relevant chunks found. Try uploading/ingesting documents "
                            "in the Documents view first, or ask a different question."
                        )
                    else:
                        # Build prompt using enriched metadata-aware context
                        is_thai_query = is_thai(q)
                        prompt = build_prompt(q, docs, sources)
                        if is_thai_query:
                            answer = generate_typhoon_answer(prompt)
                        else:
                            answer = generate_gemini_answer(prompt)
                except Exception as e:
                    answer = f"Retrieval or generation failed: {e}"

                st.markdown(answer)
                st.session_state["messages"].append(("assistant", answer))


if section == "📄 Documents":
    show_documents_page()
else:
    show_chat_page()
