import os
import streamlit as st
from dotenv import load_dotenv
from agents.knowledge_agent import KnowledgeAgent

# ==========================================================
# 🌍 Environment Setup
# ==========================================================
load_dotenv()
st.set_page_config(page_title="🧠 Knowledge Agent", layout="wide")
st.title("🧠 Intelligent Knowledge Retrieval System (RAG + Bedrock)")

# ==========================================================
# 🧩 Initialize Agent (cached to avoid re-instantiation)
# ==========================================================
@st.cache_resource
def init_agent():
    return KnowledgeAgent()

agent = init_agent()

# ==========================================================
# 🧭 Tabs
# ==========================================================
tab1, tab2, tab3 = st.tabs(["📂 Upload & Index", "❓ Ask Questions", "📊 Knowledge Base Stats"])

# ==========================================================
# 📂 TAB 1 - Upload & Index
# ==========================================================
with tab1:
    st.subheader("📂 Upload Document")
    st.write("Upload a document to process, extract, and index into your knowledge base.")

    uploaded_file = st.file_uploader(
        "Choose a document (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        key="file_uploader"
    )

    if uploaded_file and "last_uploaded" not in st.session_state:
        st.session_state["last_uploaded"] = uploaded_file.name
        st.info(f"📄 Processing and indexing `{uploaded_file.name}` ...")

        file_bytes = uploaded_file.read()

        result = agent.process_and_index_document_from_bytes(
            file_bytes,
            filename=uploaded_file.name,
            metadata={"source": "Streamlit Upload"}
        )

        st.session_state["upload_result"] = result

        if result.get("success"):
            st.success(result["message"])
            st.json({
                "document_id": result["document_id"],
                "context_chunks": len(result["indexing_result"].get("chunks", []))
            })
        else:
            st.error(result.get("error", "Unknown error occurred."))
    
    elif uploaded_file and st.session_state.get("last_uploaded") == uploaded_file.name:
        st.success(f"✅ `{uploaded_file.name}` already processed.")
        if "upload_result" in st.session_state:
            st.json({
                "document_id": st.session_state["upload_result"].get("document_id"),
                "message": st.session_state["upload_result"].get("message", "")
            })

    st.divider()

    # 🔄 Reset button
    if st.button("🔄 Reset Upload State"):
        st.session_state.pop("last_uploaded", None)
        st.session_state.pop("upload_result", None)
        st.experimental_rerun()

# ==========================================================
# ❓ TAB 2 - Ask Questions (RAG Q&A)
# ==========================================================
with tab2:
    st.subheader("❓ Ask a Question")
    st.write("Ask questions about your indexed documents — answers are generated using RAG + AWS Bedrock.")

    question = st.text_input("Enter your question:")
    col1, col2 = st.columns(2)
    with col1:
        context_limit = st.number_input("Context limit (chunks)", 1, 10, 5)
    with col2:
        doc_filter = st.text_input("Document ID (optional):")

    if st.button("Ask Question"):
        if not question.strip():
            st.warning("Please enter a question before asking.")
        else:
            with st.spinner("💬 Retrieving answer..."):
                response = agent.ask_question(question, context_limit, doc_filter)

            if response.get("success"):
                st.markdown("### 🧩 **Answer**")
                st.write(response["answer"])
                st.caption(f"Context chunks used: {response.get('context_chunks', 'N/A')}")
                if "sources" in response:
                    st.caption(f"Sources: {response['sources']}")
            else:
                st.error(response.get("error", "Failed to retrieve answer."))

    # 💡 Related question suggestions
    if question.strip():
        if st.button("💡 Suggest Related Questions"):
            with st.spinner("🧠 Generating suggestions..."):
                suggestions = agent.suggest_related_questions(question)

            if suggestions.get("success"):
                st.markdown("#### 🔁 Related Questions")
                for q in suggestions["suggestions"]:
                    st.markdown(f"- {q}")
            else:
                st.warning(suggestions.get("error", "No suggestions available."))

# ==========================================================
# 📊 TAB 3 - Knowledge Base Stats
# ==========================================================
with tab3:
    st.subheader("📊 Knowledge Base Overview")
    st.write("Monitor your indexed documents and Pinecone vector stats.")

    if st.button("📈 Refresh Knowledge Base Stats"):
        with st.spinner("Fetching knowledge base statistics..."):
            stats = agent.get_knowledge_base_stats()

        if stats.get("success"):
            st.success("✅ Knowledge Base Stats:")
            st.json({
                "Total Documents": stats["total_documents"],
                "Vector Index Stats": stats["rag_stats"],
                "Processed Documents": stats["processed_documents"]
            })
        else:
            st.error(stats.get("error", "Unable to fetch stats."))

# ==========================================================
# 🧩 Footer
# ==========================================================
st.markdown("---")
st.caption("🚀 Powered by KnowledgeAgent • AWS Bedrock • Pinecone • Streamlit")
