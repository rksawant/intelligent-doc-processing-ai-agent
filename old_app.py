import os
import sys
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Pinecone as PineconeVectorStore

# sidebar will show exactly which Python interpreter is executing
st.sidebar.text(f"Python executable: {sys.executable}")
st.sidebar.text(f"Working dir: {os.getcwd()}")

# --- Load environment variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    st.warning("⚠️ No .env file found — make sure it's in the project folder")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "local-doc-test")

st.set_page_config(page_title="📄 AI Document Search", layout="wide")
st.title("📘 AI Document Search (LangChain + Pinecone + Streamlit)")

# --- Check API key ---
if not PINECONE_API_KEY:
    st.error("❌ Missing Pinecone API Key. Please add it to your .env file.")
    st.stop()

# --- Initialize Pinecone client ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Ensure index exists ---
existing_indexes = [i.name for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    st.info(f"🪣 Creating Pinecone index '{PINECONE_INDEX_NAME}' (384-dimension)...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # matches all-MiniLM-L6-v2 embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    st.success(f"✅ Created index '{PINECONE_INDEX_NAME}'")

# --- Connect to index ---
index = pc.Index(PINECONE_INDEX_NAME)

# --- Initialize embedding model ---
embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- File uploader UI ---
st.subheader("📂 Upload a Document")
uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])

if uploaded_file is not None:
    # Save file temporarily
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # Load and process document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    else:
        loader = Docx2txtLoader(temp_path)
    documents = loader.load()

    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Create or connect vector store
    st.write("🔄 Creating embeddings and uploading to Pinecone...")
    # vectorstore = PineconeVectorStore.from_documents(docs, embedder, index_name=PINECONE_INDEX_NAME)
    

    # Get index via client
    vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embedder,
    index_name=PINECONE_INDEX_NAME
    )


    st.success("✅ Document successfully embedded and stored in Pinecone!")

# --- Search section ---
st.divider()
st.subheader("🔍 Search your knowledge base")
query = st.text_input("Enter your search query:")

if query:
    st.write("🧠 Searching Pinecone index...")
    vectorstore = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embedder)
    results = vectorstore.similarity_search(query, k=3)

    # vectorstore = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embedder)
    # results = vectorstore.similarity_search(query, k=3)

    st.subheader("📚 Top Matching Results:")
    for i, res in enumerate(results, 1):
        st.markdown(f"**Result {i}:**")
        st.write(res.page_content)
        st.divider()

st.caption("🚀 Powered by LangChain, Pinecone, and Streamlit — Python 3.12.7")
