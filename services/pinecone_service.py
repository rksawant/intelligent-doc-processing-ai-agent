"""
pinecone_service.py — Pinecone wrapper with auto dimension validation & recreation
Compatible with Python 3.12+ and Pinecone v5 SDK
"""

import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional

# Load .env variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "local-doc-test")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in environment variables")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


class PineconeService:
    """Pinecone vector database service with auto index management"""

    def __init__(self):
        self.index_name = PINECONE_INDEX_NAME
        self.region = PINECONE_REGION
        self.index = None
        self.ensure_index_exists()

    def ensure_index_exists(self, dimension: Optional[int] = None):
        """Check or create Pinecone index with proper dimension"""
        existing_indexes = [i.name for i in pc.list_indexes()]

        if self.index_name in existing_indexes:
            logger.info(f"✅ Using existing Pinecone index: {self.index_name}")
            self.index = pc.Index(self.index_name)
        else:
            if dimension is None:
                dimension = 1536  # Default safe fallback (e.g., OpenAI ada-002)
            logger.info(f"🪣 Creating Pinecone index '{self.index_name}' (dim={dimension})...")
            pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.region)
            )
            self.index = pc.Index(self.index_name)
            logger.info(f"✅ Index '{self.index_name}' created successfully.")

    def validate_index_dimension(self, embedding_dim: int):
        """Ensure Pinecone index matches the given embedding dimension"""
        try:
            stats = self.index.describe_index_stats()
            index_dim = stats.get("dimension", None)

            if index_dim is not None and index_dim != embedding_dim:
                logger.warning(
                    f"⚠️ Index dimension mismatch: index={index_dim}, embedding={embedding_dim}. "
                    f"Recreating index '{self.index_name}'..."
                )
                pc.delete_index(self.index_name)
                self.ensure_index_exists(dimension=embedding_dim)
                return True
        except Exception as e:
            logger.error(f"Error validating Pinecone index dimension: {e}")
        return False

    def index_document(self, document_id: str, content: str, embedding: List[float],
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Index a document vector with metadata
        """
        try:
            if not self.index:
                self.ensure_index_exists(len(embedding))

            # Verify or recreate index if dimension mismatched
            self.validate_index_dimension(len(embedding))

            vector = (document_id, embedding, metadata or {"content": content})
            self.index.upsert(vectors=[vector])
            logger.info(f"✅ Indexed document {document_id} successfully.")
            return True

        except Exception as e:
            logger.error(f"Error indexing document {document_id}: {e}")
            return False

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """
        Batch upsert multiple vectors into the Pinecone index.

        Each vector must be a dict with 'id', 'values', and 'metadata' keys.
        Compatible with Pinecone v5 SDK.
        """
        try:
            if not self.index:
                raise RuntimeError("Pinecone index not initialized.")

            if not vectors:
                logger.warning("⚠️ No vectors provided for upsert.")
                return False

            # Validate dimensions
            dim = len(vectors[0].get("values", []))
            self.validate_index_dimension(dim)

            # Convert dicts into Pinecone tuples (id, values, metadata)
            pinecone_vectors = [
                (v["id"], v["values"], v.get("metadata", {})) for v in vectors
            ]

            # Upsert batch
            self.index.upsert(vectors=pinecone_vectors)
            logger.info(f"✅ Upserted {len(pinecone_vectors)} vectors into Pinecone index.")
            return True

        except Exception as e:
            logger.error(f"❌ Error upserting vectors to Pinecone: {e}")
            return False


    def search_similar(self, query: str, top_k: int = 5, filters: Optional[dict] = None):
        """
        Search for documents similar to the given query vector in Pinecone.

        Args:
            query: Text or embedding vector to search for
            top_k: Number of results to return
            filters: Optional metadata filters
        """
        try:
            # Convert text to embedding if needed
            if isinstance(query, str):
                query_embedding = self.embed_text(query)
            else:
                query_embedding = query

            # Perform similarity search
            if filters:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    filter=filters,
                    include_metadata=True
                )
            else:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )

            return {"success": True, "results": results["matches"]}
        except Exception as e:
            return {"error": str(e)}

    def delete_document(self, document_id: str) -> bool:
        """Delete a document/vector by ID"""
        try:
            self.index.delete(ids=[document_id])
            logger.info(f"🗑️ Deleted document {document_id} successfully.")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    def get_index_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch Pinecone index stats"""
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return None

