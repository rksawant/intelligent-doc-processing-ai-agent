"""
RAG (Retrieval-Augmented Generation) System for document knowledge retrieval
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# AWS services
from services import BedrockService, PineconeService, S3Service
from config.config import rag_config, document_config

logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG system for document knowledge retrieval"""
    
    def __init__(self):
        self.bedrock_service = BedrockService()
        self.pinecone_service = PineconeService()
        self.s3_service = S3Service()
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=document_config.chunk_size,
            chunk_overlap=document_config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # RAG configuration
        self.vector_dimension = rag_config.vector_dimension
        self.similarity_threshold = rag_config.similarity_threshold
        self.max_context_length = rag_config.max_context_length
        self.top_k_results = rag_config.top_k_results
    
    def index_document(self, document_id: str, text_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Index a document by splitting into chunks and storing embeddings in Pinecone
        """
        try:
            # Step 1: Split into Document chunks
            chunks = self._split_text(text_content)
            logger.info(f"Indexing {len(chunks)} chunks for document {document_id}")

            # Step 2: Extract clean text list for embedding
            texts = [chunk.page_content for chunk in chunks]

            # Step 3: Generate embeddings from Bedrock
            embeddings = self.bedrock_service.generate_embeddings(texts)

            # Step 4: Prepare Pinecone vectors (include chunk text in metadata)
            vectors = []
            for i, chunk in enumerate(chunks):
                vectors.append({
                    "id": f"{document_id}_chunk_{i}",
                    "values": embeddings[i],
                    "metadata": {
                        "document_id": document_id,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                        "text": chunk.page_content,  # ✅ actual content text
                        **(metadata or {}),
                    }
                })

            # Step 5: Upsert into Pinecone
            self.pinecone_service.upsert_vectors(vectors)

            logger.info(f"✅ Indexed {len(chunks)} chunks for {document_id}")
            return {
                "success": True,
                "document_id": document_id,
                "chunk_count": len(chunks),
                "message": f"Document {document_id} indexed successfully"
            }

        except Exception as e:
            logger.error(f"❌ Error indexing document {document_id}: {e}")
            return {"error": str(e)}


    
    # def search_documents(self, query: str, top_k: Optional[int] = None, 
    #                     filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    #     """
    #     Search documents using RAG
        
    #     Args:
    #         query: Search query
    #         top_k: Number of results to return
    #         filters: Optional filters to apply
            
    #     Returns:
    #         Search results
    #     """
    #     try:
    #         top_k = top_k or self.top_k_results
            
    #         # Generate query embedding
    #         query_embedding = self.bedrock_service.generate_embeddings([query])[0]
            
    #         # Search for similar chunks
    #         search_response = self.pinecone_service.search_similar(
    #             query_embedding,
    #             top_k,
    #             filters
    #         )

    #         # Unwrap results
    #         if isinstance(search_response, dict) and "results" in search_response:
    #             similar_chunks = search_response["results"]
    #         else:
    #             similar_chunks = search_response

            
    #         # Filter by similarity threshold
    #         filtered_chunks = [
    #             chunk for chunk in similar_chunks 
    #             if chunk.get('score', 0) >= self.similarity_threshold
    #         ]
            
    #         # Group chunks by document
    #         document_chunks = self._group_chunks_by_document(filtered_chunks)
            
    #         return {
    #             'success': True,
    #             'query': query,
    #             'total_results': len(filtered_chunks),
    #             'document_results': len(document_chunks),
    #             'results': document_chunks,
    #             'similarity_threshold': self.similarity_threshold
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Error searching documents: {e}")
    #         return {'error': str(e)}

    def search_documents(self, query: str, top_k: Optional[int] = None,
                     filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search documents using RAG with improved recall and Bedrock-ready context.
        """

        try:
            top_k = top_k or getattr(self, "top_k_results", 5)
            similarity_threshold = getattr(self, "similarity_threshold", 0.55)

            # Generate embedding for query
            query_embedding = self.bedrock_service.generate_embeddings([query])[0]

            # Search for similar chunks
            search_response = self.pinecone_service.search_similar(
                query_embedding,
                top_k,
                filters
            )

            # Normalize response
            if isinstance(search_response, dict) and "results" in search_response:
                similar_chunks = search_response["results"]
            else:
                similar_chunks = search_response or []

            # Standardize chunk structure
            normalized_chunks = []
            for chunk in similar_chunks:
                metadata = chunk.get("metadata", {})
                normalized_chunks.append({
                    "id": chunk.get("id", ""),
                    "score": chunk.get("score", 0.0),
                    "content": metadata.get("text", ""),
                    "metadata": metadata
                })

            # Filter by similarity threshold
            filtered_chunks = [
                c for c in normalized_chunks
                if c["score"] >= similarity_threshold
            ]

            # If nothing survives, take top 3
            if not filtered_chunks:
                filtered_chunks = sorted(normalized_chunks, key=lambda x: x["score"], reverse=True)[:3]

            # Group chunks by document
            document_chunks = self._group_chunks_by_document(filtered_chunks)

            # Concatenate all retrieved text for Bedrock context
            context_text = "\n".join(
                [c["content"] for c in filtered_chunks if c.get("content")]
            ).strip()

            return {
                "success": True,
                "query": query,
                "total_results": len(filtered_chunks),
                "document_results": len(document_chunks),
                "results": document_chunks,
                "similarity_threshold": similarity_threshold,
                "context_text": context_text  # 🧠 usable directly in Bedrock Q&A
            }

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return {"error": str(e)}

    
    def hybrid_search(self, query: str, top_k: Optional[int] = None,
                     weight: float = 0.7) -> Dict[str, Any]:
        """
        Perform hybrid search combining vector and text search
        
        Args:
            query: Search query
            top_k: Number of results to return
            weight: Weight for vector search (0-1)
            
        Returns:
            Hybrid search results
        """
        try:
            top_k = top_k or self.top_k_results
            
            # Generate query embedding
            query_embedding = self.bedrock_service.generate_embeddings([query])[0]
            
            # Perform hybrid search
            results = self.pinecone_service.hybrid_search(
                query_embedding,
                query,
                top_k,
                weight
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.get('score', 0) >= self.similarity_threshold
            ]
            
            # Group chunks by document
            document_chunks = self._group_chunks_by_document(filtered_results)
            
            return {
                'success': True,
                'query': query,
                'total_results': len(filtered_results),
                'document_results': len(document_chunks),
                'results': document_chunks,
                'search_type': 'hybrid',
                'vector_weight': weight
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return {'error': str(e)}
    
    def answer_question(self, question: str, context_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Answer a question using RAG
        
        Args:
            question: Question to answer
            context_limit: Maximum number of context chunks to use
            
        Returns:
            Answer with context
        """
        try:
            context_limit = context_limit or self.top_k_results
            
            # Search for relevant context
            search_results = self.search_documents(question, top_k=context_limit)
            
            if 'error' in search_results:
                return search_results
            
            # Build context from search results
            context_chunks = []
            for doc_result in search_results['results']:
                for chunk in doc_result['chunks']:
                    context_chunks.append(chunk['content'])
            
            # Combine context
            context = "\n\n".join(context_chunks)
            
            # Truncate context if too long
            if len(context) > self.max_context_length:
                context = context[:self.max_context_length] + "..."
            
            # Generate answer using Bedrock
            answer = self.bedrock_service.chat_with_context(question, context)
            
            return {
                'success': True,
                'question': question,
                'answer': answer,
                'context_chunks': len(context_chunks),
                'context_length': len(context),
                'sources': [
                    {
                        'document_id': doc_result['document_id'],
                        'chunk_count': len(doc_result['chunks']),
                        'metadata': doc_result.get('metadata', {})
                    }
                    for doc_result in search_results['results']
                ]
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {'error': str(e)}
    
    def get_document_context(self, document_id: str, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Get relevant context from a specific document.
        """
        try:
            filters = {'document_id': document_id}
            search_results = self.search_documents(query, top_k, filters)

            if 'error' in search_results:
                return search_results

            if not search_results.get('results'):
                logger.warning(f"No search results for document_id={document_id}")
                return {'success': True, 'document_id': document_id, 'context_chunks': [], 'chunk_count': 0}

            context_chunks = []
            for doc_result in search_results['results']:
                doc_id = doc_result.get('document_id', '')
                # ✅ Use substring match (handles chunked IDs)
                if document_id in doc_id:
                    context_chunks = doc_result.get('chunks', [])
                    break

            logger.debug(f"Context chunks found: {len(context_chunks)} for doc_id={document_id}")
            return {
                'success': True,
                'document_id': document_id,
                'query': query,
                'context_chunks': context_chunks,
                'chunk_count': len(context_chunks)
            }

        except Exception as e:
            logger.error(f"Error getting document context: {e}")
            return {'error': str(e)}

    
    def update_document_index(self, document_id: str, text_content: str,
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update document index with new content
        
        Args:
            document_id: Document ID
            text_content: Updated text content
            metadata: Updated metadata
            
        Returns:
            Update result
        """
        try:
            # Delete existing chunks for this document
            self._delete_document_chunks(document_id)
            
            # Re-index document
            return self.index_document(document_id, text_content, metadata)
            
        except Exception as e:
            logger.error(f"Error updating document index: {e}")
            return {'error': str(e)}
    
    def delete_document_index(self, document_id: str) -> Dict[str, Any]:
        """
        Delete document from index
        
        Args:
            document_id: Document ID
            
        Returns:
            Deletion result
        """
        try:
            deleted_chunks = self._delete_document_chunks(document_id)
            
            return {
                'success': True,
                'document_id': document_id,
                'deleted_chunks': deleted_chunks
            }
            
        except Exception as e:
            logger.error(f"Error deleting document index: {e}")
            return {'error': str(e)}
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get RAG index statistics
        
        Returns:
            Index statistics
        """
        try:
            stats = self.pinecone_service.get_index_stats()
            
            if stats:
                return {
                    'success': True,
                    'stats': stats
                }
            else:
                return {'error': 'Failed to get index stats'}
                
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {'error': str(e)}
    
    def _split_text(self, text: str) -> List[Document]:
        """
        Split text into chunks
        
        Args:
            text: Text to split
            
        Returns:
            LangChain Document objects
            List of text chunks as Document objects
        """
        try:
            chunks = self.text_splitter.split_text(text)
            
            # Convert to Document objects
            documents = [
                Document(page_content=chunk, metadata={'chunk_index': i})
                for i, chunk in enumerate(chunks)
            ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            return []
    
    def _group_chunks_by_document(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group chunks by document ID and extract proper content text.
        """
        try:
            document_groups: Dict[str, Dict[str, Any]] = {}

            for chunk in chunks:
                metadata = chunk.get('metadata', {}) or {}
                doc_id = metadata.get('document_id') or chunk.get('id', 'unknown')

                if doc_id not in document_groups:
                    document_groups[doc_id] = {
                        'document_id': doc_id,
                        'metadata': metadata,
                        'chunks': []
                    }

                # ✅ Extract content safely (metadata fallback)
                content = (
                    chunk.get('content') or
                    metadata.get('text') or
                    metadata.get('chunk_text') or
                    metadata.get('page_content') or
                    ''
                )

                document_groups[doc_id]['chunks'].append({
                    'chunk_id': chunk.get('id', 'unknown'),
                    'content': content.strip(),
                    'score': chunk.get('score', 0.0),
                    'metadata': metadata
                })

            # Sort chunks within each document by score
            for doc_id in document_groups:
                document_groups[doc_id]['chunks'].sort(
                    key=lambda x: x['score'], reverse=True
                )

            grouped = list(document_groups.values())
            logger.debug(f"Grouped {len(chunks)} chunks into {len(grouped)} document(s)")

            # Optional preview log
            for doc in grouped:
                top_chunk = doc['chunks'][0] if doc['chunks'] else {}
                logger.debug(f"📄 Doc: {doc['document_id']} | Top chunk preview: {top_chunk.get('content', '')[:100]}")

            return grouped

        except Exception as e:
            logger.error(f"Error grouping chunks by document: {e}")
            return []


    
    def _delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of deleted chunks
        """
        try:
            # Search for all chunks of this document
            filters = {'document_id': document_id}
            search_results = self.pinecone_service.search_similar(
                [0.0] * self.vector_dimension,  # Dummy embedding
                1000,  # Large number to get all chunks
                filters
            )
            
            deleted_count = 0
            for chunk in search_results:
                chunk_id = chunk['document_id']
                if self.pinecone_service.delete_document(chunk_id):
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            return 0
    
    def initialize_index(self) -> Dict[str, Any]:
        """
        Initialize OpenSearch index for RAG
        
        Returns:
            Initialization result
        """
        try:
            success = self.pinecone_service.create_index(self.vector_dimension)
            
            if success:
                return {
                    'success': True,
                    'message': 'RAG index initialized successfully'
                }
            else:
                return {'error': 'Failed to initialize RAG index'}
                
        except Exception as e:
            logger.error(f"Error initializing RAG index: {e}")
            return {'error': str(e)}
