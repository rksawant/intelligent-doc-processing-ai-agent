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
from services import BedrockService, OpenSearchService, S3Service
from config.config import rag_config, document_config

logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG system for document knowledge retrieval"""
    
    def __init__(self):
        self.bedrock_service = BedrockService()
        self.opensearch_service = OpenSearchService()
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
    
    def index_document(self, document_id: str, text_content: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Index a document for RAG retrieval
        
        Args:
            document_id: Unique document identifier
            text_content: Document text content
            metadata: Optional document metadata
            
        Returns:
            Indexing result
        """
        try:
            # Split text into chunks
            chunks = self._split_text(text_content)
            
            # Generate embeddings for each chunk
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings = self.bedrock_service.generate_embeddings(chunk_texts)
            
            # Index each chunk
            indexed_chunks = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document_id}_chunk_{i}"
                
                chunk_metadata = {
                    'document_id': document_id,
                    'chunk_index': i,
                    'chunk_count': len(chunks),
                    **(metadata or {})
                }
                
                success = self.opensearch_service.index_document(
                    chunk_id,
                    chunk.page_content,
                    embedding,
                    chunk_metadata
                )
                
                if success:
                    indexed_chunks.append({
                        'chunk_id': chunk_id,
                        'chunk_index': i,
                        'content_length': len(chunk.page_content)
                    })
            
            return {
                'success': True,
                'document_id': document_id,
                'total_chunks': len(chunks),
                'indexed_chunks': len(indexed_chunks),
                'chunks': indexed_chunks
            }
            
        except Exception as e:
            logger.error(f"Error indexing document {document_id}: {e}")
            return {'error': str(e)}
    
    def search_documents(self, query: str, top_k: Optional[int] = None, 
                        filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search documents using RAG
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply
            
        Returns:
            Search results
        """
        try:
            top_k = top_k or self.top_k_results
            
            # Generate query embedding
            query_embedding = self.bedrock_service.generate_embeddings([query])[0]
            
            # Search for similar chunks
            similar_chunks = self.opensearch_service.search_similar(
                query_embedding,
                top_k,
                filters
            )
            
            # Filter by similarity threshold
            filtered_chunks = [
                chunk for chunk in similar_chunks 
                if chunk.get('score', 0) >= self.similarity_threshold
            ]
            
            # Group chunks by document
            document_chunks = self._group_chunks_by_document(filtered_chunks)
            
            return {
                'success': True,
                'query': query,
                'total_results': len(filtered_chunks),
                'document_results': len(document_chunks),
                'results': document_chunks,
                'similarity_threshold': self.similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return {'error': str(e)}
    
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
            results = self.opensearch_service.hybrid_search(
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
    
    def get_document_context(self, document_id: str, query: str, 
                           top_k: int = 3) -> Dict[str, Any]:
        """
        Get relevant context from a specific document
        
        Args:
            document_id: Document ID
            query: Search query
            top_k: Number of chunks to return
            
        Returns:
            Document context
        """
        try:
            # Search within specific document
            filters = {'document_id': document_id}
            search_results = self.search_documents(query, top_k, filters)
            
            if 'error' in search_results:
                return search_results
            
            # Extract context chunks
            context_chunks = []
            for doc_result in search_results['results']:
                if doc_result['document_id'] == document_id:
                    context_chunks = doc_result['chunks']
                    break
            
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
            stats = self.opensearch_service.get_index_stats()
            
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
        Group chunks by document ID
        
        Args:
            chunks: List of chunk results
            
        Returns:
            List of document results with grouped chunks
        """
        try:
            document_groups = {}
            
            for chunk in chunks:
                doc_id = chunk['metadata'].get('document_id', 'unknown')
                
                if doc_id not in document_groups:
                    document_groups[doc_id] = {
                        'document_id': doc_id,
                        'chunks': [],
                        'metadata': chunk['metadata']
                    }
                
                document_groups[doc_id]['chunks'].append({
                    'chunk_id': chunk['document_id'],
                    'content': chunk['content'],
                    'score': chunk['score'],
                    'metadata': chunk['metadata']
                })
            
            # Sort chunks within each document by score
            for doc_id in document_groups:
                document_groups[doc_id]['chunks'].sort(
                    key=lambda x: x['score'], reverse=True
                )
            
            return list(document_groups.values())
            
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
            search_results = self.opensearch_service.search_similar(
                [0.0] * self.vector_dimension,  # Dummy embedding
                1000,  # Large number to get all chunks
                filters
            )
            
            deleted_count = 0
            for chunk in search_results:
                chunk_id = chunk['document_id']
                if self.opensearch_service.delete_document(chunk_id):
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
            success = self.opensearch_service.create_index(self.vector_dimension)
            
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
