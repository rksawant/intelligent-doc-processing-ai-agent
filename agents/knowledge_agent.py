"""
Knowledge Retrieval Agent for Q&A and document understanding
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# AWS services
from services import BedrockService, S3Service
from .rag_system import RAGSystem
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class KnowledgeAgent:
    """Knowledge retrieval agent for Q&A and document understanding"""
    
    def __init__(self):
        self.bedrock_service = BedrockService()
        self.s3_service = S3Service()
        self.rag_system = RAGSystem()
        self.document_processor = DocumentProcessor()
        
        # Agent configuration
        self.system_prompt = """You are an intelligent knowledge retrieval agent. Your role is to:
1. Answer questions based on the provided context from company documents
2. Provide accurate, helpful, and well-structured responses
3. Cite sources when possible
4. Indicate when information is not available in the provided context
5. Suggest related topics or follow-up questions when appropriate

Always prioritize accuracy and clarity in your responses."""
    
    def process_and_index_document(self, file_path: str, 
                                  metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process and index a document for knowledge retrieval
        
        Args:
            file_path: Path to document file
            metadata: Optional document metadata
            
        Returns:
            Processing and indexing result
        """
        try:
            # Process document
            processing_result = self.document_processor.process_document(file_path, metadata)
            
            if 'error' in processing_result:
                return processing_result
            
            # Index document for RAG
            indexing_result = self.rag_system.index_document(
                processing_result['document_id'],
                processing_result['text_content'],
                processing_result['metadata']
            )
            
            if 'error' in indexing_result:
                return indexing_result
            
            return {
                'success': True,
                'document_id': processing_result['document_id'],
                'processing_result': processing_result,
                'indexing_result': indexing_result,
                'message': 'Document processed and indexed successfully'
            }
            
        except Exception as e:
            logger.error(f"Error processing and indexing document: {e}")
            return {'error': str(e)}
    
    def process_and_index_document_from_bytes(self, file_bytes: bytes, filename: str,
                                            metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process and index document from bytes
        
        Args:
            file_bytes: Document bytes
            filename: Original filename
            metadata: Optional metadata
            
        Returns:
            Processing and indexing result
        """
        try:
            # Process document
            processing_result = self.document_processor.process_document_from_bytes(
                file_bytes, filename, metadata
            )
            
            if 'error' in processing_result:
                return processing_result
            
            # Index document for RAG
            indexing_result = self.rag_system.index_document(
                processing_result['document_id'],
                processing_result['text_content'],
                processing_result['metadata']
            )
            
            if 'error' in indexing_result:
                return indexing_result
            
            return {
                'success': True,
                'document_id': processing_result['document_id'],
                'processing_result': processing_result,
                'indexing_result': indexing_result,
                'message': 'Document processed and indexed successfully'
            }
            
        except Exception as e:
            logger.error(f"Error processing and indexing document from bytes: {e}")
            return {'error': str(e)}
    
    # def ask_question(self, question: str, context_limit: Optional[int] = None,
    #                 document_filter: Optional[str] = None) -> Dict[str, Any]:
    #     """
    #     Ask a question and get an answer using RAG
        
    #     Args:
    #         question: Question to ask
    #         context_limit: Maximum number of context chunks to use
    #         document_filter: Optional document ID to limit search to
            
    #     Returns:
    #         Answer with context and sources
    #     """
    #     try:
    #         # Search for relevant context
    #         if document_filter:
    #             search_result = self.rag_system.get_document_context(
    #                 document_filter, question, context_limit or 5
    #             )
                
    #             if 'error' in search_result:
    #                 return search_result
                
    #             # Build context from document-specific results
    #             context_chunks = search_result['context_chunks']
    #             context = "\n\n".join([chunk['content'] for chunk in context_chunks])
                
    #             # Generate answer
    #             answer = self.bedrock_service.chat_with_context(
    #                 question, context, self.system_prompt
    #             )
                
    #             return {
    #                 'success': True,
    #                 'question': question,
    #                 'answer': answer,
    #                 'context_chunks': len(context_chunks),
    #                 'document_id': document_filter,
    #                 'sources': [{'document_id': document_filter, 'chunk_count': len(context_chunks)}]
    #             }
    #         else:
    #             # Use general RAG search
    #             return self.rag_system.answer_question(question, context_limit)
            
    #     except Exception as e:
    #         logger.error(f"Error asking question: {e}")
    #         return {'error': str(e)}

    def ask_question(
    self,
    question: str,
    context_limit: Optional[int] = None,
    document_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer using RAG (Retrieval-Augmented Generation).

        Improvements:
        - Uses enhanced search_documents() with context_text.
        - Auto-fallback if context is empty.
        - Supports both general and document-specific Q&A.
        - Logs retrieval stats for debugging.
        """
        try:
            logger.info(f"🧠 RAG Query: {question}")

            # --- Case 1: Ask within specific document ---
            if document_filter:
                logger.debug(f"Restricting search to document: {document_filter}")

                search_result = self.rag_system.get_document_context(
                    document_filter, question, context_limit or 5
                )

                if "error" in search_result:
                    return search_result

                context_chunks = search_result.get("context_chunks", [])
                context = "\n".join(
                    [c.get("content", "") for c in context_chunks if c.get("content")]
                ).strip()

                if not context:
                    logger.warning("⚠️ No relevant context found in the specified document.")
                    return {
                        "success": True,
                        "question": question,
                        "answer": "No relevant information found in the document.",
                        "context_chunks": 0
                    }

                # Call Bedrock for contextual Q&A
                answer = self.bedrock_service.chat_with_context(
                    question=question,
                    context=context,
                    system_prompt=self.system_prompt
                )

                return {
                    "success": True,
                    "question": question,
                    "answer": answer,
                    "context_chunks": len(context_chunks),
                    "document_id": document_filter
                }

            # --- Case 2: General search across all indexed docs ---
            else:
                search_result = self.rag_system.search_documents(
                    query=question,
                    top_k=context_limit or 5
                )

                if "error" in search_result:
                    return search_result

                context_text = search_result.get("context_text", "")
                results_count = search_result.get("total_results", 0)

                if not context_text.strip():
                    logger.warning("⚠️ No relevant context retrieved from knowledge base.")
                    return {
                        "success": True,
                        "question": question,
                        "answer": "No relevant context found to answer this question.",
                        "context_chunks": 0
                    }

                # Call Bedrock using the merged context
                logger.info(f"📚 Retrieved {results_count} context chunks. Sending to Bedrock...")
                answer = self.bedrock_service.chat_with_context(
                    question=question,
                    context=context_text,
                    system_prompt=self.system_prompt
                )

                return {
                    "success": True,
                    "question": question,
                    "answer": answer,
                    "context_chunks": results_count,
                    "documents_considered": search_result.get("document_results", 0),
                    "similarity_threshold": search_result.get("similarity_threshold")
                }

        except Exception as e:
            logger.error(f"Error in ask_question: {e}")
            return {"error": str(e)}

    
    def search_documents(self, query: str, top_k: Optional[int] = None,
                        filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search documents for relevant information
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply
            
        Returns:
            Search results
        """
        try:
            return self.rag_system.search_documents(query, top_k, filters)
            
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
            return self.rag_system.hybrid_search(query, top_k, weight)
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return {'error': str(e)}
    
    def get_document_summary(self, document_id: str, max_length: int = 500) -> Dict[str, Any]:
        """
        Get summary of a specific document
        
        Args:
            document_id: Document ID
            max_length: Maximum summary length
            
        Returns:
            Document summary
        """
        try:
            return self.document_processor.get_document_summary(document_id, max_length)
            
        except Exception as e:
            logger.error(f"Error getting document summary: {e}")
            return {'error': str(e)}
    
    def extract_key_information(self, document_id: str, 
                              information_types: List[str]) -> Dict[str, Any]:
        """
        Extract key information from a document
        
        Args:
            document_id: Document ID
            information_types: List of information types to extract
            
        Returns:
            Extracted information
        """
        try:
            return self.document_processor.extract_key_information(
                document_id, information_types
            )
            
        except Exception as e:
            logger.error(f"Error extracting key information: {e}")
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
            return self.rag_system.get_document_context(document_id, query, top_k)
            
        except Exception as e:
            logger.error(f"Error getting document context: {e}")
            return {'error': str(e)}
    
    def update_document(self, document_id: str, new_content: str,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update an existing document
        
        Args:
            document_id: Document ID
            new_content: Updated content
            metadata: Updated metadata
            
        Returns:
            Update result
        """
        try:
            # Update document index
            update_result = self.rag_system.update_document_index(
                document_id, new_content, metadata
            )
            
            if 'error' in update_result:
                return update_result
            
            # Update processed document in S3
            processed_s3_key = f"processed/{document_id}.txt"
            success = self.s3_service.upload_processed_document(
                new_content,
                f"{document_id}.txt",
                metadata or {}
            )
            
            if not success:
                return {'error': 'Failed to update processed document in S3'}
            
            return {
                'success': True,
                'document_id': document_id,
                'message': 'Document updated successfully',
                'indexing_result': update_result
            }
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return {'error': str(e)}
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document from the knowledge base
        
        Args:
            document_id: Document ID
            
        Returns:
            Deletion result
        """
        try:
            # Delete from RAG index
            index_result = self.rag_system.delete_document_index(document_id)
            
            if 'error' in index_result:
                return index_result
            
            # Delete processed document from S3
            processed_s3_key = f"processed/{document_id}.txt"
            s3_success = self.s3_service.delete_document(processed_s3_key)
            
            if not s3_success:
                logger.warning(f"Failed to delete processed document {processed_s3_key} from S3")
            
            return {
                'success': True,
                'document_id': document_id,
                'message': 'Document deleted successfully',
                'indexing_result': index_result
            }
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return {'error': str(e)}
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics
        
        Returns:
            Knowledge base statistics
        """
        try:
            # Get RAG index stats
            rag_stats = self.rag_system.get_index_stats()
            
            # Get S3 document count
            s3_documents = self.s3_service.list_documents("processed/")
            
            return {
                'success': True,
                'rag_stats': rag_stats,
                'total_documents': len(s3_documents),
                'processed_documents': s3_documents
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {'error': str(e)}
    
    def suggest_related_questions(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Suggest related questions based on the current question
        
        Args:
            question: Current question
            top_k: Number of suggestions to return
            
        Returns:
            Related question suggestions
        """
        try:
            # Search for similar content
            search_results = self.search_documents(question, top_k=10)
            
            if 'error' in search_results:
                return search_results
            
            # Generate suggestions using Bedrock
            context = " ".join([
                result['chunks'][0]['content'] 
                for result in search_results['results'][:3]
                if result['chunks']
            ])
            
            suggestion_prompt = f"""Based on the following context and the question "{question}", suggest {top_k} related questions that someone might ask:

Context: {context}

Related questions:"""
            
            suggestions_text = self.bedrock_service.generate_text(suggestion_prompt)
            
            # Parse suggestions (simple split by newlines)
            suggestions = [
                line.strip() 
                for line in suggestions_text.split('\n') 
                if line.strip() and '?' in line
            ][:top_k]
            
            return {
                'success': True,
                'original_question': question,
                'suggestions': suggestions
            }
            
        except Exception as e:
            logger.error(f"Error suggesting related questions: {e}")
            return {'error': str(e)}
    
    def initialize_knowledge_base(self) -> Dict[str, Any]:
        """
        Initialize the knowledge base
        
        Returns:
            Initialization result
        """
        try:
            # Initialize RAG index
            rag_init_result = self.rag_system.initialize_index()
            
            if 'error' in rag_init_result:
                return rag_init_result
            
            return {
                'success': True,
                'message': 'Knowledge base initialized successfully',
                'rag_initialization': rag_init_result
            }
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            return {'error': str(e)}
