"""
Pipeline Manager for orchestrating document processing workflows
"""
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import boto3
from enum import Enum

# AWS services
from services import LambdaService, S3Service
from agents import KnowledgeAgent, LegalAgent

logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """Pipeline status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PipelineType(Enum):
    """Pipeline type enumeration"""
    DOCUMENT_PROCESSING = "document_processing"
    CONTRACT_ANALYSIS = "contract_analysis"
    KNOWLEDGE_SEARCH = "knowledge_search"
    QUESTION_ANSWERING = "question_answering"
    BATCH_PROCESSING = "batch_processing"

class PipelineManager:
    """Pipeline manager for orchestrating document processing workflows"""
    
    def __init__(self):
        self.lambda_service = LambdaService()
        self.s3_service = S3Service()
        self.knowledge_agent = KnowledgeAgent()
        self.legal_agent = LegalAgent()
        
        # Pipeline configuration
        self.pipeline_configs = {
            PipelineType.DOCUMENT_PROCESSING: {
                'lambda_function': 'document-processor',
                'timeout': 300,
                'retry_count': 3
            },
            PipelineType.CONTRACT_ANALYSIS: {
                'lambda_function': 'contract-analyzer',
                'timeout': 600,
                'retry_count': 2
            },
            PipelineType.KNOWLEDGE_SEARCH: {
                'lambda_function': 'search-agent',
                'timeout': 60,
                'retry_count': 2
            },
            PipelineType.QUESTION_ANSWERING: {
                'lambda_function': 'qa-agent',
                'timeout': 120,
                'retry_count': 2
            }
        }
    
    def create_document_processing_pipeline(self, document_key: str, 
                                          document_type: str = 'pdf',
                                          metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create and execute document processing pipeline
        
        Args:
            document_key: S3 key of document
            document_type: Type of document
            metadata: Optional metadata
            
        Returns:
            Pipeline execution result
        """
        try:
            pipeline_id = self._generate_pipeline_id()
            
            # Start pipeline
            result = {
                'pipeline_id': pipeline_id,
                'pipeline_type': PipelineType.DOCUMENT_PROCESSING.value,
                'status': PipelineStatus.RUNNING.value,
                'start_time': datetime.utcnow().isoformat(),
                'document_key': document_key,
                'document_type': document_type,
                'metadata': metadata or {}
            }
            
            # Step 1: Process document
            process_result = self.knowledge_agent.process_and_index_document(
                document_key, metadata
            )
            
            if 'error' in process_result:
                result.update({
                    'status': PipelineStatus.FAILED.value,
                    'error': process_result['error'],
                    'end_time': datetime.utcnow().isoformat()
                })
                return result
            
            # Step 2: Generate summary
            document_id = process_result['document_id']
            summary_result = self.knowledge_agent.get_document_summary(document_id)
            
            # Complete pipeline
            result.update({
                'status': PipelineStatus.COMPLETED.value,
                'end_time': datetime.utcnow().isoformat(),
                'document_id': document_id,
                'processing_result': process_result,
                'summary_result': summary_result,
                'message': 'Document processing pipeline completed successfully'
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing pipeline error: {e}")
            return {
                'pipeline_id': pipeline_id,
                'status': PipelineStatus.FAILED.value,
                'error': str(e),
                'end_time': datetime.utcnow().isoformat()
            }
    
    def create_contract_analysis_pipeline(self, document_key: str,
                                        contract_type: Optional[str] = None,
                                        focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create and execute contract analysis pipeline
        
        Args:
            document_key: S3 key of contract
            contract_type: Type of contract
            focus_areas: Areas to focus on
            
        Returns:
            Pipeline execution result
        """
        try:
            pipeline_id = self._generate_pipeline_id()
            
            # Start pipeline
            result = {
                'pipeline_id': pipeline_id,
                'pipeline_type': PipelineType.CONTRACT_ANALYSIS.value,
                'status': PipelineStatus.RUNNING.value,
                'start_time': datetime.utcnow().isoformat(),
                'document_key': document_key,
                'contract_type': contract_type,
                'focus_areas': focus_areas or []
            }
            
            # Step 1: Analyze contract
            analysis_result = self.legal_agent.analyze_contract(
                document_key, contract_type, focus_areas
            )
            
            if 'error' in analysis_result:
                result.update({
                    'status': PipelineStatus.FAILED.value,
                    'error': analysis_result['error'],
                    'end_time': datetime.utcnow().isoformat()
                })
                return result
            
            document_id = analysis_result['document_id']
            
            # Step 2: Extract key information
            key_info = self.legal_agent.extract_termination_conditions(document_id)
            payment_terms = self.legal_agent.extract_payment_terms(document_id)
            liability_terms = self.legal_agent.extract_liability_terms(document_id)
            
            # Step 3: Generate comprehensive summary
            summary = self.legal_agent.generate_contract_summary(document_id)
            
            # Complete pipeline
            result.update({
                'status': PipelineStatus.COMPLETED.value,
                'end_time': datetime.utcnow().isoformat(),
                'document_id': document_id,
                'analysis_result': analysis_result,
                'key_information': {
                    'termination_conditions': key_info,
                    'payment_terms': payment_terms,
                    'liability_terms': liability_terms
                },
                'summary': summary,
                'message': 'Contract analysis pipeline completed successfully'
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Contract analysis pipeline error: {e}")
            return {
                'pipeline_id': pipeline_id,
                'status': PipelineStatus.FAILED.value,
                'error': str(e),
                'end_time': datetime.utcnow().isoformat()
            }
    
    def create_knowledge_search_pipeline(self, query: str, top_k: int = 5,
                                       filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create and execute knowledge search pipeline
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Search filters
            
        Returns:
            Pipeline execution result
        """
        try:
            pipeline_id = self._generate_pipeline_id()
            
            # Start pipeline
            result = {
                'pipeline_id': pipeline_id,
                'pipeline_type': PipelineType.KNOWLEDGE_SEARCH.value,
                'status': PipelineStatus.RUNNING.value,
                'start_time': datetime.utcnow().isoformat(),
                'query': query,
                'top_k': top_k,
                'filters': filters or {}
            }
            
            # Step 1: Search documents
            search_result = self.knowledge_agent.search_documents(query, top_k, filters)
            
            if 'error' in search_result:
                result.update({
                    'status': PipelineStatus.FAILED.value,
                    'error': search_result['error'],
                    'end_time': datetime.utcnow().isoformat()
                })
                return result
            
            # Step 2: Hybrid search for better results
            hybrid_result = self.knowledge_agent.hybrid_search(query, top_k)
            
            # Step 3: Generate suggestions
            suggestions = self.knowledge_agent.suggest_related_questions(query, 3)
            
            # Complete pipeline
            result.update({
                'status': PipelineStatus.COMPLETED.value,
                'end_time': datetime.utcnow().isoformat(),
                'search_result': search_result,
                'hybrid_result': hybrid_result,
                'suggestions': suggestions,
                'message': 'Knowledge search pipeline completed successfully'
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Knowledge search pipeline error: {e}")
            return {
                'pipeline_id': pipeline_id,
                'status': PipelineStatus.FAILED.value,
                'error': str(e),
                'end_time': datetime.utcnow().isoformat()
            }
    
    def create_question_answering_pipeline(self, question: str,
                                         document_id: Optional[str] = None,
                                         context_limit: int = 5) -> Dict[str, Any]:
        """
        Create and execute question answering pipeline
        
        Args:
            question: Question to answer
            document_id: Optional document ID
            context_limit: Context limit
            
        Returns:
            Pipeline execution result
        """
        try:
            pipeline_id = self._generate_pipeline_id()
            
            # Start pipeline
            result = {
                'pipeline_id': pipeline_id,
                'pipeline_type': PipelineType.QUESTION_ANSWERING.value,
                'status': PipelineStatus.RUNNING.value,
                'start_time': datetime.utcnow().isoformat(),
                'question': question,
                'document_id': document_id,
                'context_limit': context_limit
            }
            
            # Step 1: Answer question
            answer_result = self.knowledge_agent.ask_question(
                question, context_limit, document_id
            )
            
            if 'error' in answer_result:
                result.update({
                    'status': PipelineStatus.FAILED.value,
                    'error': answer_result['error'],
                    'end_time': datetime.utcnow().isoformat()
                })
                return result
            
            # Step 2: Generate follow-up suggestions
            suggestions = self.knowledge_agent.suggest_related_questions(question, 3)
            
            # Step 3: Get additional context if available
            additional_context = None
            if document_id:
                context_result = self.knowledge_agent.get_document_context(
                    document_id, question, 3
                )
                if 'error' not in context_result:
                    additional_context = context_result
            
            # Complete pipeline
            result.update({
                'status': PipelineStatus.COMPLETED.value,
                'end_time': datetime.utcnow().isoformat(),
                'answer_result': answer_result,
                'suggestions': suggestions,
                'additional_context': additional_context,
                'message': 'Question answering pipeline completed successfully'
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Question answering pipeline error: {e}")
            return {
                'pipeline_id': pipeline_id,
                'status': PipelineStatus.FAILED.value,
                'error': str(e),
                'end_time': datetime.utcnow().isoformat()
            }
    
    def create_batch_processing_pipeline(self, document_keys: List[str],
                                       pipeline_type: PipelineType,
                                       **kwargs) -> Dict[str, Any]:
        """
        Create and execute batch processing pipeline
        
        Args:
            document_keys: List of document S3 keys
            pipeline_type: Type of pipeline to run
            **kwargs: Additional arguments
            
        Returns:
            Batch pipeline execution result
        """
        try:
            pipeline_id = self._generate_pipeline_id()
            
            # Start pipeline
            result = {
                'pipeline_id': pipeline_id,
                'pipeline_type': PipelineType.BATCH_PROCESSING.value,
                'sub_pipeline_type': pipeline_type.value,
                'status': PipelineStatus.RUNNING.value,
                'start_time': datetime.utcnow().isoformat(),
                'document_count': len(document_keys),
                'document_keys': document_keys,
                'kwargs': kwargs
            }
            
            # Process documents in batch
            batch_results = []
            successful_count = 0
            failed_count = 0
            
            for i, document_key in enumerate(document_keys):
                try:
                    # Execute sub-pipeline based on type
                    if pipeline_type == PipelineType.DOCUMENT_PROCESSING:
                        sub_result = self.create_document_processing_pipeline(
                            document_key, **kwargs
                        )
                    elif pipeline_type == PipelineType.CONTRACT_ANALYSIS:
                        sub_result = self.create_contract_analysis_pipeline(
                            document_key, **kwargs
                        )
                    else:
                        sub_result = {'error': f'Unsupported batch pipeline type: {pipeline_type}'}
                    
                    batch_results.append({
                        'document_key': document_key,
                        'index': i,
                        'result': sub_result
                    })
                    
                    if sub_result.get('status') == PipelineStatus.COMPLETED.value:
                        successful_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Batch processing error for document {document_key}: {e}")
                    batch_results.append({
                        'document_key': document_key,
                        'index': i,
                        'result': {'error': str(e)}
                    })
                    failed_count += 1
            
            # Complete pipeline
            result.update({
                'status': PipelineStatus.COMPLETED.value,
                'end_time': datetime.utcnow().isoformat(),
                'successful_count': successful_count,
                'failed_count': failed_count,
                'batch_results': batch_results,
                'message': f'Batch processing pipeline completed: {successful_count} successful, {failed_count} failed'
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Batch processing pipeline error: {e}")
            return {
                'pipeline_id': pipeline_id,
                'status': PipelineStatus.FAILED.value,
                'error': str(e),
                'end_time': datetime.utcnow().isoformat()
            }
    
    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Get pipeline status
        
        Args:
            pipeline_id: Pipeline ID
            
        Returns:
            Pipeline status
        """
        try:
            # In a real implementation, this would query a database or storage system
            # For now, return a placeholder response
            return {
                'pipeline_id': pipeline_id,
                'status': 'unknown',
                'message': 'Pipeline status tracking not implemented'
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {'error': str(e)}
    
    def cancel_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Cancel a running pipeline
        
        Args:
            pipeline_id: Pipeline ID
            
        Returns:
            Cancellation result
        """
        try:
            # In a real implementation, this would cancel the pipeline
            # For now, return a placeholder response
            return {
                'pipeline_id': pipeline_id,
                'status': PipelineStatus.CANCELLED.value,
                'message': 'Pipeline cancellation not implemented'
            }
            
        except Exception as e:
            logger.error(f"Error cancelling pipeline: {e}")
            return {'error': str(e)}
    
    def list_pipelines(self, status: Optional[PipelineStatus] = None,
                      pipeline_type: Optional[PipelineType] = None) -> Dict[str, Any]:
        """
        List pipelines with optional filters
        
        Args:
            status: Optional status filter
            pipeline_type: Optional pipeline type filter
            
        Returns:
            List of pipelines
        """
        try:
            # In a real implementation, this would query a database
            # For now, return a placeholder response
            return {
                'pipelines': [],
                'total_count': 0,
                'message': 'Pipeline listing not implemented'
            }
            
        except Exception as e:
            logger.error(f"Error listing pipelines: {e}")
            return {'error': str(e)}
    
    def _generate_pipeline_id(self) -> str:
        """Generate unique pipeline ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return f"pipeline_{timestamp}"
    
    def execute_pipeline_async(self, pipeline_type: PipelineType, **kwargs) -> Dict[str, Any]:
        """
        Execute pipeline asynchronously using Lambda
        
        Args:
            pipeline_type: Type of pipeline
            **kwargs: Pipeline arguments
            
        Returns:
            Async execution result
        """
        try:
            # Get pipeline configuration
            config = self.pipeline_configs.get(pipeline_type)
            if not config:
                return {'error': f'Unknown pipeline type: {pipeline_type}'}
            
            # Prepare payload
            payload = {
                'action': 'execute_pipeline',
                'pipeline_type': pipeline_type.value,
                **kwargs
            }
            
            # Invoke Lambda function
            result = self.lambda_service.invoke_function(
                payload, config['lambda_function']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing pipeline async: {e}")
            return {'error': str(e)}
