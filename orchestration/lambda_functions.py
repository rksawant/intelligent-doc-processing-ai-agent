"""
Lambda functions for document processing pipeline orchestration
"""
import json
import logging
import boto3
from typing import Dict, Any, List
from datetime import datetime

# AWS services
from services import BedrockService, S3Service, OpenSearchService, TextractService
from agents import DocumentProcessor, RAGSystem, KnowledgeAgent, LegalAgent

logger = logging.getLogger()

def lambda_handler(event, context):
    """
    Main Lambda handler for document processing pipeline
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Processing result
    """
    try:
        # Parse event
        action = event.get('action', 'process_document')
        
        if action == 'process_document':
            return process_document_handler(event, context)
        elif action == 'generate_embeddings':
            return generate_embeddings_handler(event, context)
        elif action == 'search_documents':
            return search_documents_handler(event, context)
        elif action == 'analyze_contract':
            return analyze_contract_handler(event, context)
        elif action == 'answer_question':
            return answer_question_handler(event, context)
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Unknown action: {action}'})
            }
            
    except Exception as e:
        logger.error(f"Lambda handler error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def process_document_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Handle document processing requests
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Processing result
    """
    try:
        # Extract parameters
        document_key = event.get('document_key')
        document_type = event.get('document_type', 'pdf')
        bucket_name = event.get('bucket_name')
        
        if not document_key or not bucket_name:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing required parameters'})
            }
        
        # Initialize services
        s3_service = S3Service()
        document_processor = DocumentProcessor()
        
        # Download document from S3
        local_path = f"/tmp/{document_key.split('/')[-1]}"
        success = s3_service.download_document(document_key, local_path)
        
        if not success:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to download document from S3'})
            }
        
        # Process document
        result = document_processor.process_document(local_path)
        
        if 'error' in result:
            return {
                'statusCode': 500,
                'body': json.dumps(result)
            }
        
        # Clean up temporary file
        import os
        if os.path.exists(local_path):
            os.remove(local_path)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def generate_embeddings_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Handle embedding generation requests
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Embedding generation result
    """
    try:
        # Extract parameters
        text_chunks = event.get('text_chunks', [])
        document_id = event.get('document_id')
        
        if not text_chunks or not document_id:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing required parameters'})
            }
        
        # Initialize services
        bedrock_service = BedrockService()
        opensearch_service = OpenSearchService()
        
        # Generate embeddings
        embeddings = bedrock_service.generate_embeddings(text_chunks)
        
        # Index embeddings
        indexed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            
            success = opensearch_service.index_document(
                chunk_id,
                chunk,
                embedding,
                {'document_id': document_id, 'chunk_index': i}
            )
            
            if success:
                indexed_chunks.append({
                    'chunk_id': chunk_id,
                    'chunk_index': i,
                    'embedding_dimension': len(embedding)
                })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'document_id': document_id,
                'total_chunks': len(text_chunks),
                'indexed_chunks': len(indexed_chunks),
                'chunks': indexed_chunks
            })
        }
        
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def search_documents_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Handle document search requests
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Search results
    """
    try:
        # Extract parameters
        query = event.get('query')
        top_k = event.get('top_k', 5)
        filters = event.get('filters')
        
        if not query:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing query parameter'})
            }
        
        # Initialize services
        rag_system = RAGSystem()
        
        # Search documents
        if filters:
            result = rag_system.search_documents(query, top_k, filters)
        else:
            result = rag_system.search_documents(query, top_k)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Document search error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def analyze_contract_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Handle contract analysis requests
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Contract analysis result
    """
    try:
        # Extract parameters
        document_key = event.get('document_key')
        contract_type = event.get('contract_type')
        bucket_name = event.get('bucket_name')
        focus_areas = event.get('focus_areas')
        
        if not document_key or not bucket_name:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing required parameters'})
            }
        
        # Initialize services
        s3_service = S3Service()
        legal_agent = LegalAgent()
        
        # Download document from S3
        local_path = f"/tmp/{document_key.split('/')[-1]}"
        success = s3_service.download_document(document_key, local_path)
        
        if not success:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to download document from S3'})
            }
        
        # Analyze contract
        result = legal_agent.analyze_contract(local_path, contract_type, focus_areas)
        
        # Clean up temporary file
        import os
        if os.path.exists(local_path):
            os.remove(local_path)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Contract analysis error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def answer_question_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Handle question answering requests
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Answer result
    """
    try:
        # Extract parameters
        question = event.get('question')
        document_id = event.get('document_id')
        context_limit = event.get('context_limit', 5)
        
        if not question:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing question parameter'})
            }
        
        # Initialize services
        knowledge_agent = KnowledgeAgent()
        
        # Answer question
        result = knowledge_agent.ask_question(question, context_limit, document_id)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

# Additional utility functions for pipeline orchestration

def trigger_document_processing_pipeline(document_key: str, document_type: str = 'pdf') -> Dict[str, Any]:
    """
    Trigger the complete document processing pipeline
    
    Args:
        document_key: S3 key of the document
        document_type: Type of document
        
    Returns:
        Pipeline result
    """
    try:
        lambda_client = boto3.client('lambda')
        
        # Step 1: Process document
        process_payload = {
            'action': 'process_document',
            'document_key': document_key,
            'document_type': document_type,
            'bucket_name': 'ai-agent-documents'  # Configure as needed
        }
        
        process_response = lambda_client.invoke(
            FunctionName='document-processor',
            InvocationType='RequestResponse',
            Payload=json.dumps(process_payload)
        )
        
        process_result = json.loads(process_response['Payload'].read())
        
        if process_result.get('statusCode') != 200:
            return {'error': 'Document processing failed', 'details': process_result}
        
        # Step 2: Generate embeddings and index
        document_data = json.loads(process_result['body'])
        document_id = document_data.get('document_id')
        text_content = document_data.get('text_content')
        
        if document_id and text_content:
            # Split text into chunks
            from agents.rag_system import RAGSystem
            rag_system = RAGSystem()
            chunks = rag_system._split_text(text_content)
            chunk_texts = [chunk.page_content for chunk in chunks]
            
            # Generate embeddings
            embeddings_payload = {
                'action': 'generate_embeddings',
                'text_chunks': chunk_texts,
                'document_id': document_id
            }
            
            embeddings_response = lambda_client.invoke(
                FunctionName='embedding-generator',
                InvocationType='RequestResponse',
                Payload=json.dumps(embeddings_payload)
            )
            
            embeddings_result = json.loads(embeddings_response['Payload'].read())
            
            return {
                'success': True,
                'document_id': document_id,
                'processing_result': document_data,
                'embeddings_result': embeddings_result
            }
        
        return process_result
        
    except Exception as e:
        logger.error(f"Pipeline orchestration error: {e}")
        return {'error': str(e)}

def trigger_contract_analysis_pipeline(document_key: str, contract_type: str = None) -> Dict[str, Any]:
    """
    Trigger contract analysis pipeline
    
    Args:
        document_key: S3 key of the contract
        contract_type: Type of contract
        
    Returns:
        Analysis result
    """
    try:
        lambda_client = boto3.client('lambda')
        
        # Analyze contract
        analysis_payload = {
            'action': 'analyze_contract',
            'document_key': document_key,
            'contract_type': contract_type,
            'bucket_name': 'ai-agent-documents'  # Configure as needed
        }
        
        analysis_response = lambda_client.invoke(
            FunctionName='contract-analyzer',
            InvocationType='RequestResponse',
            Payload=json.dumps(analysis_payload)
        )
        
        analysis_result = json.loads(analysis_response['Payload'].read())
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Contract analysis pipeline error: {e}")
        return {'error': str(e)}

def trigger_search_pipeline(query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Trigger search pipeline
    
    Args:
        query: Search query
        top_k: Number of results
        filters: Search filters
        
    Returns:
        Search results
    """
    try:
        lambda_client = boto3.client('lambda')
        
        # Search documents
        search_payload = {
            'action': 'search_documents',
            'query': query,
            'top_k': top_k,
            'filters': filters
        }
        
        search_response = lambda_client.invoke(
            FunctionName='search-agent',
            InvocationType='RequestResponse',
            Payload=json.dumps(search_payload)
        )
        
        search_result = json.loads(search_response['Payload'].read())
        
        return search_result
        
    except Exception as e:
        logger.error(f"Search pipeline error: {e}")
        return {'error': str(e)}

def trigger_qa_pipeline(question: str, document_id: str = None, context_limit: int = 5) -> Dict[str, Any]:
    """
    Trigger Q&A pipeline
    
    Args:
        question: Question to answer
        document_id: Optional document ID to limit search
        context_limit: Context limit
        
    Returns:
        Answer result
    """
    try:
        lambda_client = boto3.client('lambda')
        
        # Answer question
        qa_payload = {
            'action': 'answer_question',
            'question': question,
            'document_id': document_id,
            'context_limit': context_limit
        }
        
        qa_response = lambda_client.invoke(
            FunctionName='qa-agent',
            InvocationType='RequestResponse',
            Payload=json.dumps(qa_payload)
        )
        
        qa_result = json.loads(qa_response['Payload'].read())
        
        return qa_result
        
    except Exception as e:
        logger.error(f"Q&A pipeline error: {e}")
        return {'error': str(e)}
