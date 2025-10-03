"""
AWS Lambda service wrapper for serverless functions
"""
import boto3
import json
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError
import logging

from config.config import aws_config, get_aws_session

logger = logging.getLogger(__name__)

class LambdaService:
    """AWS Lambda service wrapper"""
    
    def __init__(self):
        self.session = get_aws_session()
        self.lambda_client = self.session.client('lambda')
        self.function_name = aws_config.lambda_function_name
    
    def invoke_function(self, payload: Dict[str, Any], function_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke Lambda function
        
        Args:
            payload: Function payload
            function_name: Lambda function name (optional, uses default if not provided)
            
        Returns:
            Function response
        """
        try:
            func_name = function_name or self.function_name
            
            response = self.lambda_client.invoke(
                FunctionName=func_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            # Read response
            response_payload = json.loads(response['Payload'].read())
            
            if response['StatusCode'] == 200:
                logger.info(f"Successfully invoked function {func_name}")
                return response_payload
            else:
                logger.error(f"Function invocation failed with status {response['StatusCode']}")
                return {'error': 'Function invocation failed'}
                
        except ClientError as e:
            logger.error(f"Error invoking function {func_name}: {e}")
            return {'error': str(e)}
    
    def invoke_async(self, payload: Dict[str, Any], function_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke Lambda function asynchronously
        
        Args:
            payload: Function payload
            function_name: Lambda function name (optional, uses default if not provided)
            
        Returns:
            Response with status
        """
        try:
            func_name = function_name or self.function_name
            
            response = self.lambda_client.invoke(
                FunctionName=func_name,
                InvocationType='Event',
                Payload=json.dumps(payload)
            )
            
            if response['StatusCode'] == 202:
                logger.info(f"Successfully invoked function {func_name} asynchronously")
                return {'status': 'success', 'status_code': 202}
            else:
                logger.error(f"Async function invocation failed with status {response['StatusCode']}")
                return {'error': 'Async function invocation failed'}
                
        except ClientError as e:
            logger.error(f"Error invoking function {func_name} asynchronously: {e}")
            return {'error': str(e)}
    
    def create_function(self, function_name: str, role_arn: str, code_zip: bytes, 
                       handler: str = "lambda_function.lambda_handler", 
                       runtime: str = "python3.9", 
                       timeout: int = 60, 
                       memory_size: int = 512) -> bool:
        """
        Create Lambda function
        
        Args:
            function_name: Function name
            role_arn: IAM role ARN
            code_zip: Function code as zip bytes
            handler: Function handler
            runtime: Runtime environment
            timeout: Function timeout in seconds
            memory_size: Memory allocation in MB
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime=runtime,
                Role=role_arn,
                Handler=handler,
                Code={'ZipFile': code_zip},
                Timeout=timeout,
                MemorySize=memory_size,
                Environment={
                    'Variables': {
                        'AWS_REGION': aws_config.aws_region
                    }
                }
            )
            
            if response['ResponseMetadata']['HTTPStatusCode'] == 201:
                logger.info(f"Successfully created function {function_name}")
                return True
            else:
                logger.error(f"Failed to create function {function_name}")
                return False
                
        except ClientError as e:
            logger.error(f"Error creating function {function_name}: {e}")
            return False
    
    def update_function_code(self, function_name: str, code_zip: bytes) -> bool:
        """
        Update Lambda function code
        
        Args:
            function_name: Function name
            code_zip: New function code as zip bytes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=code_zip
            )
            
            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                logger.info(f"Successfully updated function code for {function_name}")
                return True
            else:
                logger.error(f"Failed to update function code for {function_name}")
                return False
                
        except ClientError as e:
            logger.error(f"Error updating function code for {function_name}: {e}")
            return False
    
    def get_function_info(self, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Get Lambda function information
        
        Args:
            function_name: Function name
            
        Returns:
            Function information or None if error
        """
        try:
            response = self.lambda_client.get_function(FunctionName=function_name)
            return response
            
        except ClientError as e:
            logger.error(f"Error getting function info for {function_name}: {e}")
            return None
    
    def list_functions(self) -> List[Dict[str, Any]]:
        """
        List all Lambda functions
        
        Returns:
            List of function information
        """
        try:
            response = self.lambda_client.list_functions()
            return response.get('Functions', [])
            
        except ClientError as e:
            logger.error(f"Error listing functions: {e}")
            return []
    
    def delete_function(self, function_name: str) -> bool:
        """
        Delete Lambda function
        
        Args:
            function_name: Function name to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.lambda_client.delete_function(FunctionName=function_name)
            
            if response['ResponseMetadata']['HTTPStatusCode'] == 204:
                logger.info(f"Successfully deleted function {function_name}")
                return True
            else:
                logger.error(f"Failed to delete function {function_name}")
                return False
                
        except ClientError as e:
            logger.error(f"Error deleting function {function_name}: {e}")
            return False
    
    def add_permission(self, function_name: str, statement_id: str, 
                      action: str, principal: str, source_arn: Optional[str] = None) -> bool:
        """
        Add permission to Lambda function
        
        Args:
            function_name: Function name
            statement_id: Statement ID
            action: Action to allow
            principal: Principal to grant permission
            source_arn: Source ARN (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            params = {
                'FunctionName': function_name,
                'StatementId': statement_id,
                'Action': action,
                'Principal': principal
            }
            
            if source_arn:
                params['SourceArn'] = source_arn
            
            response = self.lambda_client.add_permission(**params)
            
            if response['ResponseMetadata']['HTTPStatusCode'] == 201:
                logger.info(f"Successfully added permission to function {function_name}")
                return True
            else:
                logger.error(f"Failed to add permission to function {function_name}")
                return False
                
        except ClientError as e:
            logger.error(f"Error adding permission to function {function_name}: {e}")
            return False
    
    def invoke_document_processor(self, document_key: str, document_type: str = "pdf") -> Dict[str, Any]:
        """
        Invoke document processor Lambda function
        
        Args:
            document_key: S3 key of document to process
            document_type: Type of document
            
        Returns:
            Processing result
        """
        payload = {
            'document_key': document_key,
            'document_type': document_type,
            'bucket_name': aws_config.s3_bucket_name,
            'action': 'process_document'
        }
        
        return self.invoke_function(payload, 'document-processor')
    
    def invoke_embedding_generator(self, text_chunks: List[str], document_id: str) -> Dict[str, Any]:
        """
        Invoke embedding generator Lambda function
        
        Args:
            text_chunks: List of text chunks to embed
            document_id: Document ID
            
        Returns:
            Embedding result
        """
        payload = {
            'text_chunks': text_chunks,
            'document_id': document_id,
            'action': 'generate_embeddings'
        }
        
        return self.invoke_function(payload, 'embedding-generator')
    
    def invoke_search_agent(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Invoke search agent Lambda function
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Search results
        """
        payload = {
            'query': query,
            'top_k': top_k,
            'action': 'search_documents'
        }
        
        return self.invoke_function(payload, 'search-agent')
