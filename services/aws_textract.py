"""
AWS Textract service wrapper for document text extraction
"""
import boto3
import json
import time
from typing import List, Dict, Any, Optional, Union
from botocore.exceptions import ClientError
import logging
from io import BytesIO

from config.config import aws_config, get_aws_session

logger = logging.getLogger(__name__)

class TextractService:
    """AWS Textract service wrapper for document analysis"""
    
    def __init__(self):
        self.session = get_aws_session()
        self.textract = self.session.client('textract')
        self.s3_client = self.session.client('s3')
        self.bucket_name = aws_config.s3_bucket_name
    
    def extract_text_from_image(self, image_bytes: bytes) -> str:
        """
        Extract text from image using synchronous Textract
        
        Args:
            image_bytes: Image bytes
            
        Returns:
            Extracted text
        """
        try:
            response = self.textract.detect_document_text(
                Document={'Bytes': image_bytes}
            )
            
            text = self._extract_text_from_response(response)
            return text
            
        except ClientError as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def extract_text_from_s3(self, s3_key: str) -> str:
        """
        Extract text from document in S3 using synchronous Textract
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Extracted text
        """
        try:
            response = self.textract.detect_document_text(
                Document={
                    'S3Object': {
                        'Bucket': self.bucket_name,
                        'Name': s3_key
                    }
                }
            )
            
            text = self._extract_text_from_response(response)
            return text
            
        except ClientError as e:
            logger.error(f"Error extracting text from S3 document {s3_key}: {e}")
            return ""
    
    def analyze_document_async(self, s3_key: str, feature_types: List[str] = None) -> Optional[str]:
        """
        Start asynchronous document analysis
        
        Args:
            s3_key: S3 object key
            feature_types: List of features to analyze (TABLES, FORMS, QUERIES)
            
        Returns:
            Job ID if successful, None otherwise
        """
        try:
            if feature_types is None:
                feature_types = ['TABLES', 'FORMS']
            
            response = self.textract.start_document_analysis(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': self.bucket_name,
                        'Name': s3_key
                    }
                },
                FeatureTypes=feature_types
            )
            
            job_id = response['JobId']
            logger.info(f"Started Textract analysis job {job_id} for {s3_key}")
            return job_id
            
        except ClientError as e:
            logger.error(f"Error starting document analysis for {s3_key}: {e}")
            return None
    
    def get_analysis_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get asynchronous analysis result
        
        Args:
            job_id: Textract job ID
            
        Returns:
            Analysis result or None if error
        """
        try:
            response = self.textract.get_document_analysis(JobId=job_id)
            return response
            
        except ClientError as e:
            logger.error(f"Error getting analysis result for job {job_id}: {e}")
            return None
    
    def wait_for_analysis_completion(self, job_id: str, max_wait_time: int = 300) -> Optional[Dict[str, Any]]:
        """
        Wait for analysis to complete and return result
        
        Args:
            job_id: Textract job ID
            max_wait_time: Maximum wait time in seconds
            
        Returns:
            Analysis result or None if timeout/error
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = self.textract.get_document_analysis(JobId=job_id)
                status = response['JobStatus']
                
                if status == 'SUCCEEDED':
                    return response
                elif status == 'FAILED':
                    logger.error(f"Textract job {job_id} failed")
                    return None
                elif status == 'IN_PROGRESS':
                    time.sleep(5)  # Wait 5 seconds before checking again
                    continue
                    
            except ClientError as e:
                logger.error(f"Error checking job status for {job_id}: {e}")
                return None
        
        logger.warning(f"Textract job {job_id} timed out after {max_wait_time} seconds")
        return None
    
    def extract_tables(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tables from Textract analysis result
        
        Args:
            analysis_result: Textract analysis result
            
        Returns:
            List of extracted tables
        """
        try:
            blocks = analysis_result.get('Blocks', [])
            tables = []
            
            # Find all table blocks
            table_blocks = [block for block in blocks if block['BlockType'] == 'TABLE']
            
            for table_block in table_blocks:
                table = self._extract_single_table(table_block, blocks)
                if table:
                    tables.append(table)
            
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            return []
    
    def extract_forms(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract forms from Textract analysis result
        
        Args:
            analysis_result: Textract analysis result
            
        Returns:
            List of extracted form fields
        """
        try:
            blocks = analysis_result.get('Blocks', [])
            forms = []
            
            # Find all key-value blocks
            key_value_blocks = [block for block in blocks if block['BlockType'] == 'KEY_VALUE_SET']
            
            for kv_block in key_value_blocks:
                if kv_block.get('EntityTypes', [''])[0] == 'KEY':
                    form_field = self._extract_form_field(kv_block, blocks)
                    if form_field:
                        forms.append(form_field)
            
            return forms
            
        except Exception as e:
            logger.error(f"Error extracting forms: {e}")
            return []
    
    def extract_queries(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract query results from Textract analysis result
        
        Args:
            analysis_result: Textract analysis result
            
        Returns:
            List of query answers
        """
        try:
            blocks = analysis_result.get('Blocks', [])
            queries = []
            
            # Find all query result blocks
            query_blocks = [block for block in blocks if block['BlockType'] == 'QUERY_RESULT']
            
            for query_block in query_blocks:
                query = self._extract_query_result(query_block, blocks)
                if query:
                    queries.append(query)
            
            return queries
            
        except Exception as e:
            logger.error(f"Error extracting queries: {e}")
            return []
    
    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Extract text from Textract response"""
        try:
            blocks = response.get('Blocks', [])
            text_blocks = [block for block in blocks if block['BlockType'] == 'LINE']
            
            # Sort by reading order if available
            text_blocks.sort(key=lambda x: x.get('Geometry', {}).get('BoundingBox', {}).get('Top', 0))
            
            text_lines = [block['Text'] for block in text_blocks]
            return '\n'.join(text_lines)
            
        except Exception as e:
            logger.error(f"Error extracting text from response: {e}")
            return ""
    
    def _extract_single_table(self, table_block: Dict[str, Any], all_blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract a single table from table block"""
        try:
            table_id = table_block['Id']
            relationships = table_block.get('Relationships', [])
            
            # Find cells in this table
            cell_ids = []
            for relationship in relationships:
                if relationship['Type'] == 'CHILD':
                    cell_ids.extend(relationship['Ids'])
            
            # Get cell blocks
            cells = [block for block in all_blocks if block['Id'] in cell_ids and block['BlockType'] == 'CELL']
            
            # Build table structure
            table_data = []
            max_row = max((cell.get('RowIndex', 1) for cell in cells), default=0)
            max_col = max((cell.get('ColumnIndex', 1) for cell in cells), default=0)
            
            # Initialize table with empty cells
            for row in range(max_row):
                table_data.append([''] * max_col)
            
            # Fill in cell content
            for cell in cells:
                row_idx = cell.get('RowIndex', 1) - 1
                col_idx = cell.get('ColumnIndex', 1) - 1
                
                # Get cell text
                cell_relationships = cell.get('Relationships', [])
                cell_text = ""
                
                for relationship in cell_relationships:
                    if relationship['Type'] == 'CHILD':
                        word_ids = relationship['Ids']
                        words = [block for block in all_blocks if block['Id'] in word_ids and block['BlockType'] == 'WORD']
                        cell_text = ' '.join([word['Text'] for word in words])
                
                if row_idx < len(table_data) and col_idx < len(table_data[row_idx]):
                    table_data[row_idx][col_idx] = cell_text
            
            return {
                'table_id': table_id,
                'data': table_data,
                'rows': max_row,
                'columns': max_col
            }
            
        except Exception as e:
            logger.error(f"Error extracting single table: {e}")
            return None
    
    def _extract_form_field(self, key_block: Dict[str, Any], all_blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract a single form field from key block"""
        try:
            # Get key text
            key_text = self._get_text_from_block(key_block, all_blocks)
            
            # Find corresponding value block
            relationships = key_block.get('Relationships', [])
            value_id = None
            
            for relationship in relationships:
                if relationship['Type'] == 'VALUE':
                    value_ids = relationship['Ids']
                    # Find the value block
                    for vid in value_ids:
                        value_block = next((block for block in all_blocks if block['Id'] == vid), None)
                        if value_block and value_block.get('BlockType') == 'KEY_VALUE_SET':
                            value_id = vid
                            break
            
            value_text = ""
            if value_id:
                value_block = next((block for block in all_blocks if block['Id'] == value_id), None)
                if value_block:
                    value_text = self._get_text_from_block(value_block, all_blocks)
            
            return {
                'key': key_text,
                'value': value_text
            }
            
        except Exception as e:
            logger.error(f"Error extracting form field: {e}")
            return None
    
    def _extract_query_result(self, query_block: Dict[str, Any], all_blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract query result from query block"""
        try:
            query_text = self._get_text_from_block(query_block, all_blocks)
            
            return {
                'query': query_text,
                'answer': query_text  # For now, using the same text
            }
            
        except Exception as e:
            logger.error(f"Error extracting query result: {e}")
            return None
    
    def _get_text_from_block(self, block: Dict[str, Any], all_blocks: List[Dict[str, Any]]) -> str:
        """Get text content from a block"""
        try:
            relationships = block.get('Relationships', [])
            text_parts = []
            
            for relationship in relationships:
                if relationship['Type'] == 'CHILD':
                    child_ids = relationship['Ids']
                    children = [b for b in all_blocks if b['Id'] in child_ids and b['BlockType'] == 'WORD']
                    text_parts.extend([child['Text'] for child in children])
            
            return ' '.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error getting text from block: {e}")
            return ""
    
    def extract_structured_data(self, s3_key: str) -> Dict[str, Any]:
        """
        Extract all structured data from document
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Dictionary with text, tables, forms, and queries
        """
        try:
            # Start async analysis
            job_id = self.analyze_document_async(s3_key, ['TABLES', 'FORMS', 'QUERIES'])
            if not job_id:
                return {'error': 'Failed to start analysis'}
            
            # Wait for completion
            result = self.wait_for_analysis_completion(job_id)
            if not result:
                return {'error': 'Analysis failed or timed out'}
            
            # Extract different types of data
            text = self._extract_text_from_response(result)
            tables = self.extract_tables(result)
            forms = self.extract_forms(result)
            queries = self.extract_queries(result)
            
            return {
                'text': text,
                'tables': tables,
                'forms': forms,
                'queries': queries,
                'job_id': job_id
            }
            
        except Exception as e:
            logger.error(f"Error extracting structured data from {s3_key}: {e}")
            return {'error': str(e)}
