"""
AWS S3 service wrapper for document storage
"""
import boto3
import os
from typing import List, Dict, Any, Optional, BinaryIO
from botocore.exceptions import ClientError
import logging
from datetime import datetime

from config.config import aws_config, get_aws_session

logger = logging.getLogger(__name__)

class S3Service:
    """AWS S3 service wrapper for document storage"""
    
    def __init__(self):
        self.session = get_aws_session()
        self.s3_client = self.session.client('s3')
        self.bucket_name = aws_config.s3_bucket_name
        self.processed_prefix = aws_config.s3_processed_prefix
        self.raw_prefix = aws_config.s3_raw_prefix
    
    def upload_document(self, file_path: str, s3_key: str, metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Upload document to S3
        
        Args:
            file_path: Local file path
            s3_key: S3 object key
            metadata: Optional metadata to attach
            
        Returns:
            True if successful, False otherwise
        """
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key, ExtraArgs=extra_args)
            logger.info(f"Successfully uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error uploading file {file_path}: {e}")
            return False
    
    def upload_file_object(self, file_obj: BinaryIO, s3_key: str, metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Upload file object to S3
        
        Args:
            file_obj: File-like object
            s3_key: S3 object key
            metadata: Optional metadata to attach
            
        Returns:
            True if successful, False otherwise
        """
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.upload_fileobj(file_obj, self.bucket_name, s3_key, ExtraArgs=extra_args)
            logger.info(f"Successfully uploaded file object to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error uploading file object: {e}")
            return False
    
    def download_document(self, s3_key: str, local_path: str) -> bool:
        """
        Download document from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Successfully downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
            
        except ClientError as e:
            logger.error(f"Error downloading file {s3_key}: {e}")
            return False
    
    def get_document_content(self, s3_key: str) -> Optional[bytes]:
        """
        Get document content as bytes
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Document content as bytes or None if error
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read()
            
        except ClientError as e:
            logger.error(f"Error getting document content {s3_key}: {e}")
            return None
    
    def list_documents(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List documents in S3 bucket
        
        Args:
            prefix: S3 key prefix to filter
            max_keys: Maximum number of keys to return
            
        Returns:
            List of document metadata
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            documents = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    documents.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'etag': obj['ETag']
                    })
            
            return documents
            
        except ClientError as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def delete_document(self, s3_key: str) -> bool:
        """
        Delete document from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Successfully deleted s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting document {s3_key}: {e}")
            return False
    
    def get_document_metadata(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Document metadata or None if error
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            
            return {
                'content_type': response.get('ContentType'),
                'content_length': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'metadata': response.get('Metadata', {}),
                'etag': response.get('ETag')
            }
            
        except ClientError as e:
            logger.error(f"Error getting document metadata {s3_key}: {e}")
            return None
    
    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for document access
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
            
        Returns:
            Presigned URL or None if error
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
            
        except ClientError as e:
            logger.error(f"Error generating presigned URL for {s3_key}: {e}")
            return None
    
    def move_document(self, source_key: str, dest_key: str) -> bool:
        """
        Move document within S3 bucket
        
        Args:
            source_key: Source S3 key
            dest_key: Destination S3 key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Copy to new location
            copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=dest_key
            )
            
            # Delete original
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=source_key)
            
            logger.info(f"Moved s3://{self.bucket_name}/{source_key} to s3://{self.bucket_name}/{dest_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error moving document from {source_key} to {dest_key}: {e}")
            return False
    
    def upload_processed_document(
    self,
    content: Any,
    original_filename: str,
    processing_metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Upload processed document with metadata to S3.
        Handles str, bytes, dict, and mixed metadata safely.
        """
        import os, re, logging
        from datetime import datetime
        from botocore.exceptions import ClientError

        logger = logging.getLogger(__name__)

        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = os.path.splitext(original_filename)[0]
            s3_key = f"{self.processed_prefix}{filename}_{timestamp}.txt"

            # 🧠 Normalize metadata: convert all values to strings
            raw_metadata = {
                "original_filename": original_filename,
                "processed_timestamp": timestamp,
                **(processing_metadata or {})
            }
            metadata = {str(k): str(v) for k, v in raw_metadata.items() if v is not None}

            # 🧠 Prepare body content
            if isinstance(content, bytes):
                body = content
            elif isinstance(content, str):
                clean = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", content)
                body = clean.encode("utf-8", errors="replace")
            else:
                body = str(content).encode("utf-8", errors="replace")

            # Debug preview
            logger.debug(f"S3 Upload Summary:")
            logger.debug(f"  Bucket: {self.bucket_name}")
            logger.debug(f"  Key: {s3_key}")
            logger.debug(f"  Metadata: {metadata}")
            logger.debug(f"  Body Preview: {body[:200]!r}")

            # ✅ Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=body,
                ContentType="text/plain",
                Metadata=metadata
            )

            logger.info(f"✅ Uploaded successfully: s3://{self.bucket_name}/{s3_key}")
            return s3_key

        except ClientError as e:
            logger.error(f"❌ S3 ClientError: {e}")
            return None

        except Exception as e:
            logger.error(f"⚠️ Unexpected error uploading processed document: {e}")
            return None





