"""
Configuration settings for AI Document Processing & Knowledge Retrieval Agents
"""
import os
from typing import Dict, List
from pydantic import BaseSettings, Field

class AWSConfig(BaseSettings):
    """AWS Configuration"""
    aws_access_key_id: str = Field(default="", env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_DEFAULT_REGION")
    
    # Bedrock Configuration
    bedrock_model_id: str = Field(default="anthropic.claude-3-sonnet-20240229-v1:0", env="BEDROCK_MODEL_ID")
    bedrock_embeddings_model: str = Field(default="amazon.titan-embed-text-v1", env="BEDROCK_EMBEDDINGS_MODEL")
    
    # S3 Configuration
    s3_bucket_name: str = Field(default="ai-agent-documents", env="S3_BUCKET_NAME")
    s3_processed_prefix: str = Field(default="processed/", env="S3_PROCESSED_PREFIX")
    s3_raw_prefix: str = Field(default="raw/", env="S3_RAW_PREFIX")
    
    # OpenSearch Configuration
    opensearch_endpoint: str = Field(default="", env="OPENSEARCH_ENDPOINT")
    opensearch_index_name: str = Field(default="document-embeddings", env="OPENSEARCH_INDEX_NAME")
    
    # Lambda Configuration
    lambda_function_name: str = Field(default="document-processor", env="LAMBDA_FUNCTION_NAME")
    
    class Config:
        env_file = ".env"

class DocumentConfig(BaseSettings):
    """Document Processing Configuration"""
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    supported_formats: List[str] = Field(default=["pdf", "docx", "txt", "html"], env="SUPPORTED_FORMATS")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Textract Configuration
    textract_pages: List[int] = Field(default=[], env="TEXTRACT_PAGES")  # Empty means all pages
    
    class Config:
        env_file = ".env"

class RAGConfig(BaseSettings):
    """RAG Configuration"""
    vector_dimension: int = Field(default=1536, env="VECTOR_DIMENSION")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    max_context_length: int = Field(default=4000, env="MAX_CONTEXT_LENGTH")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    
    class Config:
        env_file = ".env"

class LegalAgentConfig(BaseSettings):
    """Legal Agent Specific Configuration"""
    contract_types: List[str] = Field(default=["employment", "service", "nda", "lease", "purchase"], env="CONTRACT_TYPES")
    key_clauses: List[str] = Field(default=[
        "termination", "payment", "liability", "confidentiality", 
        "intellectual_property", "governing_law", "dispute_resolution"
    ], env="KEY_CLAUSES")
    
    class Config:
        env_file = ".env"

# Global configuration instances
aws_config = AWSConfig()
document_config = DocumentConfig()
rag_config = RAGConfig()
legal_config = LegalAgentConfig()

def get_aws_session():
    """Get AWS session with configured credentials"""
    import boto3
    if aws_config.aws_access_key_id and aws_config.aws_secret_access_key:
        return boto3.Session(
            aws_access_key_id=aws_config.aws_access_key_id,
            aws_secret_access_key=aws_config.aws_secret_access_key,
            region_name=aws_config.aws_region
        )
    return boto3.Session(region_name=aws_config.aws_region)

