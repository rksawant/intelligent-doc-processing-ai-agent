"""
Configuration package for AI Document Processing & Knowledge Retrieval Agents
"""

from .config import (
    aws_config,
    document_config,
    rag_config,
    legal_config,
    get_aws_session
)

__all__ = [
    'aws_config',
    'document_config',
    'rag_config',
    'legal_config',
    'get_aws_session'
]
