"""
AWS OpenSearch service wrapper for vector search and indexing
"""
import json
import boto3
from typing import List, Dict, Any, Optional, Tuple
from botocore.exceptions import ClientError
import logging
import requests
from requests.auth import HTTPBasicAuth

from config.config import aws_config, get_aws_session

logger = logging.getLogger(__name__)

class OpenSearchService:
    """AWS OpenSearch service wrapper for vector search"""
    
    def __init__(self):
        self.session = get_aws_session()
        self.endpoint = aws_config.opensearch_endpoint
        self.index_name = aws_config.opensearch_index_name
        self.region = aws_config.aws_region
        
        # Get AWS credentials for OpenSearch authentication
        credentials = self.session.get_credentials()
        self.auth = AWS4Auth(credentials, self.region, 'es')
    
    def create_index(self, vector_dimension: int = 1536) -> bool:
        """
        Create OpenSearch index with vector mapping
        
        Args:
            vector_dimension: Dimension of vector embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            index_mapping = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "metadata": {"type": "object"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": vector_dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "l2",
                                "engine": "faiss",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 24
                                }
                            }
                        },
                        "timestamp": {"type": "date"}
                    }
                }
            }
            
            url = f"{self.endpoint}/{self.index_name}"
            response = requests.put(
                url,
                json=index_mapping,
                auth=self.auth,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully created index {self.index_name}")
                return True
            else:
                logger.error(f"Failed to create index: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    def index_document(self, document_id: str, content: str, embedding: List[float], 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Index a document with its embedding
        
        Args:
            document_id: Unique document identifier
            content: Document content
            embedding: Document embedding vector
            metadata: Optional document metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from datetime import datetime
            
            doc = {
                "document_id": document_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            url = f"{self.endpoint}/{self.index_name}/_doc/{document_id}"
            response = requests.put(
                url,
                json=doc,
                auth=self.auth,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully indexed document {document_id}")
                return True
            else:
                logger.error(f"Failed to index document: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error indexing document {document_id}: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, 
                      filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional filters to apply
            
        Returns:
            List of similar documents with scores
        """
        try:
            query_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": top_k
                        }
                    }
                },
                "_source": ["document_id", "content", "metadata", "timestamp"]
            }
            
            # Add filters if provided
            if filter_dict:
                query_body["query"] = {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_embedding,
                                        "k": top_k
                                    }
                                }
                            }
                        ],
                        "filter": []
                    }
                }
                
                for key, value in filter_dict.items():
                    query_body["query"]["bool"]["filter"].append({
                        "term": {f"metadata.{key}": value}
                    })
            
            url = f"{self.endpoint}/{self.index_name}/_search"
            response = requests.post(
                url,
                json=query_body,
                auth=self.auth,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                results = response.json()
                documents = []
                
                for hit in results.get("hits", {}).get("hits", []):
                    documents.append({
                        "document_id": hit["_source"]["document_id"],
                        "content": hit["_source"]["content"],
                        "metadata": hit["_source"]["metadata"],
                        "score": hit["_score"],
                        "timestamp": hit["_source"]["timestamp"]
                    })
                
                return documents
            else:
                logger.error(f"Search failed: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def hybrid_search(self, query_embedding: List[float], query_text: str, 
                     top_k: int = 5, weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and text matching
        
        Args:
            query_embedding: Query embedding vector
            query_text: Query text for text search
            top_k: Number of results to return
            weight: Weight for vector search (0-1)
            
        Returns:
            List of hybrid search results
        """
        try:
            query_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_embedding,
                                        "k": top_k,
                                        "boost": weight
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["content^2", "metadata.title^1.5"],
                                    "boost": 1 - weight
                                }
                            }
                        ]
                    }
                },
                "_source": ["document_id", "content", "metadata", "timestamp"]
            }
            
            url = f"{self.endpoint}/{self.index_name}/_search"
            response = requests.post(
                url,
                json=query_body,
                auth=self.auth,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                results = response.json()
                documents = []
                
                for hit in results.get("hits", {}).get("hits", []):
                    documents.append({
                        "document_id": hit["_source"]["document_id"],
                        "content": hit["_source"]["content"],
                        "metadata": hit["_source"]["metadata"],
                        "score": hit["_score"],
                        "timestamp": hit["_source"]["timestamp"]
                    })
                
                return documents
            else:
                logger.error(f"Hybrid search failed: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the index
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.endpoint}/{self.index_name}/_doc/{document_id}"
            response = requests.delete(
                url,
                auth=self.auth,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 404]:  # 404 means document doesn't exist
                logger.info(f"Successfully deleted document {document_id}")
                return True
            else:
                logger.error(f"Failed to delete document: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def update_document(self, document_id: str, content: str, embedding: List[float],
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing document
        
        Args:
            document_id: Document ID to update
            content: Updated content
            embedding: Updated embedding
            metadata: Updated metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from datetime import datetime
            
            doc = {
                "document_id": document_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            url = f"{self.endpoint}/{self.index_name}/_doc/{document_id}"
            response = requests.put(
                url,
                json=doc,
                auth=self.auth,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully updated document {document_id}")
                return True
            else:
                logger.error(f"Failed to update document: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return False
    
    def get_index_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get index statistics
        
        Returns:
            Index statistics or None if error
        """
        try:
            url = f"{self.endpoint}/{self.index_name}/_stats"
            response = requests.get(
                url,
                auth=self.auth,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get index stats: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return None

# AWS4Auth class for OpenSearch authentication
from aws_requests_auth.aws_auth import AWSRequestsAuth

class AWS4Auth(AWSRequestsAuth):
    """AWS4 authentication for OpenSearch requests"""
    
    def __init__(self, credentials, region, service):
        super().__init__(
            aws_access_key=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_token=credentials.token,
            aws_host=self._get_host_from_credentials(),
            aws_region=region,
            aws_service=service
        )
        self._host = None
    
    def _get_host_from_credentials(self):
        if self._host is None:
            # Extract host from endpoint
            from urllib.parse import urlparse
            parsed = urlparse(aws_config.opensearch_endpoint)
            self._host = parsed.hostname
        return self._host
