"""
AWS Bedrock service wrapper for LLM operations and embeddings
"""
import json
import boto3
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError
import logging

from config.config import aws_config, get_aws_session

logger = logging.getLogger(__name__)

class BedrockService:
    """AWS Bedrock service wrapper"""
    
    def __init__(self):
        self.session = get_aws_session()
        self.bedrock_runtime = self.session.client('bedrock-runtime')
        self.bedrock = self.session.client('bedrock')
        self.model_id = aws_config.bedrock_model_id
        self.embeddings_model = aws_config.bedrock_embeddings_model
    
    def generate_text(self, prompt: str, max_tokens: int = 4000, temperature: float = 0.7) -> str:
        """
        Generate text using Claude model via Bedrock
        
        Args:
            prompt: Input prompt for text generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        try:
            # Format for Claude
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        except ClientError as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Titan embeddings
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            
            for text in texts:
                body = json.dumps({
                    "inputText": text
                })
                
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.embeddings_model,
                    body=body,
                    contentType="application/json"
                )
                
                response_body = json.loads(response['body'].read())
                embedding = response_body['embedding']
                embeddings.append(embedding)
            
            return embeddings
            
        except ClientError as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def chat_with_context(self, question: str, context: str, system_prompt: Optional[str] = None) -> str:
        """
        Chat with context using RAG pattern
        
        Args:
            question: User question
            context: Retrieved context
            system_prompt: Optional system prompt
            
        Returns:
            Answer based on context
        """
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
            Use only the information in the context to answer questions. If the context doesn't contain enough 
            information to answer the question, say so clearly."""
        
        prompt = f"""Context:
{context}

Question: {question}

Please answer the question based on the provided context. If the context doesn't contain sufficient information, please indicate that clearly."""

        return self.generate_text(prompt)
    
    def summarize_document(self, text: str, max_length: int = 500) -> str:
        """
        Summarize a document
        
        Args:
            text: Document text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Document summary
        """
        prompt = f"""Please provide a concise summary of the following document in no more than {max_length} words:

{text}

Summary:"""
        
        return self.generate_text(prompt, max_tokens=max_length)
    
    def extract_key_information(self, text: str, information_types: List[str]) -> Dict[str, str]:
        """
        Extract specific types of information from text
        
        Args:
            text: Text to analyze
            information_types: List of information types to extract
            
        Returns:
            Dictionary mapping information types to extracted values
        """
        types_str = ", ".join(information_types)
        
        prompt = f"""Extract the following information from the text: {types_str}

Text:
{text}

Please provide the extracted information in the following JSON format:
{{"information_type": "extracted_value", ...}}

Extracted Information:"""
        
        response = self.generate_text(prompt)
        
        try:
            # Try to parse JSON response
            return json.loads(response)
        except json.JSONDecodeError:
            # If not JSON, return as simple text
            return {"extracted_info": response}
    
    def analyze_legal_document(self, text: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Analyze legal document for specific areas
        
        Args:
            text: Legal document text
            focus_areas: List of areas to focus on (e.g., termination, payment, liability)
            
        Returns:
            Analysis results
        """
        if focus_areas is None:
            focus_areas = ["termination", "payment", "liability", "confidentiality", "intellectual_property"]
        
        areas_str = ", ".join(focus_areas)
        
        prompt = f"""Analyze this legal document and extract information about the following areas: {areas_str}

Document:
{text}

Please provide a structured analysis in JSON format with the following structure:
{{
    "document_type": "type of document",
    "key_clauses": {{
        "area": "relevant text from document",
        ...
    }},
    "summary": "brief overview of the document",
    "risk_factors": ["list of potential risks or concerns"],
    "recommendations": ["list of recommendations"]
}}

Analysis:"""
        
        response = self.generate_text(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"analysis": response}

