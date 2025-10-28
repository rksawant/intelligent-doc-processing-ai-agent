"""
Legal Assistant Agent for contract analysis and legal document processing

✅ Uploads and embeds documents into Pinecone

✅ Retrieves top-k relevant chunks by vector similarity
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

# AWS services
from services import BedrockService, S3Service
from .rag_system import RAGSystem
from .document_processor import DocumentProcessor
from config.config import legal_config

logger = logging.getLogger(__name__)

class LegalAgent:
    """Legal assistant agent for contract analysis and legal document processing"""
    
    def __init__(self):
        self.bedrock_service = BedrockService()
        self.s3_service = S3Service()
        self.rag_system = RAGSystem()
        self.document_processor = DocumentProcessor()
        
        # Legal-specific configuration
        self.contract_types = legal_config.contract_types
        self.key_clauses = legal_config.key_clauses
        
        # Legal system prompt
        self.system_prompt = """You are an expert legal assistant specializing in contract analysis. Your role is to:
1. Analyze legal documents and contracts with precision
2. Extract key legal terms, clauses, and obligations
3. Identify potential risks and areas of concern
4. Provide clear explanations of legal language
5. Answer specific legal questions based on document content
6. Highlight important dates, parties, and financial terms
7. Suggest areas that may require legal review

Always maintain accuracy and cite specific sections when possible. If information is not available in the provided context, clearly state this limitation."""
    
    def analyze_contract(self, file_path: str, contract_type: Optional[str] = None,
                        focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a contract document
        
        Args:
            file_path: Path to contract file
            contract_type: Type of contract (employment, service, nda, etc.)
            focus_areas: Specific areas to focus on
            
        Returns:
            Contract analysis result
        """
        try:
            # Process document
            processing_result = self.document_processor.process_document(file_path)
            
            if 'error' in processing_result:
                return processing_result
            
            document_id = processing_result['document_id']
            text_content = processing_result['text_content']
            
            # Analyze contract using Bedrock
            analysis_result = self.bedrock_service.analyze_legal_document(
                text_content, focus_areas or self.key_clauses
            )
            
            # Extract specific legal information
            legal_info = self._extract_legal_information(text_content, contract_type)
            
            # Index document for future queries
            indexing_result = self.rag_system.index_document(
                document_id,
                text_content,
                {
                    'document_type': 'contract',
                    'contract_type': contract_type,
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            return {
                'success': True,
                'document_id': document_id,
                'contract_type': contract_type,
                'analysis': analysis_result,
                'legal_information': legal_info,
                'processing_result': processing_result,
                'indexing_result': indexing_result,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing contract: {e}")
            return {'error': str(e)}
    
    def analyze_contract_from_bytes(self, file_bytes: bytes, filename: str,
                                  contract_type: Optional[str] = None,
                                  focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze contract from bytes
        
        Args:
            file_bytes: Contract file bytes
            filename: Original filename
            contract_type: Type of contract
            focus_areas: Specific areas to focus on
            
        Returns:
            Contract analysis result
        """
        try:
            # Process document
            processing_result = self.document_processor.process_document_from_bytes(
                file_bytes, filename
            )
            
            if 'error' in processing_result:
                return processing_result
            
            document_id = processing_result['document_id']
            text_content = processing_result['text_content']
            
            # Analyze contract
            analysis_result = self.bedrock_service.analyze_legal_document(
                text_content, focus_areas or self.key_clauses
            )
            
            # Extract specific legal information
            legal_info = self._extract_legal_information(text_content, contract_type)
            
            # Index document
            indexing_result = self.rag_system.index_document(
                document_id,
                text_content,
                {
                    'document_type': 'contract',
                    'contract_type': contract_type,
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            return {
                'success': True,
                'document_id': document_id,
                'contract_type': contract_type,
                'analysis': analysis_result,
                'legal_information': legal_info,
                'processing_result': processing_result,
                'indexing_result': indexing_result,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing contract from bytes: {e}")
            return {'error': str(e)}
    
    def answer_legal_question(self, question: str, document_id: Optional[str] = None,
                            context_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Answer a legal question based on contract content
        
        Args:
            question: Legal question
            document_id: Optional document ID to limit search to
            context_limit: Maximum context chunks to use
            
        Returns:
            Legal answer with context
        """
        try:
            if document_id:
                # Search within specific document
                search_result = self.rag_system.get_document_context(
                    document_id, question, context_limit or 5
                )
                
                if 'error' in search_result:
                    return search_result
                
                # Build context
                context_chunks = search_result['context_chunks']
                context = "\n\n".join([chunk['content'] for chunk in context_chunks])
                
                # Generate legal answer
                answer = self.bedrock_service.chat_with_context(
                    question, context, self.system_prompt
                )
                
                return {
                    'success': True,
                    'question': question,
                    'answer': answer,
                    'document_id': document_id,
                    'context_chunks': len(context_chunks),
                    'sources': [{'document_id': document_id, 'chunk_count': len(context_chunks)}]
                }
            else:
                # Search across all documents
                search_results = self.rag_system.search_documents(question, context_limit or 5)
                
                if 'error' in search_results:
                    return search_results
                
                # Build context from search results
                context_chunks = []
                for doc_result in search_results['results']:
                    for chunk in doc_result['chunks']:
                        context_chunks.append(chunk['content'])
                
                context = "\n\n".join(context_chunks)
                
                # Generate legal answer
                answer = self.bedrock_service.chat_with_context(
                    question, context, self.system_prompt
                )
                
                return {
                    'success': True,
                    'question': question,
                    'answer': answer,
                    'context_chunks': len(context_chunks),
                    'sources': [
                        {
                            'document_id': doc_result['document_id'],
                            'chunk_count': len(doc_result['chunks']),
                            'metadata': doc_result.get('metadata', {})
                        }
                        for doc_result in search_results['results']
                    ]
                }
            
        except Exception as e:
            logger.error(f"Error answering legal question: {e}")
            return {'error': str(e)}
    
    def extract_termination_conditions(self, document_id: str) -> Dict[str, Any]:
        """
        Extract termination conditions from a contract
        
        Args:
            document_id: Document ID
            
        Returns:
            Termination conditions
        """
        try:
            question = "What are the termination conditions and procedures in this contract?"
            return self.answer_legal_question(question, document_id)
            
        except Exception as e:
            logger.error(f"Error extracting termination conditions: {e}")
            return {'error': str(e)}
    
    def extract_payment_terms(self, document_id: str) -> Dict[str, Any]:
        """
        Extract payment terms from a contract
        
        Args:
            document_id: Document ID
            
        Returns:
            Payment terms
        """
        try:
            question = "What are the payment terms, amounts, and schedules in this contract?"
            return self.answer_legal_question(question, document_id)
            
        except Exception as e:
            logger.error(f"Error extracting payment terms: {e}")
            return {'error': str(e)}
    
    def extract_liability_terms(self, document_id: str) -> Dict[str, Any]:
        """
        Extract liability and indemnification terms
        
        Args:
            document_id: Document ID
            
        Returns:
            Liability terms
        """
        try:
            question = "What are the liability, indemnification, and limitation of liability terms in this contract?"
            return self.answer_legal_question(question, document_id)
            
        except Exception as e:
            logger.error(f"Error extracting liability terms: {e}")
            return {'error': str(e)}
    
    def extract_confidentiality_terms(self, document_id: str) -> Dict[str, Any]:
        """
        Extract confidentiality and non-disclosure terms
        
        Args:
            document_id: Document ID
            
        Returns:
            Confidentiality terms
        """
        try:
            question = "What are the confidentiality, non-disclosure, and proprietary information terms in this contract?"
            return self.answer_legal_question(question, document_id)
            
        except Exception as e:
            logger.error(f"Error extracting confidentiality terms: {e}")
            return {'error': str(e)}
    
    def extract_intellectual_property_terms(self, document_id: str) -> Dict[str, Any]:
        """
        Extract intellectual property terms
        
        Args:
            document_id: Document ID
            
        Returns:
            IP terms
        """
        try:
            question = "What are the intellectual property, copyright, and ownership terms in this contract?"
            return self.answer_legal_question(question, document_id)
            
        except Exception as e:
            logger.error(f"Error extracting IP terms: {e}")
            return {'error': str(e)}
    
    def compare_contracts(self, document_id1: str, document_id2: str,
                         focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare two contracts
        
        Args:
            document_id1: First contract document ID
            document_id2: Second contract document ID
            focus_areas: Areas to compare
            
        Returns:
            Contract comparison
        """
        try:
            focus_areas = focus_areas or self.key_clauses
            
            # Get context from both documents
            context1 = self._get_document_context(document_id1)
            context2 = self._get_document_context(document_id2)
            
            if 'error' in context1 or 'error' in context2:
                return {'error': 'Failed to retrieve document contexts'}
            
            # Create comparison prompt
            comparison_prompt = f"""Compare the following two contracts focusing on these areas: {', '.join(focus_areas)}

Contract 1:
{context1}

Contract 2:
{context2}

Please provide a detailed comparison highlighting:
1. Key differences in each focus area
2. Similarities between the contracts
3. Potential risks or concerns
4. Recommendations for negotiation or review

Comparison:"""
            
            comparison = self.bedrock_service.generate_text(comparison_prompt)
            
            return {
                'success': True,
                'contract1_id': document_id1,
                'contract2_id': document_id2,
                'focus_areas': focus_areas,
                'comparison': comparison,
                'comparison_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing contracts: {e}")
            return {'error': str(e)}
    
    def identify_risk_factors(self, document_id: str) -> Dict[str, Any]:
        """
        Identify potential risk factors in a contract
        
        Args:
            document_id: Document ID
            
        Returns:
            Risk factors analysis
        """
        try:
            question = "Identify potential risk factors, red flags, or areas of concern in this contract that may require legal review or negotiation."
            return self.answer_legal_question(question, document_id)
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return {'error': str(e)}
    
    def extract_key_dates(self, document_id: str) -> Dict[str, Any]:
        """
        Extract key dates from a contract
        
        Args:
            document_id: Document ID
            
        Returns:
            Key dates
        """
        try:
            question = "Extract all important dates, deadlines, and time-sensitive terms from this contract including effective dates, expiration dates, and milestone dates."
            return self.answer_legal_question(question, document_id)
            
        except Exception as e:
            logger.error(f"Error extracting key dates: {e}")
            return {'error': str(e)}
    
    def extract_parties_and_obligations(self, document_id: str) -> Dict[str, Any]:
        """
        Extract parties and their obligations
        
        Args:
            document_id: Document ID
            
        Returns:
            Parties and obligations
        """
        try:
            question = "Identify all parties to this contract and their respective obligations, responsibilities, and deliverables."
            return self.answer_legal_question(question, document_id)
            
        except Exception as e:
            logger.error(f"Error extracting parties and obligations: {e}")
            return {'error': str(e)}
    
    def generate_contract_summary(self, document_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive contract summary
        
        Args:
            document_id: Document ID
            
        Returns:
            Contract summary
        """
        try:
            # Get document context
            document_context = self._get_document_context(document_id)
            
            if 'error' in document_context:
                return document_context
            
            # Generate summary
            summary_prompt = f"""Provide a comprehensive summary of this contract including:

1. Contract type and purpose
2. Key parties involved
3. Main obligations and deliverables
4. Financial terms and payment structure
5. Key dates and deadlines
6. Termination conditions
7. Risk factors and areas requiring attention
8. Overall assessment and recommendations

Contract:
{document_context}

Summary:"""
            
            summary = self.bedrock_service.generate_text(summary_prompt)
            
            return {
                'success': True,
                'document_id': document_id,
                'summary': summary,
                'summary_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating contract summary: {e}")
            return {'error': str(e)}
    
    def _extract_legal_information(self, text_content: str, contract_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract specific legal information from contract text
        
        Args:
            text_content: Contract text content
            contract_type: Type of contract
            
        Returns:
            Extracted legal information
        """
        try:
            legal_info = {
                'contract_type': contract_type,
                'parties': self._extract_parties(text_content),
                'dates': self._extract_dates(text_content),
                'financial_terms': self._extract_financial_terms(text_content),
                'key_clauses': self._extract_key_clauses(text_content),
                'termination_conditions': self._extract_termination_conditions_text(text_content)
            }
            
            return legal_info
            
        except Exception as e:
            logger.error(f"Error extracting legal information: {e}")
            return {'error': str(e)}
    
    def _extract_parties(self, text: str) -> List[str]:
        """Extract party names from contract text"""
        try:
            # Simple pattern matching for party names
            party_patterns = [
                r'between\s+([A-Z][a-zA-Z\s&,]+?)(?:\s+and|\s+\()',
                r'party\s+([A-Z][a-zA-Z\s&,]+?)(?:\s+and|\s+\()',
                r'([A-Z][a-zA-Z\s&,]+?)\s+\(.*?party.*?\)'
            ]
            
            parties = []
            for pattern in party_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                parties.extend([match.strip() for match in matches])
            
            return list(set(parties))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting parties: {e}")
            return []
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from contract text"""
        try:
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}\b'
            ]
            
            dates = []
            for pattern in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                dates.extend(matches)
            
            return list(set(dates))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting dates: {e}")
            return []
    
    def _extract_financial_terms(self, text: str) -> List[str]:
        """Extract financial terms from contract text"""
        try:
            financial_patterns = [
                r'\$[\d,]+(?:\.\d{2})?',
                r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|usd)\b',
                r'\b(?:payment|fee|cost|price|amount|compensation)\s*:?\s*\$?[\d,]+(?:\.\d{2})?\b'
            ]
            
            financial_terms = []
            for pattern in financial_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                financial_terms.extend(matches)
            
            return list(set(financial_terms))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting financial terms: {e}")
            return []
    
    def _extract_key_clauses(self, text: str) -> Dict[str, str]:
        """Extract key clauses from contract text"""
        try:
            clauses = {}
            
            for clause_type in self.key_clauses:
                # Create pattern for each clause type
                pattern = rf'\b{clause_type}\b.*?(?=\n\n|\n[A-Z]|\Z)'
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                
                if matches:
                    clauses[clause_type] = matches[0].strip()
            
            return clauses
            
        except Exception as e:
            logger.error(f"Error extracting key clauses: {e}")
            return {}
    
    def _extract_termination_conditions_text(self, text: str) -> List[str]:
        """Extract termination conditions from contract text"""
        try:
            termination_patterns = [
                r'termination.*?(?=\n\n|\n[A-Z]|\Z)',
                r'terminate.*?(?=\n\n|\n[A-Z]|\Z)',
                r'end\s+of\s+contract.*?(?=\n\n|\n[A-Z]|\Z)'
            ]
            
            termination_conditions = []
            for pattern in termination_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                termination_conditions.extend(matches)
            
            return list(set(termination_conditions))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting termination conditions: {e}")
            return []
    
    def _get_document_context(self, document_id: str) -> str:
        """Get full document context"""
        try:
            # Get document content from S3
            s3_key = f"processed/{document_id}.txt"
            content = self.s3_service.get_document_content(s3_key)
            
            if not content:
                return {'error': 'Document not found'}
            
            return content.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error getting document context: {e}")
            return {'error': str(e)}
