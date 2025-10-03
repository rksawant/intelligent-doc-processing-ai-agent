"""
Legal Assistant Agent Demo - Contract Analysis Example
Demonstrates how to use the legal agent to analyze contracts and answer legal questions
"""
import os
import sys
import logging
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.legal_agent import LegalAgent
from agents.knowledge_agent import KnowledgeAgent
from orchestration.pipeline_manager import PipelineManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_contract_analysis():
    """Demonstrate contract analysis capabilities"""
    print("=" * 60)
    print("LEGAL ASSISTANT AGENT DEMO - CONTRACT ANALYSIS")
    print("=" * 60)
    
    # Initialize the legal agent
    legal_agent = LegalAgent()
    
    # Example contract text (in a real scenario, this would be loaded from a file)
    sample_contract = """
    EMPLOYMENT AGREEMENT
    
    This Employment Agreement (the "Agreement") is entered into on January 15, 2024, between ABC Corporation, a Delaware corporation ("Company"), and John Smith ("Employee").
    
    1. TERM OF EMPLOYMENT
    Employee's employment shall begin on February 1, 2024, and shall continue until terminated in accordance with the terms of this Agreement.
    
    2. COMPENSATION
    Employee shall receive an annual salary of $75,000, payable in accordance with the Company's regular payroll practices.
    
    3. TERMINATION
    Either party may terminate this Agreement with thirty (30) days written notice. The Company may terminate immediately for cause, including but not limited to: (a) violation of company policies, (b) breach of confidentiality obligations, or (c) performance issues.
    
    4. CONFIDENTIALITY
    Employee agrees to maintain the confidentiality of all proprietary and confidential information of the Company, including but not limited to customer lists, business strategies, and technical information.
    
    5. INTELLECTUAL PROPERTY
    All work product, inventions, and intellectual property created by Employee during the course of employment shall be owned by the Company.
    
    6. LIABILITY AND INDEMNIFICATION
    Employee shall indemnify and hold harmless the Company from any claims arising from Employee's willful misconduct or violation of law.
    
    7. GOVERNING LAW
    This Agreement shall be governed by the laws of the State of Delaware.
    
    IN WITNESS WHEREOF, the parties have executed this Agreement on the date first written above.
    """
    
    print("\n1. ANALYZING SAMPLE EMPLOYMENT CONTRACT")
    print("-" * 40)
    
    # Analyze the contract
    analysis_result = legal_agent.analyze_contract_from_bytes(
        sample_contract.encode('utf-8'),
        "employment_agreement.txt",
        contract_type="employment",
        focus_areas=["termination", "payment", "liability", "confidentiality"]
    )
    
    if 'error' in analysis_result:
        print(f"Error analyzing contract: {analysis_result['error']}")
        return
    
    print(f"✓ Contract analyzed successfully")
    print(f"  Document ID: {analysis_result['document_id']}")
    print(f"  Contract Type: {analysis_result['contract_type']}")
    print(f"  Analysis Timestamp: {analysis_result['analysis_timestamp']}")
    
    # Display analysis results
    analysis = analysis_result.get('analysis', {})
    if isinstance(analysis, dict):
        print(f"\n  Key Analysis Points:")
        for key, value in analysis.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"    {key}: {value[:100]}...")
            else:
                print(f"    {key}: {value}")
    
    print("\n2. EXTRACTING SPECIFIC LEGAL INFORMATION")
    print("-" * 40)
    
    document_id = analysis_result['document_id']
    
    # Extract termination conditions
    print("\n  Termination Conditions:")
    termination_result = legal_agent.extract_termination_conditions(document_id)
    if 'error' not in termination_result:
        answer = termination_result.get('answer', 'No answer available')
        print(f"    {answer[:200]}...")
    
    # Extract payment terms
    print("\n  Payment Terms:")
    payment_result = legal_agent.extract_payment_terms(document_id)
    if 'error' not in payment_result:
        answer = payment_result.get('answer', 'No answer available')
        print(f"    {answer[:200]}...")
    
    # Extract liability terms
    print("\n  Liability Terms:")
    liability_result = legal_agent.extract_liability_terms(document_id)
    if 'error' not in liability_result:
        answer = liability_result.get('answer', 'No answer available')
        print(f"    {answer[:200]}...")
    
    print("\n3. ANSWERING LEGAL QUESTIONS")
    print("-" * 40)
    
    # Ask specific legal questions
    questions = [
        "What are the termination conditions in this contract?",
        "What is the employee's annual salary?",
        "What happens to intellectual property created during employment?",
        "What are the confidentiality obligations?",
        "What is the notice period for termination?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n  Question {i}: {question}")
        answer_result = legal_agent.answer_legal_question(question, document_id)
        
        if 'error' not in answer_result:
            answer = answer_result.get('answer', 'No answer available')
            print(f"    Answer: {answer[:300]}...")
        else:
            print(f"    Error: {answer_result['error']}")
    
    print("\n4. GENERATING CONTRACT SUMMARY")
    print("-" * 40)
    
    summary_result = legal_agent.generate_contract_summary(document_id)
    if 'error' not in summary_result:
        summary = summary_result.get('summary', 'No summary available')
        print(f"  Contract Summary:")
        print(f"    {summary[:500]}...")
    else:
        print(f"    Error: {summary_result['error']}")
    
    print("\n5. IDENTIFYING RISK FACTORS")
    print("-" * 40)
    
    risk_result = legal_agent.identify_risk_factors(document_id)
    if 'error' not in risk_result:
        risk_assessment = risk_result.get('answer', 'No risk assessment available')
        print(f"  Risk Assessment:")
        print(f"    {risk_assessment[:500]}...")
    else:
        print(f"    Error: {risk_result['error']}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)

def demo_pipeline_orchestration():
    """Demonstrate pipeline orchestration capabilities"""
    print("\n" + "=" * 60)
    print("PIPELINE ORCHESTRATION DEMO")
    print("=" * 60)
    
    # Initialize pipeline manager
    pipeline_manager = PipelineManager()
    
    print("\n1. DOCUMENT PROCESSING PIPELINE")
    print("-" * 40)
    
    # Example document processing pipeline
    document_key = "sample_contract.pdf"  # In real scenario, this would be an S3 key
    
    print(f"  Processing document: {document_key}")
    print("  Note: This is a demo - actual S3 document processing would require real files")
    
    # Simulate pipeline execution
    pipeline_result = {
        'pipeline_id': 'demo_pipeline_001',
        'pipeline_type': 'document_processing',
        'status': 'completed',
        'message': 'Demo pipeline completed successfully'
    }
    
    print(f"  Pipeline ID: {pipeline_result['pipeline_id']}")
    print(f"  Status: {pipeline_result['status']}")
    print(f"  Message: {pipeline_result['message']}")
    
    print("\n2. CONTRACT ANALYSIS PIPELINE")
    print("-" * 40)
    
    # Example contract analysis pipeline
    contract_key = "employment_contract.pdf"
    
    print(f"  Analyzing contract: {contract_key}")
    print("  Note: This is a demo - actual contract analysis would require real files")
    
    # Simulate pipeline execution
    contract_pipeline_result = {
        'pipeline_id': 'demo_contract_pipeline_001',
        'pipeline_type': 'contract_analysis',
        'status': 'completed',
        'message': 'Contract analysis pipeline completed successfully'
    }
    
    print(f"  Pipeline ID: {contract_pipeline_result['pipeline_id']}")
    print(f"  Status: {contract_pipeline_result['status']}")
    print(f"  Message: {contract_pipeline_result['message']}")
    
    print("\n3. KNOWLEDGE SEARCH PIPELINE")
    print("-" * 40)
    
    # Example knowledge search pipeline
    search_query = "What are the termination conditions?"
    
    print(f"  Search query: {search_query}")
    print("  Note: This is a demo - actual search would require indexed documents")
    
    # Simulate pipeline execution
    search_pipeline_result = {
        'pipeline_id': 'demo_search_pipeline_001',
        'pipeline_type': 'knowledge_search',
        'status': 'completed',
        'message': 'Knowledge search pipeline completed successfully'
    }
    
    print(f"  Pipeline ID: {search_pipeline_result['pipeline_id']}")
    print(f"  Status: {search_pipeline_result['status']}")
    print(f"  Message: {search_pipeline_result['message']}")
    
    print("\n" + "=" * 60)
    print("PIPELINE ORCHESTRATION DEMO COMPLETED")
    print("=" * 60)

def demo_knowledge_retrieval():
    """Demonstrate knowledge retrieval capabilities"""
    print("\n" + "=" * 60)
    print("KNOWLEDGE RETRIEVAL DEMO")
    print("=" * 60)
    
    # Initialize knowledge agent
    knowledge_agent = KnowledgeAgent()
    
    print("\n1. KNOWLEDGE BASE STATISTICS")
    print("-" * 40)
    
    # Get knowledge base stats
    stats_result = knowledge_agent.get_knowledge_base_stats()
    
    if 'error' not in stats_result:
        print("  Knowledge Base Statistics:")
        print(f"    Total Documents: {stats_result.get('total_documents', 0)}")
        print(f"    RAG Stats: {stats_result.get('rag_stats', 'N/A')}")
    else:
        print(f"    Error: {stats_result['error']}")
    
    print("\n2. SUGGESTING RELATED QUESTIONS")
    print("-" * 40)
    
    # Suggest related questions
    original_question = "What are the termination conditions?"
    
    print(f"  Original Question: {original_question}")
    
    suggestions_result = knowledge_agent.suggest_related_questions(original_question, 3)
    
    if 'error' not in suggestions_result:
        suggestions = suggestions_result.get('suggestions', [])
        print("  Related Questions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"    {i}. {suggestion}")
    else:
        print(f"    Error: {suggestions_result['error']}")
    
    print("\n3. DOCUMENT SEARCH")
    print("-" * 40)
    
    # Search documents
    search_query = "employment termination notice period"
    
    print(f"  Search Query: {search_query}")
    
    search_result = knowledge_agent.search_documents(search_query, top_k=3)
    
    if 'error' not in search_result:
        print("  Search Results:")
        print(f"    Total Results: {search_result.get('total_results', 0)}")
        print(f"    Document Results: {search_result.get('document_results', 0)}")
        
        results = search_result.get('results', [])
        for i, result in enumerate(results[:2], 1):  # Show first 2 results
            print(f"    Result {i}:")
            print(f"      Document ID: {result.get('document_id', 'N/A')}")
            print(f"      Chunks: {len(result.get('chunks', []))}")
    else:
        print(f"    Error: {search_result['error']}")
    
    print("\n" + "=" * 60)
    print("KNOWLEDGE RETRIEVAL DEMO COMPLETED")
    print("=" * 60)

def main():
    """Main demo function"""
    print("AI DOCUMENT PROCESSING & KNOWLEDGE RETRIEVAL AGENTS")
    print("Legal Assistant Agent Demo")
    print("=" * 60)
    
    try:
        # Run demos
        demo_contract_analysis()
        demo_pipeline_orchestration()
        demo_knowledge_retrieval()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. Configure AWS credentials and services")
        print("2. Set up S3 bucket for document storage")
        print("3. Configure OpenSearch for vector search")
        print("4. Deploy Lambda functions for orchestration")
        print("5. Test with real documents and contracts")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\nDemo failed with error: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
