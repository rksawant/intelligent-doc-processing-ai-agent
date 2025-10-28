"""
Knowledge Agent Demo - Document Processing and Q&A Example
Demonstrates how to use the knowledge agent for document processing and question answering
"""
import os
import sys
import logging
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.knowledge_agent import KnowledgeAgent
from agents.document_processor import DocumentProcessor
from orchestration.pipeline_manager import PipelineManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_document_processing():
    """Demonstrate document processing capabilities"""
    print("=" * 60)
    print("KNOWLEDGE AGENT DEMO - DOCUMENT PROCESSING")
    print("=" * 60)
    
    # Initialize the knowledge agent
    knowledge_agent = KnowledgeAgent()
    
    # Sample document content (in a real scenario, this would be loaded from a file)
    sample_document = """
    COMPANY POLICY DOCUMENT
    
    ABC Corporation Employee Handbook
    
    Section 1: Introduction
    Welcome to ABC Corporation. This handbook outlines our company policies, procedures, and expectations for all employees.
    
    Section 2: Work Hours
    Regular work hours are Monday through Friday, 9:00 AM to 5:00 PM. Flexible work arrangements may be available with manager approval.
    
    Section 3: Leave Policies
    Employees are entitled to:
    - 15 days of vacation per year
    - 10 days of sick leave per year
    - 5 personal days per year
    - Paid holidays as designated by the company
    
    Section 4: Performance Reviews
    Annual performance reviews are conducted in December. Employees will receive feedback on their performance and set goals for the upcoming year.
    
    Section 5: Code of Conduct
    All employees must adhere to the company's code of conduct, which includes:
    - Professional behavior at all times
    - Respect for colleagues and customers
    - Compliance with all applicable laws and regulations
    - Protection of confidential information
    
    Section 6: Technology Policy
    Company technology resources are for business use only. Personal use is prohibited. All data stored on company systems is the property of ABC Corporation.
    
    Section 7: Termination Policy
    Employment may be terminated by either party with 30 days written notice. The company reserves the right to terminate immediately for cause, including violation of company policies.
    
    Section 8: Contact Information
    For questions about this handbook, please contact Human Resources at hr@abccorp.com or call (555) 123-4567.
    
    This handbook is effective as of January 1, 2024, and supersedes all previous versions.
    """
    
    print("\n1. PROCESSING SAMPLE COMPANY POLICY DOCUMENT")
    print("-" * 40)
    
    # Process the document
    processing_result = knowledge_agent.process_and_index_document_from_bytes(
        sample_document.encode('utf-8'),
        "company_policy_handbook.txt",
        metadata={
            'document_type': 'policy',
            'category': 'employee_handbook',
            'effective_date': '2024-01-01'
        }
    )
    
    if 'error' in processing_result:
        print(f"Error processing document: {processing_result['error']}")
        return None
    
    print(f"✓ Document processed successfully")
    print(f"  Document ID: {processing_result['document_id']}")
    print(f"  Word Count: {processing_result['processing_result']['word_count']}")
    print(f"  Character Count: {processing_result['processing_result']['character_count']}")
    print(f"  Indexed Chunks: {processing_result['indexing_result']['indexed_chunks']}")
    
    return processing_result['document_id']
    
def demo_question_answering(document_id: str):
    """Demonstrate question answering capabilities"""
    print("\n2. QUESTION ANSWERING DEMO")
    print("-" * 40)
    
    # Initialize the knowledge agent
    knowledge_agent = KnowledgeAgent()
    
    # Example questions about the company policy
    questions = [
        "What are the regular work hours?",
        "How many vacation days do employees get per year?",
        "What is the termination policy?",
        "What are the performance review procedures?",
        "What is the technology policy?",
        "How can employees contact Human Resources?",
        "What are the sick leave benefits?",
        "What is the code of conduct?",
        "When are performance reviews conducted?",
        "What happens if someone violates company policies?"
    ]
    
    print(f"  Document ID: {document_id}")
    print(f"  Total Questions: {len(questions)}")
    
    for i, question in enumerate(questions, 1):
        print(f"\n  Question {i}: {question}")
        
        # Answer the question
        answer_result = knowledge_agent.ask_question(question, context_limit=3, document_id=document_id)
        
        if 'error' not in answer_result:
            answer = answer_result.get('answer', 'No answer available')
            context_chunks = answer_result.get('context_chunks', 0)
            sources = answer_result.get('sources', [])
            
            print(f"    Answer: {answer[:200]}...")
            print(f"    Context Chunks: {context_chunks}")
            print(f"    Sources: {len(sources)}")
        else:
            print(f"    Error: {answer_result['error']}")
    
    print("\n3. DOCUMENT SEARCH DEMO")
    print("-" * 40)
    
    # Search for specific information
    search_queries = [
        "vacation leave benefits",
        "termination procedures",
        "performance review process",
        "technology usage policy",
        "code of conduct requirements"
    ]
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n  Search Query {i}: {query}")
        
        # Search documents
        search_result = knowledge_agent.search_documents(query, top_k=3)
        
        if 'error' not in search_result:
            total_results = search_result.get('total_results', 0)
            document_results = search_result.get('document_results', 0)
            results = search_result.get('results', [])
            
            print(f"    Total Results: {total_results}")
            print(f"    Document Results: {document_results}")
            
            for j, result in enumerate(results[:2], 1):  # Show first 2 results
                print(f"    Result {j}:")
                print(f"      Document ID: {result.get('document_id', 'N/A')}")
                chunks = result.get('chunks', [])
                print(f"      Chunks: {len(chunks)}")
                if chunks:
                    print(f"      Top Score: {chunks[0].get('score', 'N/A')}")
        else:
            print(f"    Error: {search_result['error']}")
    
    print("\n4. HYBRID SEARCH DEMO")
    print("-" * 40)
    
    # Perform hybrid search
    hybrid_query = "employee benefits and leave policies"
    
    print(f"  Hybrid Search Query: {hybrid_query}")
    
    hybrid_result = knowledge_agent.hybrid_search(hybrid_query, top_k=3, weight=0.7)
    
    if 'error' not in hybrid_result:
        total_results = hybrid_result.get('total_results', 0)
        document_results = hybrid_result.get('document_results', 0)
        search_type = hybrid_result.get('search_type', 'unknown')
        vector_weight = hybrid_result.get('vector_weight', 0.7)
        
        print(f"    Search Type: {search_type}")
        print(f"    Vector Weight: {vector_weight}")
        print(f"    Total Results: {total_results}")
        print(f"    Document Results: {document_results}")
    else:
        print(f"    Error: {hybrid_result['error']}")
    
    print("\n5. RELATED QUESTIONS SUGGESTION")
    print("-" * 40)
    
    # Suggest related questions
    original_question = "What are the employee benefits?"
    
    print(f"  Original Question: {original_question}")
    
    suggestions_result = knowledge_agent.suggest_related_questions(original_question, 5)
    
    if 'error' not in suggestions_result:
        suggestions = suggestions_result.get('suggestions', [])
        print("  Suggested Related Questions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"    {i}. {suggestion}")
    else:
        print(f"    Error: {suggestions_result['error']}")

def demo_document_summary(document_id: str):
    """Demonstrate document summarization"""
    print("\n6. DOCUMENT SUMMARIZATION DEMO")
    print("-" * 40)
    
    # Initialize the knowledge agent
    knowledge_agent = KnowledgeAgent()
    
    # Get document summary
    summary_result = knowledge_agent.get_document_summary(document_id, max_length=300)
    
    if 'error' not in summary_result:
        summary = summary_result.get('summary', 'No summary available')
        original_length = summary_result.get('original_length', 0)
        summary_length = summary_result.get('summary_length', 0)
        
        print(f"  Document ID: {summary_result['document_id']}")
        print(f"  Original Length: {original_length} characters")
        print(f"  Summary Length: {summary_length} characters")
        print(f"  Summary:")
        print(f"    {summary}")
    else:
        print(f"    Error: {summary_result['error']}")

def demo_key_information_extraction(document_id: str):
    """Demonstrate key information extraction"""
    print("\n7. KEY INFORMATION EXTRACTION DEMO")
    print("-" * 40)
    
    # Initialize the knowledge agent
    knowledge_agent = KnowledgeAgent()
    
    # Extract key information
    information_types = [
        "employee benefits",
        "work hours",
        "termination policy",
        "contact information",
        "performance review process"
    ]
    
    print(f"  Document ID: {document_id}")
    print(f"  Information Types: {', '.join(information_types)}")
    
    extraction_result = knowledge_agent.extract_key_information(document_id, information_types)
    
    if 'error' not in extraction_result:
        extracted_info = extraction_result.get('extracted_information', {})
        
        print("  Extracted Information:")
        for info_type, info_value in extracted_info.items():
            if isinstance(info_value, str) and len(info_value) > 100:
                print(f"    {info_type}: {info_value[:100]}...")
            else:
                print(f"    {info_type}: {info_value}")
    else:
        print(f"    Error: {extraction_result['error']}")

def demo_knowledge_base_management():
    """Demonstrate knowledge base management"""
    print("\n8. KNOWLEDGE BASE MANAGEMENT DEMO")
    print("-" * 40)
    
    # Initialize the knowledge agent
    knowledge_agent = KnowledgeAgent()
    
    # Get knowledge base statistics
    stats_result = knowledge_agent.get_knowledge_base_stats()
    
    if 'error' not in stats_result:
        total_documents = stats_result.get('total_documents', 0)
        processed_documents = stats_result.get('processed_documents', [])
        rag_stats = stats_result.get('rag_stats', {})
        
        print("  Knowledge Base Statistics:")
        print(f"    Total Documents: {total_documents}")
        print(f"    Processed Documents: {len(processed_documents)}")
        print(f"    RAG Stats: {rag_stats}")
        
        if processed_documents:
            print("  Recent Documents:")
            for doc in processed_documents[:3]:  # Show first 3 documents
                print(f"    - {doc.get('key', 'N/A')}")
    else:
        print(f"    Error: {stats_result['error']}")
    
    # Initialize knowledge base
    print("\n  Initializing Knowledge Base...")
    init_result = knowledge_agent.initialize_knowledge_base()
    
    if 'error' not in init_result:
        print("    ✓ Knowledge base initialized successfully")
        message = init_result.get('message', '')
        if message:
            print(f"    Message: {message}")
    else:
        print(f"    Error: {init_result['error']}")

def main():
    """Main demo function"""
    print("AI DOCUMENT PROCESSING & KNOWLEDGE RETRIEVAL AGENTS")
    print("Knowledge Agent Demo")
    print("=" * 60)
    
    try:
        # Run document processing demo
        document_id = demo_document_processing()
        
        if document_id:
            # Run other demos with the processed document
            demo_question_answering(document_id)
            demo_document_summary(document_id)
            demo_key_information_extraction(document_id)
            demo_knowledge_base_management()
        
        print("\n" + "=" * 60)
        print("KNOWLEDGE AGENT DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Document processing and indexing")
        print("✓ Question answering with context")
        print("✓ Document search and retrieval")
        print("✓ Hybrid search capabilities")
        print("✓ Related question suggestions")
        print("✓ Document summarization")
        print("✓ Key information extraction")
        print("✓ Knowledge base management")
        
        print("\nNext Steps:")
        print("1. Configure AWS services (Bedrock, S3, Pinecone)")
        print("2. Set up environment variables")
        print("3. Deploy Lambda functions")
        print("4. Test with real documents")
        print("5. Scale for production use")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\nDemo failed with error: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
