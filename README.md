# AI Document Processing & Knowledge Retrieval Agents

A comprehensive AI-powered system for document processing, knowledge retrieval, and legal contract analysis using AWS services including Bedrock, S3, Pinecone (vector DB), Lambda, and Textract.

## 🚀 Features

### Core Capabilities
- **Document Processing**: Extract text from PDFs, DOCX, TXT, and HTML files
- **Knowledge Retrieval**: RAG-based question answering and document search
- **Legal Contract Analysis**: Specialized agent for contract review and analysis
- **Vector Search**: Semantic search using Pinecone (vector DB) and embeddings
- **Pipeline Orchestration**: Serverless Lambda-based workflow management

### AWS Services Integration
- **Amazon Bedrock**: LLM operations and embeddings generation
- **Amazon S3**: Document storage and management
- **Amazon Pinecone (vector DB)**: Vector search and indexing
- **AWS Lambda**: Serverless pipeline orchestration
- **Amazon Textract**: OCR and structured data extraction

### Specialized Agents
1. **Document Processor**: Handles various document formats with Textract integration
2. **Knowledge Agent**: Q&A system with RAG capabilities
3. **Legal Agent**: Contract analysis and legal document processing
4. **Pipeline Manager**: Orchestrates complex workflows

## 📋 Prerequisites

- Python 3.9+
- AWS Account with appropriate permissions
- AWS CLI configured
- Required AWS services enabled:
  - Amazon Bedrock
  - Amazon S3
  - Amazon Pinecone (vector DB)
  - AWS Lambda
  - Amazon Textract

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AIAgent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**
   ```bash
   aws configure
   ```

4. **Set up environment variables**
   ```bash
   cp config/env_example.txt .env
   # Edit .env with your AWS credentials and configuration
   ```

5. **Run setup script**
   ```bash
   python examples/setup_and_deployment.py
   ```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Knowledge     │    │   Legal         │
│   Processor     │    │   Agent         │    │   Agent         │
│                 │    │                 │    │                 │
│ • Textract      │    │ • RAG System    │    │ • Contract      │
│ • Multi-format  │    │ • Q&A Engine    │    │   Analysis      │
│ • Text Extract  │    │ • Search        │    │ • Clause        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Pipeline      │
                    │   Manager       │
                    │                 │
                    │ • Orchestration │
                    │ • Lambda        │
                    │ • Workflows     │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   AWS Services  │
                    │                 │
                    │ • Bedrock       │
                    │ • S3            │
                    │ • Pinecone (vector DB)    │
                    │ • Lambda        │
                    │ • Textract      │
                    └─────────────────┘
```

## 📚 Usage Examples

### 1. Document Processing

```python
from agents.document_processor import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process a document
result = processor.process_document("path/to/document.pdf")

if 'error' not in result:
    print(f"Document ID: {result['document_id']}")
    print(f"Word Count: {result['word_count']}")
    print(f"Text Content: {result['text_content'][:200]}...")
```

### 2. Knowledge Retrieval

```python
from agents.knowledge_agent import KnowledgeAgent

# Initialize knowledge agent
agent = KnowledgeAgent()

# Process and index document
result = agent.process_and_index_document("path/to/document.pdf")

# Ask questions
answer = agent.ask_question("What are the key points in this document?")
print(f"Answer: {answer['answer']}")
```

### 3. Legal Contract Analysis

```python
from agents.legal_agent import LegalAgent

# Initialize legal agent
legal_agent = LegalAgent()

# Analyze contract
result = legal_agent.analyze_contract(
    "path/to/contract.pdf",
    contract_type="employment",
    focus_areas=["termination", "payment", "liability"]
)

# Extract specific information
termination_conditions = legal_agent.extract_termination_conditions(
    result['document_id']
)
print(f"Termination Conditions: {termination_conditions['answer']}")
```

### 4. Pipeline Orchestration

```python
from orchestration.pipeline_manager import PipelineManager

# Initialize pipeline manager
manager = PipelineManager()

# Create document processing pipeline
result = manager.create_document_processing_pipeline(
    document_key="s3://bucket/document.pdf",
    document_type="pdf"
)

print(f"Pipeline ID: {result['pipeline_id']}")
print(f"Status: {result['status']}")
```

## 🎯 Use Cases

### Legal Assistant Agent
- **Contract Analysis**: Extract key clauses, terms, and conditions
- **Risk Assessment**: Identify potential risks and areas of concern
- **Termination Conditions**: Quickly find termination clauses
- **Payment Terms**: Extract financial obligations and schedules
- **Compliance Review**: Check for regulatory compliance issues

### Knowledge Retrieval Agent
- **Document Q&A**: Answer questions about company documents
- **Semantic Search**: Find relevant information across documents
- **Document Summarization**: Generate concise summaries
- **Information Extraction**: Extract specific data points
- **Cross-Document Analysis**: Compare information across multiple documents

### Document Processing Agent
- **Multi-Format Support**: Handle PDFs, DOCX, TXT, HTML files
- **OCR Capabilities**: Extract text from scanned documents
- **Structured Data**: Extract tables, forms, and structured information
- **Metadata Extraction**: Extract document metadata and properties
- **Batch Processing**: Process multiple documents efficiently

## 🔧 Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDINGS_MODEL=amazon.titan-embed-text-v1

# S3 Configuration
S3_BUCKET_NAME=ai-agent-documents
S3_PROCESSED_PREFIX=processed/
S3_RAW_PREFIX=raw/

# Pinecone (vector DB) Configuration
OPENSEARCH_ENDPOINT=https://your-domain.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX_NAME=document-embeddings

# Lambda Configuration
LAMBDA_FUNCTION_NAME=document-processor
```

### Document Processing Configuration

```python
# Supported file formats
SUPPORTED_FORMATS = ["pdf", "docx", "txt", "html"]

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# File size limits
MAX_FILE_SIZE_MB = 50
```

### RAG Configuration

```python
# Vector search configuration
VECTOR_DIMENSION = 1536
SIMILARITY_THRESHOLD = 0.7
MAX_CONTEXT_LENGTH = 4000
TOP_K_RESULTS = 5
```

## 🚀 Deployment

### 1. AWS Infrastructure Setup

Run the setup script to create necessary AWS resources:

```bash
python examples/setup_and_deployment.py
```

This will:
- Create S3 bucket for document storage
- Set up IAM roles and policies
- Configure Pinecone (vector DB) domain
- Create Lambda functions
- Set up environment configuration

### 2. Lambda Function Deployment

Deploy Lambda functions with the following configuration:

```bash
# Package the application
zip -r lambda-deployment.zip . -x "*.git*" "*.env*" "examples/*"

# Deploy to Lambda
aws lambda update-function-code \
  --function-name document-processor \
  --zip-file fileb://lambda-deployment.zip
```

### 3. Pinecone (vector DB) Setup

Create Pinecone (vector DB) domain with the following configuration:

```bash
# Create domain
aws opensearch create-domain \
  --domain-name ai-agent-opensearch \
  --cluster-config InstanceType=t3.small.search,InstanceCount=1 \
  --ebs-options EBSEnabled=true,VolumeType=gp3,VolumeSize=20
```

## 📊 Monitoring and Logging

### CloudWatch Integration
- Lambda function logs and metrics
- Pinecone (vector DB) cluster monitoring
- S3 access logs
- Bedrock usage tracking

### Custom Metrics
- Document processing success rate
- Search query performance
- Embedding generation latency
- Pipeline execution time

## 🔒 Security

### IAM Policies
- Principle of least privilege
- Service-specific permissions
- Resource-based policies
- Cross-service access controls

### Data Protection
- Encryption at rest and in transit
- VPC configuration for Pinecone (vector DB)
- S3 bucket policies
- Lambda execution environment isolation

## 🧪 Testing

### Run Demo Scripts

```bash
# Legal Assistant Demo
python examples/legal_assistant_demo.py

# Knowledge Agent Demo
python examples/knowledge_agent_demo.py

# Setup and Deployment
python examples/setup_and_deployment.py
```

### Unit Tests

```bash
# Run tests (when implemented)
python -m pytest tests/
```

## 📈 Performance Optimization

### Scaling Considerations
- Lambda concurrency limits
- Pinecone (vector DB) cluster sizing
- S3 transfer acceleration
- Bedrock rate limits

### Cost Optimization
- Right-size Pinecone (vector DB) instances
- Optimize Lambda memory allocation
- Use S3 Intelligent Tiering
- Monitor Bedrock usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review example scripts
3. Open an issue on GitHub
4. Contact the development team

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced document classification
- [ ] Real-time collaboration features
- [ ] Enhanced security controls
- [ ] Performance monitoring dashboard
- [ ] API gateway integration
- [ ] Mobile application support
- [ ] Advanced analytics and reporting

---

**Built with ❤️ using AWS services and Python**
"# intelligent-doc-processing-ai-agent" 

| Action          | Function Called                           | Description                                                               |
| --------------- | ----------------------------------------- | ------------------------------------------------------------------------- |
| Upload a file   | `process_and_index_document_from_bytes()` | Extracts text → Embeds → Stores in Pinecone → Uploads processed doc to S3 |
| Ask question    | `ask_question()`                          | Executes full **RAG flow** (retrieval + Bedrock LLM)                      |
| Suggest related | `suggest_related_questions()`             | Uses contextual search + LLM text generation                              |
| View stats      | `get_knowledge_base_stats()`              | Aggregates Pinecone + S3 stats                                            |

