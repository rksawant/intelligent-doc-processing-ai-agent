"""
Setup and Deployment Script for AI Document Processing & Knowledge Retrieval Agents
This script helps set up the AWS infrastructure and deploy the system
"""
import os
import sys
import json
import boto3
import logging
from typing import Dict, Any, List
from botocore.exceptions import ClientError

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import aws_config, document_config, rag_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSSetupManager:
    """Manager for AWS infrastructure setup and deployment"""
    
    def __init__(self):
        self.session = boto3.Session(
            region_name=aws_config.aws_region
        )
        self.s3_client = self.session.client('s3')
        self.lambda_client = self.session.client('lambda')
        self.iam_client = self.session.client('iam')
        self.opensearch_client = self.session.client('opensearch')
        self.bedrock_client = self.session.client('bedrock')
        
    def check_aws_credentials(self) -> bool:
        """Check if AWS credentials are configured"""
        try:
            sts_client = self.session.client('sts')
            sts_client.get_caller_identity()
            print("✓ AWS credentials are configured")
            return True
        except ClientError as e:
            print(f"✗ AWS credentials error: {e}")
            return False
    
    def check_bedrock_access(self) -> bool:
        """Check if Bedrock access is available"""
        try:
            # List available models
            response = self.bedrock_client.list_foundation_models()
            models = response.get('modelSummaries', [])
            
            print(f"✓ Bedrock access available - {len(models)} models found")
            
            # Check for specific models
            model_ids = [model['modelId'] for model in models]
            required_models = [
                aws_config.bedrock_model_id,
                aws_config.bedrock_embeddings_model
            ]
            
            for model_id in required_models:
                if model_id in model_ids:
                    print(f"  ✓ Required model available: {model_id}")
                else:
                    print(f"  ✗ Required model not found: {model_id}")
            
            return True
            
        except ClientError as e:
            print(f"✗ Bedrock access error: {e}")
            return False
    
    def setup_s3_bucket(self) -> bool:
        """Set up S3 bucket for document storage"""
        try:
            bucket_name = aws_config.s3_bucket_name
            
            # Check if bucket exists
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
                print(f"✓ S3 bucket '{bucket_name}' already exists")
                return True
            except ClientError:
                pass
            
            # Create bucket
            if aws_config.aws_region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': aws_config.aws_region}
                )
            
            print(f"✓ S3 bucket '{bucket_name}' created successfully")
            
            # Set up bucket policy for document processing
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "AllowTextractAccess",
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "textract.amazonaws.com"
                        },
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{bucket_name}/*"
                    }
                ]
            }
            
            self.s3_client.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(bucket_policy)
            )
            
            print("✓ S3 bucket policy configured for Textract access")
            return True
            
        except ClientError as e:
            print(f"✗ S3 setup error: {e}")
            return False
    
    def setup_iam_roles(self) -> Dict[str, str]:
        """Set up IAM roles for Lambda functions"""
        try:
            roles = {}
            
            # Lambda execution role
            lambda_role_name = "ai-agent-lambda-execution-role"
            
            try:
                response = self.iam_client.get_role(RoleName=lambda_role_name)
                lambda_role_arn = response['Role']['Arn']
                print(f"✓ Lambda execution role already exists: {lambda_role_arn}")
                roles['lambda_execution_role'] = lambda_role_arn
            except ClientError:
                # Create Lambda execution role
                assume_role_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "lambda.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }
                
                role_response = self.iam_client.create_role(
                    RoleName=lambda_role_name,
                    AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                    Description="Execution role for AI Agent Lambda functions"
                )
                
                lambda_role_arn = role_response['Role']['Arn']
                roles['lambda_execution_role'] = lambda_role_arn
                
                # Attach basic execution policy
                self.iam_client.attach_role_policy(
                    RoleName=lambda_role_name,
                    PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
                )
                
                # Attach additional policies
                additional_policies = [
                    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
                    "arn:aws:iam::aws:policy/AmazonBedrockFullAccess",
                    "arn:aws:iam::aws:policy/AmazonTextractFullAccess",
                    "arn:aws:iam::aws:policy/AmazonOpenSearchFullAccess"
                ]
                
                for policy_arn in additional_policies:
                    try:
                        self.iam_client.attach_role_policy(
                            RoleName=lambda_role_name,
                            PolicyArn=policy_arn
                        )
                    except ClientError as e:
                        print(f"  Warning: Could not attach policy {policy_arn}: {e}")
                
                print(f"✓ Lambda execution role created: {lambda_role_arn}")
            
            return roles
            
        except ClientError as e:
            print(f"✗ IAM setup error: {e}")
            return {}
    
    def create_lambda_functions(self, lambda_role_arn: str) -> Dict[str, str]:
        """Create Lambda functions for the system"""
        try:
            functions = {}
            
            # Lambda function configurations
            lambda_configs = {
                'document-processor': {
                    'description': 'Document processing and indexing Lambda function',
                    'timeout': 300,
                    'memory_size': 512,
                    'handler': 'orchestration.lambda_functions.lambda_handler'
                },
                'embedding-generator': {
                    'description': 'Embedding generation Lambda function',
                    'timeout': 120,
                    'memory_size': 512,
                    'handler': 'orchestration.lambda_functions.lambda_handler'
                },
                'search-agent': {
                    'description': 'Document search Lambda function',
                    'timeout': 60,
                    'memory_size': 256,
                    'handler': 'orchestration.lambda_functions.lambda_handler'
                },
                'contract-analyzer': {
                    'description': 'Contract analysis Lambda function',
                    'timeout': 600,
                    'memory_size': 1024,
                    'handler': 'orchestration.lambda_functions.lambda_handler'
                },
                'qa-agent': {
                    'description': 'Question answering Lambda function',
                    'timeout': 120,
                    'memory_size': 512,
                    'handler': 'orchestration.lambda_functions.lambda_handler'
                }
            }
            
            for function_name, config in lambda_configs.items():
                try:
                    # Check if function exists
                    response = self.lambda_client.get_function(FunctionName=function_name)
                    function_arn = response['Configuration']['FunctionArn']
                    print(f"✓ Lambda function '{function_name}' already exists: {function_arn}")
                    functions[function_name] = function_arn
                except ClientError:
                    # Create function (this would require deployment package)
                    print(f"  Note: Lambda function '{function_name}' needs to be created manually")
                    print(f"    - Use AWS CLI or Console to create the function")
                    print(f"    - Upload the deployment package")
                    print(f"    - Configure environment variables")
                    print(f"    - Set timeout: {config['timeout']} seconds")
                    print(f"    - Set memory: {config['memory_size']} MB")
                    print(f"    - Set handler: {config['handler']}")
                    print(f"    - Set role: {lambda_role_arn}")
                    print()
            
            return functions
            
        except ClientError as e:
            print(f"✗ Lambda setup error: {e}")
            return {}
    
    def setup_opensearch(self) -> bool:
        """Set up OpenSearch domain for vector search"""
        try:
            domain_name = "ai-agent-opensearch"
            
            # Check if domain exists
            try:
                response = self.opensearch_client.describe_domain(DomainName=domain_name)
                domain_status = response['DomainStatus']['Processing']
                if domain_status:
                    print(f"✓ OpenSearch domain '{domain_name}' already exists")
                    return True
            except ClientError:
                pass
            
            # Create OpenSearch domain
            print(f"  Note: OpenSearch domain '{domain_name}' needs to be created manually")
            print("    - Use AWS Console or CLI to create the domain")
            print("    - Choose appropriate instance type and storage")
            print("    - Configure access policies")
            print("    - Enable fine-grained access control")
            print("    - Set up VPC and security groups")
            print("    - Configure the endpoint in your environment variables")
            
            return True
            
        except ClientError as e:
            print(f"✗ OpenSearch setup error: {e}")
            return False
    
    def create_environment_file(self) -> bool:
        """Create environment configuration file"""
        try:
            env_content = f"""# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION={aws_config.aws_region}

# Bedrock Configuration
BEDROCK_MODEL_ID={aws_config.bedrock_model_id}
BEDROCK_EMBEDDINGS_MODEL={aws_config.bedrock_embeddings_model}

# S3 Configuration
S3_BUCKET_NAME={aws_config.s3_bucket_name}
S3_PROCESSED_PREFIX={aws_config.s3_processed_prefix}
S3_RAW_PREFIX={aws_config.s3_raw_prefix}

# OpenSearch Configuration
OPENSEARCH_ENDPOINT=https://your-opensearch-domain.{aws_config.aws_region}.es.amazonaws.com
OPENSEARCH_INDEX_NAME={aws_config.opensearch_index_name}

# Lambda Configuration
LAMBDA_FUNCTION_NAME={aws_config.lambda_function_name}

# Document Processing Configuration
MAX_FILE_SIZE_MB={document_config.max_file_size_mb}
SUPPORTED_FORMATS={document_config.supported_formats}
CHUNK_SIZE={document_config.chunk_size}
CHUNK_OVERLAP={document_config.chunk_overlap}

# RAG Configuration
VECTOR_DIMENSION={rag_config.vector_dimension}
SIMILARITY_THRESHOLD={rag_config.similarity_threshold}
MAX_CONTEXT_LENGTH={rag_config.max_context_length}
TOP_K_RESULTS={rag_config.top_k_results}
"""
            
            with open('.env', 'w') as f:
                f.write(env_content)
            
            print("✓ Environment file '.env' created")
            print("  Please update the values with your actual AWS credentials and endpoints")
            return True
            
        except Exception as e:
            print(f"✗ Environment file creation error: {e}")
            return False
    
    def run_setup(self) -> bool:
        """Run the complete setup process"""
        print("AI DOCUMENT PROCESSING & KNOWLEDGE RETRIEVAL AGENTS")
        print("AWS Infrastructure Setup")
        print("=" * 60)
        
        success = True
        
        # Check AWS credentials
        if not self.check_aws_credentials():
            success = False
        
        # Check Bedrock access
        if not self.check_bedrock_access():
            success = False
        
        # Set up S3 bucket
        if not self.setup_s3_bucket():
            success = False
        
        # Set up IAM roles
        roles = self.setup_iam_roles()
        if not roles:
            success = False
        
        # Create Lambda functions
        if roles.get('lambda_execution_role'):
            functions = self.create_lambda_functions(roles['lambda_execution_role'])
            if not functions:
                success = False
        
        # Set up OpenSearch
        if not self.setup_opensearch():
            success = False
        
        # Create environment file
        if not self.create_environment_file():
            success = False
        
        print("\n" + "=" * 60)
        if success:
            print("SETUP COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print("\nNext Steps:")
            print("1. Update the .env file with your actual AWS credentials")
            print("2. Create OpenSearch domain manually")
            print("3. Deploy Lambda functions with deployment packages")
            print("4. Test the system with sample documents")
            print("5. Configure monitoring and logging")
        else:
            print("SETUP COMPLETED WITH ERRORS")
            print("=" * 60)
            print("\nPlease review the errors above and:")
            print("1. Fix any configuration issues")
            print("2. Ensure proper AWS permissions")
            print("3. Complete manual setup steps")
            print("4. Run the setup again")
        
        return success

def main():
    """Main setup function"""
    try:
        setup_manager = AWSSetupManager()
        success = setup_manager.run_setup()
        
        if success:
            print("\n✓ Setup completed successfully!")
        else:
            print("\n✗ Setup completed with errors. Please review and fix issues.")
            
    except Exception as e:
        logger.error(f"Setup error: {e}")
        print(f"\nSetup failed with error: {e}")
        print("Please check your AWS configuration and try again.")

if __name__ == "__main__":
    main()
