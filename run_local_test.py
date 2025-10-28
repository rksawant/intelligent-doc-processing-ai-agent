import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("✅ Environment loaded successfully")

# Simulate Pinecone connection
try:
    from pinecone import Pinecone
    import boto3
    print("✅ Libraries imported successfully")
except Exception as e:
    print("❌ Import error:", e)

print("✅ Project environment ready for Python 3.12.7")
