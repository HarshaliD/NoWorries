"""Configuration for Panic & Anxiety Support Chatbot"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ Please set GOOGLE_API_KEY in your .env file")

# Paths
VECTORSTORE_PATH = "./vectorstore"
LOG_DIR = "./logs"
AUDIT_LOG = os.path.join(LOG_DIR, "conversations.jsonl")

# Create directories
os.makedirs(LOG_DIR, exist_ok=True)

# Model Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 500
RETRIEVAL_K = 3

# Safety Thresholds
MIN_CONTEXT_LENGTH = 100

# Disclaimer
DISCLAIMER = """

⚠️ **Important**: This chatbot provides educational information only. It does NOT replace professional mental health care. If you're in crisis, please contact emergency services or crisis helplines immediately."""
