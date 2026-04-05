import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class Settings:
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "NVIDIA").upper()
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    
    # Base directory refers to the 'backend' folder
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    DATA_DIR = BASE_DIR / "data"
    POLICIES_DIR = DATA_DIR / "Policies"
    EMBEDDINGS_DIR = DATA_DIR / "savedEmbeddings"
    
settings = Settings()
