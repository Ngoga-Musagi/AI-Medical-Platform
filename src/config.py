"""
Centralized Configuration Manager
All settings loaded from environment variables
Single source of truth for the entire system
"""

import os
from typing import Optional
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers - matches environment variable values"""
    # API-based (closed-source)
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    
    # Open-source via Hugging Face
    LLAMA_3_8B = "llama-3-8b"
    LLAMA_3_70B = "llama-3-70b"
    MISTRAL_7B = "mistral-7b"
    MIXTRAL_8X7B = "mixtral-8x7b"
    MEDITRON_7B = "meditron-7b"
    BIOMISTRAL_7B = "biomistral-7b"
    
    # Open-source via Ollama (local)
    OLLAMA_LLAMA3 = "ollama-llama3"
    OLLAMA_MISTRAL = "ollama-mistral"
    OLLAMA_MEDITRON = "ollama-meditron"


class Config:
    """
    Centralized configuration - ALL settings in one place
    Change .env file to update entire system
    """
    
    # ============================================
    # LLM Configuration (Choose ONE provider)
    # ============================================
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama-mistral")  # Default: ollama-mistral (Docker)
    
    # API Keys (only needed for API-based providers)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Support both GEMINI_API_KEY and GOOGLE_API_KEY (common mixup)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    # Hugging Face Token (optional, for gated models)
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    
    # Model Names (configurable per provider)
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # Ollama Configuration (Docker service or external)
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

    # LLM Settings
    USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
    LLM_DEVICE = os.getenv("LLM_DEVICE", "auto")  # "cuda", "cpu", or "auto"
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    EXTRACTION_RETRY_COUNT = int(os.getenv("EXTRACTION_RETRY_COUNT", "3"))
    EXTRACTION_RETRY_DELAY = float(os.getenv("EXTRACTION_RETRY_DELAY", "1.0"))
    
    # ============================================
    # Neo4j Configuration
    # ============================================
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # ============================================
    # File Paths
    # ============================================
    GUIDELINES_PATH = os.getenv("GUIDELINES_PATH", "outputs/guidelines.json")
    CLINICAL_NOTES_PATH = os.getenv("CLINICAL_NOTES_PATH", "outputs/clinical_notes.json")
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models/entity_extractor")
    PREDICTION_LOG_DIR = os.getenv("PREDICTION_LOG_DIR", "outputs/predictions")
    
    # ============================================
    # API Configuration
    # ============================================
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_WORKERS = int(os.getenv("API_WORKERS", "4"))
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    # ============================================
    # Dashboard Configuration
    # ============================================
    DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8050"))
    DASHBOARD_DEBUG = os.getenv("DASHBOARD_DEBUG", "false").lower() == "true"
    
    # ============================================
    # Logging Configuration
    # ============================================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "outputs/system.log")
    
    # ============================================
    # Google Cloud (Optional)
    # ============================================
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    GCP_REGION = os.getenv("GCP_REGION", "us-central1")
    
    @classmethod
    def get_llm_provider(cls) -> LLMProvider:
        """Get configured LLM provider as enum"""
        try:
            return LLMProvider(cls.LLM_PROVIDER)
        except ValueError:
            raise ValueError(
                f"Invalid LLM_PROVIDER: {cls.LLM_PROVIDER}. "
                f"Valid options: {[p.value for p in LLMProvider]}"
            )
    
    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """Get API key for configured provider"""
        provider = cls.get_llm_provider()
        
        if provider == LLMProvider.CLAUDE:
            return cls.ANTHROPIC_API_KEY
        elif provider == LLMProvider.OPENAI:
            return cls.OPENAI_API_KEY
        elif provider == LLMProvider.GEMINI:
            return cls.GEMINI_API_KEY
        else:
            return None  # Open-source models don't need API keys
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        provider = cls.get_llm_provider()
        
        # Check if API key is needed and present
        if provider in [LLMProvider.CLAUDE, LLMProvider.OPENAI, LLMProvider.GEMINI]:
            api_key = cls.get_api_key()
            if not api_key:
                print(f"⚠️  WARNING: {provider.value} requires API key but none found!")
                print(f"   Set {provider.value.upper()}_API_KEY in .env file")
                return False
        
        # Check Neo4j configuration
        if not cls.NEO4J_URI or not cls.NEO4J_PASSWORD:
            print("⚠️  WARNING: Neo4j configuration incomplete")
            return False
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without secrets)"""
        print("\n" + "="*60)
        print("SYSTEM CONFIGURATION")
        print("="*60)
        print(f"LLM Provider: {cls.LLM_PROVIDER}")
        print(f"Use Quantization: {cls.USE_QUANTIZATION}")
        print(f"Device: {cls.LLM_DEVICE}")
        print(f"Ollama Host: {cls.OLLAMA_HOST}")
        print(f"Neo4j URI: {cls.NEO4J_URI}")
        print(f"API Port: {cls.API_PORT}")
        print(f"Dashboard Port: {cls.DASHBOARD_PORT}")
        print(f"Guidelines: {cls.GUIDELINES_PATH}")
        
        # Show if API keys are set (but not the actual keys)
        if cls.ANTHROPIC_API_KEY:
            print(f"Claude API Key: {'*' * 20} (set)")
        if cls.OPENAI_API_KEY:
            print(f"OpenAI API Key: {'*' * 20} (set)")
        if cls.GEMINI_API_KEY:
            print(f"Gemini API Key: {'*' * 20} (set)")
        
        print("="*60 + "\n")


# Create singleton instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    config.print_config()
    
    if config.validate():
        print("✅ Configuration valid")
    else:
        print("❌ Configuration has issues")
