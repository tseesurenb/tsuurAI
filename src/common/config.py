"""
TsuurAI Configuration
Shared settings for batch and streaming modes
"""

import os
from pathlib import Path

# Directories
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = SCRIPT_DIR / "models"
PROMPTS_DIR = SCRIPT_DIR / "prompts"
DOMAINS_DIR = PROMPTS_DIR / "domains"
DATA_DIR = SCRIPT_DIR / "data"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Set environment variables for model caching
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR / "huggingface")
os.environ["XDG_CACHE_HOME"] = str(MODELS_DIR)

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Language codes mapping
LANGUAGE_CODES = {
    "English": {"whisper": "en", "mms": "eng"},
    "Mongolian": {"whisper": "mn", "mms": "mon"},
}

# Model information database
MODEL_INFO = {
    "whisper": {
        "tiny": {"params": "39M", "size_gb": 0.15, "speed": "~32x", "languages": 99, "quality": "Basic"},
        "base": {"params": "74M", "size_gb": 0.29, "speed": "~16x", "languages": 99, "quality": "Good"},
        "small": {"params": "244M", "size_gb": 0.97, "speed": "~6x", "languages": 99, "quality": "Better"},
        "medium": {"params": "769M", "size_gb": 3.1, "speed": "~2x", "languages": 99, "quality": "Great"},
        "large-v3": {"params": "1550M", "size_gb": 6.2, "speed": "~1x", "languages": 99, "quality": "Best"},
    },
    "meta_mms": {
        "mms-1b-all": {"params": "1000M", "size_gb": 4.0, "speed": "~2x", "languages": 1162, "quality": "Great"},
    },
}

# Model comparison data
MODEL_COMPARISON = {
    "whisper": {
        "strength": "Best zero-shot accuracy",
        "weakness": "Hard to fine-tune",
        "best_for": "Quick deployment, general use"
    },
    "meta_mms": {
        "strength": "1162 language support",
        "weakness": "Inconsistent quality",
        "best_for": "Rare/low-resource languages"
    },
}

# Local LLM models info
LOCAL_LLM_INFO = {
    "Mongolian-Llama3": {
        "model_id": "Dorjzodovsuren/Mongolian_Llama3-v0.1",
        "base_model": "unsloth/llama-3-8b-bnb-4bit",
        "size_gb": 5.0,
        "languages": "Mongolian, English",
        "description": "First Mongolian instruction-tuned LLM"
    },
    "Qwen3-8B": {
        "model_id": "Qwen/Qwen3-8B",
        "base_model": None,
        "size_gb": 5.0,
        "languages": "119 languages",
        "description": "Best multilingual coverage"
    },
}

# OpenAI API models info
API_LLM_INFO = {
    "gpt-4o": {"cost": "$2.50/$10 per 1M", "speed": "Fast", "quality": "Excellent"},
    "gpt-4o-mini": {"cost": "$0.15/$0.60 per 1M", "speed": "Very fast", "quality": "Good"},
    "gpt-4-turbo": {"cost": "$10/$30 per 1M", "speed": "Medium", "quality": "Excellent"},
    "o1-mini": {"cost": "$1.10/$4.40 per 1M", "speed": "Slower", "quality": "Best reasoning"},
    "o1": {"cost": "$7.50/$30 per 1M", "speed": "Slow", "quality": "Most advanced"},
}
