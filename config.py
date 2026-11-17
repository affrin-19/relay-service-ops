
"""
Configuration Module for AI-Powered USM System.
Centralizes application settings and environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Locate environment file
candidate_env_paths = [
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
    Path.cwd() / ".env",
]

loaded_env_file = None

for path in candidate_env_paths:
    if path.exists() and path.is_file():
        # Existing process environment variables take precedence.
        load_dotenv(dotenv_path=path, override=False)
        loaded_env_file = str(path.resolve())
        break

# Application settings
APP_NAME = os.getenv(
    "APP_NAME",
    "AI-Powered Unified Service Management System"
)

APP_ENV = os.getenv("APP_ENV", "development")

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Database - Resolve to absolute path so working directory shifts do not create duplicate databases
raw_db = os.getenv("DATABASE_URL", "").strip()
if raw_db and raw_db.startswith("sqlite:///") and not os.path.isabs(raw_db.replace("sqlite:///", "")):
    DATABASE_URL = f"sqlite:///{BASE_DIR / raw_db.replace('sqlite:///', '')}"
elif not raw_db:
    DATABASE_URL = f"sqlite:///{BASE_DIR / 'usm.db'}"
else:
    DATABASE_URL = raw_db

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
    "gpt-4o-mini"
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "text-embedding-3-small"
)

# ChromaDB configuration
CHROMA_PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY",
    str(BASE_DIR / "data" / "chroma_db")
)

KNOWLEDGE_BASE_DIR = os.getenv(
    "KNOWLEDGE_BASE_DIR",
    str(BASE_DIR / "data" / "knowledge_base")
)


def get_openai_api_key() -> str:
    """Return the currently configured API key."""
    return os.getenv("OPENAI_API_KEY", "").strip()


def is_openai_configured() -> bool:
    """Check whether an API key is configured."""
    key = get_openai_api_key()
    return bool(key and key.startswith("sk-") and len(key) > 20)


def get_masked_api_key() -> str:
    """Return a masked API key for safe diagnostics."""
    key = get_openai_api_key()

    if not key:
        return "Not Configured"

    if len(key) <= 12:
        return "Configured (Protected)"

    return f"{key[:7]}...{key[-4:]}"


def get_config_status() -> dict:
    """Return safe configuration diagnostics."""

    return {
        "environment_file": loaded_env_file or "Not found",
        "openai_configured": is_openai_configured(),
        "openai_key": get_masked_api_key(),
        "llm_model": LLM_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chroma_directory": CHROMA_PERSIST_DIRECTORY,
        "database": DATABASE_URL,
    }
