import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Check local and parent directories for .env
for env_path in [BASE_DIR / ".env", BASE_DIR.parent / ".env", Path.cwd() / ".env"]:
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)
        loaded_env_file = str(env_path.resolve())
        break
else:
    loaded_env_file = None

APP_NAME = os.getenv("APP_NAME", "Relay Operations Platform")
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Anchor sqlite relative to project root to avoid cwd drift
raw_db = os.getenv("DATABASE_URL", "").strip()
if raw_db and raw_db.startswith("sqlite:///") and not os.path.isabs(raw_db.replace("sqlite:///", "")):
    DATABASE_URL = f"sqlite:///{BASE_DIR / raw_db.replace('sqlite:///', '')}"
elif not raw_db:
    DATABASE_URL = f"sqlite:///{BASE_DIR / 'usm.db'}"
else:
    DATABASE_URL = raw_db

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "data" / "chroma_db"))
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "data" / "knowledge_base"))


def get_openai_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def is_openai_configured() -> bool:
    key = get_openai_api_key()
    return bool(key and key.startswith("sk-") and len(key) > 20)


def get_masked_api_key() -> str:
    key = get_openai_api_key()
    if not key:
        return "Not Configured"
    if len(key) <= 12:
        return "Configured"
    return f"{key[:7]}...{key[-4:]}"


def get_config_status() -> dict:
    return {
        "environment_file": loaded_env_file or "Not found",
        "openai_configured": is_openai_configured(),
        "openai_key": get_masked_api_key(),
        "llm_model": LLM_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chroma_directory": CHROMA_PERSIST_DIRECTORY,
        "database": DATABASE_URL,
    }
