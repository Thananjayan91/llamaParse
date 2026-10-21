import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Directories
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_FOLDER", "data/uploads")
PARSED_DIR = BASE_DIR / os.getenv("PARSED_FOLDER", "data/parsed")
EXTRACTED_DIR = BASE_DIR / os.getenv("EXTRACTED_FOLDER", "data/extracted")
INDEX_DIR = BASE_DIR / os.getenv("INDEX_FOLDER", "data/indices")

for directory in [DATA_DIR, UPLOAD_DIR, PARSED_DIR, EXTRACTED_DIR, INDEX_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "llx-ihwMkcLXdlCLEUCZpDodiHx461k9gvXCTjHx2HqrxG5lpAcm")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-llamaparse-key")

# Default tiers and options
DEFAULT_PARSE_TIER = "agentic_plus"
DEFAULT_PARSE_VERSION = "latest"
PARSE_TIERS = ["agentic_plus", "agentic", "cost_effective", "fast"]

