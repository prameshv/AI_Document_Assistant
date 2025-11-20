import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(os.path.dirname(app_dir))
env_path = os.path.join(project_root, '.env')

load_dotenv(dotenv_path=env_path)


class Settings:
    # Groq API Key (FREE and fast!)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Model configurations
    LLM_MODEL = "llama-3.1-8b-instant"  # Fast and free on Groq
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # File settings
    MAX_FILE_SIZE = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx']

    # Vector store settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100


settings = Settings()
