from src.config.models import get_large_language_model, get_embedding_model
from src.config.json.schema import JSON_JD_SCHEMA, JSON_RESUME_SCHEMA

from pathlib import Path 

get_llm = get_large_language_model
get_embedding_model = get_embedding_model

ROOT_DIR = Path(__file__).parent.parent.parent


RESUME_TXT_PATH = "data/raw/resume/generative-ai-resume.txt"
JD_TXT_PATH = "data/raw/jobdescription/generative-ai-jd.txt"

JSON_RESUME_SCHEMA = JSON_RESUME_SCHEMA
JSON_JD_SCHEMA = JSON_JD_SCHEMA

FAISS_RESUME_PATH = "data/vector_store/faiss_resume_index"
FAISS_JD_PATH = "data/vector_store/faiss_jd_index"

    

   