import sys
from langchain_ollama import ChatOllama, OllamaEmbeddings

OLLAMA_SERVER_IP = "172.30.7.100" 
OLLAMA_SERVER_URL = f"http://{OLLAMA_SERVER_IP}:11434"

LargeLanguageModel_NAME = "gpt-oss:20b"
EmbeddingModel_NAME = "embeddinggemma:latest"

LLM_Temperature = 0.2

def get_large_language_model():
    return ChatOllama(
        model=LargeLanguageModel_NAME,
        base_url=OLLAMA_SERVER_URL,
        temperature=LLM_Temperature,
    )

def get_embedding_model():
    return OllamaEmbeddings(
        model=EmbeddingModel_NAME,
        base_url=OLLAMA_SERVER_URL,
    )