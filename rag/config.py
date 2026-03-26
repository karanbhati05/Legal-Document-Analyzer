from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_provider: str = os.getenv("MODEL_PROVIDER", "gemini").lower()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    allow_ollama_fallback: bool = os.getenv("ALLOW_OLLAMA_FALLBACK", "true").lower() == "true"
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "5"))

    @property
    def using_openai(self) -> bool:
        return self.model_provider == "openai"

    @property
    def using_gemini(self) -> bool:
        return self.model_provider == "gemini"

    @property
    def using_ollama(self) -> bool:
        return self.model_provider == "ollama"
