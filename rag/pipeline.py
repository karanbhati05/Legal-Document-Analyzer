from __future__ import annotations

import os
from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from chromadb.config import Settings as ChromaSettings

from .config import Settings
from .loaders import load_uploaded_documents
from .prompts import EXPLANATION_PROMPT, QNA_PROMPT, SUMMARY_PROMPT, SYSTEM_INSTRUCTIONS


class LegalRAGAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.active_provider = settings.model_provider
        self.llm = self._init_llm()
        self.embeddings = self._init_embeddings()
        self.vectorstore: Chroma | None = None

    def _init_llm(self):
        if self.settings.using_openai:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=self.settings.openai_model, temperature=0.1)
        if self.settings.using_gemini:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(model=self.settings.gemini_model, temperature=0.1)
            except Exception:
                if self.settings.allow_ollama_fallback:
                    from langchain_ollama import ChatOllama

                    self.active_provider = "ollama"
                    return ChatOllama(model=self.settings.ollama_chat_model, temperature=0.1)
                raise
        if self.settings.using_ollama:
            from langchain_ollama import ChatOllama

            return ChatOllama(model=self.settings.ollama_chat_model, temperature=0.1)

        raise ValueError("Unsupported MODEL_PROVIDER. Use 'openai', 'gemini', or 'ollama'.")

    def _init_embeddings(self):
        if self.settings.using_openai:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=self.settings.openai_embedding_model)
        if self.settings.using_gemini:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(model=self.settings.gemini_embedding_model)
        if self.settings.using_ollama:
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=self.settings.ollama_embedding_model)

        raise ValueError("Unsupported MODEL_PROVIDER. Use 'openai', 'gemini', or 'ollama'.")

    def ingest(self, uploaded_files) -> Dict[str, int]:
        raw_docs = load_uploaded_documents(uploaded_files)
        if not raw_docs:
            raise ValueError("No readable content found in the uploaded documents.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(raw_docs)

        chroma_settings = ChromaSettings(anonymized_telemetry=False)

        try:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client_settings=chroma_settings,
            )
        except Exception:
            if self.settings.allow_ollama_fallback:
                from langchain_ollama import OllamaEmbeddings

                self.embeddings = OllamaEmbeddings(model=self.settings.ollama_embedding_model)
                self.active_provider = "ollama"
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    client_settings=chroma_settings,
                )
            else:
                raise

        return {"raw_documents": len(raw_docs), "chunks": len(chunks)}

    def _retrieve_context(self, query: str) -> str:
        if self.vectorstore is None:
            raise ValueError("Knowledge base not built. Upload documents first.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.settings.top_k})
        docs: List[Document] = retriever.invoke(query)

        joined_chunks = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            joined_chunks.append(f"[Chunk {idx} | Source: {source}]\n{doc.page_content}")

        return "\n\n".join(joined_chunks)

    def _run_prompt(self, prompt: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_INSTRUCTIONS),
            HumanMessage(content=prompt),
        ]
        try:
            response = self.llm.invoke(messages)
        except Exception as exc:
            should_fallback = (
                self.settings.allow_ollama_fallback
                and self.active_provider == "gemini"
                and any(token in str(exc).lower() for token in ["quota", "resourceexhausted", "429"])
            )

            if should_fallback:
                try:
                    from langchain_ollama import ChatOllama

                    self.llm = ChatOllama(model=self.settings.ollama_chat_model, temperature=0.1)
                    self.active_provider = "ollama"
                    response = self.llm.invoke(messages)
                except Exception as fallback_exc:
                    raise RuntimeError(
                        "Gemini quota is exceeded and Ollama fallback is unavailable. "
                        "Start Ollama and pull the configured model, or use a Gemini key with quota."
                    ) from fallback_exc
            else:
                raise

        return response.content if isinstance(response.content, str) else str(response.content)

    def explain(self) -> str:
        context = self._retrieve_context("Explain this legal document in depth")
        return self._run_prompt(EXPLANATION_PROMPT.format(context=context))

    def summarize(self) -> str:
        context = self._retrieve_context("Summarize the key legal points")
        return self._run_prompt(SUMMARY_PROMPT.format(context=context))

    def answer_query(self, query: str) -> str:
        context = self._retrieve_context(query)
        return self._run_prompt(QNA_PROMPT.format(query=query, context=context))
