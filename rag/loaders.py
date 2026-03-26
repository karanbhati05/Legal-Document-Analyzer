from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".rtf",
    ".csv",
    ".json",
    ".html",
    ".xml",
}


def _persist_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(uploaded_file.getvalue())
        return Path(temp.name)


def _best_effort_text_load(path: Path) -> List[Document]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if not text.strip():
        return []

    return [Document(page_content=text, metadata={"source": path.name, "loader": "fallback-text"})]


def _load_by_extension(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()
    if ext in {".txt", ".md", ".rtf", ".csv", ".json", ".html", ".xml"}:
        return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()

    # Unknown file type: try text decode fallback for best effort.
    return _best_effort_text_load(path)


def load_uploaded_documents(uploaded_files: Iterable) -> List[Document]:
    all_docs: List[Document] = []
    temp_paths: List[Path] = []

    try:
        for uploaded_file in uploaded_files:
            temp_path = _persist_upload(uploaded_file)
            temp_paths.append(temp_path)

            docs = _load_by_extension(temp_path)
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["extension"] = Path(uploaded_file.name).suffix.lower()
            all_docs.extend(docs)
    finally:
        for temp_path in temp_paths:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    return all_docs
