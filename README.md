# Legal Document Analyzer (RAG + GenAI)

A legal-document analysis app where users can upload documents and get:

- Explanation of legal content
- Summary of key points
- Answers to specific user queries

This solution uses Retrieval-Augmented Generation (RAG):

1. Ingest documents
2. Split into chunks
3. Embed chunks into a vector DB (Chroma)
4. Retrieve relevant context per task/query
5. Generate responses with Gemini (with optional local fallback)

## Features

- Multi-document upload
- RAG retrieval over uploaded files
- Separate actions for Explanation, Summary, and Query Answering
- Gemini-first generation workflow
- Legal-safe prompting (no fabrication and explicit uncertainty)

## Tech Stack

- Python + Streamlit UI
- LangChain orchestration
- Chroma vector store
- Gemini model + local embeddings

## Project Structure

```
.
├── app.py
├── requirements.txt
├── .env.example
└── rag/
    ├── __init__.py
    ├── config.py
    ├── loaders.py
    ├── pipeline.py
    └── prompts.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment template:

```bash
copy .env.example .env
```

4. Update `.env`:
- Set `MODEL_PROVIDER=gemini`
- Set `GOOGLE_API_KEY`
- Optionally tune `GEMINI_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`

## Run

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Deploy On Streamlit Community Cloud

1. Push this project to GitHub.
2. Open Streamlit Community Cloud and click `Create app`.
3. Select your repo, branch, and set main file path to `app.py`.
4. In app settings, add Secrets using this format:

```toml
GOOGLE_API_KEY = "your_gemini_api_key"
MODEL_PROVIDER = "gemini"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5
```

5. Click `Deploy`.

## Supported Document Types

Direct support:

- PDF (`.pdf`)
- DOCX (`.docx`)
- TXT/MD/RTF/CSV/JSON/HTML/XML

Unknown types are attempted as best-effort text decode.

## RAG Flow

- `load_uploaded_documents()` reads and normalizes document text.
- `RecursiveCharacterTextSplitter` chunks content.
- `Chroma.from_documents(...)` creates vector index.
- Retriever pulls top-k chunks for each task.
- Prompt templates generate explanation, summary, and Q&A.

## Notes

- This tool is for informational analysis and not legal advice.
- Accuracy depends on document quality and model selection.
- Scanned PDFs without text extraction may need OCR preprocessing.
