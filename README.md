# Legal Document Analyzer (Vercel + Gemini RAG)

Upload legal documents and get:

- Explanation of legal content
- Summary of key points
- Answers to user-specific queries

This app is Gemini-only and Vercel-deployable using Next.js.

## Live App

- Production: https://legal-document-analyzer-next.vercel.app

## Stack

- Next.js App Router
- Vercel Serverless API route
- Gemini 2.5 Flash for generation
- Gemini Embedding 1 for retrieval

## Local Run

1. Install dependencies:

```bash
npm install
```

2. Copy environment file:

```bash
copy .env.example .env
```

3. Add your API key in `.env`.

4. Run:

```bash
npm run dev
```

Open `http://localhost:3000`.

## Vercel Deploy

1. Ensure this repo is linked to Vercel (`vercel` command once).
2. Add env vars in Vercel:

```bash
vercel env add GOOGLE_API_KEY production
vercel env add GEMINI_MODEL production
vercel env add GEMINI_EMBEDDING_MODEL production
vercel env add CHUNK_SIZE production
vercel env add CHUNK_OVERLAP production
vercel env add TOP_K production
```

Recommended values:

- `GEMINI_MODEL=gemini-2.5-flash`
- `GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001`
- `CHUNK_SIZE=900`
- `CHUNK_OVERLAP=120`
- `TOP_K=3`

3. Deploy:

```bash
vercel --prod
```

## API

- Route: `POST /api/analyze`
- Input: `multipart/form-data` with `files[]` and optional `query`
- Output: `explanation`, `summary`, `answer`, and `stats`

## Supported Files

- PDF (`.pdf`)
- DOCX (`.docx`)
- TXT/MD/RTF/CSV/JSON/HTML/XML (best-effort text decode)

## Note

This tool provides informational analysis and not legal advice.
