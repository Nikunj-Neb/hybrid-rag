# Atlas Hybrid RAG

A document question-answering app with a Next.js interface and FastAPI backend.
The application source is in [`hybrid-rag-starter`](hybrid-rag-starter).

## Run in VS Code

Open this repository folder. Use **Terminal → Run Task** and run **Backend** and
**Frontend** in separate terminals after completing the setup in the
[setup guide](hybrid-rag-starter/README.md).

The frontend runs at http://localhost:3000 and API documentation at
http://localhost:8000/docs. Gemini and Qdrant credentials belong only in
`hybrid-rag-starter/backend/.env` (never commit this file).

## How it works

1. Upload PDF, TXT, or Markdown documents.
2. Python extracts the text and creates overlapping chunks with page metadata.
3. Gemini embeds the chunks; Qdrant stores their vectors. SQLite stores the text.
4. Each question searches both vectors (meaning) and BM25+ (keywords).
5. Reciprocal rank fusion combines the rankings and selects five passages.
6. Gemini answers using those passages, with numbered source excerpts in the UI.

Keyword search uses positive-IDF BM25+ and explicitly excludes passages without
matching query terms, so single-chunk collections work correctly. SQLite
connections are closed after every operation.

## Verify

From `hybrid-rag-starter/backend`, run `.venv/Scripts/python -m pytest -q`.
From `hybrid-rag-starter/frontend`, run `npm ci` and `npm run build`.
Automated backend tests use temporary databases and mocked cloud calls; they do
not require credentials. Live answer generation needs your Gemini and Qdrant keys.
