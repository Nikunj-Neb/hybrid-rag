# Atlas — Hybrid RAG Starter

A clean-from-scratch **learning-first Hybrid RAG** app: Next.js interface + FastAPI + Gemini generation/embeddings + Qdrant dense vector search + SQLite-backed BM25 keyword retrieval + reciprocal rank fusion (RRF) + citations.

**Status:** Working starter source, not production audited. Requires your own cloud credentials and installation. No real API calls were made by the author.

## Requirements

- Node.js (supported LTS), Python 3.12+ (3.12 or 3.13 recommended)
- Gemini API key from https://aistudio.google.com/apikey
- Qdrant Cloud cluster URL + API key from https://cloud.qdrant.io

No Ollama, CUDA, or Docker required.

## Setup (Windows PowerShell)

### Terminal 1: backend

```powershell
cd backend
Copy-Item .env.example .env
# Edit backend/.env, entering Gemini + Qdrant credentials. Do not commit .env.
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

Check http://localhost:8000/health and http://localhost:8000/docs.

### Terminal 2: frontend

```powershell
cd frontend
Copy-Item .env.local.example .env.local
npm install
npm run dev
```

Visit http://localhost:3000, upload a TXT/PDF/MD file and ask a question.

### Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

## Pipeline

1. Upload a text-readable PDF/TXT/MD (8MB max; PDF max 100 pages).
2. Parse per-page text, divide into overlapping character chunks.
3. Embed each chunk with Gemini `gemini-embedding-001` using `RETRIEVAL_DOCUMENT` (768 dimensions).
4. Store dense embeddings in Qdrant, plus all chunk text/metadata in local SQLite.
5. On question, embed question with `RETRIEVAL_QUERY` and search Qdrant; independently score all local chunks with BM25.
6. Fuse both top-20 rankings using RRF (k=60), select top 5.
7. Send retrieved context to Gemini Flash; return answer and source snippets.

Use the UI dropdown to compare **Hybrid**, **Semantic only**, and **Keyword only** modes.

## Learning sequence

- Day 1: Start backend/frontend, create cloud credentials, upload first file, ask first question.
- Day 2: Read `backend/core.py` (chunking + embeddings). Change chunk size and compare search.
- Day 3: Learn BM25 and why keyword-only retrieval misses paraphrases.
- Day 4: Learn cosine similarity, vector search, RRF; inspect returned citations.
- Day 5: Write 20 QA benchmark cases and compare hit rates / citations / latency.
- Later: Auth and multi-tenant isolation, durable cloud metadata, delete/reindex, OCR, streaming, tests, evaluation, rate limiting and deployment.

## Important limitations

- **Local SQLite is the BM25 index source**, while **Qdrant Cloud stores dense embeddings**. This is genuine hybrid retrieval, but not a Qdrant-native dual-index implementation. For multi-instance cloud hosting, move BM25 and metadata to a shared database/search service or Qdrant's sparse vectors.
- Local SQLite and remote Qdrant can get out of sync after a partial ingestion failure. Implement rollback/reconciliation and document deletion before production.
- Not designed for public deployment as-is: no user authentication, no document isolation, no rate limiting, no antivirus/file sanitization, no abuse protection, no scheduled backups. Do not upload confidential material or expose this API publicly.
- The document parser does not perform OCR. Scanned PDFs without embedded text cannot be processed.
- One embedding request per chunk, so long files can be slow or reach Gemini quotas. Monitor costs and rate limits.
- Citations are model-generated source labels and should be verified against displayed excerpts; they are not independently proof-checked.
- API key must remain in `backend/.env`; never embed in frontend and do not commit secrets.
- Cloud Gemini/Qdrant may incur charges depending on usage/plan.
