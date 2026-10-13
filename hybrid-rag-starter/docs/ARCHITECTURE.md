# Architecture / responsibilities

Browser (Next.js, port 3000) → FastAPI (port 8000) → Gemini / Qdrant Cloud / SQLite.

Ingestion: `main.py:/upload` → `core.py:chunk_text` → Gemini `embed` → Qdrant upsert → SQLite rows.
Retrieval: `/chat` → Qdrant dense rank + SQLite BM25 rank → `reciprocal_rank_fusion` → top 5 → Gemini answer.

For teamwork: Developer A owns frontend/UI + API client; Developer B owns Python ingestion/retrieval; both own evaluation, security and integration testing.

Quality gates: test no-document, exact term, paraphrase, citation correctness, scanned PDF, rate-limit errors, restart durability and multiple documents.
