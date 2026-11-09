# Architecture / responsibilities

Browser (Next.js, port 3000) → FastAPI (port 8000) → Gemini / Qdrant Cloud / SQLite.

Ingestion: `main.py:/upload` → `core.py:chunk_text` → Gemini `embed` → Qdrant upsert → SQLite rows.
PDF ingestion checks the extracted text layer and uses local RapidOCR on rendered pages when text is missing or corrupted. Page numbers are preserved; OCR runs locally.

Retrieval: `/chat` → up to 60 Qdrant candidates + up to 60 BM25+ candidates → reciprocal rank fusion → top 12. Keyword queries remove filler words and expand common application/use wording. For matched documents fitting a 40,000-character context budget (including citation overhead), all chunks are supplied in document order. Larger documents use ranked matches plus neighboring chunks within the same budget. Every supplied chunk retains a numbered citation.

This improves recall but cannot recover content absent from a PDF. OCR may misread formulas; verify equations against the original page.

For teamwork: Developer A owns frontend/UI + API client; Developer B owns Python ingestion/retrieval; both own evaluation, security and integration testing.

Quality gates: test no-document, exact term, paraphrase, citation correctness, scanned PDF, rate-limit errors, restart durability and multiple documents.
