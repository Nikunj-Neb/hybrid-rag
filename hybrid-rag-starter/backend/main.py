import io
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

from core import answer_question, db_connection, store_document

MAX_UPLOAD = 8 * 1024 * 1024

@asynccontextmanager
async def lifespan(app: FastAPI):
    with db_connection():
        pass
    yield

app = FastAPI(title="Hybrid RAG API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")],
                   allow_methods=["GET", "POST"], allow_headers=["*"])

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)
    mode: str = "hybrid"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/documents")
def documents():
    with db_connection() as con:
        rows = con.execute("SELECT document_id, filename, COUNT(*) AS chunks FROM chunks GROUP BY document_id, filename ORDER BY filename").fetchall()
        return {"documents": [dict(row) for row in rows]}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "document")
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "Only PDF, TXT and MD files are supported")
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Maximum upload size is 8 MB")
    try:
        if ext == ".pdf":
            pdf = PdfReader(io.BytesIO(raw))
            if len(pdf.pages) > 100:
                raise HTTPException(400, "Maximum 100 PDF pages")
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(pdf.pages)]
        else:
            pages = [(None, raw.decode("utf-8-sig"))]
        return store_document(name, pages)
    except HTTPException:
        raise
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, "Document processing failed; inspect backend logs") from exc

@app.post("/chat")
def chat(request: ChatRequest):
    if request.mode not in ("hybrid", "semantic", "keyword"):
        raise HTTPException(400, "Invalid retrieval mode")
    try:
        return answer_question(request.question, mode=request.mode)
    except Exception as exc:
        raise HTTPException(502, "Retrieval or generation failed; inspect backend logs") from exc
