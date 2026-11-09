import pytest
from fastapi.testclient import TestClient
import core
import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "DB_PATH", str(tmp_path / "chunks.sqlite3"))
    with TestClient(main.app) as client:
        yield client


def test_empty_library(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/documents").json() == {"documents": []}
    result = client.post("/chat", json={"question": "What is RAG?"})
    assert result.status_code == 200
    assert result.json()["sources"] == []


def test_request_validation(client):
    assert client.post("/chat", json={"question": "x"}).status_code == 422
    assert client.post("/chat", json={"question": "hello", "mode": "invalid"}).status_code == 400
    assert client.post("/upload", files={"file": ("data.exe", b"x")}).status_code == 400
    assert client.post("/upload", files={"file": ("data.txt", b"\xff")}).status_code == 400


def test_upload_and_list(client, monkeypatch):
    def store(filename, pages):
        assert filename == "notes.md"
        assert pages == [(None, "Hybrid retrieval combines meaning and keywords.")]
        return {"document_id": "doc", "filename": filename, "chunks": 1}
    monkeypatch.setattr(main, "store_document", store)
    result = client.post("/upload", files={"file": ("notes.md", b"Hybrid retrieval combines meaning and keywords.")})
    assert result.status_code == 200
    assert result.json()["chunks"] == 1


def test_keyword_single_chunk_and_no_match(client):
    with core.db_connection() as con:
        con.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                    ("one", "doc", "notes.txt", None, "Qdrant stores vectors"))
    assert [r["id"] for r in core.retrieve("Qdrant", mode="keyword")] == ["one"]
    assert core.retrieve("unrelated", mode="keyword") == []


def test_answer_sources(client, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(core, "retrieve", lambda *a, **kw: [
        {"filename": "notes.pdf", "page": 2, "body": "Qdrant stores vectors"}])
    monkeypatch.setattr(core, "expand_context", lambda rows: rows)
    monkeypatch.setattr(core, "ai_client", lambda: SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kw: SimpleNamespace(text="Qdrant stores vectors [1]."))))
    result = client.post("/chat", json={"question": "What does Qdrant store?"}).json()
    assert result["sources"][0]["page"] == 2
    assert result["sources"][0]["number"] == 1
    assert "[1]" in result["answer"]


def test_application_question_ignores_filler_words(client):
    with core.db_connection() as con:
        con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", [
            ("intro", "doc", "notes.pdf", 1, "Tell me about the history of thermodynamics"),
            ("scope", "doc", "notes.pdf", 9, "Practical applications include engines and refrigeration"),
        ])
    rows = core.retrieve("whats the use of thermodynamics", mode="keyword")
    assert rows[0]["id"] == "scope"


def test_small_document_context_includes_late_page(client):
    with core.db_connection() as con:
        con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", [
            ("first", "doc", "notes.pdf", 1, "Introduction"),
            ("late", "doc", "notes.pdf", 20, "Applications include heat engines"),
            ("other", "other", "private.pdf", 1, "Unrelated document"),
        ])
    sources = [{"id": "first", "document_id": "doc", "filename": "notes.pdf", "page": 1, "body": "Introduction"}]
    assert [r["id"] for r in core.expand_context(sources)] == ["first", "late"]


def test_large_context_respects_budget(client):
    rows = [{"id": str(i), "document_id": "doc", "filename": "notes.pdf", "page": i + 1, "body": "text " * 100} for i in range(8)]
    with core.db_connection() as con:
        con.executemany("INSERT INTO chunks VALUES (:id, :document_id, :filename, :page, :body)", rows)
    selected = core.expand_context([rows[3]], max_chars=1900)
    assert selected[0]["id"] == "3"
    assert len({r["id"] for r in selected}) == len(selected)
    assert sum(len(r["body"]) + len(r["filename"]) + 100 for r in selected) <= 1900


def test_broken_pdf_text_triggers_ocr():
    from pdf_text import readable_text
    assert not readable_text('!" /0&!-#5&!&B(&*(!#6!$&%�()#*! ' * 10)
    assert readable_text('Thermodynamics describes energy transfer between a system and its surroundings.')
