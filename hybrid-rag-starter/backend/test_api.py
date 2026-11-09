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
    monkeypatch.setattr(core, "ai_client", lambda: SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kw: SimpleNamespace(text="Qdrant stores vectors [1]."))))
    result = client.post("/chat", json={"question": "What does Qdrant store?"}).json()
    assert result["sources"][0]["page"] == 2
    assert result["sources"][0]["number"] == 1
    assert "[1]" in result["answer"]
