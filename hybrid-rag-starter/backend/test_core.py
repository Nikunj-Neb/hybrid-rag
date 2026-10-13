from core import chunk_text, reciprocal_rank_fusion, tokenize

def test_tokenize():
    assert tokenize("Hello, RAG World!") == ["hello", "rag", "world"]

def test_chunk_overlap():
    pieces = chunk_text("word " * 200, max_chars=100, overlap=20)
    assert len(pieces) > 1
    assert all(len(part) <= 100 for part in pieces)

def test_rrf_promotes_repeated_result():
    ranked = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert ranked[0] == "b"
