"""Quick sanity tests for the RAG core. Run: python test_rag.py"""
import pathlib

from rag import KnowledgeBase, chunk_text, extractive_answer, retrieval_query

HERE = pathlib.Path(__file__).parent
notes = (HERE / "sample_notes.txt").read_bytes()


def test_chunking():
    chunks = chunk_text(notes.decode(), size=500, overlap=100)
    assert len(chunks) > 3
    assert all(len(c) <= 500 for c in chunks), "chunk exceeds size"
    # overlap: consecutive chunks share some text
    assert any(a[-40:].split()[-1] in b for a, b in zip(chunks, chunks[1:]))


def test_retrieval():
    kb = KnowledgeBase(use_sbert=False)  # TF-IDF so the test needs no model download
    kb.add_document("sample_notes.txt", notes)
    cases = {
        "What does LIFO mean for a stack?": "Last In, First Out",
        "How are hash collisions handled?": "chaining",
        "time complexity of binary search": "O(log n)",
        "What is a doubly linked list?": "previous pointer",
    }
    for question, expected in cases.items():
        hits = kb.search(question, k=3)
        assert kb.is_relevant(hits), question
        assert any(expected in c.text for c, _ in hits), (question, expected)


def test_irrelevant_question_refused():
    kb = KnowledgeBase(use_sbert=False)
    kb.add_document("sample_notes.txt", notes)
    assert not kb.is_relevant(kb.search("Who won the cricket world cup in 2011?", k=3))


def test_multi_doc_and_remove():
    kb = KnowledgeBase(use_sbert=False)
    kb.add_document("a.txt", b"Photosynthesis converts light energy into chemical energy in chloroplasts.")
    kb.add_document("b.txt", notes)
    top = kb.search("how do plants convert light energy", k=1)[0][0]
    assert top.source == "a.txt"
    kb.remove_document("a.txt")
    assert set(kb.docs) == {"b.txt"}


def test_followup_and_extractive():
    history = [{"role": "user", "content": "What is a stack?"}, {"role": "assistant", "content": "..."}]
    assert "stack" in retrieval_query("what about queues?", history)
    kb = KnowledgeBase(use_sbert=False)
    kb.add_document("sample_notes.txt", notes)
    hits = kb.search("What is a queue?", k=3)
    assert "FIFO" in extractive_answer("What is a queue?", hits)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
