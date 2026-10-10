"""Core RAG logic for the Smart Document Knowledge Assistant.

Pipeline:  upload -> extract text -> chunk -> embed -> (query) -> top-k similarity search
           -> build prompt with retrieved chunks -> LLM answer grounded in those chunks.

Embeddings: uses sentence-transformers (semantic, understands synonyms) when installed,
otherwise falls back to TF-IDF (keyword-weighted) so the app still runs anywhere.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# 1. File reading
# --------------------------------------------------------------------------- #
SUPPORTED_TYPES = ("pdf", "txt")


def read_file(name: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return a list of (page_number, text). TXT files have page_number None."""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
        return pages
    if ext == "txt":
        return [(None, data.decode("utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: .{ext} (use {', '.join(SUPPORTED_TYPES)})")


# --------------------------------------------------------------------------- #
# 2. Chunking
# --------------------------------------------------------------------------- #
def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Sentence-aware chunking with overlap so answers spanning a boundary aren't lost."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    pieces: list[str] = []
    for sent in split_sentences(text):
        while len(sent) > size:  # very long "sentence" (tables, code): hard split
            pieces.append(sent[:size])
            sent = sent[size - overlap:]
        pieces.append(sent)

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for piece in pieces:
        if cur and cur_len + len(piece) + 1 > size:
            chunks.append(" ".join(cur))
            keep, length = [], 0  # carry the last sentences over as overlap
            for prev in reversed(cur):
                if length + len(prev) > overlap:
                    break
                keep.insert(0, prev)
                length += len(prev) + 1
            cur, cur_len = keep, sum(len(x) + 1 for x in keep)
        cur.append(piece)
        cur_len += len(piece) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    idx: int

    @property
    def label(self) -> str:
        return f"{self.source}, page {self.page}" if self.page else self.source


# --------------------------------------------------------------------------- #
# 3. Embeddings + vector store
# --------------------------------------------------------------------------- #
_SBERT_MODEL = None


def _load_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def sbert_available() -> bool:
    try:
        _load_sbert()
        return True
    except Exception:
        return False


class KnowledgeBase:
    """Holds documents, their chunks and embeddings; answers similarity queries."""

    # minimum cosine similarity for a chunk to count as "relevant"
    MIN_SCORE = {"sbert": 0.20, "tfidf": 0.03}

    def __init__(self, use_sbert: bool | None = None, chunk_size: int = 800, overlap: int = 150):
        self.kind = "sbert" if (sbert_available() if use_sbert is None else use_sbert) else "tfidf"
        self.chunk_size, self.overlap = chunk_size, overlap
        self.docs: dict[str, dict] = {}  # name -> {"chunks": [...], "emb": ndarray|None, "hash": str}
        self._vectorizer = None
        self._matrix = None
        self._all_chunks: list[Chunk] = []

    # ---- document management ----
    def add_document(self, name: str, data: bytes) -> int:
        pages = read_file(name, data)
        chunks: list[Chunk] = []
        for page, text in pages:
            for t in chunk_text(text, self.chunk_size, self.overlap):
                chunks.append(Chunk(t, name, page, len(chunks)))
        if not chunks:
            raise ValueError(f"No readable text found in {name} (scanned PDF?).")
        emb = None
        if self.kind == "sbert":
            emb = _load_sbert().encode([c.text for c in chunks], normalize_embeddings=True)
        self.docs[name] = {"chunks": chunks, "emb": emb, "hash": hashlib.md5(data).hexdigest()}
        self._rebuild()
        return len(chunks)

    def remove_document(self, name: str) -> None:
        self.docs.pop(name, None)
        self._rebuild()

    def clear(self) -> None:
        self.docs.clear()
        self._rebuild()

    @property
    def n_chunks(self) -> int:
        return len(self._all_chunks)

    def _rebuild(self) -> None:
        self._all_chunks = [c for d in self.docs.values() for c in d["chunks"]]
        self._vectorizer = self._matrix = None
        if not self._all_chunks:
            return
        if self.kind == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self._matrix = self._vectorizer.fit_transform([c.text for c in self._all_chunks])
        else:
            self._matrix = np.vstack([d["emb"] for d in self.docs.values()])

    # ---- similarity search ----
    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the top-k (chunk, cosine_similarity) pairs, best first."""
        if not self._all_chunks or not query.strip():
            return []
        if self.kind == "tfidf":
            q = self._vectorizer.transform([query])
            scores = (self._matrix @ q.T).toarray().ravel()
        else:
            q = _load_sbert().encode([query], normalize_embeddings=True)[0]
            scores = self._matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self._all_chunks[i], float(scores[i])) for i in top]

    def is_relevant(self, hits: list[tuple[Chunk, float]]) -> bool:
        return bool(hits) and hits[0][1] >= self.MIN_SCORE[self.kind]


# --------------------------------------------------------------------------- #
# 4. Generation (LLM)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a study assistant for first-year university students. Answer the question "
    "using ONLY the numbered context snippets provided. Cite the snippets you used like [1] or [2][3]. "
    "If the snippets do not contain the answer, say exactly: \"I couldn't find that in the uploaded "
    "documents.\" Do not use outside knowledge. Keep answers clear and concise, and use simple "
    "language suited to a beginner."
)

NO_ANSWER = "I couldn't find that in the uploaded documents."
_STOP = set("a an the is are was were of in on at to for and or what which who how why when does do did "
            "this that it with as by be can you me my i explain tell about define".split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{i}] ({c.label})\n{c.text}" for i, (c, _) in enumerate(hits, start=1))


def extractive_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    """No-LLM fallback: return the most question-relevant sentences from the retrieved chunks."""
    q_terms = set(_tokens(question)) - _STOP
    scored = []
    for ci, (chunk, score) in enumerate(hits, start=1):
        for sent in split_sentences(chunk.text):
            overlap = len(q_terms & set(_tokens(sent)))
            scored.append((overlap, score, ci, sent))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    best = [x for x in scored[:3] if x[0] > 0] or scored[:2]
    return " ".join(f"{s} [{ci}]" for _, _, ci, s in best) or NO_ANSWER


def _history_messages(history: list[dict], max_turns: int = 6) -> list[dict]:
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-max_turns:]]
    while msgs and msgs[0]["role"] != "user":  # APIs require the first message to be from the user
        msgs.pop(0)
    return msgs


def generate_answer(
    question: str,
    hits: list[tuple[Chunk, float]],
    history: list[dict],
    provider: str = "none",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
) -> str:
    """Send question + retrieved snippets (+ recent chat) to the chosen LLM."""
    if provider == "none" or not api_key:
        return extractive_answer(question, hits)

    user_msg = f"Context snippets:\n\n{format_context(hits)}\n\nQuestion: {question}"
    msgs = _history_messages(history) + [{"role": "user", "content": user_msg}]

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=msgs,
        )
        return resp.content[0].text.strip()

    if provider == "openai":  # also works for Gemini / Groq / OpenRouter via base_url
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url or None)
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            temperature=0.2,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + msgs,
        )
        return resp.choices[0].message.content.strip()

    raise ValueError(f"Unknown provider: {provider}")


def retrieval_query(question: str, history: list[dict]) -> str:
    """Make short follow-ups ('what about queues?') searchable by adding the previous question."""
    if history and len(question.split()) < 6:
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev:
            return f"{prev[-1]} {question}"
    return question
