# Smart Document Knowledge Assistant (Nexus Tech Recruitment 2026, Problem 02)

A Streamlit web app where you upload PDF/TXT files and chat with them. Answers come from a
Retrieval-Augmented Generation (RAG) pipeline and are grounded strictly in your files, with the
source snippets shown next to every answer so you can verify them.

> Full walkthrough (run, GitHub, Streamlit Cloud deployment, report, submission): see `DEPLOY.md`.

## Run it

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

**LLM setup** (pick one in the sidebar, or set an environment variable first):

| Option | Environment variable | Notes |
|---|---|---|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | default model `claude-haiku-4-5-20251001` |
| Groq / Gemini / OpenRouter (free tiers) | `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` | presets in the sidebar; keys can live in `.streamlit/secrets.toml` or Streamlit Cloud secrets |
| OpenAI | `OPENAI_API_KEY` | |
| No LLM | none | extractive fallback: returns the most relevant sentences; good for offline testing |

If `sentence-transformers` is not installed, retrieval falls back to TF-IDF automatically.

## Requirements checklist

| Requirement | Where |
|---|---|
| Upload and manage single/multiple .pdf/.txt files | sidebar uploader + document list with remove buttons (`app.py`) |
| Similarity search, top 3-5 chunks | `KnowledgeBase.search`, slider in sidebar (`rag.py`) |
| Pass snippets + query to an LLM API | `generate_answer` (`rag.py`) |
| Chat with follow-up questions | chat history sent to the LLM; short follow-ups are expanded with the previous question for retrieval (`retrieval_query`) |
| Show referenced snippets to verify | "Sources" expander under each answer, with file, page and similarity score |
| Grounded strictly in files | system prompt forbids outside knowledge; if the best match is below a similarity threshold the app answers "I couldn't find that in the uploaded documents." without calling the LLM |

## How it works (explain this in your demo or viva)

1. **Extract**: `pypdf` reads each PDF page (the page number is kept); TXT files are read directly.
2. **Chunk**: text is cut into ~800-character, sentence-aware chunks with ~150 characters of overlap, so an idea that crosses a boundary still appears whole in at least one chunk.
3. **Embed**: each chunk becomes a vector. With `all-MiniLM-L6-v2` the vectors capture meaning, so "LIFO structure" can match a question about "last in first out". With the TF-IDF fallback they capture weighted keywords.
4. **Retrieve**: the question is embedded the same way and compared with every chunk by cosine similarity; the top-k chunks (3-5) are kept.
5. **Generate**: the chunks are numbered `[1]..[k]` and sent to the LLM with the question and recent chat. The model must answer only from them and cite the numbers.
6. **Verify**: the same chunks are displayed under the answer.

### Likely questions and short answers

- *Why chunk at all?* Documents are far longer than an LLM prompt, and small focused chunks make retrieval precise.
- *Why overlap?* To avoid cutting an answer in half at a chunk boundary.
- *What is cosine similarity?* The angle between two vectors; 1 means pointing the same way (same meaning), 0 means unrelated.
- *How do you stop hallucination?* Restrictive prompt, citations, a relevance threshold that refuses when nothing matches, and visible sources.
- *How do follow-ups work?* The last few chat turns go to the LLM, and short questions like "what about queues?" are searched together with the previous question.
- *Limitations?* Scanned PDFs (no text layer) are not supported; tables extract poorly; very large collections would need a vector database such as FAISS or Chroma instead of in-memory arrays.

## Demo script (2 minutes)

1. Upload `sample_notes.txt` (and one of your own lecture PDFs).
2. Ask: "What is a stack and where is it used?" then the follow-up "What about queues?"
3. Ask: "How do hash tables handle collisions?" and open the Sources expander to show the matching text.
4. Ask something not in the notes, e.g. "Who won the cricket world cup?" and show that it refuses.

## Files

- `app.py`: Streamlit interface
- `rag.py`: parsing, chunking, embeddings, search, LLM calls
- `sample_notes.txt`: sample lecture notes for the demo
- `test_rag.py`: quick tests (`python test_rag.py`)
