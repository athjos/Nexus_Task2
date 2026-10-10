# From unzipped folder to submitted project

## 1. Run it locally (10 min)

```bash
cd nexus_rag
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python test_rag.py                  # should print 5 PASS lines
```

Get a **free API key** (you only need one):
- Groq: https://console.groq.com/keys  (fast, generous free tier)
- Gemini: https://aistudio.google.com/apikey

Local key setup: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste your key
(that file is git-ignored, so it will never be uploaded).

```bash
streamlit run app.py
```

Upload `sample_notes.txt`, ask "What is a stack?" then "what about queues?", then a question that is not in the notes.
If the provider says "rate limit", the app shows an extractive answer instead of crashing; wait a minute or switch provider in the sidebar.

Optional, local only: `pip install -r requirements-semantic.txt` turns on semantic (meaning-based) search.

## 2. Put it on GitHub

1. Create an empty **public** repo on github.com (e.g. `document-knowledge-assistant`), no README/gitignore.
2. In the project folder:
```bash
git init
git add .
git commit -m "Smart Document Knowledge Assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/document-knowledge-assistant.git
git push -u origin main
```
3. Check on GitHub that `.streamlit/secrets.toml` is NOT there (only `secrets.toml.example`). Never commit a real key.

## 3. Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** -> pick your repo, branch `main`, main file `app.py`.
3. **Advanced settings -> Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your-real-key"
   ```
   (or `GEMINI_API_KEY`). Choose Python 3.11 or 3.12.
4. Deploy. The first build takes a few minutes. Copy the `https://....streamlit.app` link.
5. Open the link, upload a file, ask a question to confirm it works.

Why this setup: a local LLM can't run on Streamlit's CPU-only servers, so the app calls a free-tier API and the key
lives in Streamlit secrets. The deployed app uses TF-IDF retrieval because PyTorch is too heavy for the free tier.

## 4. Screenshots for the report (take these 5)

1. Empty app with the sidebar (upload + settings)
2. Documents uploaded (sidebar list with chunk counts)
3. An answer with the Sources expander open
4. A follow-up question working ("what about queues?")
5. The refusal for an out-of-scope question

## 5. Report and submission

- Fill the highlighted fields in `Report_Problem02.docx` (name, registration number, email, links, screenshots),
  then File -> Save as PDF. PDF is fine per the organisers.
- Upload to Google Drive -> Share -> "Anyone with the link: Viewer".
- Submit the Drive link, GitHub link and deployment link at https://Nexusrecruitment.vercel.app/apply
- The stated deadline was 8 Oct 11:59 PM. The organisers' messages say you can modify and resubmit after submitting,
  but check that the form still accepts a submission. If you already submitted a Drive link, you can replace the
  file in Drive (right click -> Manage versions -> Upload new version) and the same link keeps working.
