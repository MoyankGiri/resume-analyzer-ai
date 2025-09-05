# Resume Analyzer AI (Streamlit + LangGraph + RAG)

Production-ready agentic resume analysis with Google Generative AI and an in-memory RAG knowledge base. Deployed on Render.

## 🚀 Deploy on Render
1. Create a new **Web Service** from this repo.
2. In **Environment Variables**, add `GOOGLE_API_KEY` (no quotes). Optionally set `GENAI_MODEL`, `GENAI_TEMPERATURE`, `EMBEDDINGS_MODEL`.
3. Render will build and start with `render.yaml`.

## 🖥️ Local Dev
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key
streamlit run app.py
```

## 📦 Structure
- `app.py`: Streamlit UI
- `core/`: processing, scoring models, knowledge base, agentic pipeline
- `ui/`: metrics, layout, report download

## 🔐 Security Notes
- **No hardcoded API keys**. Configure via env vars.
- Uploaded files are processed from `/tmp` and not persisted. Max size 10MB.

## ⚠️ Scraping Notes
Some sources may block scraping on PaaS. The app includes a fallback best-practices corpus; RAG still works even if external pages fail to load.
```