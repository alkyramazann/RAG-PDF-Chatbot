# 📚 RAG Chatbot — Ask Questions About Any PDF

A fully local, beginner-friendly Retrieval-Augmented Generation (RAG) chatbot.
No Docker. No PostgreSQL. No pgvector. Just Python.

---

## 🗂️ Project Structure

```
rag_chatbot/
├── rag.py            ← The entire RAG system (one file)
├── requirements.txt  ← All dependencies
├── book.pdf          ← YOUR PDF (add this yourself)
└── rag_cache.pkl     ← Auto-created after first run (speeds up restarts)
```

---

## ⚡ How to Run (Windows Step-by-Step)

### Step 1 — Make sure Python is installed
Open Command Prompt (Win + R → type `cmd` → Enter) and run:
```
python --version
```
You need Python 3.10 or higher. Download from https://python.org if needed.

---

### Step 2 — Put your PDF in the folder
Copy your PDF file into the `rag_chatbot/` folder and rename it `book.pdf`.

---

### Step 3 — Open Command Prompt in the project folder
Navigate to the folder:
```
cd C:\path\to\rag_chatbot
```
(Replace with your actual path, e.g. `cd C:\Users\YourName\Desktop\rag_chatbot`)

---

### Step 4 — Install dependencies
```
pip install -r requirements.txt
```
This installs everything needed. Takes ~2 minutes on first run.

---

### Step 5 — Run the chatbot
```
python rag.py
```

You'll be prompted for your OpenAI API key (or set it as an environment variable):
```
set OPENAI_API_KEY=sk-...your-key-here...
python rag.py
```

---

### Step 6 — Ask questions!
```
💬  You: What is the main theme of this book?
💬  You: Summarize chapter 3
💬  You: Who are the main characters?
💬  You: quit
```

---

## 🚀 What Happens on First Run vs. Subsequent Runs

| Run | What Happens |
|-----|-------------|
| First run | Reads PDF → Chunks text → Builds embeddings → Saves cache |
| Later runs | Loads cache instantly → Ready in seconds |

Delete `rag_cache.pkl` if you change the PDF.

---

## 🔑 Getting an OpenAI API Key
1. Go to https://platform.openai.com
2. Sign up / log in
3. Click your profile → "API Keys" → "Create new secret key"
4. Copy the key (starts with `sk-`)

---

## 🧠 How It Works

```
PDF → Extract Text → Split into Chunks
                          ↓
               Embed each chunk (sentence-transformers, runs locally)
                          ↓
User Question → Embed question → Cosine Similarity Search → Top 5 chunks
                                                                  ↓
                                              Send to OpenAI GPT-4o-mini
                                              with chunks as context
                                                                  ↓
                                              Human-like answer ✅
```

---

## ❓ Troubleshooting

| Error | Fix |
|-------|-----|
| `book.pdf not found` | Make sure `book.pdf` is in the same folder as `rag.py` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `AuthenticationError` | Check your OpenAI API key is correct |
| Slow first run | Normal — embedding model downloads once (~90MB) |
| `rag_cache.pkl` errors | Delete `rag_cache.pkl` and re-run |

---

## ⚙️ Configuration (edit top of rag.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per chunk. Larger = more context per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks. Helps avoid cutting mid-sentence |
| `TOP_K` | 5 | Number of chunks retrieved per question |
| `OPENAI_MODEL` | `gpt-4o-mini` | Change to `gpt-4o` for better answers |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Local embedding model (free, fast) |
