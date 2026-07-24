<h1 align="center"> RAG PDF Chatbot</h1>

<p align="center">
An AI-powered Retrieval-Augmented Generation (RAG) application that answers questions about PDF documents using semantic search and Large Language Models.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-black)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3-red)
![MIT License](https://img.shields.io/badge/License-MIT-green)

</p>

---

#  Overview

RAG PDF Chatbot allows users to ask natural language questions about any PDF document.

Instead of searching manually through long documents, the application retrieves the most relevant sections using semantic search and provides accurate answers with the help of modern Large Language Models.

The project supports multiple LLM providers, including OpenAI, Groq and Ollama.

---

#  Features

-  Ask questions about any PDF document
-  Semantic search with Sentence Transformers
-  Retrieval-Augmented Generation (RAG)
-  Multiple LLM providers (OpenAI, Groq and Ollama)
-  Automatic embedding cache for faster startup
-  Context-aware question answering
-  Beginner-friendly and lightweight implementation

---

#  Architecture

```text
PDF Document
      │
      ▼
Text Extraction (PyPDF)
      │
      ▼
Text Chunking
      │
      ▼
Sentence Transformers
      │
      ▼
Vector Embeddings
      │
      ▼
Cosine Similarity Search
      │
      ▼
Relevant Context
      │
      ▼
Groq / OpenAI / Ollama
      │
      ▼
Generated Answer
```

The application follows a standard Retrieval-Augmented Generation (RAG) pipeline.

Documents are converted into semantic embeddings, the most relevant chunks are retrieved using cosine similarity, and the selected context is sent to the language model to generate an accurate response.

---

#  Tech Stack

### Programming Language

- Python

### AI & Machine Learning

- Sentence Transformers
- Retrieval-Augmented Generation (RAG)
- Cosine Similarity Search

### LLM Providers

- Groq
- OpenAI
- Ollama

### Libraries

- PyPDF
- NumPy
- Scikit-learn

---

#  Installation

### 1. Clone the repository

```bash
git clone https://github.com/alkyramazann/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

### 4. Add your PDF

Place your PDF document inside the `data/` directory.

### 5. Run the application

```bash
python app/rag.py
```

---

#  Project Structure

```text
RAG-PDF-Chatbot/
│
├── app/
│   └── rag.py
│
├── assets/
│
├── data/
│
├── screenshots/
│
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

---

#  Author

**Ramazan Allahverdizada**

- GitHub: https://github.com/alkyramazann
- LinkedIn: https://www.linkedin.com/in/ramazan-allahverdizada-8b541431a
- Email: allahverdizade.ramazan@gmail.com
