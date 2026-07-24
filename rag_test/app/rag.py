"""
RAG Chatbot — Reads a PDF and answers questions using sentence-transformers + your choice of LLM.

Supported LLM providers (pick one):
  1. Groq   — FREE, no credit card, very fast  -> https://console.groq.com
  2. Ollama — 100% local, no internet needed   -> https://ollama.com
  3. OpenAI — Paid (requires billing credits)  -> https://platform.openai.com

Place book.pdf in the same folder as this script before running.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── 1. Dependency check ────────────────────────────────────────────────────────
REQUIRED = {
    "pypdf":                "pypdf",
    "sentence_transformers":"sentence-transformers",
    "numpy":                "numpy",
    "openai":               "openai",
    "sklearn":              "scikit-learn",
}

missing = []
for module, pkg in REQUIRED.items():
    try:
        __import__(module)
    except ImportError:
        missing.append(pkg)

if missing:
    print("\nMissing packages. Run this first:\n")
    print(f"    pip install {' '.join(missing)}\n")
    sys.exit(1)

# ── 2. Imports ─────────────────────────────────────────────────────────────────
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ── 3. Configuration ───────────────────────────────────────────────────────────
PDF_PATH      = "book.pdf"
CACHE_FILE    = "rag_cache.pkl"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
TOP_K         = 5
EMBED_MODEL   = "all-MiniLM-L6-v2"

# Current working Groq models (in order of preference)
# llama-3.3-70b-versatile is the official replacement for the old llama3-8b-8192
GROQ_PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",   # Best quality, recommended replacement
    "llama-3.1-8b-instant",      # Fastest, lightest
    "llama-4-scout-17b-16e-instruct",  # Latest Llama 4
]

OLLAMA_MODEL = "llama3"         # Run: ollama pull llama3
OPENAI_MODEL = "gpt-4o-mini"    # Requires paid credits


# ── 4. Groq: auto-pick a working model ────────────────────────────────────────
def get_groq_model(client):
    """
    Query the Groq /models endpoint and pick the best available model.
    Falls back through GROQ_PREFERRED_MODELS in order.
    This means the script will never break due to a decommissioned model.
    """
    try:
        response   = client.models.list()
        live_ids   = {m.id for m in response.data}
        for candidate in GROQ_PREFERRED_MODELS:
            if candidate in live_ids:
                return candidate
        # If none of our preferred models are live, just use the first available
        if live_ids:
            fallback = sorted(live_ids)[0]
            print(f"    Note: using fallback model '{fallback}'")
            return fallback
    except Exception as e:
        print(f"    Warning: could not query Groq models ({e}). Using default.")

    return GROQ_PREFERRED_MODELS[0]   # best-effort default


# ── 5. Provider selection ──────────────────────────────────────────────────────
def choose_provider():
    print("\n+-------------------------------------------------+")
    print("|        Choose your LLM provider                 |")
    print("+-------------------------------------------------+")
    print("|  1. Groq   -- FREE, fast, needs free API key   |")
    print("|  2. Ollama -- FREE, fully local, no internet   |")
    print("|  3. OpenAI -- Paid (requires billing credits)  |")
    print("+-------------------------------------------------+")

    while True:
        choice = input("\nEnter 1, 2, or 3: ").strip()
        if choice == "1":
            return _setup_groq()
        elif choice == "2":
            return _setup_ollama()
        elif choice == "3":
            return _setup_openai()
        else:
            print("Please enter 1, 2, or 3.")


def _setup_groq():
    print("\n-- Groq Setup ------------------------------------------")
    print("   Get a FREE key at: https://console.groq.com")
    print("   (No credit card required)\n")

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        api_key = input("Enter your Groq API key: ").strip()
    if not api_key:
        print("No key provided. Exiting.")
        sys.exit(1)

    client = OpenAI(
        api_key  = api_key,
        base_url = "https://api.groq.com/openai/v1",
    )

    print("   Checking available Groq models ...")
    model = get_groq_model(client)
    print(f"   OK  Groq ready -- model: {model}")
    return client, model, "Groq"


def _setup_ollama():
    print("\n-- Ollama Setup ----------------------------------------")
    print("   Ollama must be installed and running locally.")
    print("   Install: https://ollama.com")
    print(f"   Then run: ollama pull {OLLAMA_MODEL}\n")

    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=3)
    except Exception:
        print("ERROR: Cannot reach Ollama at http://localhost:11434")
        print("   Make sure Ollama is installed and running.")
        sys.exit(1)

    client = OpenAI(
        api_key  = "ollama",
        base_url = "http://localhost:11434/v1",
    )
    print(f"   OK  Ollama ready -- model: {OLLAMA_MODEL}")
    return client, OLLAMA_MODEL, "Ollama"


def _setup_openai():
    print("\n-- OpenAI Setup ----------------------------------------")
    print("   Requires an active billing plan.")
    print("   Top up at: https://platform.openai.com/settings/billing\n")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("No key provided. Exiting.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    print(f"   OK  OpenAI ready -- model: {OPENAI_MODEL}")
    return client, OPENAI_MODEL, "OpenAI"


# ── 6. PDF Reading ─────────────────────────────────────────────────────────────
def read_pdf(path):
    if not os.path.exists(path):
        print(f"\nERROR: File not found: {path}")
        print("   Put book.pdf in the same folder as rag.py\n")
        sys.exit(1)

    print(f"\nReading PDF: {path}")
    reader = PdfReader(path)
    pages  = [page.extract_text() for page in reader.pages if page.extract_text()]
    text   = "\n".join(pages)
    print(f"   OK  {len(reader.pages)} pages | {len(text):,} characters")
    return text


# ── 7. Chunking ────────────────────────────────────────────────────────────────
def split_into_chunks(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end   = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    print(f"   OK  {len(chunks)} chunks (size={size}, overlap={overlap})")
    return chunks


# ── 8. Embeddings ──────────────────────────────────────────────────────────────
def build_embeddings(chunks, model_name=EMBED_MODEL):
    print(f"\nLoading embedding model: {model_name}")
    model      = SentenceTransformer(model_name)
    print(f"   Encoding {len(chunks)} chunks (may take a moment first time)...")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64)
    print(f"   OK  Embeddings shape: {embeddings.shape}")
    return model, embeddings


# ── 9. Cache ───────────────────────────────────────────────────────────────────
def save_cache(chunks, embeddings, path=CACHE_FILE):
    with open(path, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
    print(f"   OK  Cache saved to {path}")


def load_cache(path=CACHE_FILE):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Cache loaded: {len(data['chunks'])} chunks from {path}")
        return data["chunks"], data["embeddings"]
    return None, None


# ── 10. Retrieval ──────────────────────────────────────────────────────────────
def retrieve(query, embed_model, chunks, embeddings, top_k=TOP_K):
    query_vec   = embed_model.encode([query])
    scores      = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# ── 11. Answer Generation ──────────────────────────────────────────────────────
def generate_answer(query, context_chunks, client, model):
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a knowledgeable reading assistant. "
        "Answer questions based on the provided book excerpts. "
        "Give clear, complete, conversational answers -- never just repeat raw text. "
        "If the context lacks enough information, say so honestly."
    )
    user_prompt = (
        f"Relevant excerpts from the book:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        f"Please give a clear, helpful answer based on the excerpts above."
    )

    response = client.chat.completions.create(
        model       = model,
        messages    = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature = 0.3,
        max_tokens  = 800,
    )
    return response.choices[0].message.content.strip()


# ── 12. Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("   RAG Chatbot -- Ask questions about your PDF")
    print("="*60)

    client, model_name, provider_name = choose_provider()

    chunks, embeddings = load_cache()

    if chunks is None:
        print("\nBuilding vector store from PDF...")
        raw_text               = read_pdf(PDF_PATH)
        chunks                 = split_into_chunks(raw_text)
        embed_model, embeddings = build_embeddings(chunks)
        save_cache(chunks, embeddings)
    else:
        print(f"\nLoading embedding model: {EMBED_MODEL}")
        embed_model = SentenceTransformer(EMBED_MODEL)

    print(f"\nReady! Using {provider_name} ({model_name})")
    print("Type your question. Type 'quit' to exit.\n")
    print("-"*60)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        print("Searching relevant passages...")
        top_chunks = retrieve(query, embed_model, chunks, embeddings, TOP_K)

        print("Generating answer...\n")
        try:
            answer = generate_answer(query, top_chunks, client, model_name)
            print(f"Answer:\n\n{answer}\n")
        except Exception as e:
            print(f"\nERROR: {e}")
            print("Check your API key / connection and try again.\n")

        print("-"*60)


if __name__ == "__main__":
    main()
