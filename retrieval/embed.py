import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# =========================
# PATH SETUP
# =========================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.normpath(os.path.join(BASE_DIR, "..", "chroma_db"))
DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "documents.json"))

# =========================
# LOAD EMBEDDING MODEL
# =========================
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# =========================
# INIT CHROMA (PERSISTENT)
# Improvement: delete the existing collection before recreating it.
#              Without this, re-running embed.py appends duplicates on top
#              of the previous run, bloating the DB and corrupting results.
# =========================
client = chromadb.PersistentClient(path=DB_PATH)

existing = [c.name for c in client.list_collections()]
if "unigraph" in existing:
    client.delete_collection("unigraph")
    print("🗑️  Deleted existing 'unigraph' collection — starting fresh.")

collection = client.create_collection(name="unigraph")

# =========================
# LOAD DATA
# =========================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} chunks")

texts     = [doc["text"]   for doc in documents]
ids       = [doc["id"]     for doc in documents]
metadatas = [{"source": doc["source"]} for doc in documents]

# =========================
# EMBED + STORE
# =========================
print("Creating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

print(f"✅ Stored {len(texts)} chunks in ChromaDB at: {DB_PATH}")
