import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerank import rerank

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.normpath(os.path.join(BASE_DIR, "..", "chroma_db"))

# =========================
# LOAD EMBEDDING MODEL
# =========================
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# =========================
# LOAD CHROMA DB
# =========================
client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="unigraph")


# =========================
# QUERY FUNCTION
# =========================
def query_system(user_query: str, top_k: int = 10) -> tuple[list[str], list[dict]]:
    query_embedding = model.encode([user_query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs    = results["documents"][0]
    sources = results["metadatas"][0]

    # Rerank with multilingual cross-encoder → keep best 3
    docs, sources = rerank(user_query, docs, sources, top_k=3)

    return docs, sources


# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    print("🔍 Query system ready  (multilingual reranker)")
    print("   Type 'exit' to quit.\n")

    while True:
        q = input("Ask: ").strip()

        if not q:
            continue
        if q.lower() == "exit":
            break

        docs, sources = query_system(q)

        print("\nTop Results (after reranking):\n")
        for i, doc in enumerate(docs):
            print(f"  {i+1}. ({sources[i]['source']}) {doc[:200]}...\n")
