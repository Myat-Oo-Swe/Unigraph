import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "chroma_db"))
PENDING_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "processed", "pending_changes.json")
)

# =========================
# LOAD EMBEDDING MODEL (singleton)
# =========================
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# =========================
# INIT CHROMA CLIENT (singleton)
# We no longer wipe the entire collection on every run.
# get_or_create_collection ensures the DB is created on first run
# and reused on all subsequent runs.
# =========================
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="unigraph")


def embed_documents() -> None:
    """Embed pending changes from pending_changes.json into ChromaDB.

    This is safe to call from both CLI and Streamlit:
      - Does nothing if there are no pending changes.
      - Never calls sys.exit; it simply returns.
    """

    if not os.path.exists(PENDING_PATH):
        print("⚠️  pending_changes.json not found.")
        print("    Run ingest.py first before running embed.py / embed_documents().")
        return

    try:
        with open(PENDING_PATH, "r", encoding="utf-8") as f:
            pending = json.load(f)
    except json.JSONDecodeError:
        print("⚠️  pending_changes.json is empty or corrupted. Skipping embedding.")
        return

    changed_filenames = pending.get("changed_filenames", [])
    new_chunks = pending.get("new_chunks", [])

    if not changed_filenames and not new_chunks:
        print("✅ Nothing to embed — all files are already up to date.")
        return

    print("📋 Pending changes:")
    print(f"   Files to update  : {changed_filenames if changed_filenames else 'none'}")
    print(f"   New chunks to add: {len(new_chunks)}")

    # =========================
    # STEP 1 — DELETE STALE CHUNKS FOR CHANGED FILES
    # For each changed file, find all its chunk IDs currently in
    # ChromaDB and delete them before adding the new version.
    # This prevents old and new versions of the same file coexisting.
    # =========================
    if changed_filenames:
        print("\n🗑️  Removing stale chunks for changed files...")
        for filename in changed_filenames:
            # Query ChromaDB for all chunks whose source matches this file
            results = collection.get(
                where={"source": {"$eq": filename}},
                include=[],  # we only need the IDs
            )
            stale_ids = results.get("ids", [])
            if stale_ids:
                collection.delete(ids=stale_ids)
                print(f"   Deleted {len(stale_ids)} stale chunks for: {filename}")
            else:
                print(f"   No existing chunks found for: {filename} (skipping delete)")

    # =========================
    # STEP 2 — EMBED + STORE NEW CHUNKS
    # =========================
    if new_chunks:
        print(f"\n⚙️  Embedding {len(new_chunks)} new chunks...")

        texts = [doc["text"] for doc in new_chunks]
        ids = [doc["id"] for doc in new_chunks]
        metadatas = [
            {
                "source": doc.get("source", "unknown"),
                "page_number": doc.get("page_number", -1),
                "section_title": doc.get("section_title", "—"),
                "document_type": doc.get("document_type", "general"),
                "language": doc.get("language", "unknown"),
                "chunk_type": doc.get("chunk_type", "text"),
            }
            for doc in new_chunks
        ]

        embeddings = model.encode(texts, show_progress_bar=True).tolist()

        # Add in batches of 500 to avoid memory spikes on large document sets
        BATCH_SIZE = 500
        for i in range(0, len(new_chunks), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(new_chunks))
            collection.add(
                documents=texts[i:batch_end],
                embeddings=embeddings[i:batch_end],
                ids=ids[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            print(
                f"   Stored batch {i // BATCH_SIZE + 1} "
                f"({i + 1}–{batch_end} of {len(new_chunks)})"
            )

    # =========================
    # STEP 3 — CLEAR PENDING FILE
    # Mark pending_changes.json as empty so re-running embed.py
    # without new ingest.py changes does nothing.
    # =========================
    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump({"changed_filenames": [], "new_chunks": []}, f)

    total = collection.count()
    print("\n✅ Done.")
    print(f"   Total chunks in ChromaDB : {total}")
    print(f"   DB path                  : {DB_PATH}")


if __name__ == "__main__":
    embed_documents()
