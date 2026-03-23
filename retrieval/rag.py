import os
import sys
import re
import chromadb
from openai import OpenAI
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerank import rerank

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.normpath(os.path.join(BASE_DIR, "..", "chroma_db"))

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

# =========================
# TYPHOON SETUP (For Thai)
# =========================
TYPHOON_API_KEY = os.environ.get("TYPHOON_API_KEY", "")
TYPHOON_MODEL   = "typhoon-v2.5-30b-a3b-instruct"

if not TYPHOON_API_KEY:
    print("⚠️  TYPHOON_API_KEY not set. Add to unigraph/.env")

typhoon = OpenAI(
    api_key=TYPHOON_API_KEY or "no-key",
    base_url="https://api.opentyphoon.ai/v1"
)

# =========================
# GEMINI SETUP (For English)
# =========================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = None

if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not set. Add to unigraph/.env")
else:
    # Initialize the new google-genai client
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# =========================
# LOAD EMBEDDING MODEL & CHROMA
# =========================
embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="unigraph")

# =========================
# LANGUAGE DETECTION
# =========================
def is_thai(text: str) -> bool:
    """Returns True if the text contains any Thai characters."""
    return bool(re.search(r'[\u0E00-\u0E7F]', text))

# =========================
# RETRIEVE
# =========================
def retrieve(query: str, top_k: int = 10):
    query_embedding = embed_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0], results["metadatas"][0]

# =========================
# BUILD PROMPT
# =========================
SYSTEM_PROMPT = (
    "You are a precise academic assistant for KMITL.\n"
    "Your primary task is to answer the user's question based strictly on the provided context.\n\n"
    "CRITICAL GUIDELINES:\n"
    "1. STRICT GROUNDING: Use ONLY the information in the context below. Do NOT use outside training knowledge.\n"
    "2. HANDLE FRAGMENTED DATA: The context may contain flattened tables or messy PDF extractions. You must piece together disjointed headers (e.g., 'Needs Improvement'), bullet points, and fragments to synthesize the answer.\n"
    "3. MISSING INFO: If the answer cannot be logically deduced from the context at all, reply EXACTLY with: \"ไม่พบข้อมูลในเอกสาร\" (if Thai) or \"Information not found in the document\" (if English).\n"
    "4. LANGUAGE: Always reply in the exact same language as the user's question.\n"
    "5. FORMATTING: Be concise. Use bullet points if listing multiple criteria or items."
)

def build_prompt(query: str, docs: list[str]) -> str:
    context = "\n\n---\n\n".join(docs)
    return (
        f"Context information is below.\n"
        f"---------------------\n"
        f"{context}\n"
        f"---------------------\n"
        f"Question: {query}\n"
        f"Answer:"
    )

# =========================
# GENERATE - TYPHOON (THAI)
# =========================
def generate_typhoon_answer(user_content: str) -> str:
    if not TYPHOON_API_KEY:
        return "❌ TYPHOON_API_KEY is not set."

    try:
        response = typhoon.chat.completions.create(
            model=TYPHOON_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Typhoon API error: {e}"

# =========================
# GENERATE - GEMINI (ENGLISH)
# =========================
def generate_gemini_answer(user_content: str) -> str:
    if not gemini_client:
        return "❌ GEMINI_API_KEY is not set or client failed to initialize."

    try:
        # Using the new google-genai syntax and the active 2.5-flash model
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1024,
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini API error: {e}"

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    print(f"🚀 UniGraph RAG Ready (Dual-Model Routing: Typhoon 🇹🇭 / Gemini 🇬🇧)")
    print("   Type 'exit' to quit.\n")

    while True:
        query = input("Ask: ").strip()

        if not query:
            continue
        if query.lower() == "exit":
            break

        # Check language
        is_thai_query = is_thai(query)

        # Step 1 — broad vector retrieval
        docs, sources = retrieve(query, top_k=10)

        # Step 2 — multilingual reranking → keep best 3
        docs, sources = rerank(query, docs, sources, top_k=3)

        print("\n🔍 Top Chunks After Reranking:\n")
        for i, doc in enumerate(docs):
            print(f"  {i+1}. ({sources[i]['source']}) {doc[:150]}...\n")

        # Step 3 — Build Prompt
        prompt = build_prompt(query, docs)

        # Step 4 — Route to correct LLM
        if is_thai_query:
            print("🤖 Routing to: Typhoon (Thai detected)...\n")
            answer = generate_typhoon_answer(prompt)
        else:
            print("🤖 Routing to: Gemini (English/Other detected)...\n")
            answer = generate_gemini_answer(prompt)

        print(answer)
        print("\n" + "─" * 60)