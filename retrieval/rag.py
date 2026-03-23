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
from query_rewriter import rewrite_query   # Agent 1

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.normpath(os.path.join(BASE_DIR, "..", "chroma_db"))

# =========================
# LOAD ENV
# =========================
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

# =========================
# TYPHOON (Thai)
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
# GEMINI (English)
# =========================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client  = None

if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not set. Add to unigraph/.env")
else:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# =========================
# EMBEDDING MODEL + CHROMA
# =========================
embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection    = chroma_client.get_collection(name="unigraph")


# =========================
# LANGUAGE DETECTION
# =========================
def is_thai(text: str) -> bool:
    return bool(re.search(r'[\u0E00-\u0E7F]', text))


# =========================
# OPTION 1 — METADATA PRE-FILTER (document_type ONLY)
#
# WHY NO LANGUAGE FILTER:
#   The query language and the document language are independent.
#   A user can ask in Thai about an English-only document, or ask
#   in English about a Thai-only document. Filtering by language
#   would exclude exactly those documents the user is looking for.
#
#   Cross-language retrieval is already handled by the bilingual
#   query rewriter producing both EN and TH search queries.
#   The metadata filter's only job is to narrow by TOPIC (document_type),
#   not by language.
#
# HOW IT WORKS:
#   Checks the query for topic-specific keywords → infers document_type
#   → passes a where={"document_type": ...} filter to ChromaDB.
#   This cuts the search space to only relevant document types,
#   improving both precision and speed.
#
#   If no type hint is found → returns None → no filtering → full search.
#   If the filter returns 0 results → retrieve_bilingual retries without it.
# =========================
QUERY_TYPE_HINTS = {
    "curriculum":  ["course", "subject", "credit", "prerequisite", "หลักสูตร", "รายวิชา", "หน่วยกิต", "บังคับก่อน"],
    "rubric":      ["grade", "criteria", "score", "rubric", "evaluation", "เกณฑ์", "คะแนน", "ประเมิน"],
    "form":        ["form", "fill", "submit", "request", "application", "แบบฟอร์ม", "กรอก", "ยื่น", "คำขอ"],
    "manual":      ["manual", "how to", "instruction", "step", "คู่มือ", "วิธี", "ขั้นตอน"],
    "regulation":  ["rule", "policy", "regulation", "allowed", "ระเบียบ", "ข้อบังคับ", "นโยบาย"],
    "schedule":    ["schedule", "timetable", "when", "date", "ตาราง", "กำหนดการ", "วันที่"],
}

def infer_metadata_filter(query: str) -> dict | None:
    """
    Returns a ChromaDB where-filter based on document_type only.
    Never filters by language — query language ≠ document language.
    """
    lower = query.lower()

    for doc_type, hints in QUERY_TYPE_HINTS.items():
        if any(hint in lower for hint in hints):
            return {"document_type": {"$eq": doc_type}}

    return None   # no filter → search entire collection


# =========================
# BILINGUAL RETRIEVE + MERGE
# =========================
def retrieve_bilingual(
    en_query:     str,
    th_query:     str,
    where_filter: dict | None = None,
    top_k:        int = 10
):
    """
    Runs two ChromaDB queries (EN + TH embeddings) with an optional
    document_type filter, then merges and deduplicates by chunk id.

    If the filtered query returns no results (filter too narrow),
    it automatically retries without the filter so the pipeline
    never returns an empty result set.
    """
    def _query(text: str):
        embedding = embed_model.encode([text])[0].tolist()
        kwargs    = dict(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = collection.query(**kwargs)
            # If filter returned nothing, fall back to unfiltered
            if not results["ids"][0]:
                raise ValueError("empty result set")
        except Exception:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas"]
            )

        return results["ids"][0], results["documents"][0], results["metadatas"][0]

    en_ids, en_docs, en_sources = _query(en_query)
    th_ids, th_docs, th_sources = _query(th_query)

    # Merge, deduplicating by chunk id — first seen wins
    seen           = set()
    merged_docs    = []
    merged_sources = []

    for chunk_id, doc, source in (
        list(zip(en_ids, en_docs, en_sources)) +
        list(zip(th_ids, th_docs, th_sources))
    ):
        if chunk_id not in seen:
            seen.add(chunk_id)
            merged_docs.append(doc)
            merged_sources.append(source)

    return merged_docs, merged_sources


# =========================
# BUILD PROMPT
# =========================
SYSTEM_PROMPT = (
    "You are a precise academic assistant for KMITL "
    "(King Mongkut's Institute of Technology Ladkrabang).\n"
    "Your primary task is to answer the user's question based strictly on the provided context.\n\n"
    "CRITICAL GUIDELINES:\n"
    "1. STRICT GROUNDING: Use ONLY the information in the context below. "
    "Do NOT use outside training knowledge under any circumstances.\n"
    "2. HANDLE FRAGMENTED DATA: The context may contain flattened tables or "
    "messy PDF extractions. Piece together disjointed fragments to synthesize the answer.\n"
    "3. MISSING INFO: If the answer cannot be deduced from the context, reply EXACTLY:\n"
    "   - Thai question   → \"ไม่พบข้อมูลในเอกสาร\"\n"
    "   - English question → \"Information not found in the document\"\n"
    "4. LANGUAGE: Reply in the exact same language as the user's question.\n"
    "5. FORMATTING: Be concise. Use bullet points when listing multiple items."
)

def build_prompt(original_query: str, docs: list[str], sources: list[dict]) -> str:
    context_parts = []
    for doc, src in zip(docs, sources):
        header = (
            f"[Source: {src.get('source', '?')} | "
            f"Type: {src.get('chunk_type', '?')} | "
            f"Section: {src.get('section_title', '?')} | "
            f"Doc type: {src.get('document_type', '?')} | "
            f"Lang: {src.get('language', '?')}]"
        )
        context_parts.append(f"{header}\n{doc}")

    context = "\n\n---\n\n".join(context_parts)
    return (
        f"Context information is below.\n"
        f"---------------------\n"
        f"{context}\n"
        f"---------------------\n"
        f"Question: {original_query}\n"
        f"Answer:"
    )


# =========================
# AGENT 2 — TYPHOON (Thai)
# =========================
def generate_typhoon_answer(user_content: str) -> str:
    if not TYPHOON_API_KEY:
        return "❌ TYPHOON_API_KEY is not set."
    try:
        response = typhoon.chat.completions.create(
            model=TYPHOON_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content}
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Typhoon API error: {e}"


# =========================
# AGENT 2 — GEMINI (English)
# =========================
def generate_gemini_answer(user_content: str) -> str:
    if not gemini_client:
        return "❌ GEMINI_API_KEY is not set or client failed to initialize."
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
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
    print("🚀 UniGraph RAG  —  2-Agent Bilingual Pipeline")
    print("   + Metadata filter : document_type only (NOT language)")
    print("   + Bilingual fetch : EN + TH queries cover all doc languages")
    print("   Type 'exit' to quit.\n")

    while True:
        original_query = input("Ask: ").strip()

        if not original_query:
            continue
        if original_query.lower() == "exit":
            break

        # ── Agent 1: bilingual rewrite ────────────────────────────────────
        en_query, th_query = rewrite_query(original_query)
        print(f"\n  🔄 EN query: \"{en_query}\"")
        print(f"  🔄 TH query: \"{th_query}\"")

        # ── Option 1: infer document_type filter (never language) ─────────
        where_filter = infer_metadata_filter(original_query)
        if where_filter:
            print(f"  🏷️  Filter: {where_filter}")
        else:
            print(f"  🏷️  Filter: none (searching all docs)")

        # ── Bilingual retrieval + merge ───────────────────────────────────
        docs, sources = retrieve_bilingual(en_query, th_query, where_filter, top_k=10)
        print(f"\n  📦 {len(docs)} unique chunks retrieved (EN + TH merged)")

        # ── Rerank with original query ────────────────────────────────────
        docs, sources = rerank(original_query, docs, sources, top_k=3)

        print("\n🔍 Top Chunks After Reranking:\n")
        for i, doc in enumerate(docs):
            src = sources[i]
            print(
                f"  {i+1}. [{src.get('chunk_type','?')} | "
                f"{src.get('document_type','?')} | "
                f"lang={src.get('language','?')}] "
                f"({src.get('source','?')}) {doc[:120]}...\n"
            )

        # ── Agent 2: route to correct LLM ────────────────────────────────
        prompt = build_prompt(original_query, docs, sources)

        if is_thai(original_query):
            print("🤖 Agent 2 → Typhoon (Thai detected)\n")
            answer = generate_typhoon_answer(prompt)
        else:
            print("🤖 Agent 2 → Gemini (English detected)\n")
            answer = generate_gemini_answer(prompt)

        print(answer)
        print("\n" + "─" * 60)