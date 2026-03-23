import os
import re
import json
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# PATH SETUP
# =========================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "raw_pdfs"))
OUTPUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "documents.json"))


# =========================
# EXTRACT
# =========================
def extract_text_from_pdf(pdf_path: str) -> str:
    doc  = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# =========================
# CLEAN
# =========================
def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse 3+ newlines → double
    text = re.sub(r"[ \t]+",  " ",   text)  # collapse horizontal whitespace
    return text.strip()


# =========================
# CHUNK
# Improvement: increased chunk_size 500 → 800 characters.
#              Larger chunks carry more complete context per retrieval hit,
#              reducing cases where the answer is split across two chunks.
#              Overlap raised proportionally 100 → 150.
# =========================
def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    return splitter.split_text(text)


# =========================
# PROCESS ALL PDFs
# =========================
def process_pdfs() -> list[dict]:
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(
            f"PDF folder not found: {DATA_PATH}\n"
            "Make sure raw PDFs are placed in data/raw_pdfs/"
        )

    pdf_files  = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    all_chunks = []

    if not pdf_files:
        print("⚠️  No PDF files found in", DATA_PATH)
        return all_chunks

    for filename in tqdm(pdf_files, desc="Ingesting PDFs"):
        filepath = os.path.join(DATA_PATH, filename)
        print(f"\nProcessing: {filename}")

        text = extract_text_from_pdf(filepath)
        text = clean_text(text)

        if not text.strip():
            print(f"  ⚠️  No extractable text in {filename}, skipping.")
            continue

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id":     f"{filename}_{i}",
                "text":   chunk,
                "source": filename
            })

        print(f"  → {len(chunks)} chunks extracted")

    return all_chunks


# =========================
# SAVE
# =========================
def save_chunks(chunks: list[dict]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Done! Saved {len(chunks)} chunks → {OUTPUT_PATH}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    chunks = process_pdfs()
    save_chunks(chunks)
