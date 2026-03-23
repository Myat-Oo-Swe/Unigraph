import os
import re
import json
import hashlib
import fitz
import pdfplumber
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# PATH SETUP
# =========================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "raw_pdfs"))
PROCESSED_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed"))
OUTPUT_PATH   = os.path.join(PROCESSED_DIR, "documents.json")
HASH_PATH     = os.path.join(PROCESSED_DIR, "file_hashes.json")
PENDING_PATH  = os.path.join(PROCESSED_DIR, "pending_changes.json")


# =========================
# OPTION B — FILE HASHING
# Compute MD5 hash of each PDF so we can detect:
#   - New files     (hash not in file_hashes.json)
#   - Changed files (hash differs from stored value)
#   - Unchanged files (hash matches → skip entirely)
# =========================
def compute_file_hash(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hashes() -> dict:
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_hashes(hashes: dict) -> None:
    with open(HASH_PATH, "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2)


# =========================
# LOAD / SAVE documents.json  (cumulative — all chunks ever processed)
# =========================
def load_all_chunks() -> list[dict]:
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_all_chunks(chunks: list[dict]) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


# =========================
# OPTION 1 — DOCUMENT TYPE INFERENCE
# =========================
DOCUMENT_TYPE_RULES = {
    "curriculum":  ["curriculum", "course", "syllabus", "หลักสูตร", "รายวิชา"],
    "rubric":      ["rubric", "grading", "criteria", "evaluation", "เกณฑ์", "ประเมิน"],
    "manual":      ["manual", "guide", "handbook", "instruction", "คู่มือ"],
    "form":        ["form", "request", "application", "แบบฟอร์ม", "คำขอ"],
    "regulation":  ["regulation", "rule", "policy", "ระเบียบ", "ข้อบังคับ"],
    "schedule":    ["schedule", "timetable", "calendar", "ตาราง", "กำหนดการ"],
}

def infer_document_type(filename: str) -> str:
    lower = filename.lower()
    for doc_type, keywords in DOCUMENT_TYPE_RULES.items():
        if any(kw in lower for kw in keywords):
            return doc_type
    return "general"


# =========================
# OPTION 1 — LANGUAGE DETECTION
# =========================
def detect_language(text: str) -> str:
    thai_chars  = len(re.findall(r'[\u0E00-\u0E7F]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total       = thai_chars + latin_chars
    if total == 0:
        return "unknown"
    ratio = thai_chars / total
    if ratio > 0.6:
        return "th"
    elif ratio < 0.15:
        return "en"
    return "mixed"


# =========================
# OPTION 1 — SECTION TITLE DETECTION
# =========================
HEADING_PATTERNS = [
    r'^\s*(?:\d+[\.\)])+\s+[A-Z\u0E00-\u0E7F]',
    r'^\s*[A-Z][A-Z\s]{4,}$',
    r'^\s*(?:chapter|section|part|หมวด|ข้อ|ตอน)\s',
    r'^\s*[ก-๙A-Z].{0,60}(?<![\.\,\:\;])\s*$',
]
_heading_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in HEADING_PATTERNS]

def extract_sections(text: str) -> list[tuple[int, str]]:
    sections = []
    for line_match in re.finditer(r'^.+$', text, re.MULTILINE):
        line = line_match.group().strip()
        if not line or len(line) > 120:
            continue
        for pattern in _heading_re:
            if pattern.match(line):
                sections.append((line_match.start(), line.strip()))
                break
    return sections

def find_section_for_offset(sections: list[tuple[int, str]], offset: int) -> str:
    title = "—"
    for sec_offset, sec_title in sections:
        if sec_offset <= offset:
            title = sec_title
        else:
            break
    return title


# =========================
# OPTION 3 — TABLE EXTRACTION
# =========================
def table_row_to_text(headers: list[str], row: list) -> str:
    parts = []
    for header, cell in zip(headers, row):
        cell_text   = str(cell).strip()   if cell   is not None else ""
        header_text = str(header).strip() if header is not None else ""
        if cell_text:
            parts.append(f"{header_text}: {cell_text}" if header_text else cell_text)
    return " | ".join(parts)

def extract_tables_from_pdf(pdf_path: str, filename: str, doc_type: str) -> list[dict]:
    table_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for table_idx, table in enumerate(page.extract_tables()):
                    if not table or len(table) < 2:
                        continue
                    headers = [
                        str(h).strip().replace("\n", " ") if h else f"Col{i}"
                        for i, h in enumerate(table[0])
                    ]
                    for row_idx, row in enumerate(table[1:], start=1):
                        row_text = table_row_to_text(headers, row)
                        if len(row_text.strip()) < 10:
                            continue
                        table_chunks.append({
                            "id":            f"{filename}_table{table_idx}_row{row_idx}_p{page_num}",
                            "text":          row_text,
                            "source":        filename,
                            "page_number":   page_num,
                            "section_title": f"Table {table_idx + 1}",
                            "document_type": doc_type,
                            "language":      detect_language(row_text),
                            "chunk_type":    "table",
                        })
    except Exception as e:
        print(f"  ⚠️  Table extraction failed for {filename}: {e}")
    return table_chunks


# =========================
# TEXT EXTRACTION + CHUNKING
# =========================
def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    doc   = fitz.open(pdf_path)
    pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    doc.close()
    return pages

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+",  " ",   text)
    return text.strip()

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    return splitter.split_text(text)


# =========================
# PROCESS A SINGLE PDF
# =========================
def process_single_pdf(filepath: str, filename: str) -> list[dict]:
    doc_type     = infer_document_type(filename)
    table_chunks = extract_tables_from_pdf(filepath, filename, doc_type)

    pages     = extract_text_from_pdf(filepath)
    full_text = clean_text("\n\n".join(text for _, text in pages))

    if not full_text.strip():
        print(f"  ⚠️  No extractable text body in {filename}.")
        return table_chunks

    sections        = extract_sections(full_text)
    text_chunks_raw = chunk_text(full_text)
    text_chunks     = []
    char_offset     = 0

    for i, chunk in enumerate(text_chunks_raw):
        chunk_start   = full_text.find(chunk[:40], char_offset)
        if chunk_start == -1:
            chunk_start = char_offset
        section_title = find_section_for_offset(sections, chunk_start)
        text_chunks.append({
            "id":            f"{filename}_text{i}",
            "text":          chunk,
            "source":        filename,
            "page_number":   -1,
            "section_title": section_title,
            "document_type": doc_type,
            "language":      detect_language(chunk),
            "chunk_type":    "text",
        })
        char_offset = max(0, chunk_start + len(chunk) - 150)

    all_chunks = table_chunks + text_chunks
    print(f"  → type={doc_type} | {len(table_chunks)} table chunks | {len(text_chunks)} text chunks")
    return all_chunks


# =========================
# MAIN — INCREMENTAL PROCESSING (OPTION B)
# =========================
def process_pdfs():
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"PDF folder not found: {DATA_PATH}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    pdf_files   = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️  No PDF files found.")
        return

    # Load existing state
    stored_hashes = load_hashes()
    all_chunks    = load_all_chunks()

    # Classify each PDF as new, changed, or unchanged
    new_files     = []
    changed_files = []
    skipped_files = []

    for filename in pdf_files:
        filepath     = os.path.join(DATA_PATH, filename)
        current_hash = compute_file_hash(filepath)
        stored_hash  = stored_hashes.get(filename)

        if stored_hash is None:
            new_files.append((filename, filepath, current_hash))
        elif stored_hash != current_hash:
            changed_files.append((filename, filepath, current_hash))
        else:
            skipped_files.append(filename)

    # Summary before processing
    print(f"\n📊 File status:")
    print(f"   ✅ Unchanged (skip) : {len(skipped_files)}")
    print(f"   🆕 New              : {len(new_files)}")
    print(f"   ✏️  Changed          : {len(changed_files)}")

    if not new_files and not changed_files:
        print("\n✅ Nothing to update — all files are up to date.")
        # Write empty pending so embed.py knows there's nothing to do
        _save_pending([], [])
        return

    # For changed files: remove their old chunks from all_chunks
    changed_names = {f for f, _, _ in changed_files}
    if changed_names:
        before = len(all_chunks)
        all_chunks = [c for c in all_chunks if c["source"] not in changed_names]
        print(f"\n🗑️  Removed {before - len(all_chunks)} stale chunks for changed files: {changed_names}")

    # Process new + changed files
    files_to_process = new_files + changed_files
    new_chunks        = []

    for filename, filepath, new_hash in tqdm(files_to_process, desc="Processing PDFs"):
        print(f"\nProcessing: {filename}")
        chunks = process_single_pdf(filepath, filename)
        new_chunks.extend(chunks)
        stored_hashes[filename] = new_hash   # update hash immediately

    # Merge new chunks into master list
    all_chunks.extend(new_chunks)

    # Persist everything
    save_all_chunks(all_chunks)
    save_hashes(stored_hashes)
    _save_pending(new_chunks, list(changed_names))

    print(f"\n✅ Done.")
    print(f"   New chunks added    : {len(new_chunks)}")
    print(f"   Total chunks in DB  : {len(all_chunks)}")
    print(f"   documents.json      → {OUTPUT_PATH}")
    print(f"   file_hashes.json    → {HASH_PATH}")
    print(f"   pending_changes.json → {PENDING_PATH}")


# =========================
# PENDING CHANGES FILE
# Tells embed.py exactly what to do:
#   - which chunk IDs to delete (from changed files)
#   - which new chunks to embed
# This means embed.py never needs to look at documents.json itself.
# =========================
def _save_pending(new_chunks: list[dict], changed_filenames: list[str]) -> None:
    pending = {
        "changed_filenames": changed_filenames,   # embed.py deletes these from ChromaDB
        "new_chunks":        new_chunks,           # embed.py embeds these
    }
    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_pdfs()