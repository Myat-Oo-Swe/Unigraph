import re
from deep_translator import GoogleTranslator

# =========================
# AGENT 1 — BILINGUAL QUERY REWRITER (deep-translator)
# =========================
# Replaced Gemini API with deep-translator (wraps Google Translate).
# Reasons:
#   - No API key needed, no model download, no GPU
#   - Google Translate quality >> any 300MB local model
#   - The rewriter only needs translation, not reasoning —
#     a full LLM was overkill for this task
#   - Eliminates the strict format parsing problems that caused
#     empty/garbage output with Gemini 2.5-flash thinking mode
#
# Strategy:
#   1. Detect source language from Unicode character ratio
#   2. Translate in the missing direction (EN→TH or TH→EN)
#   3. Extract keywords from both versions for cleaner retrieval
#   4. Always return (en_query, th_query) — never fails silently
# =========================


# =========================
# LANGUAGE DETECTION
# Fast Unicode-based detection — no external library needed.
# =========================
def _detect_language(text: str) -> str:
    thai_chars  = len(re.findall(r'[\u0E00-\u0E7F]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total       = thai_chars + latin_chars
    if total == 0:
        return "other"
    return "th" if (thai_chars / total) > 0.3 else "en"


# =========================
# KEYWORD EXTRACTION
# Strips question words and filler so ChromaDB gets clean keywords
# instead of full conversational sentences.
#
# Why this matters:
#   "I wanna know the skill sets of Su Sandi Linn for internship"
#   → "skill sets Su Sandi Linn internship"   (much better retrieval signal)
# =========================
EN_STOPWORDS = {
    "i", "want", "wanna", "know", "tell", "me", "what", "is", "are", "the",
    "a", "an", "of", "for", "to", "do", "does", "how", "which", "who",
    "please", "can", "you", "about", "give", "find", "show", "list",
    "in", "on", "at", "with", "and", "or", "not", "this", "that", "there",
    "would", "could", "should", "be", "have", "has", "any", "some", "it",
    "better", "good", "best", "more", "than",
}

TH_STOPWORDS = {
    "ฉัน", "ผม", "อยาก", "ทราบ", "รู้", "คือ", "อะไร", "บ้าง", "มี",
    "ของ", "ที่", "ใน", "และ", "หรือ", "เป็น", "ได้", "กรุณา", "ช่วย",
    "บอก", "แสดง", "หา", "ให้", "สำหรับ", "เกี่ยวกับ", "วิธี", "นี้",
    "นั้น", "จาก", "ไป", "มา", "แบบ", "ว่า", "จะ", "ต้อง", "ไหน",
    "อย่างไร", "ไหม", "ดี", "กว่า", "ดีกว่า",
}

def _extract_keywords(text: str, lang: str) -> str:
    """Remove stopwords and return space-joined keywords."""
    stopwords = TH_STOPWORDS if lang == "th" else EN_STOPWORDS
    if lang == "th":
        # Thai has no spaces between words — keep full text but remove stopwords
        words = text.split()  # split on whitespace (works for mixed text)
    else:
        words = text.lower().split()
    keywords = [w for w in words if w.lower() not in stopwords and len(w) > 1]
    return " ".join(keywords) if keywords else text


# =========================
# TRANSLATE
# =========================
def _translate(text: str, target_lang: str) -> str:
    """
    Translates text to target_lang ('en' or 'th').
    Falls back to original text if translation fails.
    """
    try:
        translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
        return translated if translated and len(translated.strip()) > 2 else text
    except Exception as e:
        print(f"  ⚠️  Translation failed ({e}) — using original text.")
        return text


# =========================
# MAIN REWRITE FUNCTION
# =========================
def rewrite_query(user_query: str) -> tuple[str, str]:
    """
    Returns (english_query, thai_query) as keyword-focused search strings.

    Logic:
      - Detect source language
      - Translate in the missing direction
      - Extract keywords from both versions
      - Return clean (en_query, th_query)

    Never raises — falls back to original query on any error.
    """
    if not user_query.strip():
        return user_query, user_query

    try:
        source_lang = _detect_language(user_query)

        if source_lang == "th":
            # Thai input → translate to English
            en_raw = _translate(user_query, "en")
            th_raw = user_query
        else:
            # English (or other) input → translate to Thai
            en_raw = user_query
            th_raw = _translate(user_query, "th")

        # Extract keywords from both versions
        en_query = _extract_keywords(en_raw, "en")
        th_query = _extract_keywords(th_raw, "th")

        # Final safety fallback
        if len(en_query.strip()) < 3:
            en_query = user_query
        if len(th_query.strip()) < 3:
            th_query = user_query

        return en_query, th_query

    except Exception as e:
        print(f"  ⚠️  Query rewriter failed ({e}) — using original query.")
        return user_query, user_query