from sentence_transformers import CrossEncoder

# =========================
# MULTILINGUAL CROSS-ENCODER
# Improvement: replaced English-only cross-encoder/ms-marco-MiniLM-L-6-v2
#              with cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 which is
#              trained on mMARCO — a multilingual MS MARCO dataset covering
#              Thai, English, and 24 other languages.
#              This means Thai queries will now be reranked correctly instead
#              of being scored by a model that has never seen Thai text.
# =========================
_reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


def rerank(query: str, docs: list[str], sources: list[dict], top_k: int = 3):
    """
    Score every (query, doc) pair and return the top_k most relevant ones.
    Both docs and sources lists are kept aligned throughout.

    Returns:
        (docs, sources) trimmed and sorted by relevance score descending.
    """
    if not docs:
        return docs, sources

    pairs  = [(query, doc) for doc in docs]
    scores = _reranker.predict(pairs)

    ranked       = sorted(zip(scores, docs, sources), key=lambda x: x[0], reverse=True)
    top          = ranked[:top_k]
    best_docs    = [doc    for _, doc,    _      in top]
    best_sources = [source for _, _,      source in top]

    return best_docs, best_sources
