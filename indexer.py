"""
ChromaDB indexing — free, local, persistent.
Embeddings: sentence-transformers all-MiniLM-L6-v2 (free, ~80 MB, auto-downloads).
"""

import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_DIR, EMBEDDING_MODEL, TOP_K


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(client: chromadb.PersistentClient, name: str = "sec_filings"):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[dict], collection) -> int:
    """Upsert chunks into ChromaDB. Safe to re-run (idempotent)."""
    if not chunks:
        return 0

    ids       = [c["chunk_id"] for c in chunks]
    documents = [c["text"]     for c in chunks]
    metadatas = [
        {k: v for k, v in c.items() if k not in ("text", "chunk_id")}
        for c in chunks
    ]

    batch = 500
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids       = ids[i:i+batch],
            documents = documents[i:i+batch],
            metadatas = metadatas[i:i+batch],
        )
    return len(ids)


def query_index(
    query:           str,
    collection,
    top_k:           int       = TOP_K,
    ticker:          str|None  = None,
    form_type:       str|None  = None,
    section:         str|None  = None,
) -> list[dict]:
    """Semantic search with optional metadata filters. Returns [{text, score, metadata}]."""
    conditions = []
    if ticker:    conditions.append({"ticker":    {"$eq": ticker.upper()}})
    if form_type: conditions.append({"form_type": {"$eq": form_type}})
    if section:   conditions.append({"section":   {"$eq": section}})

    where = ({"$and": conditions} if len(conditions) > 1
             else conditions[0] if conditions else None)

    results = collection.query(
        query_texts = [query],
        n_results   = top_k,
        where       = where,
        include     = ["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "score": round(1 - dist, 4), "metadata": meta}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def collection_stats(collection) -> dict:
    count  = collection.count()
    sample = collection.get(limit=2000, include=["metadatas"])
    metas  = sample.get("metadatas") or []
    return {
        "total_chunks": count,
        "tickers":      sorted({m["ticker"]    for m in metas}),
        "form_types":   sorted({m["form_type"] for m in metas}),
        "sections":     sorted({m["section"]   for m in metas}),
    }