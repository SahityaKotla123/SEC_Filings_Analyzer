"""
RAG pipeline using free local LLM (Ollama/Llama3) or Groq free tier.

Pipeline:
  1. Embed query → semantic search ChromaDB → retrieve top-K chunks
  2. Build grounded prompt with source context
  3. Call free LLM → return answer + cited sources
"""

from indexer import query_index
from llm_client import chat
from config import TOP_K

RAG_SYSTEM = """You are a financial analyst expert in SEC filings (10-K, 10-Q, 8-K).
Answer questions using ONLY the provided filing excerpts.
Cite sources as [Source N]. State the filing date when relevant.
If the answer is not in the excerpts, say: "Not found in the retrieved sections."
Never fabricate financial figures."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        header = f"[Source {i}] {m.get('ticker')} {m.get('form_type')} {m.get('filing_date')} | {m.get('section')}"
        # Limit each chunk to 400 chars to stay within token limits
        blocks.append(f"{header}\n{c['text'][:400]}")

    context = "\n---\n".join(blocks)
    return f"""FILING EXCERPTS:
{context}

QUESTION: {query}

Answer concisely with source citations [Source N]."""


def rag_query(
    query:      str,
    collection,
    ticker:     str|None = None,
    form_type:  str|None = None,
    section:    str|None = None,
    top_k:      int      = TOP_K,
    history:    list     = None,
) -> dict:
    """
    Full RAG query. Returns:
      { answer, sources, query }
    """
    chunks = query_index(query, collection, top_k=top_k,
                         ticker=ticker, form_type=form_type, section=section)

    if not chunks:
        return {"answer": "No indexed content found for this query. Run `ingest` first.",
                "sources": [], "query": query}

    prompt = build_prompt(query, chunks)
    answer = chat(prompt, system=RAG_SYSTEM, history=history or [])

    return {"answer": answer, "sources": chunks, "query": query}


class RAGConversation:
    """Stateful multi-turn Q&A over indexed filings."""

    def __init__(self, collection, ticker: str|None = None, form_type: str = "10-K"):
        self.collection = collection
        self.ticker     = ticker
        self.form_type  = form_type
        self.history: list[dict] = []

    def ask(self, question: str) -> dict:
        result = rag_query(
            query      = question,
            collection = self.collection,
            ticker     = self.ticker,
            form_type  = self.form_type,
            history    = self.history[-6:],
        )
        self.history += [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": result["answer"]},
        ]
        return result

    def reset(self):
        self.history = []