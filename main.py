"""
CLI entrypoint.

python main.py ingest --ticker AAPL --form 10-K --limit 3
python main.py query  --ticker AAPL --q "What are Apple's AI risks?"
python main.py chat   --ticker AAPL
python main.py drift  --ticker META --section "risk factors" --year-a 2022 --year-b 2024
python main.py eval   --output eval_report.json
python main.py stats
"""

import argparse, json
from indexer       import get_client, get_collection, index_chunks, collection_stats
from edgar_fetcher import ingest_filings
from chunker       import build_chunks
from rag_core      import rag_query, RAGConversation
from drift_detector import full_drift_report, DRIFT_SECTIONS
from evaluations   import run_eval_suite


def col():
    c = get_client()
    return get_collection(c)


def cmd_ingest(a):
    filings = ingest_filings(a.ticker, a.form, a.limit)
    chunks  = []
    for f in filings:
        fc = build_chunks(f)
        chunks.extend(fc)
        print(f"  chunked {f['filing_date']}: {len(fc)} chunks")
    n = index_chunks(chunks, col())
    print(f"\n✓ Indexed {n} chunks into ChromaDB")


def cmd_query(a):
    r = rag_query(a.q, col(), ticker=a.ticker, form_type=a.form, section=a.section)
    print("\n── Answer ───────────────────────────────────────────────")
    print(r["answer"])
    print(f"\n── Sources ({len(r['sources'])} chunks retrieved) ───────")
    for i, s in enumerate(r["sources"], 1):
        m = s["metadata"]
        print(f"  [{i}] {m['ticker']} {m['form_type']} {m['filing_date']} | {m['section']} | {s['score']}")


def cmd_chat(a):
    conv = RAGConversation(col(), ticker=a.ticker, form_type=a.form)
    print(f"\nChatting with {a.ticker} {a.form} filings. Type 'quit' to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        r = conv.ask(q)
        print(f"\nAssistant: {r['answer']}\n")


def cmd_drift(a):
    r = full_drift_report(col(), a.ticker, a.section, a.year_a, a.year_b)

    st = r["statistical"]
    print("\n── Statistical ──────────────────────────────────────────")
    print(f"  Drift score:      {st['drift_score']}/100")
    print(f"  Cosine sim:       {st['cosine_similarity']}")
    print(f"  JS divergence:    {st['js_divergence']}")
    print(f"  Emerged terms:    {', '.join(st['emerged_terms'][:8])}")
    print(f"  Dropped terms:    {', '.join(st['dropped_terms'][:8])}")

    sem = r.get("semantic", {})
    if sem and "overall_drift_score" in sem:
        print("\n── Semantic (LLM) ───────────────────────────────────────")
        print(f"  Company:          {sem.get('company_name', a.ticker)}")
        print(f"  Drift score:      {sem['overall_drift_score']}/100")
        print(f"  Tone shift:       {sem.get('tone_shift')}")
        print(f"  New concepts:     {', '.join(sem.get('new_concepts', []))}")
        print(f"  Dropped concepts: {', '.join(sem.get('dropped_concepts', []))}")
        print(f"\n  Analyst signal:\n  {sem.get('analyst_signal')}")

    print(f"\n  ▶ Blended drift score: {r['blended_drift_score']}/100")


def cmd_eval(a):
    report = run_eval_suite(col(), drift_runs=a.drift_runs)
    if a.output:
        with open(a.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved → {a.output}")


def cmd_stats(a):
    s = collection_stats(col())
    print(f"\nChromaDB index stats:")
    print(f"  Total chunks : {s['total_chunks']}")
    print(f"  Tickers      : {', '.join(s['tickers']) or 'none'}")
    print(f"  Form types   : {', '.join(s['form_types']) or 'none'}")
    print(f"  Sections     : {', '.join(s['sections']) or 'none'}")


def main():
    p   = argparse.ArgumentParser(description="SEC Filings Analyzer")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest");  s.add_argument("--ticker", required=True); s.add_argument("--form", default="10-K"); s.add_argument("--limit", type=int, default=3)
    s = sub.add_parser("query");   s.add_argument("--q", required=True); s.add_argument("--ticker"); s.add_argument("--form", default="10-K"); s.add_argument("--section", default=None)
    s = sub.add_parser("chat");    s.add_argument("--ticker"); s.add_argument("--form", default="10-K")
    s = sub.add_parser("drift");   s.add_argument("--ticker", required=True); s.add_argument("--section", default="risk factors"); s.add_argument("--year-a", required=True, dest="year_a"); s.add_argument("--year-b", required=True, dest="year_b")
    s = sub.add_parser("eval");    s.add_argument("--drift-runs", type=int, default=2, dest="drift_runs"); s.add_argument("--output", default=None)
    sub.add_parser("stats")

    args = p.parse_args()
    {"ingest": cmd_ingest, "query": cmd_query, "chat": cmd_chat,
     "drift": cmd_drift, "eval": cmd_eval, "stats": cmd_stats}[args.command](args)

if __name__ == "__main__":
    main()