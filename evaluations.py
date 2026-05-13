"""
Evaluation suite — three categories:

  1. Retrieval Precision   — right chunks retrieved for known queries?
  2. Answer Faithfulness   — LLM-as-judge (free Llama3) grounding check
  3. Drift Consistency     — score variance across multiple runs
"""

import json
import statistics
from dataclasses import dataclass, field, asdict

from indexer import query_index
from rag_core import rag_query
from drift_detector import semantic_drift
from llm_client import chat_json


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class RAGCase:
    query:             str
    ticker:            str
    expected_section:  str
    expected_keywords: list[str]
    form_type:         str = "10-K"

@dataclass
class DriftCase:
    ticker:             str
    section:            str
    year_a:             str
    year_b:             str
    expected_direction: str        # "increase" | "decrease" | "stable"
    expected_concepts:  list[str]

@dataclass
class EvalResult:
    category: str
    case:     str
    passed:   bool
    score:    float
    details:  dict = field(default_factory=dict)


# ── 1. Retrieval Precision ────────────────────────────────────────────────

def eval_retrieval(case: RAGCase, collection) -> EvalResult:
    """Precision@6: fraction of top-6 chunks from expected section."""
    chunks   = query_index(case.query, collection, top_k=6,
                           ticker=case.ticker, form_type=case.form_type)
    sections = [c["metadata"].get("section", "") for c in chunks]
    hits     = sections.count(case.expected_section)
    precision = round(hits / len(chunks), 3) if chunks else 0.0

    return EvalResult(
        category = "retrieval_precision",
        case     = f"{case.ticker}: {case.query[:55]}",
        passed   = hits > 0,
        score    = precision,
        details  = {"expected": case.expected_section,
                    "retrieved": sections, "hits": hits},
    )


# ── 2. Answer Faithfulness (LLM-as-judge, free) ───────────────────────────

JUDGE_SYSTEM = """You are evaluating RAG answer faithfulness.
Given a question, source excerpts, and an answer, score how well the answer
is grounded in ONLY the provided sources (not outside knowledge).
Return JSON: {"score": <0.0-1.0>, "faithful": <true/false>, "issues": ["..."]}"""

def eval_faithfulness(case: RAGCase, collection) -> EvalResult:
    result  = rag_query(case.query, collection,
                        ticker=case.ticker, form_type=case.form_type)
    sources = "\n---\n".join(c["text"][:400] for c in result["sources"][:4])

    prompt = f"""QUESTION: {case.query}

SOURCES:
{sources}

ANSWER:
{result['answer']}

Rate faithfulness of the answer to the provided sources only."""

    parsed = chat_json(prompt, system=JUDGE_SYSTEM)
    faith_score  = float(parsed.get("score", 0.5))

    # Keyword check
    lower   = result["answer"].lower()
    kw_hits = [kw for kw in case.expected_keywords if kw.lower() in lower]
    kw_score = len(kw_hits) / len(case.expected_keywords) if case.expected_keywords else 1.0

    final = round((faith_score + kw_score) / 2, 3)
    return EvalResult(
        category = "faithfulness",
        case     = f"{case.ticker}: {case.query[:55]}",
        passed   = final >= 0.6,
        score    = final,
        details  = {"faithfulness": faith_score, "keyword_score": kw_score,
                    "kw_hits": kw_hits,
                    "issues": parsed.get("issues", []),
                    "answer": result["answer"][:250]},
    )


# ── 3. Drift Consistency ──────────────────────────────────────────────────

def eval_drift_consistency(case: DriftCase, runs: int = 3) -> EvalResult:
    """Run semantic drift N times, check score variance + direction stability."""
    scores, directions, concept_hits = [], [], []

    for _ in range(runs):
        try:
            r      = semantic_drift(case.ticker, case.section, case.year_a, case.year_b)
            scores.append(r.get("overall_drift_score", 50))
            dirs   = [c["direction"] for c in r.get("key_theme_changes", [])]
            directions.append(max(set(dirs), key=dirs.count) if dirs else "stable")
            new_c  = [c.lower() for c in r.get("new_concepts", [])]
            concept_hits.append(
                any(any(exp.lower() in nc for nc in new_c)
                    for exp in case.expected_concepts)
            )
        except Exception as e:
            print(f"    [warn] run failed: {e}")

    if not scores:
        return EvalResult("drift_consistency",
                          f"{case.ticker} {case.section}", False, 0.0,
                          {"error": "all runs failed"})

    var       = round(statistics.variance(scores) if len(scores) > 1 else 0.0, 2)
    mean      = round(statistics.mean(scores), 1)
    dom_dir   = max(set(directions), key=directions.count)
    dir_agree = directions.count(dom_dir) / len(directions)
    concept_r = sum(concept_hits) / len(concept_hits)

    consistency = round(
        (1 - min(var / 500, 1)) * 0.5 + dir_agree * 0.3 + concept_r * 0.2, 3
    )
    return EvalResult(
        category = "drift_consistency",
        case     = f"{case.ticker} {case.section} {case.year_a}→{case.year_b}",
        passed   = var < 200 and dir_agree >= 0.67,
        score    = consistency,
        details  = {"scores": scores, "mean": mean, "variance": var,
                    "direction_agreement": dir_agree, "concept_hit_rate": concept_r},
    )


# ── Default test cases ────────────────────────────────────────────────────

DEFAULT_RAG_CASES = [
    RAGCase("What are Apple's main AI-related risks?",
            "AAPL", "risk factors", ["artificial intelligence", "competition"]),
    RAGCase("How does Microsoft describe its cloud competition?",
            "MSFT", "competition", ["cloud", "Azure", "Amazon"]),
    RAGCase("What does Meta say about regulatory risk?",
            "META", "risk factors", ["regulatory", "privacy"]),
]

DEFAULT_DRIFT_CASES = [
    DriftCase("META", "risk factors", "2022", "2024",
              "increase", ["AI", "artificial intelligence"]),
    DriftCase("MSFT", "AI technology", "2021", "2024",
              "increase", ["OpenAI", "Copilot", "generative"]),
]


# ── Run suite ─────────────────────────────────────────────────────────────

def run_eval_suite(
    collection,
    rag_cases:   list[RAGCase]   = None,
    drift_cases: list[DriftCase] = None,
    drift_runs:  int             = 1,
) -> dict:
    rag_cases   = rag_cases   or DEFAULT_RAG_CASES
    drift_cases = drift_cases or DEFAULT_DRIFT_CASES
    results: list[EvalResult] = []

    print("\n─── Retrieval Precision ──────────────────────────────")
    for c in rag_cases:
        r = eval_retrieval(c, collection)
        results.append(r)
        print(f"  {'✓' if r.passed else '✗'}  {r.case}  precision={r.score}")

    print("\n─── Answer Faithfulness ──────────────────────────────")
    for c in rag_cases:
        r = eval_faithfulness(c, collection)
        results.append(r)
        print(f"  {'✓' if r.passed else '✗'}  {r.case}  score={r.score}")

    print("\n─── Drift Consistency ────────────────────────────────")
    for c in drift_cases:
        r = eval_drift_consistency(c, runs=drift_runs)
        results.append(r)
        print(f"  {'✓' if r.passed else '✗'}  {r.case}  score={r.score}  var={r.details.get('variance')}")

    # Aggregate
    by_cat = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    summary = {}
    for cat, rs in by_cat.items():
        passed = sum(1 for r in rs if r.passed)
        summary[cat] = {
            "passed": passed, "total": len(rs),
            "pass_rate": round(passed / len(rs), 2),
            "avg_score": round(statistics.mean(r.score for r in rs), 3),
        }

    total_passed = sum(1 for r in results if r.passed)
    overall      = round(total_passed / len(results), 2) if results else 0.0

    print(f"\n─── Summary ──────────────────────────────────────────")
    print(f"  Overall: {total_passed}/{len(results)}  ({overall*100:.0f}%)")
    for cat, s in summary.items():
        print(f"  {cat}: {s['passed']}/{s['total']}  avg={s['avg_score']}")

    return {"overall_pass_rate": overall, "total": len(results),
            "by_category": summary, "results": [asdict(r) for r in results]}