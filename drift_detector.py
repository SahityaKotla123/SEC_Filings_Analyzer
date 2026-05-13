"""
Language Drift Detector — two complementary approaches:

A) Statistical  — TF-IDF cosine similarity + Jensen-Shannon divergence
                   (fully free, no LLM needed)
B) LLM semantic — Llama3 via Ollama (free) or Groq (free tier)
                   qualitative drift analysis for a named section
"""

import json
import math
import re
import statistics
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from indexer import query_index
from llm_client import chat_json
from config import DRIFT_SECTIONS


# ══════════════════════════════════════════════════════
# A) STATISTICAL DRIFT
# ══════════════════════════════════════════════════════

def get_section_chunks(collection, ticker: str, section: str, year: str) -> str:
    """Retrieve & concatenate all chunks for ticker+section in a given year."""
    results = query_index(query=section, collection=collection,
                          top_k=25, ticker=ticker, section=section)
    filtered = [r for r in results
                if r["metadata"].get("period_of_report", "").startswith(year)
                or r["metadata"].get("filing_date", "").startswith(year)]
    # Fallback: use all if no year match (happens when only one year is indexed)
    if not filtered:
        filtered = results
    return " ".join(r["text"] for r in filtered)


def tfidf_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        mat = vec.fit_transform([a, b])
        return float(cosine_similarity(mat[0], mat[1])[0][0])
    except Exception:
        return 0.0


def top_tfidf_terms(text: str, n: int = 25) -> set[str]:
    if not text.strip():
        return set()
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        mat    = vec.fit_transform([text])
        scores = zip(vec.get_feature_names_out(), mat.toarray()[0])
        return {t for t, s in sorted(scores, key=lambda x: -x[1])[:n]}
    except Exception:
        return set()


def js_divergence(a: str, b: str) -> float:
    """Jensen-Shannon divergence between unigram distributions. 0=same, 1=opposite."""
    def dist(text):
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        c = Counter(words)
        total = sum(c.values()) or 1
        return {w: v / total for w, v in c.items()}

    p, q   = dist(a), dist(b)
    vocab  = set(p) | set(q)
    eps    = 1e-10

    def kl(x, y):
        return sum(x.get(w, eps) * math.log(x.get(w, eps) / y.get(w, eps))
                   for w in vocab if x.get(w, 0) > 0)

    m = {w: (p.get(w, 0) + q.get(w, 0)) / 2 for w in vocab}
    return max(0.0, min(1.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def statistical_drift(collection, ticker: str, section: str,
                      year_a: str, year_b: str) -> dict:
    text_a = get_section_chunks(collection, ticker, section, year_a)
    text_b = get_section_chunks(collection, ticker, section, year_b)

    sim    = tfidf_similarity(text_a, text_b)
    jsd    = js_divergence(text_a, text_b)
    score  = round((1 - sim) * 100, 1)

    terms_a = top_tfidf_terms(text_a)
    terms_b = top_tfidf_terms(text_b)

    return {
        "ticker":            ticker.upper(),
        "section":           section,
        "year_a":            year_a,
        "year_b":            year_b,
        "drift_score":       score,           # 0 = identical, 100 = completely different
        "cosine_similarity": round(sim, 4),
        "js_divergence":     round(jsd, 4),
        "emerged_terms":     sorted(terms_b - terms_a)[:12],
        "dropped_terms":     sorted(terms_a - terms_b)[:12],
        "retained_terms":    sorted(terms_a & terms_b)[:10],
        "words_year_a":      len(text_a.split()),
        "words_year_b":      len(text_b.split()),
    }


# ══════════════════════════════════════════════════════
# B) LLM SEMANTIC DRIFT (free Llama3 / Groq)
# ══════════════════════════════════════════════════════

DRIFT_SYSTEM = """You are a financial NLP analyst specialising in SEC filing language.
Identify meaningful shifts in how companies describe risks, strategy, and business.
Return ONLY valid JSON. No markdown, no explanation."""

DRIFT_PROMPT = """
Analyse language drift in {ticker}'s 10-K filings — "{section}" section — comparing {year_a} vs {year_b}.

Return exactly this JSON (fill all fields, integers for scores):
{{
  "company_name": "Full company name",
  "overall_drift_score": <0-100>,
  "tone_shift": "<one sentence on overall change in tone/emphasis>",
  "key_theme_changes": [
    {{"theme": "<5-word label>", "year_a_prominence": <0-100>, "year_b_prominence": <0-100>,
      "direction": "increase|decrease|stable", "interpretation": "<one sentence>"}},
    {{"theme": "<5-word label>", "year_a_prominence": <0-100>, "year_b_prominence": <0-100>,
      "direction": "increase|decrease|stable", "interpretation": "<one sentence>"}},
    {{"theme": "<5-word label>", "year_a_prominence": <0-100>, "year_b_prominence": <0-100>,
      "direction": "increase|decrease|stable", "interpretation": "<one sentence>"}},
    {{"theme": "<5-word label>", "year_a_prominence": <0-100>, "year_b_prominence": <0-100>,
      "direction": "increase|decrease|stable", "interpretation": "<one sentence>"}}
  ],
  "new_concepts":     ["concept1", "concept2", "concept3", "concept4"],
  "dropped_concepts": ["concept1", "concept2", "concept3", "concept4"],
  "language_hardened": <true|false>,
  "representative_language": {{
    "year_a": "<1-2 sentence paraphrase of typical {year_a} language>",
    "year_b": "<1-2 sentence paraphrase of typical {year_b} language>"
  }},
  "analyst_signal": "<2-3 sentence takeaway: what does this drift signal about strategy or risk?>"
}}
"""

def semantic_drift(ticker: str, section: str, year_a: str, year_b: str) -> dict:
    prompt = DRIFT_PROMPT.format(
        ticker=ticker.upper(), section=section, year_a=year_a, year_b=year_b
    )
    return chat_json(prompt, system=DRIFT_SYSTEM)


# ══════════════════════════════════════════════════════
# C) COMBINED REPORT
# ══════════════════════════════════════════════════════

def full_drift_report(collection, ticker: str, section: str,
                      year_a: str, year_b: str) -> dict:
    """Run statistical + semantic drift and merge into one report."""
    print(f"[drift] Statistical: {ticker} | {section} | {year_a}→{year_b}")
    stats = statistical_drift(collection, ticker, section, year_a, year_b)

    print("[drift] Semantic: calling LLM…")
    try:
        semantic = semantic_drift(ticker, section, year_a, year_b)
    except Exception as e:
        print(f"  [warn] Semantic failed: {e}")
        semantic = {}

    llm_score = semantic.get("overall_drift_score", stats["drift_score"])
    blended   = round((stats["drift_score"] + llm_score) / 2, 1)

    return {
        "statistical":      stats,
        "semantic":         semantic,
        "blended_drift_score": blended,
    }