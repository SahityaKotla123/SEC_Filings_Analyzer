"""
Splits raw SEC filing text into overlapping chunks with rich metadata.
Each chunk carries: ticker, form_type, filing_date, period, accession_no,
                    detected section, chunk index.
"""

import re
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_WORDS

SECTION_PATTERNS = {
    "risk factors":          re.compile(r"risk\s+factor|risks|uncertaint", re.I),
    "management discussion": re.compile(r"management.{0,10}discussion|md&a|results of operations", re.I),
    "business overview":     re.compile(r"item\s*1[^a-z]|our\s+business|overview", re.I),
    "competition":           re.compile(r"competit|market\s+position|industry", re.I),
    "forward looking":       re.compile(r"forward[- ]looking|cautionary", re.I),
    "financials":            re.compile(r"consolidated\s+statements?|balance\s+sheet|revenue", re.I),
    "legal proceedings":     re.compile(r"legal\s+proceedings|litigation", re.I),
    "AI technology":         re.compile(r"artificial\s+intel|machine\s+learn|generative|AI\b|cloud", re.I),
    "regulation":            re.compile(r"regulat|compliance|government|privacy|data\s+protection", re.I),
    "macroeconomic":         re.compile(r"macroeconom|inflation|interest\s+rate|recession|economy", re.I),
}

def detect_section(text_window: str) -> str:
    for label, pat in SECTION_PATTERNS.items():
        if pat.search(text_window):
            return label
    return "general"

def chunk_text(text: str) -> list[str]:
    words  = text.split()
    chunks, start = [], 0
    while start < len(words):
        end   = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def build_chunks(filing: dict) -> list[dict]:
    """Convert a filing dict into a list of indexed, metadata-rich chunk dicts."""
    raw    = chunk_text(filing["text"])
    total  = len(raw)
    result = []

    for idx, text in enumerate(raw):
        section = detect_section(text[:500])
        result.append({
            "text":             text,
            "ticker":           filing.get("ticker", ""),
            "form_type":        filing.get("form_type", ""),
            "filing_date":      filing.get("filing_date", ""),
            "period_of_report": filing.get("period_of_report", ""),
            "accession_no":     filing.get("accession_no", ""),
            "section":          section,
            "chunk_index":      idx,
            "total_chunks":     total,
            "chunk_id":         f"{filing.get('ticker','')}-{filing.get('accession_no','')}-{idx}",
        })
    return result