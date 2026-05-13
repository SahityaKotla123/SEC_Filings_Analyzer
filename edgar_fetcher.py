"""
Fetches SEC filings from EDGAR (completely free, no API key needed).

Flow:
  1. Resolve ticker → CIK via EDGAR company_tickers.json
  2. Get filing submissions for that CIK
  3. Download primary HTML/text document for each filing
  4. Save raw text to data/raw/{TICKER}/{FORM}/{accession}.txt
"""

import re
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from config import EDGAR_BASE_URL, USER_AGENT, DATA_DIR

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}


# ── 1. Ticker → CIK ──────────────────────────────────────────────────────

def get_cik(ticker: str) -> str:
    """Return zero-padded 10-digit CIK string for a ticker symbol."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r   = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    for entry in r.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in EDGAR. Check the symbol.")


# ── 2. List filings ───────────────────────────────────────────────────────

def get_filings_metadata(cik: str, form_type: str = "10-K", limit: int = 5) -> list[dict]:
    url = f"{EDGAR_BASE_URL}/submissions/CIK{cik}.json"
    r   = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    data   = r.json()
    recent = data.get("filings", {}).get("recent", {})

    forms   = recent.get("form", [])
    acc_nos = recent.get("accessionNumber", [])
    dates   = recent.get("filingDate", [])
    periods = recent.get("reportDate", [])

    results = []
    cik_int = int(cik)

    for i, form in enumerate(forms):
        if form == form_type:
            acc_clean = acc_nos[i].replace("-", "")

            # Fetch the proper filing index JSON
            index_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_int}&type={form_type}&dateb=&owner=include&count=10&search_text=&output=atom"
            
            # Use the correct index URL format
            idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{acc_nos[i]}-index.htm"
            try:
                idx_r = requests.get(idx_url, headers=HEADERS, timeout=60)
                idx_r.raise_for_status()

                soup = BeautifulSoup(idx_r.content, "lxml")
                doc_url = None

                # Find the main 10-K document in the index table
                for row in soup.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) >= 3:
                        doc_type = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                        doc_name = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                        link = cells[2].find("a", href=True) if len(cells) > 2 else None

                        if link and form_type in doc_type:
                            href = link["href"]
                            doc_url = f"https://www.sec.gov{href}" if href.startswith("/") else href
                            break

                # Fallback: grab first .htm that isn't the index itself
                if not doc_url:
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        name = href.split("/")[-1].lower()
                        if (name.endswith((".htm", ".html")) and
                            "index" not in name and
                            "search" not in name and
                            len(name) > 8):
                            doc_url = f"https://www.sec.gov{href}" if href.startswith("/") else href
                            break

                if doc_url:
                    print(f"  Found: {doc_url}")
                    results.append({
                        "accession_no":     acc_nos[i],
                        "filing_date":      dates[i],
                        "period_of_report": periods[i] if i < len(periods) else "",
                        "primary_doc_url":  doc_url,
                    })
                else:
                    print(f"  [warn] No document found for {acc_nos[i]}")

            except Exception as e:
                print(f"  [warn] {acc_nos[i]}: {e}")

            if len(results) >= limit:
                break

    return results

# ── 3. Download & extract plain text ─────────────────────────────────────

def fetch_filing_text(url: str) -> str:
    # Strip EDGAR inline viewer wrapper
    if "ix?doc=" in url:
        url = "https://www.sec.gov" + url.split("ix?doc=")[1]
    
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()

    if "html" in r.headers.get("Content-Type", "") or url.lower().endswith((".htm", ".html")):
        import warnings
        from bs4 import XMLParsedAsHTMLWarning
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(r.content, "lxml")
        for tag in soup(["script", "style", "ix:header", "xbrl"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = r.text

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text  = "\n".join(lines)
    text  = re.sub(r"\n{3,}", "\n\n", text)
    return text

# ── 4. Save raw text ──────────────────────────────────────────────────────

def save_filing(ticker: str, form_type: str, accession_no: str, text: str) -> Path:
    out_dir  = DATA_DIR / ticker.upper() / form_type
    out_dir.mkdir(parents=True, exist_ok=True)
    path     = out_dir / f"{accession_no}.txt"
    path.write_text(text, encoding="utf-8")
    return path


# ── 5. Full ingest pipeline ───────────────────────────────────────────────

def ingest_filings(ticker: str, form_type: str = "10-K", limit: int = 3) -> list[dict]:
    """
    Resolve ticker → fetch filings → save text → return enriched metadata list.
    Each returned dict has keys: ticker, form_type, accession_no, filing_date,
                                  period_of_report, local_path, text
    """
    print(f"[edgar] Resolving CIK for {ticker.upper()}…")
    cik     = get_cik(ticker)
    print(f"[edgar] CIK = {cik}")

    metas   = get_filings_metadata(cik, form_type, limit)
    if not metas:
        raise RuntimeError(f"No {form_type} filings found for {ticker.upper()}")

    print(f"[edgar] Found {len(metas)} {form_type} filing(s)")
    ingested = []

    for meta in metas:
        print(f"  → {meta['filing_date']}  {meta['accession_no']}")
        try:
            text = fetch_filing_text(meta["primary_doc_url"])
            path = save_filing(ticker, form_type, meta["accession_no"], text)
            ingested.append({**meta, "ticker": ticker.upper(),
                             "form_type": form_type,
                             "local_path": str(path), "text": text})
            time.sleep(0.15)   # respect EDGAR's 10 req/s limit
        except Exception as e:
            print(f"    [warn] {e}")

    return ingested