import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from indexer import get_client, get_collection, index_chunks, collection_stats
from edgar_fetcher import ingest_filings
from chunker import build_chunks
from rag_core import RAGConversation, rag_query
from drift_detector import full_drift_report
from evaluations import run_eval_suite

st.set_page_config(
    page_title="SEC Filings Analyzer",
    page_icon="SEC",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
    }
    .source-box {
        background: #f0f7ff;
        border-left: 3px solid #1a73e8;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 13px;
    }
    .drift-box {
        background: #f0fdf4;
        border-left: 3px solid #16a34a;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #fffbeb;
        border-left: 3px solid #f59e0b;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "collection" not in st.session_state:
    client = get_client()
    st.session_state.collection = get_collection(client)
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

collection = st.session_state.collection

with st.sidebar:
    st.title("SEC Filings Analyzer")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Search and Ingest", "Chat", "Drift Detector", "Evaluations"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Index Stats**")
    try:
        stats = collection_stats(collection)
        st.metric("Total Chunks", stats["total_chunks"])
        if stats["tickers"]:
            st.markdown(f"**Tickers:** {', '.join(stats['tickers'])}")
        if stats["form_types"]:
            st.markdown(f"**Forms:** {', '.join(stats['form_types'])}")
    except Exception:
        st.warning("No data indexed yet")

    st.markdown("---")
    st.markdown("Built with SEC EDGAR, ChromaDB and LLaMA")


# ── Search and Ingest ─────────────────────────────────────────────────────

if page == "Search and Ingest":
    st.title("Search and Ingest SEC Filings")
    st.markdown("Fetch real filings directly from SEC EDGAR and index them for AI analysis.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ticker = st.text_input("Ticker", placeholder="AAPL", value="AAPL").upper()
    with col2:
        form_type = st.selectbox("Form Type", ["10-K", "10-Q", "8-K", "DEF 14A"])
    with col3:
        limit = st.number_input("Number of Filings", min_value=1, max_value=10, value=3)
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        ingest_btn = st.button("Fetch and Index", type="primary")

    if ingest_btn:
        if not ticker:
            st.error("Enter a ticker symbol")
        else:
            with st.spinner(f"Fetching {form_type} filings for {ticker} from SEC EDGAR..."):
                try:
                    filings    = ingest_filings(ticker, form_type, limit)
                    all_chunks = []
                    filing_info = []

                    for f in filings:
                        chunks = build_chunks(f)
                        all_chunks.extend(chunks)
                        filing_info.append({
                            "date":   f["filing_date"],
                            "period": f["period_of_report"],
                            "acc":    f["accession_no"],
                            "chunks": len(chunks)
                        })

                    n = index_chunks(all_chunks, collection)
                    st.success(f"Indexed {n} chunks from {len(filings)} filings successfully.")

                    st.markdown("### Filings Fetched")
                    for info in filing_info:
                        with st.expander(f"{ticker} {form_type} - Filed: {info['date']}"):
                            col_a, col_b = st.columns(2)
                            col_a.metric("Period", info["period"])
                            col_b.metric("Chunks", info["chunks"])
                            st.caption(f"Accession: {info['acc']}")

                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.markdown("### Current Index")
    try:
        stats = collection_stats(collection)
        if stats["total_chunks"] == 0:
            st.info("No filings indexed yet. Fetch some above.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Chunks", stats["total_chunks"])
            col2.metric("Tickers", len(stats["tickers"]))
            col3.metric("Form Types", len(stats["form_types"]))

            st.markdown("**Sections detected:**")
            cols = st.columns(5)
            for i, section in enumerate(stats.get("sections", [])):
                cols[i % 5].markdown(f"- {section}")
    except Exception as e:
        st.error(str(e))


# ── Chat ──────────────────────────────────────────────────────────────────

elif page == "Chat":
    st.title("Chat with SEC Filings")
    st.markdown("Ask anything about indexed filings. The AI retrieves relevant sections and answers with source citations.")

    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            stats   = collection_stats(collection)
            tickers = stats["tickers"] or ["AAPL"]
        except Exception:
            tickers = ["AAPL"]
        chat_ticker = st.selectbox("Ticker", tickers)
    with col2:
        chat_form = st.selectbox("Form Type", ["10-K", "10-Q", "8-K"])
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("New Conversation"):
            st.session_state.chat_history = []
            st.session_state.conversation = RAGConversation(
                collection, ticker=chat_ticker, form_type=chat_form
            )
            st.rerun()

    if st.session_state.conversation is None:
        st.session_state.conversation = RAGConversation(
            collection, ticker=chat_ticker, form_type=chat_form
        )

    st.markdown("**Quick questions:**")
    suggestions = [
        "What are the main risk factors?",
        "How is AI mentioned in filings?",
        "What does management say about competition?",
        "What are the key revenue drivers?",
        "How has the business changed recently?",
    ]
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"chip_{i}"):
            st.session_state._pending_question = suggestion

    st.markdown("---")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View Sources"):
                    for s in msg["sources"]:
                        m = s["metadata"]
                        st.markdown(f"""<div class="source-box">
                            <strong>[{m.get('ticker')}] {m.get('form_type')} - {m.get('filing_date')}</strong><br>
                            Section: {m.get('section')} | Relevance: {s['score']:.2f}<br>
                            <small>{s['text'][:200]}...</small>
                        </div>""", unsafe_allow_html=True)

    question = st.chat_input("Ask about the filings...")

    if hasattr(st.session_state, "_pending_question"):
        question = st.session_state._pending_question
        del st.session_state._pending_question

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching filings..."):
                try:
                    result = st.session_state.conversation.ask(question)
                    st.markdown(result["answer"])

                    if result["sources"]:
                        with st.expander("View Sources"):
                            for s in result["sources"]:
                                m = s["metadata"]
                                st.markdown(f"""<div class="source-box">
                                    <strong>[{m.get('ticker')}] {m.get('form_type')} - {m.get('filing_date')}</strong><br>
                                    Section: {m.get('section')} | Relevance: {s['score']:.2f}<br>
                                    <small>{s['text'][:200]}...</small>
                                </div>""", unsafe_allow_html=True)

                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ── Drift Detector ────────────────────────────────────────────────────────

elif page == "Drift Detector":
    st.title("Language Drift Detector")
    st.markdown("Detect how a company language shifts between filing years — a strong signal for strategy or risk changes.")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        try:
            stats   = collection_stats(collection)
            tickers = stats["tickers"] or ["AAPL"]
        except Exception:
            tickers = ["AAPL"]
        drift_ticker = st.selectbox("Ticker", tickers, key="drift_ticker")
    with col2:
        drift_section = st.selectbox("Section", [
            "risk factors", "competition", "AI technology",
            "management discussion", "regulation", "macroeconomic",
            "financials", "general"
        ])
    with col3:
        year_a = st.selectbox("From Year", ["2021", "2022", "2023", "2024"], index=1)
    with col4:
        year_b = st.selectbox("To Year", ["2022", "2023", "2024", "2025"], index=2)
    with col5:
        st.markdown("<br>", unsafe_allow_html=True)
        drift_btn = st.button("Detect Drift", type="primary")

    if drift_btn:
        if year_a == year_b:
            st.error("Select two different years")
        else:
            with st.spinner(f"Analyzing language drift for {drift_ticker} {year_a} to {year_b}..."):
                try:
                    report   = full_drift_report(collection, drift_ticker, drift_section, year_a, year_b)
                    stats_r  = report["statistical"]
                    semantic = report.get("semantic", {})

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Blended Drift Score", f"{report['blended_drift_score']}/100")
                    col2.metric("Cosine Similarity",   f"{stats_r['cosine_similarity']}")
                    col3.metric("JS Divergence",       f"{stats_r['js_divergence']}")
                    col4.metric("Emerged Terms",        len(stats_r['emerged_terms']))

                    st.markdown("---")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("### Dropped Terms")
                        if stats_r["dropped_terms"]:
                            for t in stats_r["dropped_terms"]:
                                st.markdown(f"- `{t}`")
                        else:
                            st.info("No significant dropped terms")

                    with col_b:
                        st.markdown("### Emerged Terms")
                        if stats_r["emerged_terms"]:
                            for t in stats_r["emerged_terms"]:
                                st.markdown(f"- `{t}`")
                        else:
                            st.info("No significant emerged terms")

                    if semantic and "tone_shift" in semantic:
                        st.markdown("---")
                        st.markdown("### AI Semantic Analysis")

                        st.markdown(f"""<div class="drift-box">
                            <strong>Tone Shift:</strong> {semantic.get('tone_shift', 'N/A')}
                        </div>""", unsafe_allow_html=True)

                        if semantic.get("key_theme_changes"):
                            st.markdown("**Key Theme Changes:**")
                            for theme in semantic["key_theme_changes"]:
                                direction = "up" if theme["direction"] == "increase" else "down" if theme["direction"] == "decrease" else "stable"
                                col_x, col_y = st.columns([3, 1])
                                col_x.markdown(f"**{theme['theme']}** ({direction}) - {theme.get('interpretation', '')}")
                                col_y.progress(theme["year_b_prominence"] / 100)

                        col_p, col_q = st.columns(2)
                        with col_p:
                            st.markdown(f"**New Concepts in {year_b}:**")
                            for c in semantic.get("new_concepts", []):
                                st.markdown(f"- {c}")
                        with col_q:
                            st.markdown(f"**Dropped Concepts from {year_a}:**")
                            for c in semantic.get("dropped_concepts", []):
                                st.markdown(f"- {c}")

                        if semantic.get("analyst_signal"):
                            st.markdown(f"""<div class="warning-box">
                                <strong>Analyst Signal:</strong><br>
                                {semantic['analyst_signal']}
                            </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ── Evaluations ───────────────────────────────────────────────────────────

elif page == "Evaluations":
    st.title("Evaluation Suite")
    st.markdown("Measure RAG retrieval precision, answer faithfulness and drift consistency.")

    col1, col2 = st.columns([3, 1])
    with col2:
        drift_runs = st.number_input("Drift Runs", min_value=1, max_value=3, value=1)
        run_btn    = st.button("Run Evaluations", type="primary")

    if run_btn:
        with st.spinner("Running evaluation suite, this may take a few minutes..."):
            try:
                results = run_eval_suite(collection, drift_runs=drift_runs)
                st.session_state.eval_results = results
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.session_state.eval_results:
        results = st.session_state.eval_results

        st.markdown("### Overall Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pass Rate",    f"{results['overall_pass_rate']*100:.0f}%")
        col2.metric("Total Cases",  results["total"])
        col3.metric("Cases Passed", int(results["overall_pass_rate"] * results["total"]))

        st.markdown("---")
        st.markdown("### Results by Category")

        for cat, summary in results["by_category"].items():
            status = "Pass" if summary["pass_rate"] >= 0.6 else "Partial" if summary["pass_rate"] >= 0.3 else "Fail"
            with st.expander(f"{cat.replace('_', ' ').title()} - {summary['passed']}/{summary['total']} passed ({status})"):
                col_a, col_b = st.columns(2)
                col_a.metric("Pass Rate", f"{summary['pass_rate']*100:.0f}%")
                col_b.metric("Avg Score", f"{summary['avg_score']:.3f}")

                st.markdown("**Individual cases:**")
                for r in results["results"]:
                    if r["category"] == cat:
                        status_icon = "Pass" if r["passed"] else "Fail"
                        st.markdown(f"- [{status_icon}] {r['case']} - score: {r['score']}")