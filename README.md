# SEC_Filings_Analyzer

It is a tool that lets you search, read, and analyze SEC filings using AI. You can ask questions about any public company's 10-K or 10-Q filings, detect how their language has changed over the years, and evaluate how well the system is performing.

## What it does

- Pulls real filings directly from SEC EDGAR (free, no API key needed)
- Lets you chat with the filings and get answers with source citations
- Detects language drift between filing years to spot strategy or risk changes
- Runs evaluations to measure retrieval and answer quality

## How it works

When you search for a company, it fetches the actual filing documents from SEC EDGAR and breaks them into chunks. These chunks are embedded and stored in ChromaDB. When you ask a question, it finds the most relevant chunks and sends them to an LLM to generate a grounded answer.

For drift detection, it compares the same section across two different years using TF-IDF similarity and Jensen-Shannon divergence, then uses an LLM to summarize what changed and what it might mean.

## Tech stack

- Python
- ChromaDB for vector storage
- sentence-transformers for embeddings (all-MiniLM-L6-v2)
- Groq API with LLaMA 3.3 70B for the LLM
- Ollama for local inference fallback
- Streamlit for the web interface
- SEC EDGAR API for filing data

## Setup

Clone the repo and install dependencies:

pip install -r requirements.txt

Create a .env file in the root folder:

LLM_BACKEND=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
USER_AGENT=SECAnalyzer your@email.com

A free Groq API key is available at console.groq.com. If you prefer to run everything locally without any API keys, set LLM_BACKEND=ollama and install Ollama with the llama3 model (but it will be very slow).

## Running it

To use the command line:

python main.py ingest --ticker AAPL --form 10-K --limit 3
python main.py query --ticker AAPL --q "What are Apple's main risks?"
python main.py chat --ticker AAPL
python main.py drift --ticker META --section "risk factors" --year-a 2022 --year-b 2024
python main.py eval
python main.py stats

To run the web app:

streamlit run app.py

## Project structure

config.py          settings and environment variables
edgar_fetcher.py   fetches filings from SEC EDGAR
chunker.py         splits documents into chunks with metadata
indexer.py         stores and retrieves chunks from ChromaDB
llm_client.py      handles communication with Groq or Ollama
rag_core.py        RAG pipeline and multi-turn chat
drift_detector.py  language drift analysis
evaluations.py     evaluation suite for retrieval and answer quality
main.py            command line interface
app.py             Streamlit web app

## Notes

- SEC EDGAR has a rate limit of 10 requests per second so the fetcher includes a small delay between requests
- Groq free tier has a daily token limit of 100k tokens which is enough for normal usage but may run out during large eval runs
- The data folder is not included in the repo since filing documents can get large, just run the ingest command to rebuild it
