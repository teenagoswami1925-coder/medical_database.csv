# Clinical AI Assistant

A **Retrieval-Augmented Generation (RAG)** medical chatbot powered by Google Gemini AI. Users can describe symptoms or ask medical questions and receive AI-generated responses grounded in a verified clinical knowledge base.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Frontend    │────▶│   FastAPI    │────▶│   Gemini 2.5     │
│  (HTML/UI)   │◀────│   Backend    │◀────│   Flash + Embed  │
└──────────────┘     └──────┬───────┘     └──────────────────┘
                            │
                     ┌──────▼───────┐
                     │   ChromaDB   │
                     │  Vector Store│
                     │ (local disk) │
                     └──────────────┘
```

**Two interfaces available:**
- **FastAPI + HTML** (`main.py` + `index.html`) — SSE streaming, chat history, modern UI
- **Streamlit** (`APP.py`) — simpler single-page interface

**Data ingestion pipeline:**
- `ingest_data.py` — loads `.txt` files from `./medical_knowledge_base/`, chunks, embeds, and indexes into ChromaDB
- `crawl_and_train.py` — scrapes medical web pages with BeautifulSoup, chunks, embeds, and indexes
- `medical_dataset.csv` — 8 pre-loaded clinical Q&A pairs (diabetes, burns, asthma, hypertension, etc.)

## How It Works

1. User sends a medical query
2. Query is embedded via `gemini-embedding-2`
3. ChromaDB performs cosine similarity search (top-3 results, threshold 0.75)
4. If match found → context is injected as **verified reference** → Gemini answers strictly from it
5. If no match → Gemini returns: *"No verified clinical data available. Please consult a doctor."*
6. Response is streamed back via Server-Sent Events

## Setup

### Prerequisites

- Python 3.10+
- Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/teenagoswami1925-coder/medical_database.csv.git
cd medical_database.csv

# Install dependencies
pip install -r requirements.txt

# Set your API key
export GEMINI_API_KEY="your-api-key-here"
```

### Run the FastAPI + HTML version

```bash
# Start the backend
uvicorn main:app --reload --port 8000
```

Open `index.html` in your browser (or serve it with `python -m http.server 8001`).

### Run the Streamlit version

```bash
streamlit run APP.py
```

### Ingest medical knowledge

```bash
# From web pages
python crawl_and_train.py
# Paste a medical URL when prompted

# From text files
# Place .txt files in ./medical_knowledge_base/
python ingest_data.py
```

## Project Structure

```
├── main.py                 # FastAPI backend with SSE streaming
├── APP.py                  # Streamlit frontend
├── index.html              # HTML/Tailwind CSS frontend
├── crawl_and_train.py      # Web scraper for training ChromaDB
├── ingest_data.py          # Local file ingestion pipeline
├── medical_dataset.csv     # Pre-loaded clinical Q&A dataset
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Key Features

- **RAG-based grounding** — answers are tied to verified medical data, reducing hallucination
- **SSE streaming** — real-time token-by-token response rendering
- **Dual interfaces** — FastAPI/HTML and Streamlit
- **Web scraping** — train on any medical webpage
- **Multi-session chat** — persistent chat history in localStorage
- **Verification badges** — UI shows when answer is backed by stored clinical data

## License

MIT
