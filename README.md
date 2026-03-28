# TestGen RAG — Automated Test Case Generation using RAG

A Retrieval-Augmented Generation (RAG) system that automatically generates test cases from three types of inputs:
- 🐍 **Source Code** (`.py`, `.java`) → Unit tests (PyTest / JUnit style)
- 📄 **Requirements Documents** (`.pdf`, `.docx`) → BDD acceptance tests (Gherkin)
- 🔌 **API Specifications** (`.yaml`, `.json`) → Integration / endpoint tests

---

## Architecture

```
User Input (Code / Doc / API Spec)
         ↓
  Input Type Detection
         ↓
  Parser (AST / PDF / YAML)
         ↓
  Chunking + Embedding (all-MiniLM-L6-v2)
         ↓
  FAISS Vector Store
         ↓
  RAG Retriever (top-k similar chunks)
         ↓
  Mode-Aware Prompt Engine
         ↓
  LLM (Groq Llama 3 / Gemini fallback)
         ↓
  Structured Test Case Output
```

---

## Setup

### 1. Clone the repo

```bash
git clone [https://github.com/your-team/test-gen-rag.git](https://github.com/dhanyabhat16/Test_case_generator.git)
cd Test_case_generator
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API keys

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY and GEMINI_API_KEY
```

Get free API keys:
- **Groq**: https://console.groq.com (free tier, fast Llama 3)
- **Gemini**: https://aistudio.google.com/app/apikey (free tier)

---

## Running the App

```bash
streamlit run ui/app.py
```

Then open http://localhost:8501 in your browser.

---

## Running Tests

```bash
pytest tests/ -v
```

To test with coverage:
```bash
pytest tests/ -v --cov=ingestion --cov-report=term-missing
```

---

## CLI Usage (Ingestion)

You can also ingest files directly from the command line:

```bash
# Ingest a Python file
python -m ingestion.embedder ingest data/sample_code.py

# Ingest an API spec
python -m ingestion.embedder ingest data/sample_api.yaml

# Check what's in the store
python -m ingestion.embedder stats

# Clear the store (re-index from scratch)
python -m ingestion.embedder clear
```

---

## Project Structure

```
test-gen-rag/
├── ingestion/           # Person 1 — parsing + embedding
│   ├── code_parser.py
│   ├── doc_parser.py
│   ├── api_parser.py
│   └── embedder.py
├── retrieval/           # Person 2 — RAG retrieval
│   ├── vector_store.py
│   ├── retriever.py
│   └── prompt_engine.py
├── generation/          # Person 2 — LLM generation
│   ├── llm_client.py
│   └── test_formatter.py
├── evaluation/          # Person 3 — metrics
│   ├── evaluator.py
│   └── metrics.py
├── ui/                  # Person 3 — Streamlit frontend
│   └── app.py
├── pipeline.py          # End-to-end orchestrator
├── data/                # Sample input files
├── tests/               # Unit tests
├── SCHEMA.md            # Chunk format contract (read this first)
├── .env.example         # API key template
└── requirements.txt
```

---

## Team

| Person | Responsibility | Days |
|---|---|---|
| Person 1 | Ingestion & Parsing | 1–3 |
| Person 2 | RAG & Generation | 2–5 |
| Person 3 | Evaluation & UI | 4–7 |

---

## Results

*(To be filled in by Person 3 after evaluation)*

| Mode | Input Type | Parse Rate | Execution Rate | Line Coverage |
|---|---|---|---|---|
| No RAG | Code | - | - | - |
| RAG | Code | - | - | - |
| No RAG | Requirements | - | - | - |
| RAG | Requirements | - | - | - |
| No RAG | API Spec | - | - | - |
| RAG | API Spec | - | - | - |
