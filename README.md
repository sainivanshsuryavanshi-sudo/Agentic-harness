# Agentic Harness

An explainable, self-correcting LLM reliability and semantic caching framework.

## Project Structure

```
agentic-harness/
├── app.py                # Streamlit web interface (Live Playground & Reliability Analytics)
├── harness.py            # Core orchestration pipeline (validation, retry loop, LLM judge)
├── cache.py              # Semantic cache using SQLite & SentenceTransformers (all-MiniLM-L6-v2)
├── benchmark.py          # 54-sample challenge benchmark suite
├── main.py               # Command-line entry point
├── .env                  # Environment variables (GROQ_API_KEY)
├── .gitignore            # Git ignore file for .venv, .env, *.db, __pycache__
├── pyproject.toml        # Project dependencies and configuration
└── harness_cache.db      # SQLite database for semantic cache & telemetry logs
```

## Setup & Installation

1. **Environment Setup** (using `uv` or `venv`):
   ```bash
   uv sync
   # or with standard virtualenv:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Configure API Key**:
   Create or edit `.env` in the root directory:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```
   > Note: Get your free API key at [console.groq.com](https://console.groq.com). Groq API keys typically begin with `gsk_`.

## How to Run

### 1. Command Line Interface (CLI)
Run a single prompt through the harness:
```bash
python main.py "Return ONLY valid JSON with keys 'language' and 'year' for Python."
```
or run `harness.py` directly:
```bash
python harness.py
```

### 2. Streamlit Web Dashboard
Launch the interactive web UI:
```bash
streamlit run app.py
```
This opens the Live Playground and Historical Performance Analytics tabs in your browser.

### 3. Benchmark Suite
Run the 54-sample challenge suite testing strict JSON formatting, length constraints, and code/logic precision:
```bash
python benchmark.py
```

### 4. Cache Module Test
Test the SQLite semantic caching independently:
```bash
python cache.py
```

