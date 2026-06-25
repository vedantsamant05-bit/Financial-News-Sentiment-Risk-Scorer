# FinRisk — Financial Sentiment Risk Scorer

A production-ready pipeline that ingests live financial headlines, scores them with **FinBERT** sentiment analysis, aggregates entity-level risk signals, and exposes the results through a **Streamlit dashboard** and a **FastAPI REST API**.

---

## Project Structure

```
finrisk/
├── app/                      # Streamlit entry-point
│   └── dashboard.py
├── src/                      # Core Python package
│   ├── ingestion/            # Headline scraping
│   │   ├── __init__.py
│   │   └── scraper.py
│   ├── preprocessing/        # Text cleaning & deduplication
│   │   ├── __init__.py
│   │   └── cleaner.py
│   ├── model/                # Inference & fine-tuning wrappers
│   │   ├── __init__.py
│   │   ├── fine_tune.py
│   │   └── inference.py
│   ├── risk/                 # Risk aggregation & alerting
│   │   ├── __init__.py
│   │   ├── aggregator.py
│   │   └── alerts.py
│   ├── config.py             # Path & model constants
│   ├── data_loader.py
│   ├── entity_extractor.py
│   ├── fine_tune.py          # Fine-tuning implementation
│   ├── pipeline.py           # Original pipeline runner
│   ├── risk_aggregator.py    # Aggregation implementation
│   ├── sentiment_model.py    # FinBERT inference implementation
│   └── threshold_tuner.py
├── scripts/
│   └── orchestrator.py       # End-to-end pipeline runner
├── models/
│   └── finbert-finetuned/    # Saved fine-tuned model (gitignored)
├── data/                     # CSVs & raw data (gitignored)
│   └── load_real_data.py
├── api/                      # FastAPI REST API
│   ├── __init__.py
│   └── routes.py
├── db/                       # Database layer
│   ├── __init__.py
│   ├── schema.sql
│   └── database.py
├── config/
│   └── settings.py           # Env-var backed settings
├── dashboard/
│   └── app.py                # Streamlit app implementation
├── .env                      # Local secrets (gitignored)
├── .env.example              # Template — copy to .env
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment

```bash
copy .env.example .env
# Edit .env and add your API keys (NEWSAPI_KEY, GNEWS_KEY, etc.)
```

### 3. Fetch data & run the pipeline

```bash
# Scrape live headlines + run full pipeline
python scripts/orchestrator.py

# Or use cached data (skip scraping)
python scripts/orchestrator.py --skip-scrape

# Or run the original pipeline directly
python -m src.pipeline
```

### 4. Launch the dashboard

```bash
streamlit run app/dashboard.py
# Open http://localhost:8501
```

### 5. Start the REST API

```bash
uvicorn api.routes:app --reload
# Open http://localhost:8000/docs
```

---

## Fine-tuning FinBERT

```bash
python -m src.fine_tune               # default 4 epochs
python -m src.fine_tune --epochs 6    # more epochs
python -m src.fine_tune --eval-only   # evaluate saved model
```

---

## Docker

```bash
# Build and start API + dashboard
docker compose up --build

# Also run the orchestrator (one-shot)
docker compose --profile run up orchestrator
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/risk/summary` | Entity risk summary (optional `?tier=HIGH`) |
| GET | `/risk/temporal` | Temporal z-score data (optional `?entity=Apple`) |
| GET | `/headlines` | Scored headlines (optional `?source=...&label=negative`) |

Full interactive docs at `http://localhost:8000/docs`.

---

## License

MIT
