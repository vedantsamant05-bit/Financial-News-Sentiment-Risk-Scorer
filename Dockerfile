# ── Stage 1: Python base ───────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── Stage 2: Dependencies ─────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

# ── Stage 3: Final image ──────────────────────────────────────────────────────
FROM deps AS final

# Copy source
COPY . .

# Create non-root user
RUN addgroup --system finrisk && \
    adduser --system --ingroup finrisk finrisk && \
    chown -R finrisk:finrisk /app

USER finrisk

# Expose ports for API and Streamlit dashboard
EXPOSE 8000 8501

# Default command: run the FastAPI server
# Override with: docker run ... streamlit run app/dashboard.py
CMD ["uvicorn", "api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
