# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- Dependency layer (cached unless requirements change) ----------------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code + pre-built artifact -------------------------------
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# The trained artifact is built outside the image and copied in.
# Run `views-train --postings data/postings.csv --companies data/companies/company_industries.csv`
# locally first so artifacts/views_baseline.joblib exists.
COPY artifacts ./artifacts

ENV VIEWS_MODEL_ARTIFACT=/app/artifacts/views_baseline.joblib \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn views_model.api:app --host 0.0.0.0 --port ${PORT}"]
