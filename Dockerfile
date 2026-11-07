FROM node:22-alpine AS frontend
WORKDIR /app/frontend
ENV VITE_CINEMATCH_READ_ONLY_DEMO=true
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app/src \
    CINEMATCH_ALEMBIC_CONFIG=/app/alembic.ini \
    CINEMATCH_DATABASE_URL=sqlite:////app/data/cinematch.db
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
COPY alembic.ini ./
COPY migrations/ ./migrations/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .
COPY data/raw/ ./data/raw/
COPY data/processed/week4_cold_start_recommendations.csv ./data/processed/week4_cold_start_recommendations.csv
COPY models/svd_model.pkl ./models/svd_model.pkl
COPY --from=frontend /app/frontend/dist ./frontend/dist
RUN mkdir -p /app/data && chmod 755 /app/data
EXPOSE 8000
CMD ["sh", "-c", "python -c \"from cinematch.database import initialize_database; initialize_database()\" && uvicorn cinematch.api:app --host 0.0.0.0 --port ${PORT}"]
