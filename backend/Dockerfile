# Build context is the REPOSITORY ROOT, not backend/, because the service needs
# the dataset and the trained model artifacts that live alongside it:
#
#   docker build -f backend/Dockerfile -t fmn-manufacturing-api .
#
# render.yaml sets dockerContext accordingly.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so the layer caches across code changes.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app ./backend/app
COPY data ./data
COPY models ./models

# The model artifacts are committed, so the image is ready to serve. If they are
# ever missing the app still starts and /health reports "degraded" with an
# actionable message instead of crash-looping.

WORKDIR /app/backend

EXPOSE 8000

# Render injects $PORT; default to 8000 for local runs. Bind 0.0.0.0 so the
# platform's health check can reach the container.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
