FROM python:3.12-slim

# System deps:
#   ffmpeg, libsndfile1                — audio preprocessing
#   fonts-dejavu-core                  — PDF generation (Unicode font for fpdf2)
#   tesseract-ocr (+deu/+eng), poppler — OCR fallback for scanned PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    fonts-dejavu-core \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only the deps needed for remote-backend mode (no torch/pyannote/whisper)
COPY requirements-remote.txt ./
RUN pip install --no-cache-dir -r requirements-remote.txt

# Install the app itself (editable-style so local-ai package is importable)
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# Copy static assets
COPY static/ ./static/

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/api/livez'); r.raise_for_status()" || exit 1

ENV LOCAL_AI_HOST=0.0.0.0
ENV LOCAL_AI_PORT=8000
ENV LOCAL_AI_DATA_DIR=/app/data
ENV LOCAL_AI_TRANSCRIPTION_BACKEND=remote
ENV LOCAL_AI_SUMMARY_BACKEND=openai
ENV LOCAL_AI_OTEL_ENABLED=true
ENV LOCAL_AI_OTEL_SERVICE_NAME=local-ai

# Instana GenAI observability via OpenLLMetry (traceloop-sdk)
ENV OTEL_RESOURCE_ATTRIBUTES="INSTANA_PLUGIN=genai"
ENV TRACELOOP_LOGGING_ENABLED=true
ENV TRACELOOP_METRICS_ENABLED=true

CMD ["python", "-m", "local_ai"]
