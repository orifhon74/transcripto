# Python base image
FROM python:3.11-slim

# System deps: ffmpeg for media; libsndfile1 for soundfile; build tools for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Faster startup / cleaner logs
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY . .

# Expose (optional for docs; Railway injects $PORT)
EXPOSE 5050

# Run with gunicorn (binds to Railway $PORT). Single worker is fine on 1 vCPU.
# Increase timeout for slow CPU transcriptions.
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "8", "--timeout", "600", "-b", "0.0.0.0:${PORT}", "app:app"]