# Python base image
FROM python:3.11-slim

# System deps (ffmpeg for audio/video; build tools for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Faster startup / cleaner logs
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY . .

# Railway provides $PORT; gunicorn binds to it
CMD ["python", "app.py"]600"]
