FROM python:3.10-slim

WORKDIR /app

# runtime behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# system deps for building python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# install torch cpu first (avoid heavy gpu deps / cache layer)
RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# install python deps (cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# app source
COPY . .

# runtime dirs
RUN mkdir -p /app/local_bm25_data

EXPOSE 8501

# basic liveness check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]