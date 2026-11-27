# Simple CPU image; for GPU use nvidia/cuda base and install torch accordingly.
FROM python:3.11-slim

WORKDIR /app

# System deps for extract-msg and parsing (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY ui /app/ui
COPY results /app/results
COPY logs /app/logs

ENV PYTHONUNBUFFERED=1

# default command overridden by docker-compose
CMD ["python", "-c", "print('Use docker-compose to run api or ui')"]
