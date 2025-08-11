# app/Dockerfile
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and your library module
COPY app.py ./
COPY ../market_plotting.py ./market_plotting.py

EXPOSE 7860
CMD ["python", "app.py"]
