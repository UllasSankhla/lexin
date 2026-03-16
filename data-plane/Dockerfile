FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /data /storage/transcripts /storage/recordings

ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=sqlite:////data/data_plane.db
ENV STORAGE_BASE_PATH=/storage

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
