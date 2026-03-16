FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /data /storage/context_files /storage/system_prompts

ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=sqlite:////data/control_plane.db
ENV STORAGE_BASE_PATH=/storage

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
