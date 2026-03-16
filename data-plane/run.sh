#!/bin/bash
set -e

# Copy .env if not exists
[ -f .env ] || cp .env.example .env

# Create virtual env if not exists
[ -d .venv ] || python3 -m venv .venv

source .venv/bin/activate
pip install -q -r requirements.txt

echo "Starting Data Plane on port 8001..."
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
