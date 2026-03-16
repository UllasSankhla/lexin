# Voice Appointment Booking AI

A complete voice-powered appointment booking system with three distinct components.

## Architecture

```
┌─────────────────┐     REST API      ┌──────────────────────┐
│   Web Frontend  │ ─────────────────▶│   Data Plane (:8001)  │
│  (Widget JS/HTML│ ◀── WebSocket ───  │   Voice Call Handler  │
└─────────────────┘                   └──────────┬───────────┘
                                                 │ REST API (config fetch)
                                                 ▼
                                      ┌──────────────────────┐
                                      │ Control Plane (:8000) │
                                      │  Configuration API    │
                                      └──────────────────────┘
```

### Components

| Component | Port | Description |
|-----------|------|-------------|
| `control-plane/` | 8000 | Configuration management — assistant persona, parameters, FAQs, webhooks |
| `data-plane/` | 8001 | Real-time voice call handling — STT → LLM → TTS pipeline |
| `frontend/` | — | Embeddable JavaScript widget + demo page |

## Quick Start

### 1. Setup API Keys

Set environment variables or create `.env` files:

```bash
export DEEPGRAM_API_KEY="your-deepgram-key"
export CEREBRAS_API_KEY="your-cerebras-key"
```

### 2. Start Control Plane

```bash
cd control-plane
cp .env.example .env
./run.sh
```

### 3. Seed Configuration

```bash
cd control-plane
source .venv/bin/activate
python3 seed_data.py
```

### 4. Start Data Plane

```bash
cd data-plane
cp .env.example .env
# Edit .env and add your API keys
./run.sh
```

### 5. Open Demo Page

Open `frontend/demo.html` in a browser (or serve with any static file server):

```bash
cd frontend
python3 -m http.server 5500
# Visit http://localhost:5500/demo.html
```

Click the microphone button in the bottom-right corner to start a voice call!

## Docker Compose

```bash
# Create .env with your API keys
cat > .env << EOF
DEEPGRAM_API_KEY=your-key
CEREBRAS_API_KEY=your-key
CONTROL_PLANE_API_KEY=your-secret
CORS_ORIGINS=http://localhost:5500
EOF

docker-compose up
```

## Control Plane API

Base URL: `http://localhost:8000`

| Endpoint | Description |
|----------|-------------|
| `GET/PUT/PATCH /api/v1/assistant` | Assistant persona & configuration |
| `GET/POST/PATCH/DELETE /api/v1/parameters` | Parameters to collect from callers |
| `GET/POST /api/v1/context-files` | Upload context documents |
| `GET/POST/PUT/DELETE /api/v1/faqs` | FAQ management |
| `GET/POST/PUT/DELETE /api/v1/spell-rules` | STT correction rules |
| `GET/POST/PUT/DELETE /api/v1/webhooks` | Webhook endpoint configuration |
| `GET /api/v1/config/export` | Full config export (consumed by data plane) |

## Data Plane API

Base URL: `http://localhost:8001`

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/calls/initiate` | Start a call, get session token + WS URL |
| `GET /api/v1/calls` | List all calls |
| `GET /api/v1/calls/{id}` | Get call details + gathered parameters |
| `GET /api/v1/calls/{id}/transcript` | Get call transcript text |
| `GET /api/v1/calls/{id}/analytics` | Get timing analytics |
| `WS /ws/call?token={token}` | WebSocket for live voice call |

## Widget Embedding

```html
<script
  src="https://your-domain.com/booking-widget.js"
  data-api-url="https://your-data-plane.com"
  data-theme="light"
  data-position="bottom-right"
  data-primary-color="#4F46E5"
  data-button-label="Book Appointment">
</script>
```

### Widget Data Attributes

| Attribute | Default | Description |
|-----------|---------|-------------|
| `data-api-url` | required | Data plane base URL |
| `data-theme` | `light` | `light` or `dark` |
| `data-position` | `bottom-right` | `bottom-right` or `bottom-left` |
| `data-primary-color` | `#4F46E5` | Brand accent color |
| `data-button-label` | `Book Appointment` | Text tooltip for the button |
| `data-debug` | `false` | Enable console logging |

## Voice Call Pipeline

```
Browser Mic → PCM Audio → WebSocket → Data Plane
                                          │
                                    Deepgram STT
                                          │
                                    Transcription
                                          │
                                   Cerebras LLM
                                          │
                                    AI Response
                                          │
                                    Deepgram TTS
                                          │
                              PCM Audio ← WebSocket ← Data Plane
                                          │
                                   Browser Speaker
```

## Storage

- **SQLite** (`control-plane/data/`, `data-plane/data/`) — structured configuration and call records
- **Local filesystem** (`control-plane/storage/`) — uploaded context files, system prompts
- **Local filesystem** (`data-plane/storage/`) — call transcripts, recordings

## Analytics

Each call records timing for every pipeline stage:

- `stt_session_open` — Time to establish Deepgram STT connection
- `stt_utterance_final` — Time from utterance start to final transcript
- `llm_response_complete` — Cerebras API latency
- `tts_synthesize` — Deepgram TTS latency
- `tts_playback_complete` — Total TTS delivery time

Access via: `GET /api/v1/calls/{call_id}/analytics`

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| API Framework | FastAPI + uvicorn |
| ORM | SQLAlchemy 2.0 |
| Database | SQLite (WAL mode) |
| STT | Deepgram (nova-2, streaming) |
| TTS | Deepgram (aura-2) |
| LLM | Cerebras (llama-4-scout) |
| Realtime | WebSockets |
| Frontend | Vanilla JS (no dependencies) |
| Schema Validation | Pydantic v2 |
