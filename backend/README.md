# DeltaScribe Edge Backend

FastAPI backend for longitudinal CXR comparison using MedGemma.

## Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

## Running

```bash
# Development
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/analyze` - Start analysis
- `GET /api/analyze/{job_id}/status` - Get analysis status
- `GET /api/analyze/{job_id}/result` - Get analysis result
- `GET /api/export/fhir/{job_id}` - Export FHIR bundle
- `WS /api/ws/{job_id}` - WebSocket for real-time updates

## Environment Variables

- `MODEL_NAME` - HuggingFace model name (default: google/medgemma-4b-it)
- `MODEL_PATH` - Local model path (optional)
- `DEVICE` - Inference device: auto, cuda, cpu (default: auto)
- `MAX_NEW_TOKENS` - Max tokens to generate (default: 2048)
