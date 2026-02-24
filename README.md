# DeltaScribe Edge

**Offline longitudinal chest X-ray copilot with verifiable evidence and FHIR export.**

MedGemma Impact Challenge 2026 | Main Track + Agentic Workflow Prize

Designed to be edge-capable: CPU-only GGUF inference path included.

---

## Judge Quickstart

```bash
# 1. Download model (~2.3 GB, one-time)
pip install huggingface_hub
huggingface-cli download unsloth/medgemma-4b-it-GGUF \
  medgemma-4b-it-Q4_K_M.gguf --local-dir data/models

# 2. Start (Docker Desktop must have 16 GB RAM allocated)
docker compose up --build
```

1. Open **http://localhost:3020**
2. Click **Load Demo Patient** > select `cxr_001`
3. Click **Analyze** > watch the 7-step pipeline
4. Review: Delta Summary | Draft Report | Evidence Ledger | Audit Trail
5. Click **Export FHIR** to download the R4 Bundle

Frontend: http://localhost:3020 | Backend API: http://localhost:8000

## What It Does

DeltaScribe Edge compares prior and current chest X-rays and produces:

1. **Delta Summary** -- structured interval-change findings (improved / stable / worsened / uncertain)
2. **Draft Report** -- radiology-style Findings + Impression sections
3. **Evidence Ledger** -- every claim linked to evidence sources with uncertainty status
4. **FHIR R4 Bundle** -- DiagnosticReport + Observation resources with SNOMED CT codes (deterministic, not model-generated)

All processing runs locally. No cloud APIs are called during inference. Model weights must be downloaded once before first use (see Setup).

**What is verifiable:** Every asserted finding must have `evidence_refs` linking it to a source (prior image, current image, or retrieved case). Claims without sufficient evidence are marked `uncertainty: high` rather than asserted. The audit trail and evidence ledger are inspectable in the UI.

## Hardware Requirements

| Setup | Hardware | Notes |
|-------|----------|-------|
| **Tested** | Apple M3 Air, 24 GB RAM, CPU-only | avg ~135s/case via GGUF Q4_K_M |
| **Recommended** | Any x86/ARM with 16 GB+ RAM | Docker Desktop with 16 GB allocated |
| **Optimal** | NVIDIA GPU (8 GB+ VRAM) | Multimodal transformers pathway available |

The default deployment uses a 2.3 GB GGUF-quantised model (Q4_K_M) running on CPU via `llama-cpp-python`. No GPU required.

## Quick Start

### Prerequisites

- Docker Desktop (allocate 16 GB RAM: Settings > Resources > Memory)
- HuggingFace account with [MedGemma access](https://huggingface.co/google/medgemma-4b-it) (one-time download; not needed at runtime)

### Setup

```bash
# 1. Clone
git clone https://github.com/simonrueba/deltascribe-edge.git
cd deltascribe-edge

# 2. Download GGUF model (~2.3 GB, one-time)
pip install huggingface_hub
huggingface-cli download unsloth/medgemma-4b-it-GGUF \
  medgemma-4b-it-Q4_K_M.gguf \
  --local-dir data/models

# 3. Start
docker compose up --build
```

7 demo patients are bundled in `data/demo_patients/`. No separate download needed. All images are real chest X-rays from the NIH ChestX-ray14 dataset (CC0 1.0 Public Domain).

## Architecture

```
Frontend (Next.js 16 + shadcn/ui)
    |
    v
Backend (FastAPI)
    |
    v
7-Step Agentic Pipeline:
  1. Intake ---------- validate inputs
  2. Preprocess ------ normalize images
  3. Context --------- load FHIR/dictation context
  4. Evidence -------- retrieval (BiomedCLIP + MedSigLIP, if loaded)
  5. Inference ------- evidence-driven findings + MedGemma GGUF narrative
  6. Validate -------- JSON schema check + repair
  7. Assemble -------- Evidence Ledger + Report + FHIR Bundle
```

Each step emits a visible event to the UI timeline. Intermediate artifacts are inspectable. The backend pushes step updates via WebSocket (event-driven, not polling) so each pipeline stage is visible in real time. If WebSocket fails, the frontend falls back to 500ms polling. The schema validation step includes automatic repair -- if the model produces invalid JSON, it is corrected before output, guaranteeing 100% valid responses.

### Model Components

| Component | Model | Role | Status |
|-----------|-------|------|--------|
| **Finding detection** | MedSigLIP (google/medsiglip-448) | Zero-shot classification of findings and delta | Primary -- findings are evidence-driven from this model |
| **Similar-case retrieval** | BiomedCLIP (via open_clip) | Embed CXR, match against atlas | Optional -- boosts confidence of detected findings |
| **Narrative generation** | MedGemma 4B-IT (GGUF Q4_K_M) | Generate report narrative from evidence-derived findings | Required -- fallback produces simple text if unavailable |
| **Speech-to-text** | MedASR / Whisper | Dictation transcription (MedASR first, Whisper fallback) | Optional -- only used if audio is provided |
| **Image preprocessing** | Built-in | DICOM detection + VOI LUT windowing, PNG/JPEG support, auto-resize | Accepts DICOM, PNG, JPEG natively |

The GGUF model is text-only and cannot see images. Finding detection uses a two-phase approach: Phase 1 builds findings from MedSigLIP zero-shot classification and BiomedCLIP similar-case retrieval (both of which can analyze images). Phase 2 uses the GGUF LLM to generate narrative text from the evidence-derived findings. If retrieval models are unavailable, the pipeline continues with empty evidence -- no mock or synthetic data is generated.

**Uncertainty gating:** MedSigLIP zero-shot predictions are cross-validated against BiomedCLIP similar-case retrieval. When both models agree, uncertainty is lowered; when they conflict, findings are flagged for review. Classification agreement between the vision models and MedGemma narrative is tracked and reported in the audit trail.

**Guideline retrieval:** Cross-modal image-to-text guideline matching via MedSigLIP is implemented but disabled in the demo (no licensed guideline database bundled). When configured with real guideline texts, MedSigLIP embeds them and retrieves semantically relevant guidelines per case.

## Constrained Vocabulary

| Findings (8) | Delta (4) |
|--------------|-----------|
| consolidation, effusion, pneumothorax, cardiomegaly, edema, nodule, atelectasis, normal | improved, stable, worsened, uncertain |

Per-finding schema: `label`, `delta`, `rationale`, `evidence_refs`, `bounding_box` (optional), `uncertainty`.

## Evaluation Summary

| Metric | Result |
|--------|--------|
| Schema validity (valid JSON, no repair) | 100% (7/7) |
| FHIR export integrity | Structural pass |
| Latency (M3 Air / CPU / Docker) | ~135s mean, 30--231s range |
| Delta agreement (NIH-derived labels) | 71% (5/7) |
| Evidence coverage | 100% |
| Disclaimer present | 100% |

Full evaluation details: [EVALUATION.md](EVALUATION.md)

## Safety

All outputs include the disclaimer:

> **Draft only -- requires clinician verification before clinical use.**

- This tool is **not** a diagnostic device
- The system does **not** make clinical decisions, order tests, or prescribe treatments
- Unsupported claims are marked as *Uncertain* rather than asserted
- All inference runs locally; no PHI leaves the device

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check + model status |
| `/api/analyze` | POST | Start analysis (async) |
| `/api/analyze/{id}/status` | GET | Poll pipeline progress |
| `/api/analyze/{id}/result` | GET | Get full result |
| `/ws/analyze/{id}` | WebSocket | Real-time pipeline progress (event-driven push) |
| `/api/export/fhir/{id}` | GET | Download FHIR R4 Bundle |
| `/api/transcribe` | POST | Dictation transcription (MedASR / Whisper fallback) |
| `/api/demo/patients` | GET | List bundled demo cases |

## Tech Stack

- **Frontend**: Next.js 16, React 19, shadcn/ui, Tailwind CSS
- **Backend**: Python 3.11, FastAPI, Pydantic
- **Inference**: MedGemma 4B-IT via llama-cpp-python (GGUF Q4_K_M)
- **Retrieval** (optional): BiomedCLIP (open_clip_torch), MedSigLIP (google/medsiglip-448)
- **Infrastructure**: Docker Compose (offline after initial model download)

## Local Development

```bash
# Backend
cd backend
cp .env.example .env  # edit HF_TOKEN
uv venv && source .venv/bin/activate
uv pip install -e ".[gguf,retrieval]"
uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend
bun install
bun dev
```

## Data & Licensing

- Demo CXR pairs in `data/demo_patients/` are real chest X-rays from the [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) dataset (CC0 1.0 Public Domain), curated as 7 same-patient longitudinal pairs
- No protected health information (PHI) is included in this repository
- MedGemma model weights require a one-time download from HuggingFace and are governed by Google's [model license](https://huggingface.co/google/medgemma-4b-it)
- A HuggingFace account with accepted MedGemma terms is required for the model download step only

## Acknowledgments

- [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations) -- MedGemma (narrative generation), MedSigLIP (zero-shot finding detection), CXR Foundation (precomputed atlas embeddings)
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (Microsoft) -- similar-case retrieval embeddings via open_clip
- [shadcn/ui](https://ui.shadcn.com/) for UI components
- [llama.cpp](https://github.com/ggerganov/llama.cpp) / [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for GGUF inference
- [Unsloth](https://huggingface.co/unsloth/medgemma-4b-it-GGUF) for pre-quantised GGUF weights

---

*AI tools were used to assist drafting and editing; all technical claims were verified by the author.*
