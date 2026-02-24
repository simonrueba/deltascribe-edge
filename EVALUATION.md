# Evaluation

## Setup

- **Hardware**: Apple M3 Air, 24 GB RAM
- **Runtime**: Docker Desktop (16 GB allocated), CPU-only
- **Model**: MedGemma 4B-IT, GGUF Q4_K_M quantisation (~2.3 GB) via `llama-cpp-python`
- **Retrieval**: BiomedCLIP (open_clip) for similar-case retrieval + MedSigLIP (google/medsiglip-448) for zero-shot classification
- **Input formats**: DICOM (with VOI LUT windowing), PNG, JPEG
- **Real-time updates**: WebSocket event-driven push (500ms polling fallback)
- **Date**: February 2026

## Demo Patients

Seven real CXR pairs from the NIH ChestX-ray14 dataset (CC0 1.0) in `data/demo_patients/`. Images are real chest X-rays; pairs are same-patient sequential studies selected to cover different clinical trajectories.

**Ground truth methodology:** The NIH ChestX-ray14 dataset provides per-image finding labels (e.g. "Effusion", "Cardiomegaly") extracted from radiologist reports via NLP. The expected delta for each pair is derived logically from the per-image NIH labels: if findings resolve between studies → improved; if new findings appear → worsened; if the same findings persist → stable. This is not a direct radiologist annotation of interval change, but is grounded in the dataset's official per-image labels.

| Patient | Prior NIH Labels | Current NIH Labels | Derived Delta | Reasoning |
|---------|------------------|-------------------|---------------|-----------|
| cxr_001 | Atelectasis, Cardiomegaly, Consolidation, Effusion | No Finding | improved | All findings resolved |
| cxr_002 | No Finding | Effusion, Infiltration | worsened | New findings appeared |
| cxr_003 | Cardiomegaly | Cardiomegaly, Emphysema | worsened | Stable finding + new finding |
| cxr_004 | No Finding | Pneumothorax | worsened | New finding appeared |
| cxr_005 | Effusion | No Finding | improved | Finding resolved |
| cxr_006 | Atelectasis | Atelectasis | stable | Same finding in both |
| cxr_007 | No Finding | No Finding | stable | Normal in both |

## Protocol

Each demo patient was submitted via `POST /api/analyze` with prior and current CXR images (base64-encoded). The job was polled via `/api/analyze/{id}/status` until completion. Full results were retrieved from `/api/analyze/{id}/result`.

All cases ran sequentially in a single Docker session. The first case includes cold model load time for the GGUF model (~30s) plus retrieval model downloads (BiomedCLIP + MedSigLIP, ~60s total on first run).

## Results

### Per-Patient Breakdown

| Patient | Time (s) | Findings Detected | Delta | Expected | Match | Valid JSON | FHIR |
|---------|----------|-------------------|-------|----------|-------|------------|------|
| cxr_001 | 231* | effusion, cardiomegaly | improved | improved | Yes | Yes | Pass |
| cxr_002 | 130 | pneumothorax | worsened | worsened | Yes | Yes | Pass |
| cxr_003 | 161 | cardiomegaly | worsened | worsened | Yes | Yes | Pass |
| cxr_004 | 191 | pneumothorax | improved | worsened | No | Yes | Pass |
| cxr_005 | 171 | consolidation | worsened | improved | No | Yes | Pass |
| cxr_006 | 34 | normal | stable | stable | Yes | Yes | Pass |
| cxr_007 | 30 | normal | stable | stable | Yes | Yes | Pass |

\* cxr_001 includes cold model load (~30s GGUF + ~60s retrieval models). cxr_006/007 ran with warm models.

### Aggregate Metrics

| Category | Metric | Result |
|----------|--------|--------|
| Reliability | Schema validity (valid JSON without repair) | **100%** (7/7) |
| Reliability | FHIR export integrity (structural) | **Pass** |
| Feasibility | Latency, mean (M3 Air / CPU / Docker) | **~135s** |
| Feasibility | Latency, range | **30 -- 231s** |
| Trust | Evidence coverage (findings with evidence_refs) | **100%** |
| Trust | Retrieval evidence per case | **5 similar cases, 13 zero-shot predictions** |
| Trust | Disclaimer present in output | **100%** |
| Trust | Classification agreement (MedSigLIP vs MedGemma) | Tracked per case |
| Reliability | Schema validation + auto-repair | Built-in (0 repairs triggered) |
| Interoperability | FHIR R4 with SNOMED CT codes | Deterministic mapping |
| Consistency | Delta agreement vs. NIH-derived labels | **71%** (5/7) |

### Notes

- **Schema validity**: All 7 cases returned structurally valid JSON on the first inference pass. No schema repair was triggered.
- **Latency**: The range (30--231s) reflects warm vs. cold model state. The first case (cxr_001, 231s) includes cold model load for all three models (GGUF ~30s + BiomedCLIP + MedSigLIP ~60s). Normal cases (no findings) complete in ~30s with warm models; cases with findings take 130--190s. On GPU hardware, latency would be significantly lower.
- **Retrieval evidence**: Every case received 5 similar cases from the BiomedCLIP atlas and 13 zero-shot classification predictions from MedSigLIP. Retrieval evidence is used as the primary finding detection mechanism and uncertainty gate. (Guideline retrieval via cross-modal image-to-text matching is implemented but disabled in the demo -- no licensed guideline database is bundled. When configured with real guideline texts, MedSigLIP embeds them and retrieves semantically relevant guidelines per case.)
- **Uncertainty gating**: Each finding passes through an uncertainty gate that cross-references MedSigLIP zero-shot predictions, BiomedCLIP similar-case matches, and MedGemma output. Conflicting evidence raises uncertainty; corroborating evidence lowers it. Classification agreement between models is tracked per case.
- **Evidence coverage**: Every asserted finding includes at least one `evidence_refs` entry linking it to an evidence source (current image, prior image, or retrieved case).
- **FHIR export**: The FHIR R4 Bundle is generated deterministically from the structured JSON output. The model does not write FHIR directly. Each bundle contains a `DiagnosticReport` with linked `Observation` resources using SNOMED CT codes.
- **Finding detection**: The system detects findings via MedSigLIP zero-shot classification. Detected findings may not match NIH per-image labels exactly (e.g. cxr_002 NIH labels indicate Effusion/Infiltration but MedSigLIP detects Pneumothorax). MedSigLIP was not trained on NIH ChestX-ray14 labels specifically, so some label disagreement is expected.
- **Delta agreement**: 5 of 7 cases match the expected delta derived from NIH per-image labels (see Ground truth methodology above). The 2 mismatches (cxr_004, cxr_005) involve MedSigLIP delta predictions inverting the direction — predicting "improved" for a worsening case and vice versa. MedSigLIP's zero-shot delta prompts ("delta:improved", "delta:worsened") have limited discriminative power for subtle interval changes compared to its finding-level classification.
- **Limitations**: (1) NIH labels are NLP-extracted from reports, not direct radiologist annotations of each image; (2) delta is derived logically from per-image labels, not from a radiologist's interval-change assessment; (3) 7 cases is too small for statistical significance. A formal evaluation would use MS-CXR-T (1,326 radiologist-annotated temporal pairs on PhysioNet) as a benchmark.

## Reproduce

```bash
# 1. Start services
docker compose up

# 2. Wait for backend health
curl http://localhost:8000/api/health

# 3. Submit a case (example: cxr_001)
# Encode prior.png and current.png as base64, POST to /api/analyze
# See backend/app/api/routes.py for the request schema

# 4. Poll status
curl http://localhost:8000/api/analyze/{job_id}/status

# 5. Get result
curl http://localhost:8000/api/analyze/{job_id}/result
```

All demo patient data is included in the repository under `data/demo_patients/`.
