"""API route definitions."""

import asyncio
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.orchestrator.workflow import AnalysisOrchestrator
from app.schemas.analysis import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisStatus,
    HealthResponse,
    StepStatus,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionTimestamp,
    WorkflowStep,
    WorkflowStepUpdate,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

# In-memory job storage (for demo - replace with Redis/DB for production)
jobs: dict[UUID, AnalysisStatus] = {}
results: dict[UUID, AnalysisResult] = {}
# Per-job events for real-time WebSocket push (avoids polling delay)
job_events: dict[UUID, asyncio.Event] = {}
# Track insertion order for cleanup
_job_order: list[UUID] = []
_MAX_STORED_JOBS = 3
orchestrator = AnalysisOrchestrator()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health and model status."""
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        model_loaded=orchestrator.is_model_loaded(),
        device=str(orchestrator.device),
    )


@router.post("/analyze", response_model=AnalysisStatus)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
) -> AnalysisStatus:
    """
    Start an analysis job for a pair of CXR images.

    Returns a job ID that can be used to poll for status or connect via WebSocket.
    """
    logger.info("Starting analysis", patient_id=request.patient_id)

    # Create job status
    status = AnalysisStatus(
        job_id=orchestrator.generate_job_id(),
        status="pending",
        current_step=WorkflowStep.INTAKE,
        steps=[
            WorkflowStepUpdate(
                step=WorkflowStep.INTAKE,
                status=StepStatus.PENDING,
            )
        ],
        progress_percent=0,
    )
    jobs[status.job_id] = status
    _job_order.append(status.job_id)

    # Evict old jobs to prevent memory growth
    while len(_job_order) > _MAX_STORED_JOBS:
        old_id = _job_order.pop(0)
        jobs.pop(old_id, None)
        results.pop(old_id, None)
        job_events.pop(old_id, None)

    # Run analysis in background
    background_tasks.add_task(run_analysis, status.job_id, request)

    return status


async def run_analysis(job_id: UUID, request: AnalysisRequest) -> None:
    """Execute the analysis workflow."""
    status = jobs[job_id]

    try:
        status.status = "processing"

        # Run the orchestrated workflow
        result = await orchestrator.analyze(
            job_id=job_id,
            prior_image=request.prior_image,
            current_image=request.current_image,
            patient_id=request.patient_id,
            dictation_audio=request.dictation_audio,
            fhir_context=request.fhir_context,
            status_callback=lambda update: update_job_status(job_id, update),
        )

        # Store result
        results[job_id] = result
        status.status = "complete"
        status.progress_percent = 100

        logger.info("Analysis complete", job_id=str(job_id))

    except Exception as e:
        logger.error("Analysis failed", job_id=str(job_id), error=str(e))
        status.status = "error"
        status.error_message = str(e)


def update_job_status(job_id: UUID, update: WorkflowStepUpdate) -> None:
    """Update job status with workflow step progress."""
    if job_id not in jobs:
        return

    status = jobs[job_id]
    status.current_step = update.step
    status.steps.append(update)

    # Calculate progress based on completed steps
    step_weights = {
        WorkflowStep.INTAKE: 5,
        WorkflowStep.PREPROCESS: 10,
        WorkflowStep.RETRIEVE_CONTEXT: 10,
        WorkflowStep.RETRIEVE_EVIDENCE: 15,  # CXR Foundation + MedSigLIP
        WorkflowStep.INFERENCE: 50,
        WorkflowStep.VALIDATE: 5,
        WorkflowStep.ASSEMBLE: 5,
    }

    completed_weight = sum(
        step_weights.get(s.step, 0)
        for s in status.steps
        if s.status == StepStatus.COMPLETE
    )
    status.progress_percent = min(completed_weight, 100)

    # Signal any waiting WebSocket to push immediately
    event = job_events.get(job_id)
    if event:
        event.set()


@router.get("/analyze/{job_id}/status", response_model=AnalysisStatus)
async def get_analysis_status(job_id: UUID) -> AnalysisStatus:
    """Get the current status of an analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@router.get("/analyze/{job_id}/result", response_model=AnalysisResult)
async def get_analysis_result(job_id: UUID) -> AnalysisResult:
    """Get the result of a completed analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    status = jobs[job_id]
    if status.status != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {status.status}",
        )

    if job_id not in results:
        raise HTTPException(status_code=404, detail="Result not found")

    return results[job_id]


@router.get("/export/fhir/{job_id}")
async def export_fhir_bundle(job_id: UUID) -> JSONResponse:
    """Export analysis result as a FHIR R4 bundle."""
    if job_id not in results:
        raise HTTPException(status_code=404, detail="Result not found")

    from app.output.fhir import build_fhir_bundle

    result = results[job_id]
    bundle = build_fhir_bundle(result)

    return JSONResponse(
        content=bundle,
        headers={
            "Content-Disposition": f'attachment; filename="deltascribe-{job_id}.fhir.json"',
            "Content-Type": "application/fhir+json",
        },
    )


@router.websocket("/ws/analyze/{job_id}")
async def websocket_status(websocket: WebSocket, job_id: UUID) -> None:
    """WebSocket endpoint for real-time status updates.

    Uses event-driven push: every emit_status() call signals immediately
    rather than relying on polling, so even sub-second steps show their
    "running" state.
    """
    await websocket.accept()

    if job_id not in jobs:
        await websocket.close(code=4004, reason="Job not found")
        return

    # Register event for this job so update_job_status can signal us
    event = asyncio.Event()
    job_events[job_id] = event

    try:
        # Send current state immediately on connect
        status = jobs.get(job_id)
        if status:
            await websocket.send_json(status.model_dump(mode="json"))

        while True:
            # Wait for next update signal or timeout (heartbeat)
            try:
                await asyncio.wait_for(event.wait(), timeout=5.0)
                event.clear()
            except TimeoutError:
                pass  # Send heartbeat/current state anyway

            status = jobs.get(job_id)
            if status is None:
                await websocket.close(code=4004, reason="Job not found")
                return

            await websocket.send_json(status.model_dump(mode="json"))

            # Stop once terminal state is reached
            if status.status in ("complete", "error"):
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", job_id=str(job_id))
    finally:
        job_events.pop(job_id, None)


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest) -> TranscriptionResponse:
    """
    Transcribe audio using medical speech recognition.

    Attempts transcription in order:
    1. MedASR (Google's medical ASR) if available
    2. Whisper (local) if installed

    Raises 503 if no ASR engine is available.

    Returns transcription with confidence and word-level timestamps.
    """
    from app.audio.transcriber import transcribe_audio as do_transcribe

    logger.info("Transcription request received")

    try:
        result = await do_transcribe(
            audio_base64=request.audio,
            language=request.language,
        )

        return TranscriptionResponse(
            text=result.text,
            confidence=result.confidence,
            timestamps=[
                TranscriptionTimestamp(
                    start=ts["start"],
                    end=ts["end"],
                    text=ts["text"],
                )
                for ts in result.timestamps
            ],
            source=result.source,
        )

    except Exception as e:
        logger.error("Transcription failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}") from e


@router.get("/transcribe/status")
async def get_transcription_status() -> dict[str, bool]:
    """Check which transcription engines are available."""
    from app.audio.transcriber import get_transcription_status

    return await get_transcription_status()


@router.get("/demo/patients")
async def list_demo_patients() -> list[dict[str, Any]]:
    """List available demo patients for testing."""
    import json

    from app.core.config import settings

    patients = []
    demo_dir = settings.demo_patients_dir

    if demo_dir.exists():
        for patient_dir in demo_dir.iterdir():
            if patient_dir.is_dir():
                manifest_path = patient_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        patients.append(json.load(f))

    return patients


@router.get("/demo/patients/{patient_id}/images")
async def get_demo_patient_images(patient_id: str) -> dict[str, str]:
    """Get demo patient images as base64-encoded strings for frontend loading."""
    import base64

    from app.core.config import settings

    patient_dir = settings.demo_patients_dir / patient_id

    if not patient_dir.exists():
        raise HTTPException(status_code=404, detail=f"Demo patient '{patient_id}' not found")

    prior_path = patient_dir / "prior.png"
    current_path = patient_dir / "current.png"

    if not prior_path.exists() or not current_path.exists():
        raise HTTPException(status_code=404, detail="Patient images not found")

    with open(prior_path, "rb") as f:
        prior_b64 = base64.b64encode(f.read()).decode("utf-8")

    with open(current_path, "rb") as f:
        current_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "patient_id": patient_id,
        "prior_image": prior_b64,
        "current_image": current_b64,
    }
