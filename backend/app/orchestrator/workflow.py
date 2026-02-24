"""Analysis workflow orchestrator."""

import asyncio
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import structlog

from app.schemas.analysis import (
    AnalysisResult,
    AuditEntry,
    DraftReport,
    EvidenceSourceType,
    Finding,
    StepStatus,
    UncertaintyLevel,
    WorkflowStep,
    WorkflowStepUpdate,
)

logger = structlog.get_logger(__name__)


class AnalysisOrchestrator:
    """
    Orchestrates the analysis workflow with deterministic step execution.

    Workflow (Maximum Credibility - 4 HAI-DEF Models):
    1. Intake - Validate inputs
    2. Preprocess - Normalize images
    3. Retrieve Context - Load FHIR/dictation (MedASR)
    4. Retrieve Evidence - CXR Foundation + MedSigLIP retrieval
    5. Inference - Run MedGemma longitudinal comparison
    6. Validate - Schema validation + uncertainty gating
    7. Assemble - Build outputs (report, delta, audit, FHIR)
    """

    def __init__(self) -> None:
        self._model_loaded = False
        self._device = "cpu"
        self._model = None
        self._processor = None
        self._retrieval_evidence = None  # Cached evidence for assembly

    def is_model_loaded(self) -> bool:
        """Check if the model is available (file exists or already in memory)."""
        from pathlib import Path

        from app.core.config import settings

        # Check if GGUF model is already loaded in memory
        if settings.use_gguf:
            from app.inference.medgemma_gguf import _llm

            if _llm is not None:
                return True
            # Check if model file exists on disk (lazy-loads on first inference)
            return Path(settings.gguf_model_path).exists()

        # For transformers backend, check if model is in memory
        return self._model_loaded

    @property
    def device(self) -> str:
        """Get the current device."""
        return self._device

    def generate_job_id(self) -> UUID:
        """Generate a new job ID."""
        return uuid4()

    async def analyze(
        self,
        job_id: UUID,
        prior_image: str,
        current_image: str,
        patient_id: str,
        dictation_audio: str | None = None,
        fhir_context: dict | None = None,
        status_callback: Callable[[WorkflowStepUpdate], None] | None = None,
    ) -> AnalysisResult:
        """
        Execute the full analysis workflow.

        Args:
            job_id: Unique job identifier
            prior_image: Base64-encoded prior CXR
            current_image: Base64-encoded current CXR
            patient_id: Patient identifier
            dictation_audio: Optional base64-encoded audio
            fhir_context: Optional FHIR bundle
            status_callback: Callback for status updates
        """
        async def emit_status(
            step: WorkflowStep,
            status: StepStatus,
            message: str | None = None,
            duration_ms: int | None = None,
        ) -> None:
            if status_callback:
                status_callback(WorkflowStepUpdate(
                    step=step,
                    status=status,
                    timestamp=datetime.utcnow(),
                    message=message,
                    duration_ms=duration_ms,
                ))
            # Yield to event loop after every emission so WebSocket
            # can push each state change before the next step starts.
            await asyncio.sleep(0)

        # Step 1: Intake
        await emit_status(WorkflowStep.INTAKE, StepStatus.RUNNING, "Validating inputs")
        start = time.time()

        self._validate_inputs(prior_image, current_image)

        await emit_status(
            WorkflowStep.INTAKE,
            StepStatus.COMPLETE,
            "Inputs validated",
            int((time.time() - start) * 1000),
        )

        # Step 2: Preprocess
        await emit_status(WorkflowStep.PREPROCESS, StepStatus.RUNNING, "Normalizing images")
        start = time.time()

        prior_processed, current_processed = await self._preprocess_images(
            prior_image, current_image
        )

        await emit_status(
            WorkflowStep.PREPROCESS,
            StepStatus.COMPLETE,
            "Images normalized",
            int((time.time() - start) * 1000),
        )

        # Step 3: Retrieve Context
        await emit_status(
            WorkflowStep.RETRIEVE_CONTEXT, StepStatus.RUNNING, "Loading context",
        )
        start = time.time()

        context = await self._retrieve_context(fhir_context, dictation_audio)

        await emit_status(
            WorkflowStep.RETRIEVE_CONTEXT,
            StepStatus.COMPLETE,
            f"Context loaded ({len(context)} items)",
            int((time.time() - start) * 1000),
        )

        # Step 4: Retrieve Evidence (CXR Foundation + MedSigLIP)
        await emit_status(
            WorkflowStep.RETRIEVE_EVIDENCE,
            StepStatus.RUNNING,
            "Retrieving evidence (CXR Foundation + MedSigLIP)",
        )
        start = time.time()

        retrieval_evidence = await self._retrieve_evidence(
            prior_processed, current_processed
        )
        self._retrieval_evidence = retrieval_evidence

        # Add evidence to context for MedGemma
        if retrieval_evidence:
            context["retrieval_evidence"] = retrieval_evidence

        await emit_status(
            WorkflowStep.RETRIEVE_EVIDENCE,
            StepStatus.COMPLETE,
            "Evidence retrieved"
            f" (confidence: {retrieval_evidence.get('overall_confidence', 0):.0%})",
            int((time.time() - start) * 1000),
        )

        # Step 5: Inference
        await emit_status(
            WorkflowStep.INFERENCE, StepStatus.RUNNING, "Running MedGemma analysis",
        )
        start = time.time()

        raw_output = await self._run_inference(
            prior_processed, current_processed, context
        )

        await emit_status(
            WorkflowStep.INFERENCE,
            StepStatus.COMPLETE,
            "Analysis complete",
            int((time.time() - start) * 1000),
        )

        # Step 6: Validate
        await emit_status(
            WorkflowStep.VALIDATE, StepStatus.RUNNING, "Validating output schema",
        )
        start = time.time()

        validated_output = await self._validate_and_repair(raw_output)

        await emit_status(
            WorkflowStep.VALIDATE,
            StepStatus.COMPLETE,
            "Output validated",
            int((time.time() - start) * 1000),
        )

        # Step 7: Assemble
        await emit_status(WorkflowStep.ASSEMBLE, StepStatus.RUNNING, "Building outputs")
        start = time.time()

        result = await self._assemble_result(
            job_id=job_id,
            patient_id=patient_id,
            validated_output=validated_output,
        )

        await emit_status(
            WorkflowStep.ASSEMBLE,
            StepStatus.COMPLETE,
            "Outputs ready",
            int((time.time() - start) * 1000),
        )

        return result

    def _validate_inputs(self, prior_image: str, current_image: str) -> None:
        """Validate input images."""
        import base64

        for name, img in [("prior", prior_image), ("current", current_image)]:
            if not img:
                raise ValueError(f"{name} image is required")
            try:
                # Attempt to decode to verify it's valid base64
                decoded = base64.b64decode(img)
                if len(decoded) < 100:
                    raise ValueError(f"{name} image appears too small")
            except Exception as e:
                raise ValueError(f"Invalid {name} image: {e}") from e

    async def _preprocess_images(
        self, prior_image: str, current_image: str
    ) -> tuple[Any, Any]:
        """
        Normalize images for model input.

        Returns processed image tensors/arrays.
        """
        from app.imaging.preprocessor import preprocess_cxr

        prior_processed = await preprocess_cxr(prior_image, label="prior")
        current_processed = await preprocess_cxr(current_image, label="current")

        return prior_processed, current_processed

    async def _retrieve_context(
        self, fhir_context: dict | None, dictation_audio: str | None
    ) -> dict[str, Any]:
        """Retrieve and organize context from FHIR and dictation."""
        context: dict[str, Any] = {}

        if fhir_context:
            from app.output.fhir import extract_minimal_context
            context["fhir"] = extract_minimal_context(fhir_context)

        if dictation_audio:
            from app.audio.transcriber import transcribe_audio

            try:
                result = await transcribe_audio(dictation_audio)
                context["dictation"] = {
                    "transcript": result.text,
                    "timestamps": result.timestamps,
                    "confidence": result.confidence,
                    "source": result.source,
                }
                logger.info(
                    "Dictation transcribed",
                    source=result.source,
                    confidence=result.confidence,
                )
            except Exception as e:
                logger.warning("Dictation transcription failed", error=str(e))
                context["dictation"] = {"transcript": "", "timestamps": [], "error": str(e)}

        return context

    async def _run_inference(
        self,
        prior_processed: Any,
        current_processed: Any,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run MedGemma inference for longitudinal comparison.

        Uses GGUF model by default for fast CPU/Metal inference.
        Falls back to transformers if GGUF is disabled.

        Returns raw model output as dict.
        """
        from app.core.config import settings

        if settings.use_gguf:
            from app.inference.medgemma_gguf import run_longitudinal_comparison
            logger.info("Using GGUF inference backend")
        else:
            from app.inference.medgemma import run_longitudinal_comparison
            logger.info("Using transformers inference backend")

        return await run_longitudinal_comparison(
            prior_image=prior_processed,
            current_image=current_processed,
            context=context,
        )

    async def _retrieve_evidence(
        self,
        prior_processed: Any,
        current_processed: Any,
    ) -> dict[str, Any]:
        """
        Retrieve evidence from CXR Foundation and MedSigLIP.

        Returns dict with evidence data for context enrichment.
        """
        from app.retrieval.evidence import get_evidence_aggregator

        aggregator = get_evidence_aggregator()

        # Convert processed images to PIL Images if needed
        prior_image = self._to_pil_image(prior_processed)
        current_image = self._to_pil_image(current_processed)

        # Gather evidence from both retrieval models
        evidence = await aggregator.gather_evidence(
            current_image=current_image,
            prior_image=prior_image,
        )

        # Convert to dict for context
        return {
            "similar_cases": [
                {
                    "case_id": c.case_id,
                    "similarity_score": c.similarity_score,
                    "known_findings": c.known_findings,
                    "delta_label": c.delta_label,
                    "source": c.source,
                    "description": c.metadata.get("description", ""),
                }
                for c in evidence.similar_cases
            ],
            "zero_shot_predictions": [
                {
                    "label": p.label,
                    "confidence": p.confidence,
                }
                for p in evidence.zero_shot_predictions
            ],
            "guideline_matches": [
                {
                    "title": g.title,
                    "source": g.source,
                    "text": g.text,
                    "similarity_score": g.similarity_score,
                    "finding_type": g.finding_type,
                    "citation": g.citation,
                }
                for g in evidence.guideline_matches
            ],
            "case_retrieval_confidence": evidence.case_retrieval_confidence,
            "overall_confidence": evidence.overall_confidence,
            "evidence_summary": evidence.evidence_summary,
            "classification_agreement": evidence.classification_agreement.get(
                "confidence", None
            ) if evidence.classification_agreement else None,
            "prompt_context": aggregator.format_for_prompt(evidence),
        }

    def _to_pil_image(self, processed: Any) -> Any:
        """Convert processed image data to PIL Image."""
        import io

        from PIL import Image

        if isinstance(processed, Image.Image):
            return processed

        # If it's a dict with 'pil_image' key (from preprocessor)
        if isinstance(processed, dict) and "pil_image" in processed:
            img_data = processed["pil_image"]
            if isinstance(img_data, Image.Image):
                return img_data

        # Legacy: check for 'image' key
        if isinstance(processed, dict) and "image" in processed:
            img_data = processed["image"]
            if isinstance(img_data, Image.Image):
                return img_data

        # If it's bytes or base64
        if isinstance(processed, bytes):
            return Image.open(io.BytesIO(processed))

        # If it's a numpy array
        try:
            import numpy as np
            if isinstance(processed, np.ndarray):
                return Image.fromarray(processed.astype("uint8"))
        except ImportError:
            pass

        # Fallback: try to get pil_image attribute
        if hasattr(processed, "pil_image"):
            return processed.pil_image

        logger.warning("Could not convert processed image to PIL, using placeholder")
        return Image.new("RGB", (224, 224), color="gray")

    async def _validate_and_repair(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        """
        Validate output against schema and attempt repair if needed.
        """
        from app.validation.schema import validate_and_repair_output

        return await validate_and_repair_output(raw_output)

    async def _assemble_result(
        self,
        job_id: UUID,
        patient_id: str,
        validated_output: dict[str, Any],
    ) -> AnalysisResult:
        """Assemble the final analysis result."""
        from app.output.audit import build_audit_trail
        from app.output.delta import build_delta_summary
        from app.output.report import build_draft_report
        from app.schemas.analysis import (
            GuidelineMatchResult,
            RetrievalEvidenceResult,
            SimilarCaseResult,
            ZeroShotResult,
        )

        # Build findings from validated output
        findings = [
            Finding(**f) for f in validated_output.get("findings", [])
        ]

        # Build delta summary
        delta_summary = build_delta_summary(findings)

        # Build draft report
        draft_report = DraftReport(
            findings=build_draft_report(findings, section="findings"),
            impression=build_draft_report(findings, section="impression"),
        )

        # Build audit trail
        audit_trail = build_audit_trail(findings)

        # Build retrieval evidence result from cached evidence
        retrieval_evidence_result = None
        if self._retrieval_evidence:
            ev = self._retrieval_evidence

            # Add similar case evidence to audit trail
            for case in ev.get("similar_cases", [])[:2]:
                audit_trail.append(
                    AuditEntry(
                        claim=(
                            f"Similar case supports "
                            f"{', '.join(case['known_findings'])} "
                            f"({case['delta_label']})"
                        ),
                        source_type=EvidenceSourceType.SIMILAR_CASE,
                        source_ref=f"{case['case_id']} ({case['source']})",
                        uncertainty=UncertaintyLevel.LOW
                        if case["similarity_score"] > 0.7
                        else UncertaintyLevel.MEDIUM,
                    )
                )

            # Add zero-shot evidence to audit trail
            for pred in ev.get("zero_shot_predictions", [])[:3]:
                if pred["confidence"] > 0.1 and not pred["label"].startswith("delta:"):
                    audit_trail.append(
                        AuditEntry(
                            claim=f"Zero-shot classification: {pred['label']}",
                            source_type=EvidenceSourceType.ZERO_SHOT,
                            source_ref=f"MedSigLIP ({pred['confidence']:.0%} confidence)",
                            uncertainty=UncertaintyLevel.LOW
                            if pred["confidence"] > 0.5
                            else UncertaintyLevel.MEDIUM,
                        )
                    )

            # Build structured retrieval evidence result
            retrieval_evidence_result = RetrievalEvidenceResult(
                similar_cases=[
                    SimilarCaseResult(
                        case_id=c["case_id"],
                        similarity_score=c["similarity_score"],
                        known_findings=c["known_findings"],
                        delta_label=c["delta_label"],
                        source=c["source"],
                        description=c.get("description"),
                    )
                    for c in ev.get("similar_cases", [])
                ],
                zero_shot_predictions=[
                    ZeroShotResult(
                        label=p["label"],
                        confidence=p["confidence"],
                    )
                    for p in ev.get("zero_shot_predictions", [])
                ],
                guideline_matches=[
                    GuidelineMatchResult(
                        title=g["title"],
                        source=g["source"],
                        text=g["text"],
                        similarity_score=g["similarity_score"],
                        finding_type=g["finding_type"],
                        citation=g["citation"],
                    )
                    for g in ev.get("guideline_matches", [])
                ],
                case_retrieval_confidence=ev.get("case_retrieval_confidence", 0.0),
                overall_confidence=ev.get("overall_confidence", 0.0),
                evidence_summary=ev.get("evidence_summary", ""),
                classification_agreement=ev.get("classification_agreement"),
            )

            logger.info(
                "Retrieval evidence assembled",
                similar_cases=len(retrieval_evidence_result.similar_cases),
                zero_shot=len(retrieval_evidence_result.zero_shot_predictions),
                guidelines=len(retrieval_evidence_result.guideline_matches),
            )

        # Extract model narrative (raw LLM output)
        model_narrative = validated_output.get("raw_text") or None

        return AnalysisResult(
            job_id=job_id,
            patient_id=patient_id,
            findings=findings,
            delta_summary=delta_summary,
            draft_report=draft_report,
            audit_trail=audit_trail,
            retrieval_evidence=retrieval_evidence_result,
            model_narrative=model_narrative,
        )
