"""MedGemma GGUF inference for fast CPU/Metal inference.

Uses llama-cpp-python with quantized GGUF model for efficient inference
on devices without dedicated GPUs.

Architecture: The GGUF model is text-only and cannot see images. We use a
two-phase approach:
  Phase 1: Build findings from retrieval evidence (MedSigLIP zero-shot +
           CXR Foundation similar cases), which CAN analyze images.
  Phase 2: Use the GGUF LLM to generate narrative rationales and report text
           based on the evidence-derived findings.
"""

import asyncio
import re
from typing import Any

import structlog

from app.core.config import settings
from app.schemas.analysis import DeltaStatus, FindingLabel

logger = structlog.get_logger(__name__)

# Global model instance (lazy loaded)
_llm = None
_load_lock = asyncio.Lock()

# Mapping from MedSigLIP labels to our FindingLabel enum
_SIGLIP_TO_FINDING = {
    "pleural effusion": FindingLabel.EFFUSION,
    "consolidation": FindingLabel.CONSOLIDATION,
    "pneumothorax": FindingLabel.PNEUMOTHORAX,
    "cardiomegaly": FindingLabel.CARDIOMEGALY,
    "pulmonary edema": FindingLabel.EDEMA,
    "pulmonary nodule": FindingLabel.NODULE,
    "atelectasis": FindingLabel.ATELECTASIS,
    "normal no abnormalities": FindingLabel.NORMAL,
}

# Mapping from MedSigLIP delta labels to our DeltaStatus
_SIGLIP_TO_DELTA = {
    "delta:improved": DeltaStatus.IMPROVED,
    "delta:stable": DeltaStatus.STABLE,
    "delta:worsened": DeltaStatus.WORSENED,
    "delta:new": DeltaStatus.WORSENED,
    "delta:resolved": DeltaStatus.IMPROVED,
}

# Minimum confidence for zero-shot predictions to become findings
_ZERO_SHOT_THRESHOLD = 0.12


def _get_model_path() -> str:
    """Get the GGUF model path."""
    if hasattr(settings, "gguf_model_path") and settings.gguf_model_path:
        return settings.gguf_model_path
    return "/data/models/medgemma-4b-it-Q4_K_M.gguf"


def _load_model_sync():
    """Synchronously load the GGUF model."""
    from llama_cpp import Llama

    model_path = _get_model_path()
    logger.info("Loading MedGemma GGUF model", model_path=model_path)

    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False,
    )

    logger.info("MedGemma GGUF model loaded")
    return llm


async def get_model():
    """Lazy load the GGUF model (non-blocking)."""
    global _llm

    if _llm is not None:
        return _llm

    async with _load_lock:
        if _llm is not None:
            return _llm

        try:
            _llm = await asyncio.to_thread(_load_model_sync)
            return _llm
        except Exception as e:
            logger.error("Failed to load MedGemma GGUF model", error=str(e))
            raise


def _build_findings_from_evidence(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Build findings from retrieval evidence (Phase 1).

    Uses MedSigLIP zero-shot predictions and CXR Foundation similar cases
    to determine which findings are actually present in the images.
    """
    ev = context.get("retrieval_evidence", {})

    zero_shot = ev.get("zero_shot_predictions", [])
    similar_cases = ev.get("similar_cases", [])

    # Determine delta from MedSigLIP delta predictions
    delta_preds = {
        label: pred["confidence"]
        for pred in zero_shot
        if (label := pred.get("label", "")) in _SIGLIP_TO_DELTA
    }
    best_delta = DeltaStatus.UNCERTAIN
    if delta_preds:
        best_delta_label = sorted(delta_preds, key=lambda k: delta_preds[k], reverse=True)[0]
        best_delta = _SIGLIP_TO_DELTA.get(best_delta_label, DeltaStatus.UNCERTAIN)

    # Cross-reference delta with similar cases
    if similar_cases:
        top_case = similar_cases[0]
        case_delta = top_case.get("delta_label", "").lower()
        case_delta_mapping = {
            "improved": DeltaStatus.IMPROVED,
            "stable": DeltaStatus.STABLE,
            "worsened": DeltaStatus.WORSENED,
        }
        case_delta_status = case_delta_mapping.get(case_delta)
        if case_delta_status and best_delta == DeltaStatus.UNCERTAIN:
            best_delta = case_delta_status

    # Collect findings from zero-shot (above threshold, excluding delta labels)
    detected: dict[FindingLabel, float] = {}
    for pred in zero_shot:
        label = pred.get("label", "")
        conf = pred.get("confidence", 0)
        if label.startswith("delta:"):
            continue
        if conf < _ZERO_SHOT_THRESHOLD:
            continue
        finding_label = _SIGLIP_TO_FINDING.get(label)
        if finding_label and finding_label != FindingLabel.NORMAL:
            detected[finding_label] = conf

    # Cross-reference with similar cases: boost confidence for findings
    # confirmed by similar cases, but do NOT add new findings from cases alone.
    # A similar case having a finding doesn't mean this image has it.
    case_findings: set[str] = set()
    for case in similar_cases[:3]:
        for kf in case.get("known_findings", []):
            case_findings.add(kf.lower())

    for label in list(detected):
        if label.value in case_findings:
            detected[label] = min(detected[label] + 0.05, 1.0)

    # If nothing detected above threshold, report normal
    if not detected:
        return [{
            "label": FindingLabel.NORMAL.value,
            "delta": DeltaStatus.STABLE.value,
            "rationale": "No significant abnormalities detected by image analysis models.",
            "evidence_refs": ["current_image"],
            "bounding_box": None,
            "uncertainty": "low",
        }]

    # Build findings with confidence-based uncertainty
    findings = []
    for label, conf in sorted(detected.items(), key=lambda x: -x[1]):
        if conf >= 0.20:
            uncertainty = "medium"
        elif conf >= 0.15:
            uncertainty = "high"
        else:
            uncertainty = "high"

        findings.append({
            "label": label.value,
            "delta": best_delta.value,
            "rationale": f"{label.value.title()} detected (confidence: {conf:.0%}).",
            "evidence_refs": ["current_image", "prior_image"],
            "bounding_box": None,
            "uncertainty": uncertainty,
        })

    return findings


def _build_narrative_prompt(
    findings: list[dict[str, Any]],
    context: dict[str, Any],
) -> str:
    """Build a prompt for the LLM to generate narrative text (Phase 2).

    The LLM's job is to write a natural-language comparison based on the
    evidence-derived findings, NOT to detect findings itself.
    """
    findings_desc = []
    for f in findings:
        findings_desc.append(
            f"- {f['label'].title()}: {f['delta']} (uncertainty: {f['uncertainty']})"
        )

    ev = context.get("retrieval_evidence", {})
    evidence_summary = ev.get("evidence_summary", "No additional evidence available.")

    # Add FHIR context if available
    patient_context = ""
    if "fhir" in context:
        fhir = context["fhir"]
        if isinstance(fhir, dict):
            parts = [f"- {k}: {v}" for k, v in fhir.items()]
            patient_context = "\nPatient context:\n" + "\n".join(parts)

    # Add dictation if available
    dictation = ""
    if "dictation" in context and context["dictation"].get("transcript"):
        dictation = f"\nRadiologist notes: {context['dictation']['transcript']}"

    prompt = (
        "<start_of_turn>user\n"
        "You are a radiology AI assistant. Based on the findings below\n"
        "(detected by image analysis models), write a brief narrative\n"
        "comparison between the prior and current chest X-rays.\n\n"
        "Detected findings:\n"
        f"{chr(10).join(findings_desc)}\n\n"
        f"Evidence: {evidence_summary}\n"
        f"{patient_context}{dictation}\n\n"
        "Write a 2-3 sentence narrative comparison. "
        "Be concise and clinical. Do NOT add findings not listed above.\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    return prompt


def _run_inference_sync(llm, prompt: str, max_tokens: int = 512) -> str:
    """Run inference synchronously."""
    logger.info("Running GGUF inference")

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.9,
        stop=["```", "<end_of_turn>"],
    )

    response = output["choices"][0]["text"]
    logger.info("GGUF inference complete", response_length=len(response))

    return response


async def run_longitudinal_comparison(
    prior_image: dict[str, Any],  # noqa: ARG001 — interface compat, GGUF is text-only
    current_image: dict[str, Any],  # noqa: ARG001 — interface compat, GGUF is text-only
    context: dict[str, Any],
) -> dict[str, Any]:
    """Run longitudinal comparison using evidence + GGUF narrative.

    Phase 1: Build findings from retrieval evidence (MedSigLIP + CXR Foundation).
             These models CAN analyze images; the GGUF model cannot.
    Phase 2: Use the GGUF LLM to generate narrative report text.
    """
    logger.info("Running longitudinal comparison with GGUF (evidence-driven)")

    # Phase 1: Evidence-driven findings
    findings = _build_findings_from_evidence(context)
    logger.info(
        "Evidence-derived findings",
        num_findings=len(findings),
        labels=[f["label"] for f in findings],
    )

    # Phase 2: LLM narrative generation
    try:
        llm = await get_model()
        prompt = _build_narrative_prompt(findings, context)
        max_tokens = getattr(settings, "max_new_tokens", 256)
        raw_text = await asyncio.to_thread(_run_inference_sync, llm, prompt, max_tokens)
        raw_text = raw_text.strip()
    except Exception as e:
        logger.warning("GGUF narrative generation failed, using default", error=str(e))
        # Fallback: generate simple narrative from findings
        parts = []
        for f in findings:
            parts.append(f"{f['label'].title()} is {f['delta']}.")
        raw_text = " ".join(parts)

    # Update rationales with LLM narrative if available
    if raw_text and len(raw_text) > 20:
        for f in findings:
            label = f["label"]
            # Look for the finding in the narrative for a better rationale
            for sentence in re.split(r'[.;]', raw_text):
                if label in sentence.lower():
                    rationale = sentence.strip()
                    if len(rationale) > 10:
                        f["rationale"] = rationale[:500]
                    break

    return {
        "findings": findings,
        "raw_text": raw_text,
    }
