"""MedGemma inference for longitudinal CXR comparison."""

import asyncio
import json
from typing import Any

import structlog

from app.core.config import settings
from app.inference.prompts import build_longitudinal_prompt
from app.schemas.analysis import DeltaStatus, FindingLabel

logger = structlog.get_logger(__name__)

# Global model instance (lazy loaded)
_model = None
_processor = None
_load_lock = asyncio.Lock()


def _load_model_sync():
    """Synchronously load the MedGemma model (runs in thread pool)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    # Determine device
    if settings.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.device

    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(settings.torch_dtype, torch.float16)

    logger.info(
        "Loading MedGemma model in thread pool",
        model_name=settings.model_name,
        device=device,
    )

    # Load model - this is the blocking operation
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
        token=settings.hf_token,
    )

    if device == "cpu":
        model = model.to(device)

    # Load processor
    processor = AutoProcessor.from_pretrained(
        settings.model_name,
        trust_remote_code=True,
        token=settings.hf_token,
    )

    logger.info("MedGemma model loaded", device=device, dtype=str(torch_dtype))

    return model, processor


async def get_model():
    """Lazy load the MedGemma model (non-blocking)."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    # Use lock to prevent multiple concurrent load attempts
    async with _load_lock:
        # Double-check after acquiring lock
        if _model is not None:
            return _model, _processor

        logger.info("Loading MedGemma model", model_name=settings.model_name)

        try:
            # Run the blocking model loading in a thread pool to avoid blocking the event loop
            _model, _processor = await asyncio.to_thread(_load_model_sync)
            return _model, _processor

        except Exception as e:
            logger.error("Failed to load MedGemma model", error=str(e))
            raise


def _run_inference_sync(model, processor, prompt, prior_pil, current_pil) -> str:
    """Synchronously run model inference (runs in thread pool)."""
    import torch

    logger.info("Running MedGemma inference in thread pool")

    # Process inputs
    inputs = processor(
        text=prompt,
        images=[prior_pil, current_pil],
        return_tensors="pt",
    )

    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate - this is CPU-bound and blocking
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            do_sample=settings.temperature > 0,
        )

    # Decode output
    response = processor.decode(outputs[0], skip_special_tokens=True)
    logger.info("MedGemma inference complete", response_length=len(response))

    return response


async def run_longitudinal_comparison(
    prior_image: dict[str, Any],
    current_image: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Run longitudinal comparison between prior and current CXR images.

    Args:
        prior_image: Preprocessed prior image dict
        current_image: Preprocessed current image dict
        context: Additional context (FHIR, dictation)

    Returns:
        Raw model output as structured dict
    """
    logger.info("Running longitudinal comparison")

    try:
        model, processor = await get_model()
    except Exception as e:
        logger.error("Model not available", error=str(e))
        raise RuntimeError(
            "MedGemma model failed to load. Ensure model weights are downloaded "
            "and MODEL_PATH is configured. See README for setup instructions."
        ) from e

    # Build the prompt
    prompt = build_longitudinal_prompt(context)

    # Prepare images
    prior_pil = prior_image.get("pil_image")
    current_pil = current_image.get("pil_image")

    if prior_pil is None or current_pil is None:
        raise ValueError("PIL images not available in preprocessed data")

    # Run inference in thread pool to avoid blocking the event loop
    response = await asyncio.to_thread(
        _run_inference_sync, model, processor, prompt, prior_pil, current_pil
    )

    # Parse structured output from response
    return parse_model_response(response)




def parse_model_response(response: str) -> dict[str, Any]:
    """
    Parse the model's text response into structured output.

    The model is instructed to output JSON, but may include surrounding text.
    """
    logger.debug("Parsing model response", response_length=len(response))

    # Try to extract JSON from the response
    json_start = response.find("{")
    json_end = response.rfind("}") + 1

    if json_start != -1 and json_end > json_start:
        json_str = response[json_start:json_end]
        try:
            parsed = json.loads(json_str)
            logger.info("Successfully parsed JSON from response")
            return parsed
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON from response", error=str(e))

    # If JSON parsing fails, try to extract structured data from text
    return _extract_findings_from_text(response)


def _extract_findings_from_text(response: str) -> dict[str, Any]:
    """
    Extract findings from free-text response when JSON parsing fails.

    This is a fallback that creates a basic structure with estimated bounding boxes.
    """
    logger.info("Extracting findings from free text")

    # Look for finding keywords in the response
    findings = []
    response_lower = response.lower()

    # Check for each finding type
    finding_keywords = {
        FindingLabel.CONSOLIDATION: ["consolidation", "opacity", "infiltrate"],
        FindingLabel.EFFUSION: ["effusion", "pleural fluid"],
        FindingLabel.PNEUMOTHORAX: ["pneumothorax"],
        FindingLabel.CARDIOMEGALY: ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
        FindingLabel.EDEMA: ["edema", "pulmonary congestion"],
        FindingLabel.NODULE: ["nodule", "mass", "lesion"],
        FindingLabel.ATELECTASIS: ["atelectasis", "collapse"],
    }

    delta_keywords = {
        DeltaStatus.IMPROVED: ["improved", "better", "resolved", "decreased", "improving"],
        DeltaStatus.WORSENED: ["worsened", "worse", "increased", "progressed", "new"],
        DeltaStatus.STABLE: ["stable", "unchanged", "no change", "similar"],
    }

    for label, keywords in finding_keywords.items():
        for keyword in keywords:
            if keyword in response_lower:
                # Determine delta status
                delta = DeltaStatus.UNCERTAIN
                for delta_status, delta_words in delta_keywords.items():
                    for word in delta_words:
                        if word in response_lower:
                            delta = delta_status
                            break
                    if delta != DeltaStatus.UNCERTAIN:
                        break

                findings.append({
                    "label": label.value,
                    "delta": delta.value,
                    "rationale": f"Finding detected in model response: {keyword}",
                    "evidence_refs": ["current_image", "prior_image"],
                    "bounding_box": None,
                    "uncertainty": "high",
                })
                break

    # If no findings detected, add normal
    if not findings:
        findings.append({
            "label": FindingLabel.NORMAL.value,
            "delta": DeltaStatus.STABLE.value,
            "rationale": "No significant abnormalities detected.",
            "evidence_refs": ["current_image"],
            "bounding_box": None,
            "uncertainty": "medium",
        })

    return {
        "findings": findings,
        "raw_text": response,
    }
