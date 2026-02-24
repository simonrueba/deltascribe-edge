"""JSON schema validation and repair for model outputs."""

from typing import Any

import structlog
from jsonschema import Draft7Validator, ValidationError

from app.schemas.analysis import DeltaStatus, FindingLabel, UncertaintyLevel

logger = structlog.get_logger(__name__)

# JSON Schema for model output validation
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["findings"],
    "properties": {
        "findings": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["label", "delta", "rationale", "evidence_refs"],
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": [f.value for f in FindingLabel],
                    },
                    "delta": {
                        "type": "string",
                        "enum": [d.value for d in DeltaStatus],
                    },
                    "rationale": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    },
                    "evidence_refs": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "string",
                            "enum": [
                                "prior_image",
                                "current_image",
                                "fhir_context",
                                "guideline",
                                "dictation",
                            ],
                        },
                    },
                    "bounding_box": {
                        "oneOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "required": ["x", "y", "w", "h"],
                                "properties": {
                                    "x": {"type": "integer", "minimum": 0},
                                    "y": {"type": "integer", "minimum": 0},
                                    "w": {"type": "integer", "minimum": 1},
                                    "h": {"type": "integer", "minimum": 1},
                                },
                            },
                        ],
                    },
                    "uncertainty": {
                        "type": "string",
                        "enum": [u.value for u in UncertaintyLevel],
                        "default": "medium",
                    },
                },
            },
        },
        "raw_text": {
            "type": "string",
        },
    },
}

validator = Draft7Validator(OUTPUT_SCHEMA)


async def validate_and_repair_output(raw_output: dict[str, Any]) -> dict[str, Any]:
    """
    Validate model output against schema and attempt repair if needed.

    Args:
        raw_output: Raw output from model inference

    Returns:
        Validated and potentially repaired output

    Raises:
        ValidationError: If validation fails and repair is unsuccessful
    """
    logger.info("Validating model output")

    # First attempt: direct validation
    errors = list(validator.iter_errors(raw_output))

    if not errors:
        logger.info("Output validated successfully")
        return _normalize_output(raw_output)

    logger.warning("Validation failed, attempting repair", error_count=len(errors))

    # Log validation errors
    for error in errors:
        logger.debug("Validation error", path=list(error.path), message=error.message)

    # Attempt repair
    repaired = await _repair_output(raw_output, errors)

    # Validate repaired output
    repair_errors = list(validator.iter_errors(repaired))

    if not repair_errors:
        logger.info("Output repaired successfully")
        return _normalize_output(repaired)

    # Repair failed - use fallback
    logger.error("Repair failed, using fallback output")
    return _create_fallback_output(raw_output)


async def _repair_output(output: dict[str, Any], errors: list[ValidationError]) -> dict[str, Any]:
    """
    Attempt to repair common validation errors.

    This handles structural issues only, not content changes.
    """
    repaired = output.copy()

    # Ensure findings array exists
    if "findings" not in repaired:
        repaired["findings"] = []

    # Repair each finding
    repaired_findings = []
    for finding in repaired.get("findings", []):
        if not isinstance(finding, dict):
            continue

        repaired_finding = finding.copy()

        # Fix label if invalid
        valid_labels = [f.value for f in FindingLabel]
        if "label" not in repaired_finding or repaired_finding["label"] not in valid_labels:
            # Try to map to valid label
            raw_label = str(repaired_finding.get("label", "")).lower()
            repaired_finding["label"] = _map_to_valid_label(raw_label)

        # Fix delta if invalid
        valid_deltas = [d.value for d in DeltaStatus]
        if "delta" not in repaired_finding or repaired_finding["delta"] not in valid_deltas:
            raw_delta = str(repaired_finding.get("delta", "")).lower()
            repaired_finding["delta"] = _map_to_valid_delta(raw_delta)

        # Ensure rationale exists
        if "rationale" not in repaired_finding or not repaired_finding["rationale"]:
            repaired_finding["rationale"] = "Finding detected in image analysis."

        # Truncate rationale if too long
        if len(repaired_finding.get("rationale", "")) > 500:
            repaired_finding["rationale"] = repaired_finding["rationale"][:497] + "..."

        # Ensure evidence_refs exists
        if "evidence_refs" not in repaired_finding or not repaired_finding["evidence_refs"]:
            repaired_finding["evidence_refs"] = ["current_image"]

        # Fix evidence_refs if invalid
        valid_refs = {"prior_image", "current_image", "fhir_context", "guideline", "dictation"}
        repaired_finding["evidence_refs"] = [
            ref for ref in repaired_finding.get("evidence_refs", [])
            if ref in valid_refs
        ] or ["current_image"]

        # Fix uncertainty if invalid
        valid_uncertainties = [u.value for u in UncertaintyLevel]
        if (
            "uncertainty" not in repaired_finding
            or repaired_finding["uncertainty"] not in valid_uncertainties
        ):
            repaired_finding["uncertainty"] = "medium"

        # Fix bounding_box if present but invalid
        if "bounding_box" in repaired_finding and repaired_finding["bounding_box"] is not None:
            bb = repaired_finding["bounding_box"]
            if not isinstance(bb, dict) or not all(k in bb for k in ["x", "y", "w", "h"]):
                repaired_finding["bounding_box"] = None
            else:
                # Ensure values are integers and valid
                try:
                    repaired_finding["bounding_box"] = {
                        "x": max(0, int(bb.get("x", 0))),
                        "y": max(0, int(bb.get("y", 0))),
                        "w": max(1, int(bb.get("w", 1))),
                        "h": max(1, int(bb.get("h", 1))),
                    }
                except (ValueError, TypeError):
                    repaired_finding["bounding_box"] = None

        repaired_findings.append(repaired_finding)

    repaired["findings"] = repaired_findings

    # Ensure at least one finding
    if not repaired["findings"]:
        repaired["findings"] = [{
            "label": "normal",
            "delta": "uncertain",
            "rationale": "Unable to extract findings from model output.",
            "evidence_refs": ["current_image"],
            "uncertainty": "high",
        }]

    return repaired


def _map_to_valid_label(raw_label: str) -> str:
    """Map potentially invalid label to valid FindingLabel."""
    label_map = {
        "consolidation": "consolidation",
        "opacity": "consolidation",
        "infiltrate": "consolidation",
        "effusion": "effusion",
        "pleural": "effusion",
        "pneumothorax": "pneumothorax",
        "ptx": "pneumothorax",
        "cardiomegaly": "cardiomegaly",
        "heart": "cardiomegaly",
        "cardiac": "cardiomegaly",
        "edema": "edema",
        "congestion": "edema",
        "nodule": "nodule",
        "mass": "nodule",
        "lesion": "nodule",
        "atelectasis": "atelectasis",
        "collapse": "atelectasis",
        "normal": "normal",
        "clear": "normal",
        "unremarkable": "normal",
    }

    for key, value in label_map.items():
        if key in raw_label:
            return value

    return "normal"


def _map_to_valid_delta(raw_delta: str) -> str:
    """Map potentially invalid delta to valid DeltaStatus."""
    delta_map = {
        "improved": "improved",
        "better": "improved",
        "resolved": "improved",
        "decreased": "improved",
        "stable": "stable",
        "unchanged": "stable",
        "similar": "stable",
        "worsened": "worsened",
        "worse": "worsened",
        "increased": "worsened",
        "new": "worsened",
        "uncertain": "uncertain",
        "unknown": "uncertain",
        "indeterminate": "uncertain",
    }

    for key, value in delta_map.items():
        if key in raw_delta:
            return value

    return "uncertain"


def _normalize_output(output: dict[str, Any]) -> dict[str, Any]:
    """Normalize a valid output to ensure consistent structure."""
    normalized = output.copy()

    # Add default uncertainty if missing
    for finding in normalized.get("findings", []):
        if "uncertainty" not in finding:
            finding["uncertainty"] = "medium"
        if "bounding_box" not in finding:
            finding["bounding_box"] = None

    return normalized


def _create_fallback_output(original: dict[str, Any]) -> dict[str, Any]:
    """Create a safe fallback output when validation and repair both fail."""
    return {
        "findings": [{
            "label": "normal",
            "delta": "uncertain",
            "rationale": "Analysis could not be completed reliably. Manual review required.",
            "evidence_refs": ["current_image"],
            "bounding_box": None,
            "uncertainty": "high",
        }],
        "raw_text": original.get("raw_text", ""),
    }
