"""Prompt templates for MedGemma inference."""

from typing import Any

# Constrained vocabulary for findings
VALID_FINDINGS = [
    "consolidation",
    "effusion",
    "pneumothorax",
    "cardiomegaly",
    "edema",
    "nodule",
    "atelectasis",
    "normal",
]

# Constrained vocabulary for delta status
VALID_DELTAS = ["improved", "stable", "worsened", "uncertain"]

# Base system prompt for longitudinal comparison
SYSTEM_PROMPT = """You are a radiology AI assistant specialized in chest X-ray analysis.
You are helping generate a DRAFT report that requires clinician verification.
You do NOT make clinical decisions or diagnoses.

Your task is to compare a PRIOR chest X-ray with a CURRENT chest X-ray and identify changes.

IMPORTANT CONSTRAINTS:
- Use ONLY these finding labels: {findings}
- Use ONLY these delta values: {deltas}
- If you cannot confidently identify a finding, use "uncertain" as the delta
- Always cite which image(s) support each finding
- Output your analysis in the specified JSON format
"""

# Instruction template for longitudinal comparison
COMPARISON_INSTRUCTION = """Compare the two chest X-ray images provided.

IMAGE 1 (PRIOR): <start_of_image>
IMAGE 2 (CURRENT): <start_of_image>

Analyze the images and identify:
1. Findings present in BOTH images (and whether they improved, worsened, or are stable)
2. NEW findings in the CURRENT image not present in PRIOR
3. Findings RESOLVED (present in PRIOR but not in CURRENT)

{context_section}

Output your analysis as JSON in this exact format:
{{
  "findings": [
    {{
      "label": "<one of: {findings}>",
      "delta": "<one of: {deltas}>",
      "rationale": "<brief explanation of the finding and change, max 500 chars>",
      "evidence_refs": ["prior_image", "current_image"],
      "bounding_box": {{"x": 0, "y": 0, "w": 100, "h": 100}} or null,
      "uncertainty": "<one of: low, medium, high>"
    }}
  ]
}}

Rules:
- Include ALL relevant findings, even if stable
- Set uncertainty to "high" if you are not confident
- If no abnormalities detected, include a single finding with label "normal" and delta "stable"
- Bounding box coordinates are relative to the image (0-512 range)
- bounding_box can be null if you cannot localize the finding

Begin your analysis:"""


def build_longitudinal_prompt(context: dict[str, Any] | None = None) -> str:
    """
    Build the full prompt for longitudinal CXR comparison.

    Args:
        context: Optional context dict with FHIR data, dictation, etc.

    Returns:
        Complete prompt string
    """
    # Format system prompt
    system = SYSTEM_PROMPT.format(
        findings=", ".join(VALID_FINDINGS),
        deltas=", ".join(VALID_DELTAS),
    )

    # Build context section if available
    context_section = ""
    if context:
        context_parts = []

        if "fhir" in context and context["fhir"]:
            fhir_data = context["fhir"]
            if fhir_data.get("problems"):
                context_parts.append(f"Patient problems: {', '.join(fhir_data['problems'])}")
            if fhir_data.get("medications"):
                context_parts.append(f"Current medications: {', '.join(fhir_data['medications'])}")
            if fhir_data.get("prior_impression"):
                context_parts.append(f"Prior report impression: {fhir_data['prior_impression']}")

        if "dictation" in context and context["dictation"].get("transcript"):
            context_parts.append(f"Clinician notes: {context['dictation']['transcript']}")

        if context_parts:
            context_section = (
                "ADDITIONAL CONTEXT:\n"
                + "\n".join(f"- {p}" for p in context_parts)
                + "\n"
            )

    # Format instruction
    instruction = COMPARISON_INSTRUCTION.format(
        context_section=context_section,
        findings=", ".join(VALID_FINDINGS),
        deltas=", ".join(VALID_DELTAS),
    )

    return f"{system}\n\n{instruction}"


def build_repair_prompt(invalid_json: str, error_message: str) -> str:
    """
    Build a prompt to repair invalid JSON output.

    Args:
        invalid_json: The malformed JSON string
        error_message: The validation error message

    Returns:
        Prompt for repair pass
    """
    return f"""The following JSON output has validation errors. Fix ONLY the structural issues.
Do NOT change the clinical content or findings.

ERROR: {error_message}

INVALID JSON:
{invalid_json}

Valid finding labels: {", ".join(VALID_FINDINGS)}
Valid delta values: {", ".join(VALID_DELTAS)}
Valid uncertainty values: low, medium, high
Valid evidence_refs: prior_image, current_image, fhir_context, guideline, dictation

Output the corrected JSON only, no explanation:"""
