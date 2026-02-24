"""Audit trail builder for claims-to-evidence mapping."""

from app.schemas.analysis import (
    AuditEntry,
    EvidenceSourceType,
    Finding,
)


def build_audit_trail(
    findings: list[Finding],
) -> list[AuditEntry]:
    """
    Build an audit trail mapping each claim to its evidence source.

    Args:
        findings: List of validated findings

    Returns:
        List of audit entries for transparency
    """
    audit_entries = []

    for finding in findings:
        # Create main claim for the finding
        claim_text = _build_claim_text(finding)

        # Determine primary source
        primary_source = _get_primary_source(finding)

        # Add audit entry for the finding
        audit_entries.append(
            AuditEntry(
                claim=claim_text,
                source_type=primary_source,
                source_ref=_get_source_ref(primary_source, finding),
                uncertainty=finding.uncertainty,
            )
        )

        # If delta involves comparison, add comparison claim
        if finding.delta.value in ["improved", "worsened"]:
            comparison_claim = _build_comparison_claim(finding)
            audit_entries.append(
                AuditEntry(
                    claim=comparison_claim,
                    source_type=EvidenceSourceType.PRIOR_IMAGE,
                    source_ref="prior_image:comparison",
                    uncertainty=finding.uncertainty,
                )
            )

    return audit_entries


def _build_claim_text(finding: Finding) -> str:
    """Build a human-readable claim from a finding."""
    label = finding.label.value.title()
    delta = finding.delta.value

    templates = {
        "improved": f"{label} has improved compared to prior study",
        "worsened": f"{label} has worsened or is newly present",
        "stable": f"{label} is unchanged from prior study",
    }
    return templates.get(delta, f"{label} is present (comparison uncertain)")


def _build_comparison_claim(finding: Finding) -> str:
    """Build a comparison-specific claim."""
    label = finding.label.value.title()

    if finding.delta.value == "improved":
        return f"Prior study showed more pronounced {label.lower()}"
    else:
        return f"Prior study showed less pronounced or absent {label.lower()}"


def _get_primary_source(finding: Finding) -> EvidenceSourceType:
    """Determine the primary evidence source for a finding."""
    refs = finding.evidence_refs

    # Prioritize current image
    if EvidenceSourceType.CURRENT_IMAGE in refs:
        return EvidenceSourceType.CURRENT_IMAGE

    # Then prior image
    if EvidenceSourceType.PRIOR_IMAGE in refs:
        return EvidenceSourceType.PRIOR_IMAGE

    # Then other sources
    if refs:
        return refs[0]

    return EvidenceSourceType.CURRENT_IMAGE


def _get_source_ref(source_type: EvidenceSourceType, finding: Finding) -> str:
    """Get a specific reference string for the source."""
    if source_type == EvidenceSourceType.CURRENT_IMAGE:
        if finding.bounding_box:
            bb = finding.bounding_box
            return f"current_image:region:{bb.x},{bb.y},{bb.w},{bb.h}"
        return "current_image:full"

    elif source_type == EvidenceSourceType.PRIOR_IMAGE:
        return "prior_image:comparison"

    elif source_type == EvidenceSourceType.FHIR_CONTEXT:
        return "fhir:patient_context"

    elif source_type == EvidenceSourceType.GUIDELINE:
        return "guideline:clinical_reference"

    elif source_type == EvidenceSourceType.DICTATION:
        return "dictation:clinician_notes"

    return f"{source_type.value}:unspecified"
