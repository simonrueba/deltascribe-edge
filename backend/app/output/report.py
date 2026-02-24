"""Draft radiology report builder."""

from app.schemas.analysis import DeltaStatus, Finding, FindingLabel


def build_draft_report(findings: list[Finding], section: str) -> str:
    """
    Build a section of the draft radiology report.

    Args:
        findings: List of validated findings
        section: Either "findings" or "impression"

    Returns:
        Report section text
    """
    if section == "findings":
        return _build_findings_section(findings)
    elif section == "impression":
        return _build_impression_section(findings)
    else:
        raise ValueError(f"Unknown section: {section}")


def _build_findings_section(findings: list[Finding]) -> str:
    """Build the FINDINGS section of the report."""
    if not findings:
        return "No findings available."

    lines = ["COMPARISON: Prior chest radiograph.", ""]

    # Group findings by anatomical region (simplified)
    cardiac_findings = [f for f in findings if f.label in [FindingLabel.CARDIOMEGALY]]
    pulmonary_findings = [
        f for f in findings
        if f.label in [
            FindingLabel.CONSOLIDATION,
            FindingLabel.EDEMA,
            FindingLabel.ATELECTASIS,
            FindingLabel.NODULE,
        ]
    ]
    pleural_findings = [
        f for f in findings
        if f.label in [FindingLabel.EFFUSION, FindingLabel.PNEUMOTHORAX]
    ]
    normal_findings = [f for f in findings if f.label == FindingLabel.NORMAL]

    # Cardiac
    lines.append("HEART AND MEDIASTINUM:")
    if cardiac_findings:
        for f in cardiac_findings:
            lines.append(f"- {_format_finding(f)}")
    else:
        lines.append("- Cardiac silhouette is within normal limits.")
    lines.append("")

    # Lungs
    lines.append("LUNGS AND AIRWAYS:")
    if pulmonary_findings:
        for f in pulmonary_findings:
            lines.append(f"- {_format_finding(f)}")
    else:
        lines.append("- Lungs are clear without focal consolidation, effusion, or pneumothorax.")
    lines.append("")

    # Pleura
    if pleural_findings:
        lines.append("PLEURA:")
        for f in pleural_findings:
            lines.append(f"- {_format_finding(f)}")
        lines.append("")

    # Overall normal
    if normal_findings and len(findings) == 1:
        lines = [
            "COMPARISON: Prior chest radiograph.",
            "",
            "FINDINGS:",
            "Heart size is normal. Lungs are clear without focal consolidation.",
            "No pleural effusion or pneumothorax. No acute osseous abnormality.",
        ]

    return "\n".join(lines)


def _build_impression_section(findings: list[Finding]) -> str:
    """Build the IMPRESSION section of the report."""
    if not findings:
        return "Unable to generate impression."

    # Check for all normal
    if all(f.label == FindingLabel.NORMAL for f in findings):
        return "No acute cardiopulmonary abnormality. No significant change from prior examination."

    impression_parts = []

    # Prioritize worsened/new findings
    worsened = [f for f in findings if f.delta == DeltaStatus.WORSENED]
    improved = [f for f in findings if f.delta == DeltaStatus.IMPROVED]
    stable_abnormal = [
        f for f in findings
        if f.delta == DeltaStatus.STABLE and f.label != FindingLabel.NORMAL
    ]

    # Add numbered impressions
    impression_num = 1

    for f in worsened:
        status = "new" if "new" in f.rationale.lower() else "worsened"
        impression_parts.append(f"{impression_num}. {f.label.value.title()}, {status}.")
        impression_num += 1

    for f in improved:
        impression_parts.append(f"{impression_num}. {f.label.value.title()}, improved from prior.")
        impression_num += 1

    for f in stable_abnormal:
        impression_parts.append(f"{impression_num}. {f.label.value.title()}, stable.")
        impression_num += 1

    if not impression_parts:
        return "No acute findings. Stable examination."

    return "\n".join(impression_parts)


def _format_finding(finding: Finding) -> str:
    """Format a single finding for the report."""
    label = finding.label.value.title()

    # Build the sentence
    if finding.delta == DeltaStatus.IMPROVED:
        if "resolved" in finding.rationale.lower():
            return f"Previously noted {label.lower()} has resolved."
        return f"{label} shows interval improvement compared to prior."
    elif finding.delta == DeltaStatus.WORSENED:
        if "new" in finding.rationale.lower():
            return f"New {label.lower()} identified."
        return f"{label} has worsened compared to prior examination."
    elif finding.delta == DeltaStatus.STABLE:
        return f"{label} is unchanged from prior."
    else:
        return f"{label} noted; comparison with prior is uncertain."


