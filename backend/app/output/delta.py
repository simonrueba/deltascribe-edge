"""Delta summary builder."""

from app.schemas.analysis import DeltaStatus, Finding


def build_delta_summary(findings: list[Finding]) -> str:
    """
    Build a bullet-point delta summary from findings.

    Focuses on changes (improved/worsened), with stable findings mentioned briefly.

    Args:
        findings: List of validated findings

    Returns:
        Markdown-formatted delta summary
    """
    if not findings:
        return "- No findings to report."

    # Separate findings by delta status
    improved = [f for f in findings if f.delta == DeltaStatus.IMPROVED]
    worsened = [f for f in findings if f.delta == DeltaStatus.WORSENED]
    stable = [f for f in findings if f.delta == DeltaStatus.STABLE]
    uncertain = [f for f in findings if f.delta == DeltaStatus.UNCERTAIN]

    lines = []

    # Worsened first (most important)
    if worsened:
        lines.append("**Worsened/New findings:**")
        for f in worsened:
            tag = " [uncertain]" if f.uncertainty.value == "high" else ""
            summary = _summarize_rationale(f.rationale)
            lines.append(f"- {f.label.value.title()}: {summary}{tag}")
        lines.append("")

    # Improved
    if improved:
        lines.append("**Improved findings:**")
        for f in improved:
            tag = " [uncertain]" if f.uncertainty.value == "high" else ""
            summary = _summarize_rationale(f.rationale)
            lines.append(f"- {f.label.value.title()}: {summary}{tag}")
        lines.append("")

    # Stable (brief)
    if stable:
        stable_labels = [f.label.value.title() for f in stable if f.label.value != "normal"]
        if stable_labels:
            lines.append(f"**Stable:** {', '.join(stable_labels)}")
            lines.append("")

    # Uncertain
    if uncertain:
        lines.append("**Requires verification:**")
        for f in uncertain:
            lines.append(f"- {f.label.value.title()}: {_summarize_rationale(f.rationale)}")
        lines.append("")

    # Check if all normal and stable
    if all(f.label.value == "normal" and f.delta == DeltaStatus.STABLE for f in findings):
        lines = ["- No significant change from prior examination."]

    return "\n".join(lines).strip()


def _summarize_rationale(rationale: str, max_length: int = 200) -> str:
    """Summarize a rationale string to a maximum length."""
    if len(rationale) <= max_length:
        return rationale

    # Find a good break point
    truncated = rationale[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]

    return truncated.rstrip(".,") + "..."
