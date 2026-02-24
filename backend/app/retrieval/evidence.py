"""Evidence aggregation service combining CXR Foundation and MedSigLIP retrieval.

This service:
- Orchestrates retrieval from both models
- Aggregates evidence for the audit trail
- Provides uncertainty gating for MedGemma claims
"""

from dataclasses import dataclass, field
from typing import Any

import structlog
from PIL import Image

from app.retrieval.cxr_foundation import (
    CXRFoundationService,
    CXRSimilarCase,
    get_cxr_foundation_service,
)
from app.retrieval.medsiglip import (
    GuidelineMatch,
    MedSigLIPService,
    ZeroShotPrediction,
    get_medsiglip_service,
)

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalEvidence:
    """Aggregated evidence from retrieval models."""

    # From CXR Foundation
    similar_cases: list[CXRSimilarCase] = field(default_factory=list)
    case_retrieval_confidence: float = 0.0

    # From MedSigLIP
    guideline_matches: list[GuidelineMatch] = field(default_factory=list)
    zero_shot_predictions: list[ZeroShotPrediction] = field(default_factory=list)
    classification_agreement: dict[str, Any] = field(default_factory=dict)

    # Combined metrics
    overall_confidence: float = 0.0
    evidence_summary: str = ""


@dataclass
class UncertaintyGate:
    """Result of uncertainty gating for a claim."""

    claim: str
    original_uncertainty: str  # low, medium, high
    gated_uncertainty: str  # Adjusted based on evidence
    supporting_evidence: list[str]
    conflicting_evidence: list[str]
    recommendation: str  # "accept", "flag_for_review", "reject"


class EvidenceAggregator:
    """Aggregates and analyzes evidence from retrieval models.

    Combines:
    - CXR Foundation similar case retrieval
    - MedSigLIP guideline retrieval and zero-shot classification

    Provides uncertainty gating for MedGemma claims.
    """

    def __init__(
        self,
        cxr_service: CXRFoundationService | None = None,
        siglip_service: MedSigLIPService | None = None,
    ) -> None:
        self._cxr_service = cxr_service or get_cxr_foundation_service()
        self._siglip_service = siglip_service or get_medsiglip_service()

    async def gather_evidence(
        self,
        current_image: Image.Image,
        prior_image: Image.Image | None = None,
        finding_filter: list[str] | None = None,
        medgemma_findings: list[str] | None = None,
    ) -> RetrievalEvidence:
        """Gather evidence from all retrieval sources.

        Args:
            current_image: Current CXR image
            prior_image: Optional prior CXR image
            finding_filter: Optional filter for specific finding types
            medgemma_findings: Optional MedGemma findings for agreement analysis

        Returns:
            Aggregated retrieval evidence
        """
        logger.info("Gathering retrieval evidence")

        evidence = RetrievalEvidence()

        # CXR Foundation: Similar case retrieval
        try:
            evidence.similar_cases = await self._cxr_service.retrieve_similar_cases(
                current_image=current_image,
                prior_image=prior_image,
                top_k=5,
                finding_filter=finding_filter,
            )
            evidence.case_retrieval_confidence = (
                self._cxr_service.get_retrieval_confidence(evidence.similar_cases)
            )
            logger.info(
                "CXR Foundation retrieval complete",
                num_cases=len(evidence.similar_cases),
                confidence=evidence.case_retrieval_confidence,
            )
        except Exception as e:
            logger.error("CXR Foundation retrieval failed", error=str(e))

        # MedSigLIP: Zero-shot classification
        try:
            evidence.zero_shot_predictions = (
                await self._siglip_service.zero_shot_classify(
                    image=current_image,
                    include_delta=prior_image is not None,
                )
            )
            logger.info(
                "MedSigLIP zero-shot complete",
                num_predictions=len(evidence.zero_shot_predictions),
            )
        except Exception as e:
            logger.error("MedSigLIP zero-shot failed", error=str(e))

        # MedSigLIP: Guideline retrieval
        try:
            evidence.guideline_matches = (
                await self._siglip_service.retrieve_guidelines(
                    image=current_image,
                    top_k=3,
                )
            )
            logger.info(
                "MedSigLIP guideline retrieval complete",
                num_guidelines=len(evidence.guideline_matches),
            )
        except Exception as e:
            logger.error("MedSigLIP guideline retrieval failed", error=str(e))

        # Classification agreement analysis
        if medgemma_findings:
            evidence.classification_agreement = (
                self._siglip_service.get_classification_agreement(
                    medgemma_findings=medgemma_findings,
                    zero_shot_predictions=evidence.zero_shot_predictions,
                )
            )
            logger.info(
                "Classification agreement analyzed",
                agreement_rate=evidence.classification_agreement.get(
                    "agreement_rate", 0
                ),
            )

        # Calculate overall confidence
        evidence.overall_confidence = self._calculate_overall_confidence(evidence)

        # Generate summary
        evidence.evidence_summary = self._generate_summary(evidence)

        return evidence

    def _calculate_overall_confidence(self, evidence: RetrievalEvidence) -> float:
        """Calculate overall confidence from all evidence sources."""
        scores = []

        # CXR Foundation confidence
        if evidence.similar_cases:
            scores.append(evidence.case_retrieval_confidence)

        # Zero-shot top confidence
        if evidence.zero_shot_predictions:
            top_zs_conf = max(p.confidence for p in evidence.zero_shot_predictions)
            scores.append(top_zs_conf)

        # Guideline match confidence
        if evidence.guideline_matches:
            top_gl_conf = max(g.similarity_score for g in evidence.guideline_matches)
            scores.append(top_gl_conf)

        # Classification agreement
        if evidence.classification_agreement:
            agreement_rate = evidence.classification_agreement.get("agreement_rate", 0)
            scores.append(agreement_rate)

        if not scores:
            return 0.5  # Neutral if no evidence

        return sum(scores) / len(scores)

    def _generate_summary(self, evidence: RetrievalEvidence) -> str:
        """Generate human-readable evidence summary."""
        parts = []

        # Similar cases
        if evidence.similar_cases:
            top_case = evidence.similar_cases[0]
            parts.append(
                f"Top similar case: {', '.join(top_case.known_findings)} "
                f"({top_case.delta_label}) with {top_case.similarity_score:.0%} similarity"
            )

        # Zero-shot findings
        if evidence.zero_shot_predictions:
            top_findings = [
                p.label
                for p in evidence.zero_shot_predictions[:3]
                if p.confidence > 0.1 and not p.label.startswith("delta:")
            ]
            if top_findings:
                parts.append(f"Zero-shot findings: {', '.join(top_findings)}")

        # Guidelines
        if evidence.guideline_matches:
            top_guideline = evidence.guideline_matches[0]
            parts.append(f"Relevant guideline: {top_guideline.title}")

        # Agreement
        if evidence.classification_agreement:
            conf = evidence.classification_agreement.get("confidence", "unknown")
            parts.append(f"Classification agreement: {conf}")

        return "; ".join(parts) if parts else "No retrieval evidence available"

    def gate_claims(
        self,
        claims: list[dict[str, Any]],
        evidence: RetrievalEvidence,
    ) -> list[UncertaintyGate]:
        """Apply uncertainty gating to MedGemma claims.

        Args:
            claims: List of claims from MedGemma (with 'text', 'finding', 'uncertainty')
            evidence: Retrieved evidence

        Returns:
            List of gated claims with adjusted uncertainty
        """
        gated_claims = []

        # Build evidence lookup
        similar_findings = set()
        for case in evidence.similar_cases:
            similar_findings.update(f.lower() for f in case.known_findings)

        zero_shot_findings = {
            p.label.lower(): p.confidence
            for p in evidence.zero_shot_predictions
            if p.confidence > 0.1
        }

        for claim in claims:
            claim_text = claim.get("text", "")
            finding = claim.get("finding", "").lower()
            original_uncertainty = claim.get("uncertainty", "medium")

            supporting = []
            conflicting = []

            # Check similar cases
            if finding in similar_findings:
                supporting.append(f"Similar case supports '{finding}'")
            elif similar_findings and finding not in similar_findings:
                # Only conflict if we have cases but they don't match
                top_similar = list(similar_findings)[:2]
                conflicting.append(
                    f"Similar cases show {', '.join(top_similar)} instead"
                )

            # Check zero-shot
            if finding in zero_shot_findings:
                conf = zero_shot_findings[finding]
                supporting.append(f"Zero-shot confirms '{finding}' ({conf:.0%})")
            elif zero_shot_findings:
                top_zs = sorted(
                    zero_shot_findings.items(), key=lambda x: x[1], reverse=True
                )[:2]
                if top_zs[0][0] != finding:
                    conflicting.append(
                        f"Zero-shot suggests '{top_zs[0][0]}' ({top_zs[0][1]:.0%})"
                    )

            # Determine gated uncertainty
            if len(supporting) >= 2 and not conflicting:
                gated_uncertainty = "low"
                recommendation = "accept"
            elif supporting and not conflicting:
                gated_uncertainty = (
                    "low" if original_uncertainty == "medium" else original_uncertainty
                )
                recommendation = "accept"
            elif conflicting and not supporting:
                gated_uncertainty = "high"
                recommendation = "flag_for_review"
            elif conflicting and supporting:
                gated_uncertainty = "medium"
                recommendation = "flag_for_review"
            else:
                gated_uncertainty = original_uncertainty
                recommendation = "accept" if original_uncertainty != "high" else "flag_for_review"

            gated_claims.append(
                UncertaintyGate(
                    claim=claim_text,
                    original_uncertainty=original_uncertainty,
                    gated_uncertainty=gated_uncertainty,
                    supporting_evidence=supporting,
                    conflicting_evidence=conflicting,
                    recommendation=recommendation,
                )
            )

        return gated_claims

    def format_for_prompt(self, evidence: RetrievalEvidence) -> str:
        """Format evidence as context for MedGemma prompt.

        Args:
            evidence: Retrieved evidence

        Returns:
            Formatted string for prompt inclusion
        """
        sections = []

        # Similar cases section
        if evidence.similar_cases:
            case_lines = ["[Similar Cases from Atlas]"]
            for i, case in enumerate(evidence.similar_cases[:3], 1):
                case_lines.append(
                    f"{i}. {', '.join(case.known_findings)} - {case.delta_label} "
                    f"(similarity: {case.similarity_score:.0%}, source: {case.source})"
                )
            sections.append("\n".join(case_lines))

        # Zero-shot section
        if evidence.zero_shot_predictions:
            findings = [
                f"{p.label} ({p.confidence:.0%})"
                for p in evidence.zero_shot_predictions[:5]
                if p.confidence > 0.05 and not p.label.startswith("delta:")
            ]
            if findings:
                sections.append(
                    "[Zero-Shot Classification]\n" + ", ".join(findings)
                )

        # Guidelines section
        if evidence.guideline_matches:
            guide_lines = ["[Relevant Clinical Guidelines]"]
            for match in evidence.guideline_matches[:2]:
                guide_lines.append(f"- {match.title}: {match.text[:200]}...")
            sections.append("\n".join(guide_lines))

        return "\n\n".join(sections) if sections else ""


# Singleton instance
_evidence_aggregator: EvidenceAggregator | None = None


def get_evidence_aggregator() -> EvidenceAggregator:
    """Get the singleton evidence aggregator instance."""
    global _evidence_aggregator
    if _evidence_aggregator is None:
        _evidence_aggregator = EvidenceAggregator()
    return _evidence_aggregator
