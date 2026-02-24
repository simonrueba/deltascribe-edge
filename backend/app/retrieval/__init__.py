"""Retrieval module for evidence-based medical imaging analysis.

This module provides:
- CXR Foundation: Image-image retrieval for similar chest X-ray cases
- MedSigLIP: Cross-modal retrieval for guidelines and zero-shot classification
"""

from app.retrieval.cxr_foundation import CXRFoundationService, CXRSimilarCase
from app.retrieval.evidence import EvidenceAggregator, RetrievalEvidence
from app.retrieval.medsiglip import GuidelineMatch, MedSigLIPService, ZeroShotPrediction

__all__ = [
    "CXRFoundationService",
    "CXRSimilarCase",
    "MedSigLIPService",
    "GuidelineMatch",
    "ZeroShotPrediction",
    "EvidenceAggregator",
    "RetrievalEvidence",
]
