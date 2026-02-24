"""MedSigLIP service for cross-modal medical image-text embeddings.

Uses Google's MedSigLIP model for:
- Zero-shot finding classification
- Semantic guideline retrieval (image→text)
- Cross-modal evidence grounding
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from PIL import Image

logger = structlog.get_logger(__name__)

# MedSigLIP expected input size (448x448 for MedSigLIP)
MEDSIGLIP_TARGET_SIZE = (448, 448)

# Predefined finding labels for zero-shot classification
FINDING_LABELS = [
    "chest X-ray showing consolidation",
    "chest X-ray showing pleural effusion",
    "chest X-ray showing pneumothorax",
    "chest X-ray showing cardiomegaly",
    "chest X-ray showing pulmonary edema",
    "chest X-ray showing pulmonary nodule",
    "chest X-ray showing atelectasis",
    "normal chest X-ray with no abnormalities",
]

# Delta-specific prompts for longitudinal comparison
DELTA_PROMPTS = {
    "improved": "chest X-ray showing improvement compared to prior",
    "stable": "chest X-ray showing no significant change from prior",
    "worsened": "chest X-ray showing worsening compared to prior",
    "new": "chest X-ray showing new finding not present on prior",
    "resolved": "chest X-ray showing resolution of prior finding",
}


@dataclass
class GuidelineMatch:
    """A guideline snippet matched via cross-modal retrieval."""

    title: str
    source: str
    text: str
    similarity_score: float
    finding_type: str
    citation: str


@dataclass
class ZeroShotPrediction:
    """Zero-shot classification prediction."""

    label: str
    confidence: float
    raw_label: str  # Original prompt used


class MedSigLIPService:
    """Service for MedSigLIP cross-modal embeddings and retrieval.

    Features:
    - 400M parameter vision + 400M parameter text transformers
    - 448x448 image resolution
    - Shared embedding space for image-text matching
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model = None
        self._processor = None
        self._model_loaded = False
        self._load_attempted = False
        self._model_path = model_path or "google/medsiglip-448"
        self._device = "cpu"
        self._guideline_embeddings: dict[str, Any] = {}

    async def load_model(self) -> None:
        """Load MedSigLIP model from Hugging Face."""
        if self._model_loaded or self._load_attempted:
            return

        self._load_attempted = True
        logger.info("Loading MedSigLIP model", model=self._model_path)

        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            from app.core.config import settings

            # Detect device
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            # Load model and processor in thread to avoid blocking
            def _load():
                processor = AutoProcessor.from_pretrained(
                    self._model_path,
                    token=settings.hf_token,
                    trust_remote_code=True,
                )
                model = AutoModel.from_pretrained(
                    self._model_path,
                    token=settings.hf_token,
                    trust_remote_code=True,
                )
                return processor, model

            self._processor, self._model = await asyncio.to_thread(_load)
            self._model = self._model.to(self._device)
            self._model.eval()

            self._model_loaded = True
            logger.info(
                "MedSigLIP model loaded",
                device=self._device,
            )

        except ImportError as e:
            logger.warning(
                "MedSigLIP dependencies not installed",
                error=str(e),
            )
            import traceback
            traceback.print_exc()
        except Exception as e:
            logger.error("Failed to load MedSigLIP model", error=str(e))
            import traceback
            traceback.print_exc()

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image using TensorFlow-compatible method for reproducibility."""
        try:
            from tensorflow.image import resize as tf_resize

            img_array = np.array(image.convert("RGB"))
            resized = tf_resize(
                images=img_array,
                size=list(MEDSIGLIP_TARGET_SIZE),
                method="bilinear",
                antialias=False,
            )
            return Image.fromarray(resized.numpy().astype(np.uint8))
        except ImportError:
            # Fallback to PIL resize
            return image.convert("RGB").resize(
                MEDSIGLIP_TARGET_SIZE, Image.Resampling.LANCZOS
            )

    async def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate image embedding.

        Args:
            image: PIL Image

        Returns:
            Image embedding as numpy array
        """
        if not self._model_loaded:
            await self.load_model()

        if not self._model_loaded:
            raise RuntimeError(
                "MedSigLIP model not loaded. Ensure google/medsiglip-448 is "
                "accessible and HF_TOKEN is configured."
            )

        import torch

        # Resize image
        image_resized = self._resize_image(image)

        # Process
        inputs = self._processor(images=image_resized, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
            # Handle both tensor and model output objects
            if hasattr(outputs, 'pooler_output'):
                embedding = outputs.pooler_output.cpu().numpy()
            elif hasattr(outputs, 'cpu'):
                embedding = outputs.cpu().numpy()
            else:
                embedding = outputs[0].cpu().numpy()

        return embedding.flatten().astype(np.float32)

    async def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding.

        Args:
            text: Text string

        Returns:
            Text embedding as numpy array
        """
        if not self._model_loaded:
            await self.load_model()

        if not self._model_loaded:
            raise RuntimeError(
                "MedSigLIP model not loaded. Cannot generate text embeddings."
            )

        import torch

        # Process
        inputs = self._processor(
            text=[text], padding="max_length", return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)
            # Handle both tensor and model output objects
            if hasattr(outputs, 'pooler_output'):
                embedding = outputs.pooler_output.cpu().numpy()
            elif hasattr(outputs, 'cpu'):
                embedding = outputs.cpu().numpy()
            else:
                embedding = outputs[0].cpu().numpy()

        return embedding.flatten()

    async def compute_similarity(
        self, image: Image.Image, texts: list[str]
    ) -> list[tuple[str, float]]:
        """Compute image-text similarity scores.

        Args:
            image: PIL Image
            texts: List of text prompts

        Returns:
            List of (text, similarity_score) tuples sorted by score
        """
        if not self._model_loaded:
            await self.load_model()

        if not self._model_loaded:
            raise RuntimeError(
                "MedSigLIP model not loaded. Cannot compute image-text similarity."
            )

        import torch

        # Resize image
        image_resized = self._resize_image(image)

        # Process together
        inputs = self._processor(
            text=texts,
            images=image_resized,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        results = [(text, float(prob)) for text, prob in zip(texts, probs, strict=False)]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    async def zero_shot_classify(
        self,
        image: Image.Image,
        labels: list[str] | None = None,
        include_delta: bool = True,
    ) -> list[ZeroShotPrediction]:
        """Zero-shot classification using label prompts.

        Args:
            image: PIL Image
            labels: Custom labels or use predefined FINDING_LABELS
            include_delta: Include delta-specific prompts

        Returns:
            List of predictions sorted by confidence
        """
        if labels is None:
            labels = FINDING_LABELS.copy()

        if include_delta:
            labels.extend(list(DELTA_PROMPTS.values()))

        similarities = await self.compute_similarity(image, labels)

        # Parse results
        predictions: list[ZeroShotPrediction] = []
        for raw_label, confidence in similarities:
            # Extract clean label
            clean_label = raw_label.replace("chest X-ray showing ", "").replace(
                "chest X-ray with ", ""
            )

            # Check if it's a delta label
            for delta_key, delta_prompt in DELTA_PROMPTS.items():
                if raw_label == delta_prompt:
                    clean_label = f"delta:{delta_key}"
                    break

            predictions.append(
                ZeroShotPrediction(
                    label=clean_label,
                    confidence=confidence,
                    raw_label=raw_label,
                )
            )

        return predictions

    async def precompute_guideline_embeddings(
        self, guidelines: list[dict[str, str]]
    ) -> None:
        """Precompute embeddings for guideline texts.

        Args:
            guidelines: List of guideline dicts with 'text' and 'id' keys
        """
        logger.info("Precomputing guideline embeddings", count=len(guidelines))

        for guideline in guidelines:
            guideline_id = guideline.get("id") or guideline.get("title", "")
            text = guideline.get("text", "")

            if text:
                embedding = await self.get_text_embedding(text)
                self._guideline_embeddings[guideline_id] = {
                    "embedding": embedding,
                    "metadata": guideline,
                }

        logger.info(
            "Guideline embeddings computed", count=len(self._guideline_embeddings)
        )

    async def retrieve_guidelines(
        self,
        image: Image.Image,
        guidelines: list[dict[str, Any]] | None = None,
        top_k: int = 3,
    ) -> list[GuidelineMatch]:
        """Retrieve relevant guidelines via cross-modal similarity.

        Args:
            image: Query image
            guidelines: Guidelines to search (or use precomputed)
            top_k: Number of guidelines to retrieve

        Returns:
            List of matched guidelines sorted by relevance
        """
        # Precompute if guidelines provided
        if guidelines and not self._guideline_embeddings:
            await self.precompute_guideline_embeddings(guidelines)

        # Use demo guidelines if none available
        if not self._guideline_embeddings:
            await self._load_demo_guideline_embeddings()

        # Get image embedding
        image_embedding = await self.get_image_embedding(image)
        image_embedding = image_embedding / np.linalg.norm(image_embedding)

        # Calculate similarities
        similarities: list[tuple[str, float, dict]] = []
        for guideline_id, data in self._guideline_embeddings.items():
            guideline_embedding = data["embedding"]
            guideline_embedding = guideline_embedding / np.linalg.norm(
                guideline_embedding
            )

            similarity = float(np.dot(image_embedding, guideline_embedding))
            similarities.append((guideline_id, similarity, data["metadata"]))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results: list[GuidelineMatch] = []
        for guideline_id, score, meta in similarities[:top_k]:
            results.append(
                GuidelineMatch(
                    title=meta.get("title", guideline_id),
                    source=meta.get("source", "Clinical Guidelines"),
                    text=meta.get("text", ""),
                    similarity_score=max(0.0, min(1.0, (score + 1) / 2)),
                    finding_type=meta.get("finding_type", "general"),
                    citation=meta.get("citation", ""),
                )
            )

        logger.info(
            "Guidelines retrieved",
            num_results=len(results),
            top_score=results[0].similarity_score if results else 0,
        )

        return results

    async def _load_demo_guideline_embeddings(self) -> None:
        """Load guideline embeddings — currently a no-op.

        The hardcoded guideline database was removed because the text was
        paraphrased, not directly quoted from published guidelines.
        A production system should load real, licensed guideline texts here.
        """
        logger.info("Guideline embeddings skipped (no guideline database configured)")

    def get_classification_agreement(
        self,
        medgemma_findings: list[str],
        zero_shot_predictions: list[ZeroShotPrediction],
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        """Check agreement between MedGemma and zero-shot predictions.

        Args:
            medgemma_findings: Finding labels from MedGemma
            zero_shot_predictions: Zero-shot predictions from MedSigLIP
            threshold: Minimum confidence for zero-shot to count

        Returns:
            Agreement analysis dict
        """
        # Normalize MedGemma findings
        medgemma_set = {f.lower() for f in medgemma_findings}

        # Extract confident zero-shot findings
        zero_shot_findings: dict[str, float] = {}
        for pred in zero_shot_predictions:
            if pred.confidence >= threshold and not pred.label.startswith("delta:"):
                # Normalize label
                normalized = pred.label.lower().replace("no abnormalities", "normal")
                zero_shot_findings[normalized] = pred.confidence

        # Calculate agreement
        agreed = []
        medgemma_only = []
        zero_shot_only = []

        for finding in medgemma_set:
            if finding in zero_shot_findings:
                agreed.append(
                    {"finding": finding, "confidence": zero_shot_findings[finding]}
                )
            else:
                medgemma_only.append(finding)

        for finding, conf in zero_shot_findings.items():
            if finding not in medgemma_set:
                zero_shot_only.append({"finding": finding, "confidence": conf})

        agreement_rate = len(agreed) / max(len(medgemma_set), 1)

        return {
            "agreement_rate": agreement_rate,
            "agreed_findings": agreed,
            "medgemma_only": medgemma_only,
            "zero_shot_only": zero_shot_only,
            "confidence": (
                "high" if agreement_rate > 0.7
                else "medium" if agreement_rate > 0.4
                else "low"
            ),
        }


# Singleton instance
_medsiglip_service: MedSigLIPService | None = None


def get_medsiglip_service() -> MedSigLIPService:
    """Get the singleton MedSigLIP service instance."""
    global _medsiglip_service
    if _medsiglip_service is None:
        _medsiglip_service = MedSigLIPService()
    return _medsiglip_service
