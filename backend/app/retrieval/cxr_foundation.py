"""CXR Foundation service for chest X-ray image embeddings and retrieval.

Uses Google's CXR Foundation precomputed embeddings for retrieval
and BiomedCLIP for generating embeddings from new images.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from PIL import Image

logger = structlog.get_logger(__name__)

# Expected input size for BiomedCLIP
CXR_TARGET_SIZE = (224, 224)

# Finding label mappings from CXR Foundation labels.csv
FINDING_COLUMNS = [
    "AIRSPACE_OPACITY",
    "FRACTURE",
    "PNEUMOTHORAX",
    "CONSOLIDATION",
    "EFFUSION",
    "PULMONARY_EDEMA",
    "ATELECTASIS",
    "CARDIOMEGALY",
]

# Map to standardized finding names
FINDING_MAP = {
    "AIRSPACE_OPACITY": "opacity",
    "FRACTURE": "fracture",
    "PNEUMOTHORAX": "pneumothorax",
    "CONSOLIDATION": "consolidation",
    "EFFUSION": "effusion",
    "PULMONARY_EDEMA": "edema",
    "ATELECTASIS": "atelectasis",
    "CARDIOMEGALY": "cardiomegaly",
}


@dataclass
class CXRSimilarCase:
    """A similar case retrieved from the CXR atlas."""

    case_id: str
    similarity_score: float  # Cosine similarity 0-1
    known_findings: list[str]
    delta_label: str  # "improved", "stable", "worsened", "new", "resolved"
    image_path: str  # Path to atlas image
    source: str  # Dataset source (e.g., "NIH-CXR14", "CheXpert")
    metadata: dict[str, Any] = field(default_factory=dict)


class CXRFoundationService:
    """Service for CXR Foundation embeddings and similar case retrieval.

    Uses:
    - BiomedCLIP for encoding new images
    - Precomputed CXR Foundation embeddings for retrieval
    """

    def __init__(self) -> None:
        self._clip_model = None
        self._clip_preprocess = None
        self._tokenizer = None
        self._model_loaded = False
        self._load_attempted = False
        self._device = "cpu"

        # Atlas data
        self._atlas_embeddings: np.ndarray | None = None
        self._atlas_ids: list[str] = []
        self._atlas_metadata: dict[str, dict] = {}
        self._atlas_loaded = False

    async def load_model(self) -> None:
        """Load BiomedCLIP model for embedding generation."""
        if self._model_loaded or self._load_attempted:
            return

        self._load_attempted = True
        logger.info("Loading BiomedCLIP model for CXR embeddings")

        try:
            import open_clip
            import torch

            self._device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            # Load BiomedCLIP
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            self._tokenizer = open_clip.get_tokenizer(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )

            self._clip_model = self._clip_model.to(self._device)
            self._clip_model.eval()

            self._model_loaded = True
            logger.info("BiomedCLIP model loaded", device=self._device)

        except Exception as e:
            logger.error("Failed to load BiomedCLIP model", error=str(e))
            import traceback
            traceback.print_exc()

    async def load_atlas(self) -> None:
        """Load CXR Foundation precomputed embeddings from HuggingFace."""
        if self._atlas_loaded:
            return

        logger.info("Loading CXR Foundation atlas embeddings")

        try:
            import pandas as pd
            from huggingface_hub import hf_hub_download

            from app.core.config import settings

            token = settings.hf_token
            cache_dir = str(settings.data_dir / "models")

            # Download embeddings and labels
            emb_path = await asyncio.to_thread(
                hf_hub_download,
                "google/cxr-foundation",
                "precomputed_embeddings/embeddings.npz",
                token=token,
                cache_dir=cache_dir,
            )

            labels_path = await asyncio.to_thread(
                hf_hub_download,
                "google/cxr-foundation",
                "precomputed_embeddings/labels.csv",
                token=token,
                cache_dir=cache_dir,
            )

            # Load embeddings
            emb_data = np.load(emb_path)
            logger.info("Loaded embeddings file", num_keys=len(emb_data.keys()))

            # Load labels
            labels_df = pd.read_csv(labels_path)
            logger.info("Loaded labels", num_rows=len(labels_df))

            # Build atlas
            embeddings_list = []
            for _, row in labels_df.iterrows():
                image_id = row["image_id"]

                if image_id not in emb_data:
                    continue

                embedding = emb_data[image_id]
                if embedding.ndim > 1:
                    embedding = embedding.flatten()

                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                embeddings_list.append(embedding)
                self._atlas_ids.append(image_id)

                # Extract findings
                findings = []
                for col in FINDING_COLUMNS:
                    if col in row and row[col] == 1.0:
                        findings.append(FINDING_MAP.get(col, col.lower()))

                if not findings:
                    findings = ["normal"]

                self._atlas_metadata[image_id] = {
                    "case_id": image_id,
                    "patient_id": row.get("patient_id", ""),
                    "known_findings": findings,
                    "delta_label": "unknown",  # Per-image label only; no longitudinal data
                    "source": "NIH-CXR14",
                    "split": row.get("split", ""),
                }

            self._atlas_embeddings = np.array(embeddings_list, dtype=np.float32)
            self._atlas_loaded = True

            logger.info(
                "CXR Foundation atlas loaded",
                num_cases=len(self._atlas_ids),
                embedding_dim=(
                    self._atlas_embeddings.shape[1] if len(self._atlas_embeddings) > 0 else 0
                ),
            )

        except Exception as e:
            logger.error("Failed to load CXR Foundation atlas", error=str(e))
            import traceback
            traceback.print_exc()
            # Atlas unavailable — retrieval will return empty results

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    async def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate embedding for an image using BiomedCLIP.

        Args:
            image: PIL Image

        Returns:
            Numpy array embedding (512-dim for BiomedCLIP)
        """
        if not self._model_loaded:
            await self.load_model()

        if not self._model_loaded:
            raise RuntimeError(
                "BiomedCLIP model not loaded. Install open_clip_torch and ensure "
                "network access for first-time model download."
            )

        import torch

        # Preprocess image
        image_rgb = image.convert("RGB")
        image_tensor = self._clip_preprocess(image_rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            image_features = self._clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten()

        return embedding.astype(np.float32)

    async def retrieve_similar_cases(
        self,
        current_image: Image.Image,
        prior_image: Image.Image | None = None,
        top_k: int = 5,
        finding_filter: list[str] | None = None,
    ) -> list[CXRSimilarCase]:
        """Retrieve similar cases from the atlas.

        Args:
            current_image: Current CXR image
            prior_image: Optional prior CXR for delta context
            top_k: Number of cases to retrieve
            finding_filter: Optional filter for specific finding types

        Returns:
            List of similar cases sorted by similarity
        """
        # Ensure atlas is loaded
        if not self._atlas_loaded:
            await self.load_atlas()

        # Get embedding for current image
        query_embedding = await self.get_embedding(current_image)

        # If prior image provided, create combined embedding
        if prior_image is not None:
            prior_embedding = await self.get_embedding(prior_image)
            # Combine: emphasize current with delta signal
            delta = query_embedding - prior_embedding
            query_embedding = 0.7 * query_embedding + 0.3 * delta
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Calculate similarities using matrix multiplication
        if self._atlas_embeddings is not None and len(self._atlas_embeddings) > 0:
            # Ensure query has correct shape for matmul
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Handle dimension mismatch by projecting or using demo
            atlas_dim = self._atlas_embeddings.shape[1]
            query_dim = query_embedding.shape[1]

            if atlas_dim != query_dim:
                logger.warning(
                    "Embedding dimension mismatch, using cosine on truncated vectors",
                    atlas_dim=atlas_dim,
                    query_dim=query_dim,
                )
                # Use the smaller dimension
                min_dim = min(atlas_dim, query_dim)
                similarities = np.dot(
                    self._atlas_embeddings[:, :min_dim],
                    query_embedding[:, :min_dim].T
                ).flatten()
            else:
                similarities = np.dot(self._atlas_embeddings, query_embedding.T).flatten()
        else:
            # No atlas, return empty
            return []

        # Apply finding filter
        valid_indices = []
        for i, case_id in enumerate(self._atlas_ids):
            if finding_filter:
                case_findings = self._atlas_metadata[case_id].get("known_findings", [])
                if not any(f in case_findings for f in finding_filter):
                    continue
            valid_indices.append(i)

        if not valid_indices:
            valid_indices = list(range(len(self._atlas_ids)))

        # Get top-k from valid indices
        valid_similarities = [(i, similarities[i]) for i in valid_indices]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results: list[CXRSimilarCase] = []
        for idx, score in valid_similarities[:top_k]:
            case_id = self._atlas_ids[idx]
            meta = self._atlas_metadata[case_id]

            # Convert similarity to 0-1 range (cosine can be -1 to 1)
            normalized_score = max(0.0, min(1.0, (score + 1) / 2))

            results.append(
                CXRSimilarCase(
                    case_id=case_id,
                    similarity_score=normalized_score,
                    known_findings=meta.get("known_findings", []),
                    delta_label=meta.get("delta_label", "unknown"),
                    image_path=meta.get("dicom_file", ""),
                    source=meta.get("source", "NIH-CXR14"),
                    metadata={
                        "patient_id": meta.get("patient_id", ""),
                        "split": meta.get("split", ""),
                    },
                )
            )

        logger.info(
            "Similar cases retrieved",
            num_results=len(results),
            top_score=results[0].similarity_score if results else 0,
        )

        return results

    def get_retrieval_confidence(self, similar_cases: list[CXRSimilarCase]) -> float:
        """Calculate overall retrieval confidence."""
        if not similar_cases:
            return 0.0

        scores = [c.similarity_score for c in similar_cases]

        # Factor 1: Top score
        top_score = scores[0]

        # Factor 2: Score gap (larger gap = more confident)
        score_gap = scores[0] - scores[1] if len(scores) > 1 else 0.0

        # Factor 3: Finding agreement
        finding_counts: dict[str, int] = {}
        for case in similar_cases:
            for finding in case.known_findings:
                finding_counts[finding] = finding_counts.get(finding, 0) + 1

        max_agreement = max(finding_counts.values()) / len(similar_cases) if finding_counts else 0.0

        # Combined confidence
        confidence = (0.5 * top_score) + (0.2 * score_gap * 5) + (0.3 * max_agreement)
        return max(0.0, min(1.0, confidence))


# Singleton instance
_cxr_foundation_service: CXRFoundationService | None = None


def get_cxr_foundation_service() -> CXRFoundationService:
    """Get the singleton CXR Foundation service instance."""
    global _cxr_foundation_service
    if _cxr_foundation_service is None:
        _cxr_foundation_service = CXRFoundationService()
    return _cxr_foundation_service
