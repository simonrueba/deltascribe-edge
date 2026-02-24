"""Image preprocessing for CXR analysis."""

import base64
import io
from datetime import datetime
from typing import Any

import structlog
from PIL import Image

logger = structlog.get_logger(__name__)

# MedGemma expected input size
TARGET_SIZE = (512, 512)

# DICOM magic bytes
DICOM_MAGIC = b"DICM"
DICOM_MAGIC_OFFSET = 128


def is_dicom(data: bytes) -> bool:
    """Check if data is a DICOM file by looking for magic bytes."""
    if len(data) < DICOM_MAGIC_OFFSET + 4:
        return False
    return data[DICOM_MAGIC_OFFSET : DICOM_MAGIC_OFFSET + 4] == DICOM_MAGIC


def extract_dicom_image(data: bytes) -> tuple[Image.Image, dict[str, Any]]:
    """
    Extract pixel data and metadata from DICOM file.

    Args:
        data: Raw DICOM file bytes

    Returns:
        Tuple of (PIL Image, metadata dict)
    """
    try:
        import numpy as np
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM support. Install with: pip install pydicom"
        ) from None

    # Read DICOM from bytes
    ds = pydicom.dcmread(io.BytesIO(data))

    # Extract metadata
    metadata: dict[str, Any] = {}

    # Patient info
    metadata["patient_id"] = str(getattr(ds, "PatientID", "unknown"))
    metadata["patient_name"] = str(getattr(ds, "PatientName", "unknown"))

    # Study info
    metadata["study_date"] = str(getattr(ds, "StudyDate", ""))
    metadata["study_description"] = str(getattr(ds, "StudyDescription", ""))
    metadata["series_description"] = str(getattr(ds, "SeriesDescription", ""))

    # Image info
    metadata["modality"] = str(getattr(ds, "Modality", ""))
    metadata["rows"] = int(getattr(ds, "Rows", 0))
    metadata["columns"] = int(getattr(ds, "Columns", 0))
    metadata["bits_stored"] = int(getattr(ds, "BitsStored", 0))

    # Acquisition date/time
    acq_date = getattr(ds, "AcquisitionDate", None)
    acq_time = getattr(ds, "AcquisitionTime", None)
    if acq_date:
        try:
            if acq_time:
                metadata["acquisition_datetime"] = datetime.strptime(
                    f"{acq_date}{acq_time[:6]}", "%Y%m%d%H%M%S"
                ).isoformat()
            else:
                metadata["acquisition_datetime"] = datetime.strptime(
                    acq_date, "%Y%m%d"
                ).isoformat()
        except ValueError:
            metadata["acquisition_datetime"] = acq_date

    logger.info(
        "DICOM metadata extracted",
        modality=metadata.get("modality"),
        study_date=metadata.get("study_date"),
    )

    # Extract pixel array
    pixel_array = ds.pixel_array

    # Apply VOI LUT (window/level) if available
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        pixel_array = apply_voi_lut(pixel_array, ds)

    # Normalize to 8-bit range
    pixel_array = pixel_array.astype(np.float32)
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
    pixel_array = (pixel_array * 255).astype(np.uint8)

    # Handle photometric interpretation (invert if needed)
    photometric = str(getattr(ds, "PhotometricInterpretation", ""))
    if photometric == "MONOCHROME1":
        # MONOCHROME1 means higher values are darker, so invert
        pixel_array = 255 - pixel_array

    # Convert to PIL Image
    image = Image.fromarray(pixel_array, mode="L")

    return image, metadata


async def preprocess_cxr(image_b64: str, label: str = "image") -> dict[str, Any]:
    """
    Preprocess a chest X-ray image for model input.

    Supports PNG, JPEG, and DICOM formats.

    Args:
        image_b64: Base64-encoded image data
        label: Label for logging (prior/current)

    Returns:
        Dict containing processed image data and metadata
    """
    logger.info("Preprocessing image", label=label)

    # Decode base64
    image_data = base64.b64decode(image_b64)

    # Check if DICOM
    dicom_metadata = None
    if is_dicom(image_data):
        logger.info("DICOM file detected", label=label)
        image, dicom_metadata = extract_dicom_image(image_data)
    else:
        image = Image.open(io.BytesIO(image_data))

    # Store original info
    original_size = image.size
    original_mode = image.mode

    # Convert to RGB if needed (e.g., from RGBA or L)
    if image.mode != "RGB":
        if image.mode == "L":
            # Grayscale - convert to RGB
            image = image.convert("RGB")
        elif image.mode == "RGBA":
            # Remove alpha channel
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")

    # Resize to target size (maintain aspect ratio with padding)
    image = resize_with_padding(image, TARGET_SIZE)

    # Normalize pixel values to [0, 1]
    # (actual normalization happens at model input time)

    # Convert back to base64 for transport (or could return tensor)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    processed_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    logger.info(
        "Image preprocessed",
        label=label,
        original_size=original_size,
        original_mode=original_mode,
        target_size=TARGET_SIZE,
        is_dicom=dicom_metadata is not None,
    )

    result = {
        "image_b64": processed_b64,
        "label": label,
        "original_size": original_size,
        "processed_size": TARGET_SIZE,
        "pil_image": image,  # Keep PIL image for direct model input
    }

    # Include DICOM metadata if available
    if dicom_metadata:
        result["dicom_metadata"] = dicom_metadata
        result["is_dicom"] = True
    else:
        result["is_dicom"] = False

    return result


def resize_with_padding(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """
    Resize image to target size maintaining aspect ratio with black padding.
    """
    # Calculate scaling factor
    width, height = image.size
    target_width, target_height = target_size

    scale = min(target_width / width, target_height / height)

    # Resize maintaining aspect ratio
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create black background and paste resized image centered
    result = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    result.paste(image, (paste_x, paste_y))

    return result


async def load_image_from_file(file_path: str) -> str:
    """Load image from file and return as base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
