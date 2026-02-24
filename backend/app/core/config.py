"""Application configuration settings."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Settings
    api_title: str = "DeltaScribe Edge API"
    api_version: str = "0.1.0"
    api_prefix: str = "/api"
    debug: bool = False

    # Model Settings
    model_path: Path = Path("/data/models/medgemma-1.5-4b")
    model_name: str = "google/medgemma-4b-it"  # Hugging Face model ID
    device: Literal["cuda", "cpu", "auto"] = "auto"
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    max_new_tokens: int = 512  # Narrative generation token budget
    hf_token: str | None = None  # Hugging Face API token for gated models

    # GGUF Settings (for fast CPU/Metal inference)
    use_gguf: bool = True  # Use GGUF model by default for speed
    gguf_model_path: str = "/data/models/medgemma-4b-it-Q4_K_M.gguf"
    gguf_n_ctx: int = 4096  # Context window
    gguf_n_threads: int = 8  # CPU threads

    # Inference Settings
    inference_timeout: int = 120  # seconds
    temperature: float = 0.1  # Low for deterministic outputs
    top_p: float = 0.9

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3010", "http://localhost:3020"]

    # Paths
    data_dir: Path = Path("/data")
    demo_patients_dir: Path = Path("/data/demo_patients")
    guidelines_dir: Path = Path("/data/guidelines")


settings = Settings()
