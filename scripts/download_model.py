#!/usr/bin/env python3
"""
Download MedGemma model weights for offline use.

This script downloads the MedGemma 1.5 4B model from Hugging Face
and saves it locally for offline inference.

Usage:
    python scripts/download_model.py

Requirements:
    - Hugging Face account
    - Accepted MedGemma license agreement
    - HF_TOKEN environment variable set (for gated models)

The script saves the model to:
    data/models/medgemma-1.5-4b/
"""

import os
import sys
from pathlib import Path


def main() -> None:
    """Main entry point."""
    print("DeltaScribe Edge - Model Download")
    print("=" * 50)

    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\nWARNING: HF_TOKEN environment variable not set.")
        print("You may need this to download gated models like MedGemma.")
        print("\nTo set your token:")
        print("  export HF_TOKEN=your_huggingface_token")
        print("\nContinuing without token...")

    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError:
        print("\nRequired packages not installed. Run:")
        print("  pip install huggingface-hub transformers torch")
        sys.exit(1)

    # Model configuration
    model_name = "google/medgemma-4b-it"  # Update if model name changes

    # Alternative models to try if MedGemma isn't available
    fallback_models = [
        "google/gemma-2-2b-it",  # Smaller Gemma model
        "google/paligemma-3b-pt-224",  # Vision-language model
    ]

    # Determine output directory
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "data" / "models" / "medgemma-1.5-4b"

    print(f"\nModel: {model_name}")
    print(f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading model (this may take a while)...")
    print("Model size: ~8GB for 4B parameter model")

    try:
        # Download model snapshot
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            token=hf_token,
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        print(f"\nModel downloaded successfully to: {output_dir}")

    except Exception as e:
        print(f"\nFailed to download {model_name}: {e}")
        print("\nNote: MedGemma may require license acceptance on Hugging Face.")
        print("Visit: https://huggingface.co/google/medgemma-4b-it")
        print("\nTrying fallback model for testing...")

        # Try fallback model
        for fallback in fallback_models:
            try:
                print(f"\nTrying: {fallback}")
                fallback_dir = script_dir / "data" / "models" / fallback.split("/")[-1]
                fallback_dir.mkdir(parents=True, exist_ok=True)

                snapshot_download(
                    repo_id=fallback,
                    local_dir=fallback_dir,
                    token=hf_token,
                    resume_download=True,
                    local_dir_use_symlinks=False,
                )
                print(f"\nFallback model downloaded to: {fallback_dir}")
                print("\nNote: Update MODEL_NAME in your .env to use this model")
                break

            except Exception as fallback_error:
                print(f"  Failed: {fallback_error}")
                continue

    # Verify download
    print("\n" + "=" * 50)
    print("Verifying download...")

    config_path = output_dir / "config.json"
    if config_path.exists():
        print("  config.json: OK")
    else:
        print("  config.json: MISSING")

    safetensors = list(output_dir.glob("*.safetensors"))
    if safetensors:
        print(f"  Model weights: {len(safetensors)} file(s)")
        total_size = sum(f.stat().st_size for f in safetensors)
        print(f"  Total size: {total_size / (1024**3):.2f} GB")
    else:
        print("  Model weights: MISSING")

    print("\n" + "=" * 50)
    print("Done!")
    print("\nTo use the model:")
    print("  1. Set MODEL_PATH in your .env file")
    print("  2. Run: docker compose up")


if __name__ == "__main__":
    main()
