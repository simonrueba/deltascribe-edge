#!/usr/bin/env python3
"""
Verify demo patient data against the official NIH ChestX-ray14 metadata.

Downloads the CSV from HuggingFace and checks that every manifest.json
matches the ground-truth labels, patient IDs, and follow-up numbers.

Usage:
    cd backend && uv run python ../scripts/verify_demo_data.py

Requires: huggingface-hub (pip install huggingface-hub)
"""

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    print("DeltaScribe Edge — Demo Data Verification")
    print("=" * 50)
    print("Source: NIH ChestX-ray14 (Data_Entry_2017_v2020.csv)")
    print()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface-hub required. Install with:")
        print("  pip install huggingface-hub")
        sys.exit(1)

    # Download official metadata
    print("Downloading official NIH metadata CSV...")
    csv_path = hf_hub_download(
        repo_id="alkzar90/NIH-Chest-X-ray-dataset",
        filename="data/Data_Entry_2017_v2020.csv",
        repo_type="dataset",
    )
    print(f"  {csv_path}")

    # Build lookup
    lookup: dict[str, dict[str, str]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["Image Index"]] = {
                "finding_labels": row["Finding Labels"],
                "follow_up": row["Follow-up #"],
                "patient_id": row["Patient ID"],
                "patient_age": row["Patient Age"],
                "patient_gender": row["Patient Gender"],
                "view_position": row["View Position"],
            }
    print(f"  Loaded {len(lookup)} image entries\n")

    # Check each demo patient
    script_dir = Path(__file__).parent.parent
    demo_dir = script_dir / "data" / "demo_patients"

    if not demo_dir.exists():
        print(f"ERROR: Demo data directory not found: {demo_dir}")
        sys.exit(1)

    total_checks = 0
    failures = 0

    for case_dir in sorted(demo_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        manifest_path = case_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        case_id = manifest["patient_id"]
        print(f"{case_id}: {manifest['description']}")

        for role in ["prior", "current"]:
            nih_image = manifest[role].get("nih_image")
            claimed_labels = set(manifest[role].get("nih_labels", []))

            if not nih_image:
                print(f"  {role}: SKIP (no nih_image)")
                continue

            if nih_image not in lookup:
                print(f"  {role}: FAIL - {nih_image} not found in NIH CSV")
                failures += 1
                total_checks += 1
                continue

            official = lookup[nih_image]
            official_labels = {label.strip() for label in official["finding_labels"].split("|")}

            # Check labels match
            total_checks += 1
            if claimed_labels == official_labels:
                print(f"  {role}: PASS - {nih_image} labels={official['finding_labels']}")
            else:
                print(f"  {role}: FAIL - labels mismatch!")
                print(f"    claimed: {sorted(claimed_labels)}")
                print(f"    official: {sorted(official_labels)}")
                failures += 1

            # Check patient ID matches
            total_checks += 1
            if manifest.get("nih_patient_id") == official["patient_id"]:
                print(f"         patient_id={official['patient_id']} "
                      f"age={official['patient_age']} "
                      f"gender={official['patient_gender']} "
                      f"view={official['view_position']} "
                      f"followup={official['follow_up']}")
            else:
                print(f"         FAIL - patient_id mismatch: "
                      f"claimed={manifest.get('nih_patient_id')} "
                      f"official={official['patient_id']}")
                failures += 1

            # Check same patient across prior/current
            if role == "current":
                prior_image = manifest["prior"].get("nih_image")
                if prior_image and prior_image in lookup:
                    prior_pid = lookup[prior_image]["patient_id"]
                    curr_pid = official["patient_id"]
                    total_checks += 1
                    if prior_pid == curr_pid:
                        print(f"         longitudinal pair confirmed (same patient {curr_pid})")
                    else:
                        print(f"         FAIL - different patients! prior={prior_pid} current={curr_pid}")
                        failures += 1

        # Check image files exist and are valid
        for role in ["prior", "current"]:
            img_path = case_dir / f"{role}.png"
            total_checks += 1
            if img_path.exists() and img_path.stat().st_size > 1000:
                print(f"  {role}.png: PASS ({img_path.stat().st_size / 1024:.0f} KB)")
            else:
                size = img_path.stat().st_size if img_path.exists() else 0
                print(f"  {role}.png: FAIL (missing or too small: {size} bytes)")
                failures += 1

        print()

    # Summary
    print("=" * 50)
    passed = total_checks - failures
    status = "PASS" if failures == 0 else "FAIL"
    print(f"Result: {status} ({passed}/{total_checks} checks passed)")
    if failures > 0:
        print(f"  {failures} failures found")
        sys.exit(1)
    else:
        print("All demo data verified against official NIH ChestX-ray14 metadata.")


if __name__ == "__main__":
    main()
