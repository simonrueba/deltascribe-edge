#!/usr/bin/env python3
"""
Download real CXR pairs from the NIH ChestX-ray14 dataset for DeltaScribe Edge.

Downloads metadata CSV from HuggingFace, finds same-patient longitudinal pairs,
then extracts only the needed images from the remote zip via HTTP range requests
(~6 MB total instead of downloading the full 2 GB zip).

Usage:
    cd backend && uv run python ../scripts/download_demo_data.py

Creates:
    data/demo_patients/
        cxr_001/  (prior.png, current.png, manifest.json)
        cxr_002/  ...

License: NIH Clinical Center, CC0 1.0 Public Domain.
"""

import csv
import json
import shutil
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path

# --- Curated longitudinal case definitions ---
DESIRED_CASES = [
    {
        "id": "cxr_001",
        "description": "Consolidation and effusion improving",
        "prior_target": {"Consolidation"},
        "current_target": {"No Finding"},
        "expected_delta": "improved",
        "prior_finding": "Consolidation, effusion, atelectasis",
        "current_finding": "Consolidation and effusion resolved",
    },
    {
        "id": "cxr_002",
        "description": "New effusion and infiltration",
        "prior_target": {"No Finding"},
        "current_target": {"Effusion"},
        "expected_delta": "worsened",
        "prior_finding": "Clear lungs",
        "current_finding": "New effusion with infiltration",
    },
    {
        "id": "cxr_003",
        "description": "Stable cardiomegaly with new emphysema",
        "prior_target": {"Cardiomegaly"},
        "current_target": {"Cardiomegaly"},
        "expected_delta": "worsened",
        "prior_finding": "Cardiomegaly",
        "current_finding": "Stable cardiomegaly, new emphysema",
    },
    {
        "id": "cxr_004",
        "description": "New pneumothorax",
        "prior_target": {"No Finding"},
        "current_target": {"Pneumothorax"},
        "expected_delta": "worsened",
        "prior_finding": "Clear lungs",
        "current_finding": "New pneumothorax",
    },
    {
        "id": "cxr_005",
        "description": "Effusion improving",
        "prior_target": {"Effusion"},
        "current_target": {"No Finding"},
        "expected_delta": "improved",
        "prior_finding": "Pleural effusion",
        "current_finding": "Effusion resolved",
    },
    {
        "id": "cxr_006",
        "description": "Stable atelectasis",
        "prior_target": {"Atelectasis"},
        "current_target": {"Atelectasis"},
        "expected_delta": "stable",
        "prior_finding": "Atelectasis",
        "current_finding": "Stable atelectasis",
    },
    {
        "id": "cxr_007",
        "description": "Normal to normal comparison",
        "prior_target": {"No Finding"},
        "current_target": {"No Finding"},
        "expected_delta": "stable",
        "prior_finding": "Normal",
        "current_finding": "Normal",
    },
]

HF_REPO = "alkzar90/NIH-Chest-X-ray-dataset"
CSV_FILENAME = "data/Data_Entry_2017_v2020.csv"
ZIP_FILENAME = "data/images/images_001.zip"


def download_metadata_csv() -> Path:
    """Download the NIH metadata CSV via huggingface_hub."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=HF_REPO, filename=CSV_FILENAME, repo_type="dataset")
    )


def find_matching_pairs(csv_path: Path) -> list[dict]:
    """Scan CSV to find same-patient longitudinal pairs matching desired cases."""
    print("Scanning metadata for longitudinal pairs...")

    patient_index: dict[str, list] = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["Patient ID"]
            labels = {label.strip() for label in row["Finding Labels"].split("|")}
            patient_index[pid].append({
                "image": row["Image Index"],
                "labels": labels,
                "followup": int(row["Follow-up #"]),
                "view": row["View Position"],
                "age": row["Patient Age"],
                "gender": row["Patient Gender"],
            })

    multi = {pid: imgs for pid, imgs in patient_index.items() if len(imgs) >= 2}
    print(f"  {len(patient_index)} patients, {len(multi)} with 2+ images")

    used_patients: set[str] = set()
    matches = []

    for case in DESIRED_CASES:
        found = False
        for pid, images in multi.items():
            if pid in used_patients:
                continue
            pa = sorted(
                [i for i in images if i["view"] == "PA"],
                key=lambda x: x["followup"],
            )
            if len(pa) < 2:
                continue

            for j in range(len(pa)):
                if found:
                    break
                for k in range(j + 1, len(pa)):
                    prior, curr = pa[j], pa[k]
                    if case["prior_target"].issubset(
                        prior["labels"]
                    ) and case["current_target"].issubset(curr["labels"]):
                        matches.append({
                            **case,
                            "prior_image": prior["image"],
                            "current_image": curr["image"],
                            "prior_labels": sorted(prior["labels"]),
                            "current_labels": sorted(curr["labels"]),
                            "nih_patient_id": pid,
                        })
                        used_patients.add(pid)
                        found = True
                        print(
                            f"  {case['id']}: patient {pid}, "
                            f"{prior['image']} -> {curr['image']}"
                        )
                        break

        if not found:
            print(f"  WARNING: No match for {case['id']} ({case['description']})")

    return matches


def extract_images_from_remote_zip(
    needed_files: set[str], out_dir: Path
) -> dict[str, Path]:
    """
    Extract specific files from a remote zip using HTTP range requests.
    Downloads only the central directory + requested files (~6 MB vs 2 GB).
    """
    import requests
    from huggingface_hub import hf_hub_url

    zip_url = hf_hub_url(
        repo_id=HF_REPO, filename=ZIP_FILENAME, repo_type="dataset"
    )

    # Get total size
    head = requests.head(zip_url, allow_redirects=True, timeout=30)
    total_size = int(head.headers["content-length"])
    print(f"  Remote zip: {total_size / 1e9:.2f} GB")

    # Read End of Central Directory
    eocd_size = min(65536 + 22, total_size)
    r = requests.get(
        zip_url,
        headers={"Range": f"bytes={total_size - eocd_size}-{total_size - 1}"},
        timeout=60,
    )
    eocd_data = r.content
    eocd_pos = eocd_data.rfind(b"\x50\x4b\x05\x06")
    eocd = eocd_data[eocd_pos:]
    _, _, _, _, _, cd_size, cd_offset = struct.unpack_from("<IHHHHII", eocd, 0)

    # Read central directory
    r = requests.get(
        zip_url,
        headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
        timeout=60,
    )
    cd_data = r.content

    # Parse central directory to find our files
    offset = 0
    found: dict[str, dict] = {}
    while offset < len(cd_data) - 46:
        sig = struct.unpack_from("<I", cd_data, offset)[0]
        if sig != 0x02014B50:
            break
        fields = struct.unpack_from("<4xHHHHHHIIIHHHHHII", cd_data, offset)
        comp_size = fields[6]
        name_len, extra_len, comment_len = fields[9], fields[10], fields[11]
        local_offset = fields[15]
        name = cd_data[offset + 46 : offset + 46 + name_len].decode("utf-8")
        basename = name.split("/")[-1]
        if basename in needed_files:
            found[basename] = {
                "offset": local_offset,
                "comp_size": comp_size,
                "comp_method": fields[3],
            }
        offset += 46 + name_len + extra_len + comment_len

    print(f"  Found {len(found)}/{len(needed_files)} files in central directory")

    # Download each file
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}
    total_downloaded = 0

    for basename, info in sorted(found.items()):
        fetch_size = 30 + 512 + info["comp_size"]
        r = requests.get(
            zip_url,
            headers={
                "Range": f"bytes={info['offset']}-{info['offset'] + fetch_size - 1}"
            },
            timeout=60,
        )
        local_data = r.content
        total_downloaded += len(local_data)

        local_name_len = struct.unpack_from("<H", local_data, 26)[0]
        local_extra_len = struct.unpack_from("<H", local_data, 28)[0]
        data_start = 30 + local_name_len + local_extra_len
        file_data = local_data[data_start : data_start + info["comp_size"]]

        if info["comp_method"] == 0:
            raw_data = file_data
        elif info["comp_method"] == 8:
            raw_data = zlib.decompress(file_data, -15)
        else:
            print(f"    ERROR: Unknown compression for {basename}")
            continue

        out_path = out_dir / basename
        with open(out_path, "wb") as f:
            f.write(raw_data)
        results[basename] = out_path
        print(f"    {basename} ({len(raw_data) / 1024:.0f} KB)")

    print(
        f"  Downloaded {total_downloaded / 1e6:.1f} MB "
        f"(vs {total_size / 1e9:.2f} GB full zip)"
    )
    return results


def save_cases(
    matches: list[dict], image_paths: dict[str, Path], data_dir: Path
) -> None:
    """Organize images into case directories with manifest files."""
    dates = [
        ("2024-01-15", "2024-03-10"),
        ("2024-02-01", "2024-04-15"),
        ("2024-01-20", "2024-04-20"),
        ("2024-03-01", "2024-03-05"),
        ("2024-02-10", "2024-04-10"),
        ("2024-01-05", "2024-05-05"),
        ("2024-02-20", "2024-06-20"),
    ]

    for i, case in enumerate(matches):
        case_dir = data_dir / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        prior_src = image_paths.get(case["prior_image"])
        current_src = image_paths.get(case["current_image"])
        if not prior_src or not current_src:
            print(f"  WARNING: Missing images for {case['id']}")
            continue

        shutil.copy2(prior_src, case_dir / "prior.png")
        shutil.copy2(current_src, case_dir / "current.png")

        prior_date, current_date = dates[i] if i < len(dates) else ("2024-01-01", "2024-06-01")

        manifest = {
            "patient_id": case["id"],
            "description": case["description"],
            "prior": {
                "filename": "prior.png",
                "finding": case["prior_finding"],
                "date": prior_date,
                "nih_image": case["prior_image"],
                "nih_labels": case["prior_labels"],
            },
            "current": {
                "filename": "current.png",
                "finding": case["current_finding"],
                "date": current_date,
                "nih_image": case["current_image"],
                "nih_labels": case["current_labels"],
            },
            "expected_delta": case["expected_delta"],
            "source": "NIH ChestX-ray14 (CC0 1.0 Public Domain)",
            "nih_patient_id": case["nih_patient_id"],
        }

        with open(case_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  {case['id']}: {case['description']} ({case['expected_delta']})")


def main() -> None:
    print("DeltaScribe Edge — Real CXR Demo Data Download")
    print("=" * 55)
    print("Source: NIH ChestX-ray14 (CC0 1.0 Public Domain)")
    print()

    try:
        import requests  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        print("ERROR: Required packages missing. Install with:")
        print("  uv pip install huggingface-hub requests pillow")
        sys.exit(1)

    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data" / "demo_patients"

    if data_dir.exists():
        print(f"Removing existing demo data in {data_dir}...")
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download metadata
    print("\nStep 1: Downloading metadata CSV...")
    csv_path = download_metadata_csv()
    print(f"  Metadata: {csv_path}")

    # Step 2: Find matching pairs
    print("\nStep 2: Finding longitudinal pairs...")
    matches = find_matching_pairs(csv_path)
    if not matches:
        print("\nERROR: No matching pairs found!")
        sys.exit(1)

    # Step 3: Collect all needed image filenames
    needed = set()
    for m in matches:
        needed.add(m["prior_image"])
        needed.add(m["current_image"])
    print(f"\nStep 3: Extracting {len(needed)} images from remote zip...")
    raw_dir = data_dir / "_raw"
    image_paths = extract_images_from_remote_zip(needed, raw_dir)

    # Step 4: Organize into case directories
    print("\nStep 4: Organizing cases...")
    save_cases(matches, image_paths, data_dir)

    # Cleanup
    shutil.rmtree(raw_dir)

    # Summary
    print("\n" + "=" * 55)
    print(f"Downloaded {len(matches)} real CXR cases to {data_dir}")
    print("All images are from the NIH ChestX-ray14 dataset (CC0 1.0).")
    print("Each case contains real same-patient longitudinal CXR pairs.")


if __name__ == "__main__":
    main()
