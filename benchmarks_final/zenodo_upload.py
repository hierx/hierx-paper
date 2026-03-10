#!/usr/bin/env python3
"""Upload hierx-paper data files to Zenodo.

Reads zenodo_manifest.txt, creates a Zenodo deposit, uploads all files,
sets metadata, and optionally publishes. After upload, updates the manifest
with SHA256 checksums and writes the DOI to stdout.

Usage:
    # Create draft deposit (review on Zenodo before publishing)
    python -m benchmarks_final.zenodo_upload --token $ZENODO_TOKEN

    # Create and publish in one step
    python -m benchmarks_final.zenodo_upload --token $ZENODO_TOKEN --publish

    # Use Zenodo sandbox for testing
    python -m benchmarks_final.zenodo_upload --token $ZENODO_SANDBOX_TOKEN --sandbox

    # Update an existing deposit (new version)
    python -m benchmarks_final.zenodo_upload --token $ZENODO_TOKEN --deposit-id 1234567

Environment variables:
    ZENODO_TOKEN          API token (alternative to --token)
    ZENODO_SANDBOX_TOKEN  Sandbox API token (used with --sandbox)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

ZENODO_API = "https://zenodo.org/api"
ZENODO_SANDBOX_API = "https://sandbox.zenodo.org/api"

# Metadata for the Zenodo deposit
DEPOSIT_METADATA = {
    "title": (
        "Data archive for: HierX: Fast Multi-Scale Distance-Decay "
        "Interaction on Million-Node Networks"
    ),
    "upload_type": "dataset",
    "description": (
        "<p>Archived network data and pre-computed results for the hierx paper.</p>"
        "<p>Contains:</p>"
        "<ul>"
        "<li>London road and walking network snapshots (.graphml) extracted from "
        "OpenStreetMap</li>"
        "<li>GB driving network (.pkl) and Census activity vectors (.npz)</li>"
        "<li>Pre-computed accessibility node arrays (.npz) and benchmark results "
        "(.json)</li>"
        "</ul>"
        "<p>These files are used by the "
        '<a href="https://github.com/hierx/hierx-paper">'
        "hierx-paper</a> reproducibility pipeline. "
        "See reproduce.sh for usage.</p>"
    ),
    "creators": [
        {"name": "Hellervik, Alexander", "affiliation": "Chalmers University of Technology"},
        {"name": "Bohlin, Joakim", "affiliation": "Chalmers University of Technology"},
        {"name": "Andersson, Claes", "affiliation": "Chalmers University of Technology"},
    ],
    "keywords": [
        "spatial networks",
        "hierarchical approximation",
        "accessibility",
        "London",
        "OpenStreetMap",
    ],
    "license": "MIT",
    "related_identifiers": [
        {
            "identifier": "https://github.com/hierx/hierx",
            "relation": "isSupplementTo",
            "resource_type": "software",
            "scheme": "url",
        },
        {
            "identifier": "https://github.com/hierx/hierx-paper",
            "relation": "isSupplementTo",
            "resource_type": "other",
            "scheme": "url",
        },
    ],
    "access_right": "open",
}


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Parse zenodo_manifest.txt into a list of entries."""
    entries = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            print(f"WARNING: Skipping malformed manifest line: {line}", file=sys.stderr)
            continue
        entries.append({
            "zenodo_name": parts[0],
            "local_path": parts[1],
            "sha256": parts[2],
        })
    return entries


def create_deposit(api_base: str, token: str) -> dict:
    """Create a new empty Zenodo deposit."""
    r = requests.post(
        f"{api_base}/deposit/depositions",
        json={},
        params={"access_token": token},
    )
    r.raise_for_status()
    return r.json()


def new_version(api_base: str, token: str, deposit_id: int) -> dict:
    """Create a new version of an existing deposit."""
    r = requests.post(
        f"{api_base}/deposit/depositions/{deposit_id}/actions/newversion",
        params={"access_token": token},
    )
    r.raise_for_status()
    # Follow the link to the new draft
    new_url = r.json()["links"]["latest_draft"]
    r2 = requests.get(new_url, params={"access_token": token})
    r2.raise_for_status()
    return r2.json()


def upload_file(bucket_url: str, token: str, filepath: Path, zenodo_name: str) -> dict:
    """Upload a single file to the deposit bucket."""
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  Uploading {zenodo_name} ({size_mb:.1f} MB)...", flush=True)
    with open(filepath, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{zenodo_name}",
            data=f,
            params={"access_token": token},
        )
    r.raise_for_status()
    print(f"  OK: {zenodo_name}")
    return r.json()


def set_metadata(api_base: str, token: str, deposit_id: int, metadata: dict) -> dict:
    """Set deposit metadata."""
    r = requests.put(
        f"{api_base}/deposit/depositions/{deposit_id}",
        json={"metadata": metadata},
        params={"access_token": token},
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.json()


def publish_deposit(api_base: str, token: str, deposit_id: int) -> dict:
    """Publish the deposit (irreversible)."""
    r = requests.post(
        f"{api_base}/deposit/depositions/{deposit_id}/actions/publish",
        params={"access_token": token},
    )
    r.raise_for_status()
    return r.json()


def update_manifest_checksums(manifest_path: Path, checksums: dict[str, str]) -> None:
    """Rewrite manifest with computed SHA256 checksums."""
    lines = manifest_path.read_text().splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] in checksums:
            parts[2] = checksums[parts[0]]
            new_lines.append("\t".join(parts))
        else:
            new_lines.append(line)
    manifest_path.write_text("\n".join(new_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload hierx-paper data to Zenodo")
    parser.add_argument(
        "--token",
        default=None,
        help="Zenodo API token (or set ZENODO_TOKEN / ZENODO_SANDBOX_TOKEN env var)",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use Zenodo sandbox (sandbox.zenodo.org) for testing",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the deposit after upload (irreversible)",
    )
    parser.add_argument(
        "--deposit-id",
        type=int,
        default=None,
        help="Existing deposit ID to create a new version of",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "zenodo_manifest.txt",
        help="Path to zenodo_manifest.txt",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for resolving local_path entries (default: manifest parent dir)",
    )
    parser.add_argument(
        "--update-checksums",
        action="store_true",
        help="Update manifest with computed SHA256 checksums after upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files exist and compute checksums without uploading",
    )
    args = parser.parse_args()

    # Resolve token
    import os

    token = args.token
    if token is None:
        env_var = "ZENODO_SANDBOX_TOKEN" if args.sandbox else "ZENODO_TOKEN"
        token = os.environ.get(env_var)
    if token is None and not args.dry_run:
        parser.error(
            "Zenodo API token required. Pass --token or set "
            f"{'ZENODO_SANDBOX_TOKEN' if args.sandbox else 'ZENODO_TOKEN'} env var."
        )

    api_base = ZENODO_SANDBOX_API if args.sandbox else ZENODO_API
    base_dir = args.base_dir or args.manifest.parent

    # Parse manifest
    entries = parse_manifest(args.manifest)
    if not entries:
        print("ERROR: No entries found in manifest", file=sys.stderr)
        sys.exit(1)
    print(f"Manifest: {len(entries)} files")

    # Validate all files exist
    missing = []
    for entry in entries:
        filepath = base_dir / entry["local_path"]
        if not filepath.is_file():
            missing.append(entry["local_path"])
    if missing:
        print(f"\nERROR: {len(missing)} files not found:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)
    print("All files present.")

    # Compute checksums
    print("\nComputing SHA256 checksums...")
    checksums: dict[str, str] = {}
    for entry in entries:
        filepath = base_dir / entry["local_path"]
        sha = sha256_file(filepath)
        checksums[entry["zenodo_name"]] = sha
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  {entry['zenodo_name']}: {sha[:16]}... ({size_mb:.1f} MB)")

    total_mb = sum(
        (base_dir / e["local_path"]).stat().st_size for e in entries
    ) / (1024 * 1024)
    print(f"\nTotal upload size: {total_mb:.0f} MB ({len(entries)} files)")

    if args.dry_run:
        print("\n--- DRY RUN: no upload performed ---")
        if args.update_checksums:
            update_manifest_checksums(args.manifest, checksums)
            print(f"Updated checksums in {args.manifest}")
        return

    # Create or version the deposit
    env_label = "SANDBOX" if args.sandbox else "PRODUCTION"
    print(f"\nTarget: {env_label} ({api_base})")

    if args.deposit_id:
        print(f"Creating new version of deposit {args.deposit_id}...")
        deposit = new_version(api_base, token, args.deposit_id)
        # Delete existing files from the new version draft
        for f in deposit.get("files", []):
            requests.delete(
                f"{api_base}/deposit/depositions/{deposit['id']}/files/{f['id']}",
                params={"access_token": token},
            )
    else:
        print("Creating new deposit...")
        deposit = create_deposit(api_base, token)

    deposit_id = deposit["id"]
    bucket_url = deposit["links"]["bucket"]
    print(f"Deposit ID: {deposit_id}")
    print(f"Bucket URL: {bucket_url}")
    print(f"Draft URL:  {deposit['links']['html']}")

    # Upload files
    print(f"\nUploading {len(entries)} files...")
    for entry in entries:
        filepath = base_dir / entry["local_path"]
        upload_file(bucket_url, token, filepath, entry["zenodo_name"])

    # Set metadata
    print("\nSetting metadata...")
    result = set_metadata(api_base, token, deposit_id, DEPOSIT_METADATA)
    print(f"Title: {result['metadata']['title']}")

    # Optionally publish
    if args.publish:
        print("\nPublishing deposit (this is irreversible)...")
        result = publish_deposit(api_base, token, deposit_id)
        doi = result["doi"]
        record_id = result["id"]
        print(f"\nPublished!")
        print(f"  DOI:       {doi}")
        print(f"  Record ID: {record_id}")
        print(f"  URL:       https://doi.org/{doi}")

        # Output DOI for CI consumption
        print(f"\n::set-output name=doi::{doi}")
        print(f"::set-output name=record_id::{record_id}")
        # Also use the newer GITHUB_OUTPUT approach
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"doi={doi}\n")
                f.write(f"record_id={record_id}\n")
    else:
        print(f"\nDeposit created as DRAFT (not published).")
        print(f"  Review at: {deposit['links']['html']}")
        print(f"  To publish, use the Zenodo web UI or re-run with --publish.")

    # Update checksums in manifest
    if args.update_checksums:
        update_manifest_checksums(args.manifest, checksums)
        print(f"\nUpdated checksums in {args.manifest}")

    print("\nDone.")


if __name__ == "__main__":
    main()
