#!/usr/bin/env python3
"""
Download completed Landsat composite exports from YOUR OWN GCS bucket
(gs://dsai-lab-project/wildfire_satellite/raw/landsat/...) to local disk.

This is NOT the public Landsat archive — it's the GEE-exported cloud-masked
median composites that fetch_landsat.py produces. Since it's a private
bucket, this uses your authenticated gcloud/service-account credentials,
not an anonymous client.

Resumable + integrity-checked, same as before:
  - Skips files already present AND checksum-verified.
  - Resumes partial downloads with an HTTP range request.
  - Verifies every completed download against GCS md5 metadata; re-downloads
    on mismatch.

Install:
    pip install google-cloud-storage tqdm

Auth (one of):
    gcloud auth application-default login
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/keyfile.json

Usage:
    python download_exports.py \
        --project iitm-dsai-lab \
        --bucket dsai-lab-project \
        --prefix wildfire_satellite/raw/landsat \
        --dest ./data/wildfire_satellite/raw/landsat
"""

import argparse
import base64
import hashlib
import sys
import time
from pathlib import Path

from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

CHUNK_SIZE = 8 * 1024 * 1024
LOG_FILE_NAME = "_download_manifest.log"
BAD_LOG_NAME = "_download_failed.log"


def load_set(path: Path) -> set:
    return set(path.read_text().splitlines()) if path.exists() else set()


def append_line(path: Path, line: str) -> None:
    with open(path, "a") as f:
        f.write(line + "\n")


def local_md5(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("utf-8")


def verify_local_file(blob, local_path: Path) -> bool:
    if not local_path.exists():
        return False
    if blob.size is not None and local_path.stat().st_size != blob.size:
        return False
    if blob.md5_hash:
        return local_md5(local_path) == blob.md5_hash
    return True


def download_with_resume(blob, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")

    remote_size = blob.size
    if remote_size is None:
        blob.reload()
        remote_size = blob.size

    start_byte = tmp_path.stat().st_size if tmp_path.exists() else 0
    if start_byte >= remote_size:
        start_byte = 0
        tmp_path.unlink(missing_ok=True)

    mode = "ab" if start_byte else "wb"
    with open(tmp_path, mode) as f:
        pos = start_byte
        pbar = tqdm(total=remote_size, initial=start_byte, unit="B", unit_scale=True,
                    desc=local_path.name, leave=False)
        while pos < remote_size:
            end = min(pos + CHUNK_SIZE, remote_size) - 1
            data = blob.download_as_bytes(start=pos, end=end, retry=DEFAULT_RETRY)
            f.write(data)
            pos += len(data)
            if hasattr(pbar, "update"):
                pbar.update(len(data))
        if hasattr(pbar, "close"):
            pbar.close()

    tmp_path.rename(local_path)


def process_one(blob, dest_dir: Path, prefix: str, max_retries: int, verify_only: bool):
    rel_path = blob.name[len(prefix):].lstrip("/") if blob.name.startswith(prefix) else blob.name
    local_path = dest_dir / rel_path
    done_log = dest_dir / LOG_FILE_NAME
    bad_log = dest_dir / BAD_LOG_NAME

    if verify_only:
        if verify_local_file(blob, local_path):
            append_line(done_log, blob.name)
            return "verified_ok"
        append_line(bad_log, blob.name)
        return "failed"

    for attempt in range(1, max_retries + 1):
        try:
            if verify_local_file(blob, local_path):
                append_line(done_log, blob.name)
                return "skipped_ok"
            download_with_resume(blob, local_path)
            if verify_local_file(blob, local_path):
                append_line(done_log, blob.name)
                return "downloaded_ok"
            print(f"\n[checksum mismatch] {blob.name} -> removing and retrying")
            local_path.unlink(missing_ok=True)
            raise IOError("post-download checksum verification failed")
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"\n[retry {attempt}/{max_retries}] {blob.name}: {e} (waiting {wait}s)")
            time.sleep(wait)

    append_line(bad_log, blob.name)
    return "failed"


def main():
    parser = argparse.ArgumentParser(description="Download completed exports from your private GCS bucket")
    parser.add_argument("--project", required=True, help="GCP project id (for billing/auth context)")
    parser.add_argument("--bucket", required=True, help="Your GCS bucket, e.g. dsai-lab-project")
    parser.add_argument("--prefix", required=True,
                         help="Prefix to fetch, e.g. wildfire_satellite/raw/landsat")
    parser.add_argument("--dest", required=True, help="Local destination directory")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--verify-only", action="store_true",
                         help="Audit existing local files against GCS checksums; download nothing new")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    client = storage.Client(project=args.project)  # uses your authenticated credentials
    bucket = client.bucket(args.bucket)

    print(f"Listing gs://{args.bucket}/{args.prefix} ...")
    blobs = list(client.list_blobs(bucket, prefix=args.prefix))
    print(f"Found {len(blobs)} objects.")

    counts = {"skipped_ok": 0, "downloaded_ok": 0, "verified_ok": 0, "failed": 0}
    for blob in tqdm(blobs, desc="Overall progress", unit="file"):
        result = process_one(blob, dest_dir, args.prefix, args.max_retries, args.verify_only)
        counts[result] += 1

    print("\n--- Summary ---")
    if args.verify_only:
        print(f"Verified OK: {counts['verified_ok']}, Failed/missing: {counts['failed']}")
    else:
        print(f"Downloaded: {counts['downloaded_ok']}, already had: {counts['skipped_ok']}, "
              f"failed: {counts['failed']}")
        if counts["failed"]:
            print(f"See {dest_dir / BAD_LOG_NAME}. Re-run the same command to retry only these.")


if __name__ == "__main__":
    main()