from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    tmp = Path(raw_tmp)
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_shard_manifest(
    output_path: Path,
    metadata: dict[str, Any],
    config_hash: str,
    manifest_dir: Path,
) -> Path:
    manifest = {
        **metadata,
        "config_hash": config_hash,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
        },
    }
    year = metadata.get("year")
    partition = manifest_dir / f"year={year}" if year is not None else manifest_dir
    destination = partition / f"{output_path.name}.manifest.json"
    atomic_json(manifest, destination)
    return destination


@contextlib.contextmanager
def claim_shard(
    claim_path: Path, worker: str, ttl_hours: float = 24, force: bool = False
) -> Iterator[None]:
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    if force:
        claim_path.unlink(missing_ok=True)
    if claim_path.exists():
        try:
            current = json.loads(claim_path.read_text(encoding="utf-8"))
            started = datetime.fromisoformat(current["started_at_utc"])
            if datetime.now(UTC) - started > timedelta(hours=ttl_hours):
                claim_path.unlink()
        except (KeyError, ValueError, json.JSONDecodeError):
            raise RuntimeError(f"Invalid or active shard claim: {claim_path}") from None

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(claim_path, flags)
    except FileExistsError:
        current = claim_path.read_text(encoding="utf-8")
        raise RuntimeError(f"Shard already claimed: {claim_path}\n{current}") from None
    try:
        payload = {
            "worker": worker,
            "pid": os.getpid(),
            "started_at_utc": datetime.now(UTC).isoformat(),
        }
        os.write(fd, json.dumps(payload, indent=2).encode("utf-8"))
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
        claim_path.unlink(missing_ok=True)


def run_command(arguments: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def gcs_list(uri: str, timeout: int = 60) -> list[str]:
    try:
        result = run_command(["gsutil", "ls", uri], timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Timed out after {timeout}s while listing {uri}") from exc
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise PermissionError(f"Cannot list {uri}: {message}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gcs_copy(uri: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["gsutil", "-q", "cp", uri, str(destination)], check=False)
    if result.returncode:
        raise RuntimeError(f"gsutil cp failed for {uri}")
    return destination


def _gcs_read_csv_once(uri: str, wanted_columns: set[str]) -> pd.DataFrame:
    """Stream a GCS CSV through gcloud without persisting the raw object locally."""
    process = subprocess.Popen(
        ["gcloud", "storage", "cat", uri],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError(f"Failed to open streaming pipes for {uri}")
    try:
        frame = pd.read_csv(
            process.stdout,
            usecols=lambda column: column in wanted_columns,
            low_memory=False,
        )
    except BaseException:
        process.terminate()
        process.wait(timeout=10)
        raise
    finally:
        process.stdout.close()
    error_text = process.stderr.read().decode("utf-8", errors="replace").strip()
    process.stderr.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Streaming GCS CSV failed for {uri} (exit={return_code}): {error_text}")
    return frame


def gcs_read_csv(uri: str, wanted_columns: set[str], max_attempts: int = 3) -> pd.DataFrame:
    """Stream a GCS CSV with bounded retries and no persistent raw file."""
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            return _gcs_read_csv_once(uri, wanted_columns)
        except (OSError, RuntimeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise RuntimeError(f"Failed to stream {uri} after {max_attempts} attempts") from last_error
