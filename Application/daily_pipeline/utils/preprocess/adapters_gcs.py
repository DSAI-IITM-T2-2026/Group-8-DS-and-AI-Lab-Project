"""GCS adapters for flat dated raw files."""

from __future__ import annotations

import io
import logging
import re
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger("preprocess.adapters_gcs")

S2_FLAT_RE = re.compile(r"s2feat_(\d{8})_(\d{8})\.(csv|parquet)$")
S5P_FLAT_RE = re.compile(r"s5pfeat_(\d{8})_(\d{8})\.(csv|parquet)$")
ERA5_NC_RE = re.compile(r"era5_(\d{4})_(\d{2})_(\d{2})\.nc$")


def _prefer_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """If both CSV and parquet exist for a window, keep parquet only."""
    if df.empty or "layout" not in df.columns:
        return df
    out = df.copy()
    out["_rank"] = out["layout"].map({"flat_parquet": 0, "flat_csv": 1}).fillna(2)
    out = out.sort_values("_rank").drop_duplicates(["year", "window"], keep="first")
    return out.drop(columns=["_rank"])


def gsutil_ls(prefix_uri: str) -> list[str]:
    try:
        proc = subprocess.run(
            ["gsutil", "ls", prefix_uri],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return _sdk_list(prefix_uri)
    if proc.returncode != 0:
        return _sdk_list(prefix_uri)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _sdk_list(prefix_uri: str) -> list[str]:
    from google.cloud import storage

    if not prefix_uri.startswith("gs://"):
        return []
    rest = prefix_uri[5:]
    bucket, _, prefix = rest.partition("/")
    client = storage.Client()
    return [f"gs://{bucket}/{b.name}" for b in client.list_blobs(bucket, prefix=prefix)]


def download_blob(uri: str, dest: Path) -> Path:
    """Download a GCS object. Prefer the Python client — gsutil sliced cp often hangs."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for leftover in dest.parent.glob(dest.name + "*gstmp*"):
        try:
            leftover.unlink()
        except OSError:
            pass

    logger.info("Downloading %s → %s", uri, dest)
    rest = uri[5:]
    bucket_name, _, blob_name = rest.partition("/")
    try:
        from google.cloud import storage

        blob = storage.Client().bucket(bucket_name).blob(blob_name)
        size = blob.size
        if size:
            logger.info("  %s is %.1f MB", dest.name, size / 1e6)
        blob.download_to_filename(str(dest))
    except Exception as exc:
        logger.warning("SDK download failed (%s); gsutil without sliced download", exc)
        subprocess.check_call(
            [
                "gsutil",
                "-o",
                "GSUtil:sliced_object_download_max_components=1",
                "cp",
                uri,
                str(dest),
            ]
        )
    logger.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest


def read_parquet_uri(uri: str, local_cache: Path) -> pd.DataFrame:
    name = uri.rsplit("/", 1)[-1]
    local = local_cache / name
    download_blob(uri, local)
    return pd.read_parquet(local)


def era5_blob_candidates(day: date, cfg: dict) -> list[str]:
    """Primary wildfire-detection-first, fallback legacy dsai-lab-project."""
    fname = f"era5_{day.year:04d}_{day.month:02d}_{day.day:02d}.nc"
    bucket = cfg["gcs"]["bucket"]
    prefix = cfg["gcs"]["prefixes"]["era5"]
    primary = f"gs://{bucket}/{prefix}/{day.year:04d}/{fname}"
    legacy_bucket = cfg["gcs"]["era5_legacy_bucket"]
    legacy_prefix = cfg["gcs"]["era5_legacy_prefix"]
    legacy = f"gs://{legacy_bucket}/{legacy_prefix}/{day.year:04d}/{fname}"
    return [primary, legacy]


def resolve_era5_uri(day: date, cfg: dict) -> str | None:
    for uri in era5_blob_candidates(day, cfg):
        blobs = gsutil_ls(uri)
        if blobs:
            return blobs[0]
        # exact path check
        try:
            from google.cloud import storage

            rest = uri[5:]
            bucket, _, name = rest.partition("/")
            if storage.Client().bucket(bucket).blob(name).exists():
                return uri
        except Exception:
            continue
    return None


def list_flat_s2(bucket: str, prefix: str) -> pd.DataFrame:
    uris = gsutil_ls(f"gs://{bucket}/{prefix.rstrip('/')}/")
    rows: list[dict] = []
    for uri in uris:
        name = uri.rsplit("/", 1)[-1]
        m = S2_FLAT_RE.match(name)
        if not m:
            continue
        start = pd.Timestamp(m.group(1))
        end = pd.Timestamp(m.group(2))
        rows.append({
            "year": start.year,
            "month": start.month,
            "window": int((start - pd.Timestamp("2018-01-01")).days // 5) + 1,
            "blob_name": f"{prefix.rstrip('/')}/{name}",
            "gs_uri": uri,
            "window_start": start,
            "window_end": end,
            "layout": "flat_parquet" if name.endswith(".parquet") else "flat_csv",
        })
    return _prefer_parquet(pd.DataFrame(rows))


def list_flat_s5p(bucket: str, prefix: str) -> pd.DataFrame:
    uris = gsutil_ls(f"gs://{bucket}/{prefix.rstrip('/')}/")
    rows: list[dict] = []
    for uri in uris:
        name = uri.rsplit("/", 1)[-1]
        m = S5P_FLAT_RE.match(name)
        if not m:
            continue
        day = pd.Timestamp(m.group(1))
        rows.append({
            "year": day.year,
            "month": day.month,
            "window": day.dayofyear,
            "blob_name": f"{prefix.rstrip('/')}/{name}",
            "gs_uri": uri,
            "window_start": day,
            "window_end": day,
            "layout": "flat_parquet" if name.endswith(".parquet") else "flat_csv",
        })
    return _prefer_parquet(pd.DataFrame(rows))


def load_flat_feature_table(uri: str, local_cache: Path, columns: list[str] | None = None) -> pd.DataFrame:
    name = uri.rsplit("/", 1)[-1]
    local = local_cache / name
    download_blob(uri, local)
    if name.endswith(".parquet"):
        df = pd.read_parquet(local)
    else:
        df = pd.read_csv(local)
    if columns:
        keep = [c for c in ["grid_id", "latitude", "longitude", "window_start", "window_end", *columns] if c in df.columns]
        df = df[keep]
    for col in ("window_start", "window_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.normalize()
    return df


def upload_parquet(df: pd.DataFrame, uri: str) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from google.cloud import storage

    rest = uri[5:]
    bucket, _, blob_name = rest.partition("/")
    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    client = storage.Client()
    payload = buf.getvalue()
    blob = client.bucket(bucket).blob(blob_name)
    blob.upload_from_string(payload, content_type="application/octet-stream")
    blob.reload(client)
    if not blob.size or int(blob.size) != len(payload):
        raise RuntimeError(
            f"GCS parquet verification failed for {uri}: expected {len(payload)} bytes, got {blob.size}"
        )
    return uri
