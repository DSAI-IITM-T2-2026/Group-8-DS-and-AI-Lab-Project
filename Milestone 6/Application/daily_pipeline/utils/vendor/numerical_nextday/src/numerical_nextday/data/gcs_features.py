"""Robust GCS list/download for S2/S5P feature CSVs (anon SDK can fail mid-run)."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_FEATURES_RE = re.compile(
    r"gs://([^/]+)/(.*/year=(\d+)/month=(\d+)/window=(\d+)/features\.csv)$"
)


def list_feature_files_gsutil(
    bucket: str,
    prefix: str,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """List Hive features.csv via gsutil (works with GS_NO_SIGN_REQUEST=YES)."""
    years = years or []
    rows: list[dict] = []
    year_list = years if years else list(range(2016, 2030))
    for year in year_list:
        # Prefer month-scoped listing (much faster than year-wide -r)
        for month in range(1, 13):
            uri = f"gs://{bucket}/{prefix.rstrip('/')}/year={year}/month={month:02d}/"
            cmd = ["gsutil", "ls", "-r", uri]
            logger.info("gsutil ls -r %s", uri)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except FileNotFoundError as e:
                raise RuntimeError("gsutil not found on PATH") from e
            if proc.returncode != 0:
                # month may be missing
                continue
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line.endswith("features.csv"):
                    continue
                m = _FEATURES_RE.match(line)
                if not m:
                    m2 = re.search(r"year=(\d+)/month=(\d+)/window=(\d+)/features\.csv$", line)
                    if not m2:
                        continue
                    y, mo, w = map(int, m2.groups())
                    blob_name = line.replace(f"gs://{bucket}/", "", 1)
                else:
                    blob_name = m.group(2)
                    y, mo, w = int(m.group(3)), int(m.group(4)), int(m.group(5))
                if years and y not in years:
                    continue
                rows.append(
                    {
                        "year": y,
                        "month": mo,
                        "window": w,
                        "blob_name": blob_name,
                        "gs_uri": f"gs://{bucket}/{blob_name}",
                        "size": None,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No feature CSVs via gsutil under gs://%s/%s", bucket, prefix)
    else:
        logger.info("gsutil indexed %d feature files under %s/%s", len(df), bucket, prefix)
        df = df.sort_values(["year", "month", "window"]).drop_duplicates(
            ["year", "month", "window"], keep="last"
        ).reset_index(drop=True)
    return df


def cache_csv_gsutil(
    bucket: str,
    blob_name: str,
    dest: Path,
    columns: list[str] | None = None,
) -> Path:
    """Download features.csv with gsutil, then write parquet."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp_csv = dest.with_suffix(".csv.partial")
    uri = f"gs://{bucket}/{blob_name}"
    logger.info("gsutil cp %s → %s", uri, tmp_csv)
    subprocess.check_call(["gsutil", "-q", "cp", uri, str(tmp_csv)])
    usecols = None
    if columns:
        base = ["grid_id", "latitude", "longitude", "window_start", "window_end"]
        usecols = list(dict.fromkeys(base + columns))
    try:
        df = pd.read_csv(tmp_csv, usecols=lambda c: usecols is None or c in usecols)
    except ValueError:
        df = pd.read_csv(tmp_csv)
        keep = [c for c in (usecols or df.columns) if c in df.columns]
        df = df[keep]
    df["window_start"] = pd.to_datetime(df["window_start"]).dt.normalize()
    df["window_end"] = pd.to_datetime(df["window_end"]).dt.normalize()
    df.to_parquet(dest, index=False)
    try:
        tmp_csv.unlink()
    except OSError:
        pass
    return dest


def list_feature_files_robust(nf, bucket: str, prefix: str, years: list[int] | None = None) -> pd.DataFrame:
    """Prefer gsutil (reliable with GS_NO_SIGN_REQUEST); fall back to SDK."""
    try:
        df = list_feature_files_gsutil(bucket, prefix, years=years)
        if not df.empty:
            return df
    except Exception as e:
        logger.warning("gsutil list failed (%s) — trying SDK", e)
    return nf.list_feature_files(bucket, prefix, years=years)


def cache_csv_robust(
    nf,
    bucket: str,
    blob_name: str,
    dest: Path,
    columns: list[str] | None = None,
) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        return cache_csv_gsutil(bucket, blob_name, dest, columns=columns)
    except Exception as e:
        logger.warning("gsutil download failed (%s) — trying SDK", e)
        return nf.cache_csv_to_parquet(bucket, blob_name, dest, columns=columns)
