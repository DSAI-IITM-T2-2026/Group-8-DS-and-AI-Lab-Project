"""Bootstrap local imports so daily_pipeline is self-contained (no PYTHONPATH)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def utils_root() -> Path:
    return Path(__file__).resolve().parent


def pipeline_root() -> Path:
    return utils_root().parent


def _load_dotenv() -> None:
    env_path = pipeline_root() / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return
    load_dotenv(env_path, override=False)


def bootstrap() -> Path:
    """Add utils/ + vendored M4 to sys.path; load .env; set FIRMS GCS anon env."""
    _load_dotenv()
    uroot = utils_root()
    m4_src = uroot / "vendor" / "numerical_nextday" / "src"

    for path in (uroot, m4_src):
        s = str(path)
        if path.is_dir() and s not in sys.path:
            sys.path.insert(0, s)

    os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")
    return pipeline_root()
