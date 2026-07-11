"""Load multiple CDS API credentials and create clients."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cdsapi
import yaml


DEFAULT_CDS_URL = "https://cds.climate.copernicus.eu/api"


@dataclass(frozen=True)
class CdsAccount:
    name: str
    url: str
    key: str


def _parse_rc_file(path: Path) -> tuple[str, str]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    url = data.get("url", DEFAULT_CDS_URL)
    key = data.get("key")
    if not key:
        raise ValueError(f"No 'key' found in CDS credentials file: {path}")
    return url, key


def load_account(account_cfg: dict) -> CdsAccount:
    name = account_cfg["name"]

    if "key_env" in account_cfg:
        key = os.environ.get(account_cfg["key_env"])
        if not key:
            raise ValueError(f"Environment variable {account_cfg['key_env']} is not set")
        url = account_cfg.get("url", DEFAULT_CDS_URL)
        return CdsAccount(name=name, url=url, key=key)

    rc_path = Path(os.path.expanduser(account_cfg["rc_file"]))
    if not rc_path.exists():
        raise FileNotFoundError(f"CDS credentials file not found: {rc_path}")
    url, key = _parse_rc_file(rc_path)
    return CdsAccount(name=name, url=url, key=key)


def load_accounts(account_configs: list[dict]) -> list[CdsAccount]:
    if not account_configs:
        default_rc = Path.home() / ".cdsapirc"
        if default_rc.exists():
            url, key = _parse_rc_file(default_rc)
            return [CdsAccount(name="primary", url=url, key=key)]
        raise ValueError("No CDS accounts configured and ~/.cdsapirc not found")

    accounts = [load_account(cfg) for cfg in account_configs]
    keys = [a.key for a in accounts]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate API keys detected — each cds_accounts entry must use a different CDS account"
        )
    return accounts


def create_client(account: CdsAccount) -> cdsapi.Client:
    return cdsapi.Client(url=account.url, key=account.key)
