import json
from pathlib import Path
import sys

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(UTILS))

import config_loader


def test_feature_contract_rejects_legacy_schema(monkeypatch, tmp_path):
    path = tmp_path / "champion.json"
    path.write_text(json.dumps({"feature_columns": [f"f{i}" for i in range(86)]}))
    monkeypatch.setattr(config_loader, "resolve_path", lambda *_: path)

    with pytest.raises(ValueError, match="Feature contract mismatch"):
        config_loader.load_feature_contract({})


def test_feature_contract_accepts_exact_unique_champion_order(monkeypatch, tmp_path):
    features = [f"f{i}" for i in range(86)]
    path = tmp_path / "champion.json"
    path.write_text(json.dumps({"feature_prune": {"kept_features": features}}))
    monkeypatch.setattr(config_loader, "resolve_path", lambda *_: path)

    assert config_loader.load_feature_contract({})["feature_prune"]["kept_features"] == features
