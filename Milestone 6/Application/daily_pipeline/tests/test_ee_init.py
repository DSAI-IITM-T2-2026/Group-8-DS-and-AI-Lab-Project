from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from download.ee_init import initialize_ee  # noqa: E402


def test_worker_never_starts_interactive_auth_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    authenticate_called = False

    def initialize(*, project: str) -> None:
        raise RuntimeError(f"no Earth Engine access for {project}")

    def authenticate() -> None:
        nonlocal authenticate_called
        authenticate_called = True

    monkeypatch.delenv("WILDFIRE_ALLOW_INTERACTIVE_EE_AUTH", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "ee",
        types.SimpleNamespace(Initialize=initialize, Authenticate=authenticate),
    )

    with pytest.raises(RuntimeError, match="Interactive browser authentication is disabled"):
        initialize_ee("example-project")

    assert authenticate_called is False
