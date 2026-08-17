from __future__ import annotations

import re
import os
from pathlib import Path


def test_maintained_application_uses_only_california_time_wording():
    application = Path(__file__).resolve().parents[2]
    milestone = application.parent
    roots = [application, milestone / "USER_GUIDE.md", milestone / "DEVELOPER_GUIDE.md"]
    excluded = {"node_modules", "dist", ".git", ".pytest_cache", "__pycache__", ".venv", ".state"}
    disallowed = re.compile(
        "|".join(("Ind" + "ia", "Ind" + "ian", r"\b" + "I" + "ST" + r"\b")),
        re.IGNORECASE,
    )
    matches = []
    for root in roots:
        if root.is_file():
            paths = [root]
        else:
            paths = []
            for directory, dirnames, filenames in os.walk(root):
                dirnames[:] = [name for name in dirnames if name not in excluded]
                paths.extend(Path(directory) / name for name in filenames)
        for path in paths:
            if not path.is_file() or any(part in excluded for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if disallowed.search(text):
                matches.append(str(path))
    assert matches == []
