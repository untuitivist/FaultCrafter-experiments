from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_reference = payload.pop("base_config", None)
    if base_reference is None:
        return payload
    base_path = Path(str(base_reference))
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    merged = load_config(base_path)
    merged.update(payload)
    return merged
