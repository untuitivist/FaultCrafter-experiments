from __future__ import annotations

from pathlib import Path
from typing import Any

from .bjtu import BJTUAdapter
from .catalog import AdapterCollection
from .cwru import CWRUAdapter
from .hitsm import HITSMAdapter


def collection_from_config(config: dict[str, Any]) -> AdapterCollection:
    roots = config["roots"]
    return AdapterCollection(
        (
            BJTUAdapter(Path(roots["bjtu_rao"]), include_left_compounds=bool(config.get("bjtu_include_left_compounds", False))),
            CWRUAdapter(Path(roots["cwru"])),
            HITSMAdapter(Path(roots["hit_sm"])),
        )
    )
