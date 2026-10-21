from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research/protocols/task_taxonomy_and_protocol_20261018_zh.md"
DATA_ROOT = Path(os.environ.get("FAULTCRAFTER_DATA_ROOT", "Z:/DATASETS/Bearing"))
RUN_ROOT = Path(os.environ.get("FAULTCRAFTER_RUN_ROOT", ROOT / "outputs/comparisons_v19_20261020"))
SPLITS = (42, 43, 44, 45, 46)
T1_DEVICES = ("hdu", "dirg", "hust", "cwru")
T2_DEVICES = ("bjtu", "hit_sq", "hit_self", "pu")
METHODS = ("diffucd", "ccgan", "iccdm", "time_weaver", "flowts")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def identity(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def seed_for(*parts: object) -> int:
    return int(identity({"protocol": "v19", "root_seed": 42, "parts": parts})[:16], 16)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def log(event: str, **values) -> None:
    print(json.dumps({"time": timestamp(), "event": event, **values}, ensure_ascii=False, allow_nan=False), flush=True)
