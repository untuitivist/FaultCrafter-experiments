from __future__ import annotations

import os
import shutil
import uuid

from pathlib import Path

from .common import canonical, file_hash, identity, read_json, timestamp


def publish_directory(source, destination, receipt_name, identity_field):
    original = read_json(source / receipt_name)
    if destination.exists():
        existing_path = destination / receipt_name
        if not existing_path.exists() or read_json(existing_path).get(identity_field) != original.get(identity_field):
            raise RuntimeError(f"Publish target contains another or unfinished case: {destination}")
        return False
    root = destination.parent.parent.resolve()
    staging = root / ".incoming" / uuid.uuid4().hex
    if not destination.resolve().is_relative_to(root) or not staging.resolve().is_relative_to(root):
        raise ValueError("Publish path is outside its explicitly selected result directory")
    staging.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging)
    if file_hash(staging / receipt_name) != file_hash(source / receipt_name):
        raise RuntimeError("Published receipt checksum differs from worker receipt")
    os.rename(staging, destination)
    return True


def publish_case(worker, primary, job):
    if worker.resolve() == primary.resolve():
        raise ValueError("Worker and primary result directories must differ")
    if job["parent"]:
        worker_generator = read_json(worker / "generators" / job["parent"] / "result.json")
        primary_generator = read_json(primary / "generators" / job["parent"] / "result.json")
        if worker_generator["best_checkpoint_sha256"] != primary_generator["best_checkpoint_sha256"]:
            raise RuntimeError("Worker and primary must use the same frozen generator asset")
        synthetic_id = identity({key: job.get(key) for key in ("parent", "device", "channel", "track", "target", "split")})[:24]
        publish_directory(worker / "synthetic" / synthetic_id, primary / "synthetic" / synthetic_id,
                          "manifest.json", "signals_sha256")
    published = publish_directory(worker / "jobs" / job["id"], primary / "jobs" / job["id"], "result.json", "identity")
    if not published:
        return
    with (primary / "worker_imports.jsonl").open("ab") as stream:
        stream.write(canonical({"time": timestamp(), "worker": str(worker), "job": job["id"],
                                "split": job["split"], "result_sha256": file_hash(worker / "jobs" / job["id"] / "result.json")}) + b"\n")
