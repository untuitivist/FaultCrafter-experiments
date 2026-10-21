from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile


REPO_ID = "untuitivist/faultcrafter"
RELEASE = "prepared-v19-20261021"
BLOCK = 8 * 1024 * 1024


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*path.parts)


def destination(root, value):
    target = root / relative_path(value)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes output directory: {value!r}")
    return target


def restore(snapshot, output, release=RELEASE):
    release_root = snapshot / "releases" / relative_path(release)
    manifest = json.loads((release_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "faultcrafter-prepared-v1" or manifest["release"] != release:
        raise ValueError("Unrecognized dataset manifest")
    output.mkdir(parents=True, exist_ok=True)
    expected_files = set()
    for archive in manifest["archives"]:
        source = destination(release_root, archive["path"])
        if source.stat().st_size != archive["bytes"] or digest(source) != archive["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {source.name}")
        expected = {member["path"]: member for member in archive["members"]}
        if len(expected) != len(archive["members"]) or expected_files.intersection(expected):
            raise ValueError("Duplicate files in manifest")
        expected_files.update(expected)
        print(f"Verified archive: {source.name}", flush=True)
        with zipfile.ZipFile(source) as bundle:
            members = bundle.infolist()
            if len(members) != len(expected) or {member.filename for member in members} != set(expected):
                raise ValueError(f"Archive member mismatch: {source.name}")
            for member in members:
                metadata = expected[member.filename]
                if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Unsupported member: {member.filename}")
                if member.file_size != metadata["bytes"]:
                    raise ValueError(f"Member size mismatch: {member.filename}")
                target = destination(output, member.filename)
                if target.exists():
                    if target.is_file() and target.stat().st_size == metadata["bytes"] and digest(target) == metadata["sha256"]:
                        continue
                    raise FileExistsError(f"Existing file differs; use a new output directory: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".restoring")
                checksum = hashlib.sha256()
                with bundle.open(member) as incoming, temporary.open("xb") as outgoing:
                    while block := incoming.read(BLOCK):
                        checksum.update(block)
                        outgoing.write(block)
                if checksum.hexdigest() != metadata["sha256"]:
                    raise ValueError(f"Extracted checksum mismatch: {member.filename}")
                os.replace(temporary, target)
        print(f"Restored: {source.name}", flush=True)
    if len(expected_files) != manifest["file_count"]:
        raise ValueError("Dataset file count mismatch")
    receipt = {"release": release, "file_count": len(expected_files),
               "uncompressed_bytes": manifest["uncompressed_bytes"],
               "manifest_sha256": digest(release_root / "manifest.json")}
    receipt_path = output / "dataset_restore_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"restored": str(output.resolve()), **receipt}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description="Download and verify the prepared FaultCrafter dataset.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", default=RELEASE)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--download", action="store_true")
    arguments = parser.parse_args()
    relative_path(arguments.release)
    if arguments.download:
        from modelscope.hub.snapshot_download import dataset_snapshot_download

        dataset_snapshot_download(
            REPO_ID, revision=arguments.revision, local_dir=str(arguments.snapshot),
            allow_patterns=[f"releases/{arguments.release}/*", "restore_dataset.py", "README.md"],
            token=os.environ.get("MODELSCOPE_API_TOKEN"), max_workers=4,
        )
    restore(arguments.snapshot, arguments.output, arguments.release)


if __name__ == "__main__":
    main()
