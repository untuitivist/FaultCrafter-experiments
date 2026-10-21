from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from faultcrafter import __version__


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="faultcrafter", description="Vibration generation and device adaptation")
    root.add_argument("--version", action="version", version=f"FaultCrafter {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-root", type=Path, help="Compact model directory; defaults to FAULTCRAFTER_MODEL_ROOT")
    common.add_argument("--user-data-root", type=Path, help="Device and job storage; defaults to FAULTCRAFTER_DATA_ROOT")
    common.add_argument("--compute-device", choices=("cpu", "cuda"))
    inspect = commands.add_parser("inspect", parents=[common], help="Print model identity, routes and units as JSON")
    inspect.add_argument("--output", type=Path)
    for command, description in (("generate", "Generate one signal as JSON"), ("batch", "Generate a batch as a ZIP archive")):
        action = commands.add_parser(command, parents=[common], help=description)
        action.add_argument("--request", required=True, type=Path)
        action.add_argument("--output", required=True, type=Path)
    gui = commands.add_parser("gui", parents=[common], help="Start the local browser GUI")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--no-browser", action="store_true")
    device = commands.add_parser("device", help="Prepare, train or inspect an uploaded device")
    actions = device.add_subparsers(dest="device_action", required=True)
    prepare = actions.add_parser("prepare", parents=[common], help="Copy and validate source training and validation records")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--signals", type=Path, required=True)
    for action in ("train", "status"):
        command = actions.add_parser(action, parents=[common])
        command.add_argument("job_id")
    return root


def read_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("JSON request must be an object")
    return payload


def write_result(payload: Any, destination: Path | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if destination is None:
        sys.stdout.write(text)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)


def execute(arguments: argparse.Namespace) -> None:
    if arguments.command == "gui":
        from faultcrafter.gui.server import main as launch_gui
        options = ["--port", str(arguments.port)]
        for option in ("model_root", "user_data_root", "compute_device"):
            value = getattr(arguments, option)
            if value is not None:
                options.extend(["--" + option.replace("_", "-"), str(value)])
        if arguments.no_browser:
            options.append("--no-browser")
        launch_gui(options)
        return
    from faultcrafter.services.application import RuntimeApplication
    output = getattr(arguments, "output", None)
    if output is not None and output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    application = RuntimeApplication(arguments.model_root, compute_device=arguments.compute_device,
                                     user_data_root=arguments.user_data_root)
    if arguments.command == "inspect":
        write_result(application.public_manifest(), output)
    elif arguments.command == "generate":
        write_result(application.generate(read_request(arguments.request)), output)
    elif arguments.command == "batch":
        job = application.start_batch(read_request(arguments.request))
        status = application.batch_jobs.wait(job["job_id"])
        if status["status"] != "completed":
            raise RuntimeError(status.get("error") or "Batch generation failed")
        archive, _ = application.batch_jobs.archive(job["job_id"])
        output.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("rb") as source, output.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        write_result({"output": str(output.resolve()), "job": status})
    else:
        manager = application.device_jobs()
        if arguments.device_action == "prepare":
            payload = read_request(arguments.manifest)
            from faultcrafter.device_onboarding import validate_manifest
            checked = validate_manifest(payload, application.compact_root)
            signals_root = arguments.signals.resolve()
            for row in checked["records"]:
                candidate = (signals_root / row["file"]).resolve()
                if not candidate.is_relative_to(signals_root) or not candidate.is_file():
                    raise ValueError(f"Source record missing or outside the signals directory: {row['file']}")
            created = manager.create(payload)
            for row in checked["records"]:
                source = signals_root / row["file"]
                with source.open("rb") as stream:
                    manager.upload(created["job_id"], row["file"], stream, source.stat().st_size)
            write_result(manager.validate(created["job_id"]))
        elif arguments.device_action == "train":
            manager.start(arguments.job_id)
            status = manager.wait(arguments.job_id)
            write_result(status)
            if status["state"] != "ready":
                raise RuntimeError(status.get("error") or "Device training failed")
        else:
            write_result(manager.status(arguments.job_id))


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        execute(arguments)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        message = str(error)
        if isinstance(error, ImportError):
            message += "; install training support with: pip install 'faultcrafter[training]'"
        sys.stderr.write(json.dumps({"error": message}, ensure_ascii=False) + "\n")
        return 2
    return 0
