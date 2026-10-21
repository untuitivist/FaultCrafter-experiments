from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path

import torch

from .artifact_selection import dependencies, registered_jobs
from .common import ROOT, file_hash, read_json, timestamp, write_json
from .inference import collect


JOB_FILES = ("best.pt", "job.json", "result.json", "history.jsonl", "progress.json", "cost.json", "timing.json",
             "source_manifest.json", "target_manifest.json", "seen_manifest.json", "target_predictions.npz",
             "seen_predictions.npz", "generation.json", "failure.json")
GENERATOR_FILES = ("best.pt", "job.json", "result.json", "cost.json", "source_manifest.json", "auxiliary_training.json", "history.jsonl", "progress.json", "failure.json")
ROOT_FILES = ("coverage.json", "generator_coverage.json", "runtime.json", "dependencies.json", "report.md", "report.json",
              "report.csv", "summary.json", "summary.csv", "generator_report.csv", "generator_summary.json",
              "generator_summary.csv", "diagnostic_groups.json", "diagnostic_group_summary.json", "diagnostic_group_summary.csv",
              "quality_summary.json", "audit.json", "condition_preflight.json", "preflight.json", "paper_tables.json",
              "comparison_experiments.xlsx", "workbook.json", "workbook_data.json", "analysis.json",
              "parallel_workers.json", "worker_imports.jsonl", "evaluation_cache.json", "evaluation_cache_equivalence.json",
              "status.json", "execution.log", "quality.log", "eta_projection.json", "training_recipes.json", "training_budgets.json",
              "selection.json", "launcher.json")
FIGURE_CATALOG = [
    {"figure": "Time waveforms, FFT, PSD, envelope spectra and time-frequency maps", "data": "inference/**/signals.npz; sample rate and time/frequency axes", "status": "frozen_model_inference"},
    {"figure": "Amplitude histograms, 12 time-domain indicators and indicator correlations", "data": "inference/**/*_statistics.npz; normalized/original-scale real waveform panels", "status": "frozen_model_inference"},
    {"figure": "Joint UMAP/PCA, nearest neighbors, content/style and distribution distances", "data": "inference/**/*_features.npz; source-fitted transforms; quality/*.json", "status": "frozen_model_inference"},
    {"figure": "ROC, precision-recall, component and fault-set confusion matrices, per-component F1", "data": "jobs/*/{target,seen}_predictions.npz; label dictionaries; diagnostic_groups", "status": "saved_predictions"},
    {"figure": "Learning curves, training-loss distributions and source-validation selection", "data": "jobs/*/history.jsonl; generators/*/history.jsonl; auxiliary phases, learning rates and stop reasons", "status": "saved_training_history"},
    {"figure": "T1-T4 comparison tables, five-run SD, paired gains and statistical tests", "data": "all per-run result.json, prediction arrays, group identities and summary tables", "status": "saved_results"},
    {"figure": "Parameters, FLOPs, NFE, adaptation/generation time and memory", "data": "generator and CNN cost/result/runtime records", "status": "saved_costs"},
    {"figure": "Learned mixing weights, generator branch features and controlled condition scans", "data": "all final weights; sample.py module hooks and saved request/seed/condition traces", "status": "rerunnable_inference; architecture-specific branches require the final proposed model"},
    {"figure": "Architecture ablations and content/style interventions", "data": "results and configurations of the corresponding registered core experiments", "status": "requires_final_model; no new training is added by export"},
    {"figure": "Platform photos and method architecture diagrams", "data": "existing manuscript assets and final method definition", "status": "static_paper_assets; not experiment-generated"},
]


class Transcript:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def payload_files(run, data_run, artifacts, jobs):
    diagnostics, generators = dependencies(run, jobs)
    entries = {}

    def add(path, name):
        if path.is_file():
            if name in entries and entries[name] != path:
                raise RuntimeError(f"Archive name collision: {name}")
            entries[name] = path

    for name in ROOT_FILES:
        add(run / name, "results/" + name)
    analysis_link = run / "analysis.json"
    if analysis_link.exists():
        analysis = Path(read_json(analysis_link)["directory"])
        for path in analysis.rglob("*"):
            if any(part.startswith("interrupted_") for part in path.relative_to(analysis).parts):
                continue
            if path.is_file() and path.suffix.lower() in (".json", ".jsonl", ".npz", ".pt", ".log", ".csv", ".png", ".pdf"):
                add(path, "analysis/" + path.relative_to(analysis).as_posix())
    for job in diagnostics:
        for name in JOB_FILES:
            add(run / "jobs" / job["id"] / name, f"results/jobs/{job['id']}/{name}")
        for directory in ("quality", "diagnostic_groups"):
            add(run / directory / (job["id"] + ".json"), f"results/{directory}/{job['id']}.json")
    for job in generators:
        for name in GENERATOR_FILES:
            add(run / "generators" / job["id"] / name, f"results/generators/{job['id']}/{name}")
    for name in ("inventory.json", "plan.json", "data_ready.json", "preflight.json"):
        add(data_run / name, "metadata/" + name)
    for split in sorted({job["split"] for job in jobs}):
        add(data_run / "partitions" / f"{split}.json", f"metadata/partitions/{split}.json")
        add(data_run / "cache/windows" / str(split) / "manifest.json", f"metadata/window_manifests/{split}.json")
    for path in artifacts.rglob("*"):
        add(path, "inference/" + path.relative_to(artifacts).as_posix())
    roots = (ROOT / "research/benchmarks", ROOT / "faultcrafter", ROOT / "research/protocols",
             ROOT / "third_party/comparison_upstream", ROOT / "third_party/comparison_sources/ccgan")
    for folder in roots:
        for path in folder.rglob("*"):
            if any(part in (".git", "__pycache__", ".venv", "node_modules") for part in path.parts):
                continue
            if path.name in ("AGENTS.md", "SKILL.md", ".NOTE.md"):
                continue
            if path.suffix.lower() in (".py", ".mjs", ".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".cmd") or path.name.startswith("LICENSE"):
                add(path, "code/" + path.relative_to(ROOT).as_posix())
    for name in ("pyproject.toml", "LICENSE"):
        add(ROOT / name, "code/" + name)
    return entries


def write_archive(destination, entries, metadata):
    if destination.exists() or destination.with_suffix(destination.suffix + ".partial").exists():
        raise FileExistsError("Use a fresh archive filename; existing exports are preserved")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    manifest = []
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        def add_document(name, content):
            payload = content.encode("utf-8")
            archive.writestr(name, payload)
            manifest.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})

        for index, (name, path) in enumerate(sorted(entries.items()), 1):
            before = path.stat()
            checksum = hashlib.sha256()
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in (".npz", ".pdf", ".png") else zipfile.ZIP_DEFLATED
            entry._compresslevel = 1
            entry.file_size = before.st_size
            with path.open("rb") as source, archive.open(entry, "w", force_zip64=True) as target:
                while block := source.read(2**20):
                    checksum.update(block)
                    target.write(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"An input changed while exporting: {path}")
            manifest.append({"path": name, "bytes": before.st_size, "sha256": checksum.hexdigest()})
            if index % 100 == 0:
                print(f"archive {index}/{len(entries)} files", flush=True)
        add_document("figure_catalog.json", json.dumps(FIGURE_CATALOG, ensure_ascii=False, indent=2))
        add_document("README.txt", "FaultCrafter complete research delivery\n\n"
            "results/: all final generator/CNN weights, histories, per-sample predictions, five-repeat tables and costs.\n"
            "inference/: all saved representations/statistics, balanced waveform panels, physical scales for real signals, figures.\n"
            "metadata/: dataset/condition/label dictionaries, split groups, normalization and sampling identities.\n"
            "code/: frozen inference/export/plotting code, required model implementations and upstream licenses.\n\n"
            "To generate again after extraction, use code as the working directory and run:\n"
            "python -m research.benchmarks.sample --generator ../results/generators/ID --inventory ../metadata/inventory.json --requests requests.json --output ../new_samples\n"
            "Each request supplies device, channel, state, coordinates, count, and native length 1024.\n"
            "To redraw locally: python -m research.benchmarks.figures --artifacts ../inference --run ../results\n"
            "To diagnose saved signals: python -m research.benchmarks.diagnose --checkpoint ../results/jobs/ID/best.pt --signals ../inference/references/REFERENCE/target_waveforms.npz --output ../new_diagnostics\n"
            "No original datasets, optimizer/resume snapshots or disposable training caches are required for these operations.\n"
            "The saved raw representations permit a joint UMAP using optional umap-learn; PCA is available without it.\n"
            "Payload file hashes are listed in manifest.json. The archive checksum is supplied beside this ZIP.\n")
        archive.writestr("manifest.json", json.dumps({**metadata, "files": manifest}, ensure_ascii=False, indent=2))
    os.replace(temporary, destination)
    checksum = file_hash(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(checksum + "  " + destination.name + "\n", encoding="utf-8")
    write_json(destination.with_suffix(destination.suffix + ".json"), {**metadata, "archive": str(destination),
               "bytes": destination.stat().st_size, "sha256": checksum, "files": len(manifest)})
    return destination


def deliver(run, data_run, destination, *, device="auto"):
    if destination.exists() or destination.with_suffix(destination.suffix + ".partial").exists():
        raise FileExistsError("Use a fresh archive filename; existing exports are preserved")
    status = read_json(run / "status.json")
    if status["state"] != "complete":
        raise RuntimeError("The fixed complete delivery starts only after the comparison queue finishes")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    from .audit import audit
    from .diagnostic_groups import evaluate
    from .report import generate_report

    print("Delivery: checking completed jobs, all five repeats, checkpoint hashes and paired data", flush=True)
    if not audit(run)["passed"]:
        raise RuntimeError("The completed run failed the execution audit")
    evaluate(run)
    from .quality import evaluate as evaluate_quality

    evaluate_quality(data_run, run, device="cpu")
    generate_report(run)
    from .workbook import generate_workbook

    generate_workbook(run, data_run)
    jobs = list(registered_jobs(run))
    artifacts = run / "inference_artifacts"
    collect(run, data_run, artifacts, jobs, device=device)
    from .figures import render

    render(artifacts, run)
    log_path = run / "delivery.log"
    if log_path.exists():
        (artifacts / "delivery_prearchive.log").write_bytes(log_path.read_bytes())
    metadata = {"created": timestamp(), "policy": "fixed_complete_research_delivery_v1", "jobs": len(jobs),
                "seeds": sorted({job["split"] for job in jobs}), "standard_deviation": "sample SD, ddof=1",
                "coverage": read_json(run / "coverage.json")["sampling"], "inference_sha256": file_hash(artifacts / "manifest.json"),
                "not_included": ["original datasets", "rebuildable raw/window caches", "optimizer/resume snapshots"]}
    entries = payload_files(run, data_run, artifacts, jobs)
    print(f"Delivery: packaging {len(entries)} files; {sum(path.stat().st_size for path in entries.values()) / 2**30:.2f} GiB before ZIP compression", flush=True)
    return write_archive(destination, entries, metadata)


def main():
    parser = argparse.ArgumentParser(description="Create one fixed, complete research ZIP after all jobs finish.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    arguments.run.mkdir(parents=True, exist_ok=True)
    with (arguments.run / "delivery.log").open("a", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Transcript(sys.stdout, logfile)), contextlib.redirect_stderr(Transcript(sys.stderr, logfile)):
            try:
                print(deliver(arguments.run.resolve(), arguments.data_run.resolve(), arguments.output.resolve(), device=arguments.device))
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == "__main__":
    main()
