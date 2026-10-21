from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from faultcrafter.experiment0 import canonical_digest
from faultcrafter.models import CNN1D
from faultcrafter.semantic_pooling import NATIVE_SEMANTIC_POOLING, configure_semantic_pooling


def load_frozen_source_classifier(
    path: Path, labels: tuple[str, ...], split_sha256: str, experiment0: dict,
) -> tuple[CNN1D, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["labels"]) != labels or tuple(checkpoint["fault_labels"]) != labels:
        raise ValueError("source semantic classifier fault order differs")
    if checkpoint["split_sha256"] != split_sha256 or checkpoint["config_sha256"] != canonical_digest(experiment0):
        raise ValueError("source semantic classifier belongs to different source split/configuration")
    if int(checkpoint["seed"]) != int(experiment0["seed"]):
        raise ValueError("source semantic classifier seed differs")
    if checkpoint["input_normalization"] != "record_max_abs_before_windowing":
        raise ValueError("source semantic classifier normalization differs")
    with torch.random.fork_rng(devices=[]):
        classifier = CNN1D(num_classes=len(labels))
    classifier.load_state_dict(checkpoint["model"], strict=True)
    classifier.requires_grad_(False).eval()
    return classifier, {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "seed": checkpoint["seed"], "split_sha256": split_sha256, "labels": list(labels),
                        "config_sha256": checkpoint["config_sha256"], "input_normalization": checkpoint["input_normalization"],
                        "frozen_training_only": True, "exported": False}


def confidence_weighted_semantic_kl(
    source_logits: torch.Tensor, prediction_logits: torch.Tensor, fault: torch.Tensor, time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if source_logits.ndim != 2 or prediction_logits.shape != source_logits.shape or fault.shape != (len(source_logits),) or time.shape != fault.shape:
        raise ValueError("semantic KL tensor shapes differ")
    if fault.dtype != torch.long or torch.any((fault < 0) | (fault >= source_logits.shape[1])):
        raise ValueError("semantic KL fault is outside classifier vocabulary")
    if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("semantic KL time must lie in[0,1]")
    with torch.no_grad():
        reference = source_logits.softmax(dim=1)
        correct = reference.gather(1, fault[:, None]).squeeze(1)
        alternatives = reference.clone().scatter_(1, fault[:, None], 0.0)
        confidence = (correct - alternatives.amax(dim=1)).clamp_min(0.0)
        eligible = source_logits.argmax(dim=1) == fault
    divergence = F.kl_div(prediction_logits.log_softmax(dim=1), reference, reduction="none").sum(dim=1).clamp_min(0.0)
    return divergence * confidence * (1.0 - time).square(), confidence, eligible


@dataclass(frozen=True, slots=True)
class SourceSemanticTeachers:
    classifiers: tuple[CNN1D, ...]
    record_maxima: tuple[torch.Tensor, ...]
    fault_offsets: tuple[int, ...]
    view_length: int
    audit: dict

    @classmethod
    def from_devices(cls, devices: Sequence[Any], experiment0: dict, compute_device: torch.device,
                     *, pooling_policy: str = NATIVE_SEMANTIC_POOLING):
        classifiers, record_maxima, offsets, audit = [], [], [], []
        specifications = {item["run_id"]: item for item in experiment0["experiments"]}
        expected_offset = 0
        for device in devices:
            if device.index != len(classifiers) or device.fault_offset != expected_offset or getattr(device, "model_fault_labels", None) is not None:
                raise ValueError("source semantic training requires contiguous single-fault device vocabularies")
            source_conditions = set(specifications[device.run_id]["source_conditions"])
            if any(record.record.condition_label not in source_conditions for record in device.corpus.records):
                raise ValueError("semantic teacher normalization must never read target records")
            labels = tuple(device.corpus.fault_labels)
            path = Path(experiment0["output_root"]) / device.run_id / "best_cnn.pt"
            classifier, metadata = load_frozen_source_classifier(path, labels, device.split_sha256, experiment0)
            configure_semantic_pooling(classifier, pooling_policy)
            maxima = np.asarray([np.max(np.abs(record.signal)) for record in device.corpus.records], dtype=np.float32)
            if not len(maxima) or not np.isfinite(maxima).all() or np.any(maxima <= 0):
                raise ValueError("source semantic record scales must be finite and positive")
            classifiers.append(classifier.to(compute_device))
            record_maxima.append(torch.from_numpy(maxima).to(compute_device))
            offsets.append(device.fault_offset)
            audit.append({**metadata, "device_id": device.device_id, "source_record_count": len(maxima)})
            expected_offset += len(labels)
        if not classifiers or int(experiment0["window_length"]) < 16:
            raise ValueError("source semantic teachers need source devices and valid CNN window length")
        metadata = {"teachers": audit, "view_length": int(experiment0["window_length"]),
                    "normalization": "parent_amplitude_over_source_record_max_abs",
                    "runtime_dependencies_added": False, "record_scales_saved": False,
                    "confidence_gate": "positive_true_probability_minus_largest_other", "time_gate": "(1-time)^2"}
        if pooling_policy != NATIVE_SEMANTIC_POOLING:
            metadata["pooling_policy"] = pooling_policy
        return cls(tuple(classifiers), tuple(record_maxima), tuple(offsets), int(experiment0["window_length"]), metadata)

    def objective(
        self, prediction: torch.Tensor, clean: torch.Tensor, fault: torch.Tensor, device_id: torch.Tensor,
        record_index: torch.Tensor, parent_amplitude: torch.Tensor, time: torch.Tensor,
        *, prediction_normalization: str = "source_parent",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if prediction_normalization not in {"source_parent", "generated_record"}:
            raise ValueError("unknown semantic prediction normalization")
        if clean.shape != prediction.shape or clean.ndim != 3 or clean.shape[1] != 1 or clean.shape[-1] < 16:
            raise ValueError("source semantic views require equal [batch,1,length>=16] signals")
        if any(value.shape != (len(clean),) for value in (fault, device_id, record_index, parent_amplitude, time)):
            raise ValueError("source semantic routing shapes differ")
        if fault.dtype != torch.long or device_id.dtype != torch.long or record_index.dtype != torch.long:
            raise ValueError("source semantic routing must use int64")
        if torch.any((device_id < 0) | (device_id >= len(self.classifiers))):
            raise ValueError("source semantic device is outside source vocabulary")
        if prediction_normalization == "generated_record" and (not torch.isfinite(prediction).all() or torch.any(prediction.abs().amax(dim=(1, 2)) > 1.00001)):
            raise ValueError("generated semantic input must use its own record max_abs normalization")
        length = min(self.view_length, clean.shape[-1])
        start = (clean.shape[-1] - length) // 2
        losses, confidence_values, eligible_values = [], [], []
        for device_index, classifier in enumerate(self.classifiers):
            selected = device_id == device_index
            if not torch.any(selected):
                continue
            if classifier.training or any(parameter.requires_grad for parameter in classifier.parameters()):
                raise ValueError("source semantic classifiers must stay frozen in eval mode")
            indices = record_index[selected]
            if torch.any((indices < 0) | (indices >= len(self.record_maxima[device_index]))):
                raise ValueError("source semantic record index is outside source corpus")
            scale = parent_amplitude[selected] / self.record_maxima[device_index][indices]
            if not torch.isfinite(scale).all() or torch.any(scale <= 0):
                raise ValueError("semantic parent amplitude must belong to the declared source record")
            reference = clean[selected, :, start:start + length] * scale[:, None, None]
            if torch.any(reference.abs().amax(dim=(1, 2)) > 1.00001):
                raise ValueError("semantic source view exceeds its declared source record maximum")
            generated = prediction[selected, :, start:start + length]
            if prediction_normalization == "source_parent":
                generated = generated * scale[:, None, None]
            with torch.no_grad():
                source_logits = classifier(reference)
            prediction_logits = classifier(generated)
            losses_for_device, confidence, eligible = confidence_weighted_semantic_kl(
                source_logits, prediction_logits, fault[selected] - self.fault_offsets[device_index], time[selected],
            )
            losses.append(losses_for_device)
            confidence_values.append(confidence)
            eligible_values.append(eligible)
        loss = torch.cat(losses).mean()
        return loss, {"source_semantic_loss": float(loss.detach()),
                      "source_semantic_confidence_mean": float(torch.cat(confidence_values).mean()),
                      "source_semantic_correct_fraction": float(torch.cat(eligible_values).float().mean())}
