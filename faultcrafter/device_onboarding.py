from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from faultcrafter.condition_units import condition_field, convert_condition_values


IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
FAULT_GROUPS = {"cage": 0, "inner_race": 1, "normal": 2, "outer_race": 3, "rolling_element": 4}
MAXIMUM_UPLOAD_BYTES = 128 * 1024 * 1024


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("标识只能包含英文字母、数字、下划线和短横线，最长128字符")
    return value


def frozen_fault_tokens(compact_root: Path) -> dict[str, dict[str, int]]:
    candidates: dict[int, list[int]] = {}
    for path in sorted(compact_root.glob("*/signal_profile.json")):
        for fault in json.loads(path.read_text(encoding="utf-8"))["faults"].values():
            candidates.setdefault(int(fault["fault_group_index"]), []).append(int(fault["fault_index"]))
    if set(candidates) != set(FAULT_GROUPS.values()):
        raise ValueError("发布权重不包含所需的冻结故障词表")
    return {label: {"fault_index": min(candidates[group]), "fault_group_index": group}
            for label, group in FAULT_GROUPS.items()}


def validate_manifest(payload: dict[str, Any], compact_root: Path) -> dict[str, Any]:
    manifest = json.loads(json.dumps(payload, allow_nan=False))
    if manifest.get("schema") != "faultcrafter_device_upload_v1":
        raise ValueError("请选择版本1设备数据清单")
    identifier(manifest.get("device_id"))
    for name in ("dataset_id", "sensor_location", "channel", "physical_unit"):
        value = manifest.get(name)
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise ValueError(f"请填写{name}")
    sample_rate = manifest.get("sample_rate_hz")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (float, int)) or not 1 <= sample_rate <= 10_000_000:
        raise ValueError("采样率须在1到10000000 Hz之间")
    if manifest.get("sampling_policy") != "declared_rate_sample_index":
        raise ValueError("本版使用声明采样率及样本顺序；不自动从不规则时间戳推算或重采样")
    names = manifest.get("parameter_names", [])
    if not 1 <= len(names) <= 2 or len(set(names)) != len(names):
        raise ValueError("32支持一到两个独立工况字段")
    for name in names:
        condition_field(name)
    speed_names = [name for name in names if name in ("shaft_speed_rpm", "shaft_speed_hz")]
    if len(speed_names) != 1:
        raise ValueError("新设备须提供实际轴转速，不能用驱动频率冒充轴转速")
    if len({condition_field(name).quantity for name in names}) != len(names):
        raise ValueError("同一物理量不能重复输入")
    fixed = manifest.setdefault("fixed_conditions", {})
    if not isinstance(fixed, dict) or set(fixed) & set(names):
        raise ValueError("固定工况和可变工况不能重叠")
    for name, value in fixed.items():
        condition_field(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError("固定工况必须是有限数值，使用字段声明的原生单位")
    records = manifest.get("records", [])
    if not isinstance(records, list) or not 2 <= len(records) <= 10000:
        raise ValueError("请提供2到10000条采集记录")
    tokens = frozen_fault_tokens(compact_root)
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    cells: dict[str, set[tuple[Any, ...]]] = {"train": set(), "validation": set()}
    for row in records:
        record_id = identifier(row.get("record_id"))
        filename = row.get("file", "")
        if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.(npy|npz|csv|mat)", filename):
            raise ValueError("文件名须为ASCII且不包含路径；支持npy、npz、csv、mat")
        if record_id in seen_ids or filename in seen_files:
            raise ValueError("每条采集记录和文件只能出现一次，不能跨训练和验证")
        seen_ids.add(record_id)
        seen_files.add(filename)
        role, label = row.get("split"), row.get("fault")
        if role not in cells or label not in tokens:
            raise ValueError("仅接收train/validation及已支持的故障语义，不接收测试记录")
        values = row.get("conditions", {})
        if set(values) != set(names):
            raise ValueError("每条记录须提供全部声明工况")
        raw = [values[name] for name in names]
        if any(isinstance(value, bool) or not isinstance(value, (float, int)) for value in raw):
            raise ValueError("工况必须为数值")
        converted = convert_condition_values(names, raw, input_units=manifest.get("condition_units"))
        if not np.isfinite(converted).all() or converted[names.index(speed_names[0])] <= 0:
            raise ValueError("工况须有限且实际转速须大于零")
        if any(condition_field(name).quantity in ("radial_force", "power") and converted[index] < 0 for index, name in enumerate(names)):
            raise ValueError("径向力与标注功率不能为负数")
        row["conditions"] = dict(zip(names, map(float, converted), strict=True))
        cells[role].add((label, *converted.tolist()))
    if cells["train"] != cells["validation"]:
        raise ValueError("每个故障×工况组合都须有独立训练和验证记录")
    manifest["condition_units"] = {name: condition_field(name).native_unit for name in names}
    manifest["fault_tokens"] = {label: tokens[label] for label in sorted({row["fault"] for row in records})}
    manifest["speed_parameter_index"] = names.index(speed_names[0])
    manifest["speed_to_hz"] = 1.0 / 60 if speed_names[0].endswith("rpm") else 1.0
    seed = manifest.setdefault("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("种子须为非负32位整数")
    verification = manifest.setdefault("verification_conditions", [])
    if not isinstance(verification, list) or len(verification) > 16:
        raise ValueError("导出校验最多接收16组工况")
    for values in verification:
        if len(values) != len(names) or not np.isfinite(np.asarray(values, dtype=float)).all() or values[manifest["speed_parameter_index"]] <= 0:
            raise ValueError("导出校验工况无效，须使用原生单位")
    return manifest


def read_signal(path: Path, *, channel: str, array_key: str = "signal") -> tuple[np.ndarray, dict[str, Any]]:
    audit: dict[str, Any] = {"sampling_policy": "declared_rate_sample_index"}
    if path.suffix == ".npy":
        signal = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            signal = archive[array_key]
    elif path.suffix == ".csv":
        signal = np.loadtxt(path, delimiter=",", encoding="utf-8")
    elif path.suffix == ".mat":
        from scipy.io import loadmat

        content = loadmat(path, simplify_cells=True)
        if array_key in content:
            signal = content[array_key]
        else:
            roots = [value for name, value in content.items() if not name.startswith("__")]
            if len(roots) != 1 or not isinstance(roots[0], dict):
                raise ValueError("MAT须包含signal数组或唯一的具名通道结构")
            record = roots[0]
            channels = record["Y"]
            if isinstance(channels, dict):
                channels = [channels]
            matches = [value for value in channels if value["Name"] == channel]
            if len(matches) != 1:
                raise ValueError("MAT中找不到唯一的指定通道")
            selected = matches[0]
            signal = selected["Data"]
            if "XIndex" in selected:
                axes = record["X"]
                if isinstance(axes, dict):
                    axes = [axes]
                times = np.asarray(axes[int(selected["XIndex"]) - 1]["Data"], dtype=float).reshape(-1)
                if times.size != np.size(signal) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                    raise ValueError("时间轴不递增或与通道长度不同")
                audit.update({"timestamp_duration_seconds": float(times[-1] - times[0]),
                              "timestamp_interval_min": float(np.diff(times).min()),
                              "timestamp_interval_max": float(np.diff(times).max()),
                              "timestamp_interval_median": float(np.median(np.diff(times)))})
    else:
        raise ValueError("不支持的信号文件类型")
    values = np.asarray(signal)
    if np.iscomplexobj(values) or values.dtype.kind not in "fiu" or values.ndim not in (1, 2) or (values.ndim == 2 and 1 not in values.shape):
        raise ValueError("请选择单通道实数一维信号，不能自动混合通道")
    values = values.reshape(-1).astype(np.float32)
    if not 16384 <= values.size <= 10_000_000 or not np.isfinite(values).all() or np.max(np.abs(values)) <= 0:
        raise ValueError("信号须含16384到10000000个有限采样点且不能全零")
    audit["samples"] = int(values.size)
    audit["signal_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
    return values, audit
