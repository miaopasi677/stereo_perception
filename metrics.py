"""M0 frame-rate metrics; no capture, inference, or acceptance claims.

Only measured (non-warmup) records belong here. Metadata is retained verbatim;
recommended keys are hardware, input, implementation (actual CPU/GPU execution),
scope_limitations, acceptance, and expected_stages. Metadata must be JSON-safe.
Missing stages/camera timings are not imputed as zero. Percentiles use linear
interpolation, matching numpy.percentile's default method, without requiring
NumPy at runtime.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import math
from numbers import Integral, Real
from pathlib import Path


_REQUIRED_FIELDS = (
    "frame_id", "completed_s", "latency_ms", "stages_ms",
    "object_count", "target_count", "json_bytes",
)


def _nonnegative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _percentile(ordered: list[float], percentile: float) -> float | None:
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timings(values: list[float], frame_count: int, *, p99: bool = True) -> dict:
    ordered = sorted(values)
    result = {
        "count": len(values),
        "missing_count": frame_count - len(values),
        "mean": math.fsum(value / len(values) for value in values) if values else None,
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
    }
    if p99:
        result["p99"] = _percentile(ordered, 99)
    return result


def _counts(values: list[int]) -> dict:
    total = sum(values)
    return {
        "total": total,
        "mean": total / len(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _worst_one_second_count(completed_s: list[float], elapsed_s: float) -> int | None:
    """Sweep every change in [s, s+1), not just integer-aligned bins.

    A completion at t enters just AFTER s=t-1 and leaves just AFTER s=t.
    Apply simultaneous events together. Evaluate s=0 separately; events at
    elapsed_s-1 must NOT be applied, since starts beyond that are invalid.
    Fractions of decimal timestamp representations avoid rounding t-1 onto the
    wrong side of a boundary. No epsilon or timestamp binning hides real gaps.
    Complexity: O(n log n) time, O(n) space.
    """
    final_start = Fraction(str(elapsed_s)) - 1
    if final_start < 0:
        return None
    current = 0
    events: dict[Fraction, int] = {}
    for value in completed_s:
        timestamp = Fraction(str(value))
        if 0 <= timestamp < 1:
            current += 1
        for start, change in ((timestamp - 1, 1), (timestamp, -1)):
            if 0 <= start < final_start:
                events[start] = events.get(start, 0) + change
    worst = current
    for start in sorted(events):
        current += events[start]
        worst = min(worst, current)
    return worst


def summarize_run(records: list[dict], elapsed_s: float, metadata: dict) -> dict:
    """Summarize measured records using the full supplied wall-clock duration.

    Empty runs have zero throughput and null timing statistics. Runs shorter
    than one second have no complete window and cannot pass either FPS gate.
    Nonempty zero-duration runs and invalid/nonfinite measurements are rejected.
    ``expected_stages`` may name stages that never appeared in the records.
    Passing an FPS gate never changes ``m0_only`` or certifies full acceptance.
    """
    if not isinstance(records, list) or not isinstance(metadata, dict):
        raise TypeError("records must be a list and metadata must be a dict")
    duration = _nonnegative_number(elapsed_s, "elapsed_s")
    if records and duration == 0:
        raise ValueError("a nonempty run requires elapsed_s > 0")
    expected_stages = metadata.get("expected_stages", [])
    if not isinstance(expected_stages, (list, tuple)) or any(
        not isinstance(stage, str) for stage in expected_stages
    ):
        raise ValueError("metadata.expected_stages must be a sequence of stage names")

    completed: list[float] = []
    latencies: list[float] = []
    camera_latencies: list[float] = []
    stages: dict[str, list[float]] = {name: [] for name in expected_stages}
    counts: dict[str, list[int]] = {
        "object_count": [], "target_count": [], "json_bytes": [],
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be a dict")
        missing = [name for name in _REQUIRED_FIELDS if name not in record]
        if missing:
            raise ValueError(f"records[{index}] missing fields: {', '.join(missing)}")
        timestamp = _nonnegative_number(record["completed_s"], "completed_s")
        if timestamp > duration:
            raise ValueError("completed_s must not exceed elapsed_s")
        completed.append(timestamp)
        latencies.append(_nonnegative_number(record["latency_ms"], "latency_ms"))
        if record.get("camera_latency_ms") is not None:
            camera_latencies.append(
                _nonnegative_number(record["camera_latency_ms"], "camera_latency_ms")
            )
        stage_values = record["stages_ms"]
        if not isinstance(stage_values, dict):
            raise ValueError("stages_ms must be a dict")
        for name, value in stage_values.items():
            if not isinstance(name, str):
                raise ValueError("stage names must be strings")
            timings = stages.setdefault(name, [])
            if value is not None:
                timings.append(_nonnegative_number(value, f"stages_ms.{name}"))
        for name in counts:
            counts[name].append(_nonnegative_integer(record[name], name))

    frame_count = len(records)
    throughput = frame_count / duration if duration else 0.0
    worst = _worst_one_second_count(completed, duration)
    return {
        "schema_version": 1,
        "m0_only": True,
        "frame_count": frame_count,
        "elapsed_s": duration,
        "throughput_fps": throughput,
        "completed_s": {
            "first": min(completed) if completed else None,
            "last": max(completed) if completed else None,
        },
        "latency_ms": _timings(latencies, frame_count),
        "camera_latency_ms": _timings(camera_latencies, frame_count),
        "stages_ms": {
            name: _timings(values, frame_count, p99=False)
            for name, values in sorted(stages.items())
        },
        **{name: _counts(values) for name, values in counts.items()},
        "worst_1s_count": worst,
        "full_1s_window_available": worst is not None,
        "pass_5_fps": throughput >= 5 and worst is not None and worst >= 5,
        "pass_10_fps": throughput >= 10 and worst is not None and worst >= 5,
        "metadata": deepcopy(metadata),
    }


def write_report(summary: dict, output_dir: Path) -> None:
    """Write UTF-8 summary.json and a Chinese report.txt; do not infer hardware.

    Every metadata field and metric is included without truncation. In
    particular, acceptance evidence/limitations supplied by the caller survive
    unchanged. The FPS pass flags are never presented as full acceptance.
    """
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    metadata = summary.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("summary.metadata must be a dict")

    def display(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)

    unspecified = "未提供，不能推断或视为已验证"
    implementation = metadata.get(
        "implementation", metadata.get("actual_implementation", unspecified)
    )
    metrics = {key: value for key, value in summary.items() if key != "metadata"}
    report = "\n".join([
        "双目视觉 M0 帧率基准报告",
        "=" * 64,
        "【重要：仅 M0 帧率基准，不是完整验收报告】",
        "FPS 门槛通过不等于文档要求的 1800 秒（30 分钟）验收通过。",
        "真实摄像头标定、三目标场景亦不能由本帧率统计证明已验收。",
        "是否实测及尚未验收的事实，以以下原样元数据和范围限制为准。",
        "",
        "【硬件（hardware）】",
        display(metadata.get("hardware", unspecified)),
        "【输入及数据类型（input）】",
        display(metadata.get("input", unspecified)),
        "【实际实现 CPU/GPU（implementation）】",
        display(implementation),
        "拥有 GPU 不代表使用了 GPU；不根据硬件配置推断执行后端。",
        "【全部范围限制（scope_limitations）】",
        display(metadata.get("scope_limitations", "未提供限制声明，不表示没有限制")),
        "",
        "【统计口径】",
        "吞吐量 throughput_fps = frame_count / elapsed_s（完整实测墙钟时间）。",
        "不平均瞬时 FPS；不以末帧时间或耗时总和替代吞吐分母；不含 warmup。",
        "latency_ms：从开始读取至输出 JSONL flush 完成的端到端耗时（毫秒）。",
        "camera_latency_ms：可选摄像头耗时（毫秒），只统计实际提供的样本。",
        "耗时包含均值、p50、p95、p99；stages_ms 为各阶段均值、p50、p95。",
        "分位数使用线性插值；count/missing_count 标记已记录/缺失数量。",
        "object_count / target_count / json_bytes：总量、均值、最小值、最大值。",
        "最差窗口遍历 0 <= start <= elapsed_s - 1 的全部 [start, start+1)。",
        "时间戳按十进制表示精确比较；同一端点事件一起处理，不按整数秒分箱。",
        "不足 1 秒无完整窗口（null），两项 FPS 门槛均不通过；缺失耗时不填零。",
        "pass_5_fps：真实吞吐 >= 5 且任意完整 1 秒窗口完成次数 >= 5。",
        "pass_10_fps：真实吞吐 >= 10 且任意完整 1 秒窗口完成次数 >= 5。",
        "m0_only 始终为 true；这两项门槛不宣称长测、摄像头标定或三目标验收。",
        "",
        "【记录的全部 metrics】",
        display(metrics),
        "",
        "【完整原样 metadata：硬件、输入类型、阶段缺失与验收事实】",
        display(metadata),
        "",
    ])
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(summary_json + "\n", encoding="utf-8")
    (directory / "report.txt").write_text(report, encoding="utf-8")
