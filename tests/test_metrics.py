"""Synthetic metrics tests; these are not hardware benchmark results."""

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from metrics import summarize_run, write_report


def make_records(times, latencies=None):
    if latencies is None:
        latencies = [20.0] * len(times)
    return [
        {
            "frame_id": index,
            "completed_s": timestamp,
            "latency_ms": latency,
            "stages_ms": {"read": latency / 4, "json_flush": latency / 2},
            "object_count": index % 4,
            "target_count": index % 2,
            "json_bytes": 100 + index,
        }
        for index, (timestamp, latency) in enumerate(zip(times, latencies))
    ]


def brute_force_worst(times, duration):
    """Independent oracle: evaluate endpoints and all intervening intervals."""
    end = Fraction(str(duration)) - 1
    if end < 0:
        return None
    timestamps = [Fraction(str(value)) for value in times]
    points = {Fraction(0), end}
    for timestamp in timestamps:
        for event in (timestamp, timestamp - 1):
            if 0 <= event <= end:
                points.add(event)
    ordered = sorted(points)
    starts = ordered + [(left + right) / 2 for left, right in zip(ordered, ordered[1:])]
    return min(sum(start <= value < start + 1 for value in timestamps) for start in starts)


class SummarizeRunTests(unittest.TestCase):
    def test_regular_ten_fps(self):
        summary = summarize_run(make_records([index / 10 for index in range(100)]), 10, {})
        self.assertEqual(summary["frame_count"], 100)
        self.assertEqual(summary["throughput_fps"], 10)
        self.assertEqual(summary["worst_1s_count"], 10)
        self.assertTrue(summary["pass_5_fps"])
        self.assertTrue(summary["pass_10_fps"])
        self.assertTrue(summary["m0_only"])

    def test_ten_fps_with_completion_at_elapsed_boundary(self):
        summary = summarize_run(make_records([index / 10 for index in range(1, 101)]), 10, {})
        self.assertEqual(summary["throughput_fps"], 10)
        # [0, 1) excludes the frame completed exactly at 1 second.
        self.assertEqual(summary["worst_1s_count"], 9)
        self.assertTrue(summary["pass_10_fps"])

    def test_noninteger_sliding_window_detects_drop(self):
        times = [0.05, 0.1, 0.15, 0.2, 0.25, 1.75, 1.8, 1.85, 1.9, 1.95]
        self.assertEqual(sum(0 <= value < 1 for value in times), 5)
        self.assertEqual(sum(1 <= value < 2 for value in times), 5)
        summary = summarize_run(make_records(times), 2, {})
        self.assertEqual(summary["throughput_fps"], 5)
        self.assertEqual(summary["worst_1s_count"], 0)
        self.assertFalse(summary["pass_5_fps"])
        self.assertFalse(summary["pass_10_fps"])

    def test_ten_fps_gate_requires_only_five_in_worst_window(self):
        for minimum, expected_pass in ((5, True), (4, False)):
            with self.subTest(minimum=minimum):
                times = [0.25] * minimum + [1.25] * (20 - minimum)
                summary = summarize_run(make_records(times), 2, {})
                self.assertEqual(summary["throughput_fps"], 10)
                self.assertEqual(summary["worst_1s_count"], minimum)
                self.assertEqual(summary["pass_10_fps"], expected_pass)
                self.assertEqual(summary["pass_5_fps"], expected_pass)

    def test_short_run_has_no_worst_window_and_no_pass(self):
        summary = summarize_run(make_records([0.01] * 20), 0.5, {})
        self.assertEqual(summary["throughput_fps"], 40)
        self.assertIsNone(summary["worst_1s_count"])
        self.assertFalse(summary["full_1s_window_available"])
        self.assertFalse(summary["pass_5_fps"])
        self.assertFalse(summary["pass_10_fps"])
        self.assertTrue(summary["m0_only"])

    def test_empty_input(self):
        for duration, expected_worst in ((0, None), (0.5, None), (1, 0), (1800, 0)):
            with self.subTest(duration=duration):
                summary = summarize_run([], duration, {"expected_stages": ["inference"]})
                self.assertEqual(summary["throughput_fps"], 0)
                self.assertEqual(summary["frame_count"], 0)
                self.assertEqual(summary["worst_1s_count"], expected_worst)
                self.assertIsNone(summary["latency_ms"]["mean"])
                self.assertIsNone(summary["latency_ms"]["p99"])
                self.assertIsNone(summary["stages_ms"]["inference"]["p95"])
                self.assertEqual(summary["json_bytes"]["total"], 0)
                self.assertIsNone(summary["object_count"]["mean"])
                self.assertFalse(summary["pass_5_fps"])
                self.assertFalse(summary["pass_10_fps"])
                self.assertTrue(summary["m0_only"])
                json.dumps(summary, allow_nan=False)

    def test_percentiles_and_wall_clock_denominator(self):
        records = make_records([0.1, 0.2, 0.3, 0.4], [10, 20, 30, 40])
        summary = summarize_run(records, 2, {})
        self.assertEqual(summary["throughput_fps"], 2)  # Not 4/0.4 or 1000/25.
        self.assertEqual(summary["latency_ms"]["mean"], 25)
        self.assertEqual(summary["latency_ms"]["p50"], 25)
        self.assertAlmostEqual(summary["latency_ms"]["p95"], 38.5)
        self.assertAlmostEqual(summary["latency_ms"]["p99"], 39.7)
        read = summary["stages_ms"]["read"]
        self.assertEqual(read["mean"], 6.25)
        self.assertEqual(read["p50"], 6.25)
        self.assertAlmostEqual(read["p95"], 9.625)
        self.assertEqual(summary["object_count"], {"total": 6, "mean": 1.5, "min": 0, "max": 3})
        self.assertEqual(summary["target_count"]["total"], 2)
        self.assertEqual(summary["json_bytes"]["total"], 406)
        self.assertEqual(summary["completed_s"], {"first": 0.1, "last": 0.4})

    def test_single_sample_percentiles(self):
        summary = summarize_run(make_records([0.25], [7]), 1, {})
        for name in ("mean", "p50", "p95", "p99"):
            self.assertEqual(summary["latency_ms"][name], 7)

    def test_boundary_windows(self):
        cases = [
            # Exactly one full window: include zero, exclude the right endpoint.
            ([0, 0, 0.5, 1, 1], 1, 3),
            # Do not process events AFTER the final permissible start (0.5).
            ([0.5], 1.5, 1),
            ([0, 0.5, 1.5], 1.5, 1),
            # The isolated start=0 state must be checked before entering t=1.
            ([1], 1.25, 0),
            # There are no frames in the trailing complete windows.
            ([0.25], 1.5, 0),
            # Equal-time entries/exits must be processed together.
            ([0.25] * 5 + [1.25] * 5, 1.5, 5),
            ([0] * 5 + [1] * 5 + [2] * 5, 2, 5),
            # Decimal boundaries should not gain an artificial gap from t-1.
            ([0.2] * 5 + [1.2] * 5, 1.4, 5),
            # Genuine, even very small, gaps must not be hidden by an epsilon.
            ([0.2] * 5 + [1.2000000001] * 5, 1.4, 0),
            # A noninteger final start is part of the domain.
            ([0, 0.1, 1.4, 1.5], 1.5, 0),
        ]
        for times, duration, expected in cases:
            with self.subTest(times=times, duration=duration):
                summary = summarize_run(make_records(times), duration, {})
                self.assertEqual(summary["worst_1s_count"], expected)
                self.assertEqual(summary["worst_1s_count"], brute_force_worst(times, duration))

    def test_sweep_matches_oracle_for_unsorted_duplicate_timestamps(self):
        rng = random.Random(2026)
        for _ in range(200):
            duration_ticks = rng.randint(10, 50)
            duration = duration_ticks / 10
            times = [rng.randint(0, duration_ticks) / 10 for _ in range(rng.randint(0, 40))]
            summary = summarize_run(make_records(times), duration, {})
            self.assertEqual(summary["worst_1s_count"], brute_force_worst(times, duration))

    def test_optional_camera_and_missing_stages_preserve_metadata(self):
        records = make_records([0.2, 0.4, 0.6])
        records[0]["camera_latency_ms"] = 6
        records[1]["camera_latency_ms"] = 10
        records[2]["camera_latency_ms"] = None
        del records[1]["stages_ms"]["read"]
        records[2]["stages_ms"]["read"] = None
        metadata = {
            "input": {"data_type": "合成测试"},
            "expected_stages": ["read", "rectify"],
            "scope_limitations": ["标定阶段未实现"],
            "m0_only": False,  # Cannot override the top-level M0 scope.
        }
        original_records = deepcopy(records)
        original_metadata = deepcopy(metadata)
        summary = summarize_run(records, 1, metadata)
        self.assertEqual(summary["camera_latency_ms"]["mean"], 8)
        self.assertEqual(summary["camera_latency_ms"]["missing_count"], 1)
        self.assertEqual(summary["stages_ms"]["read"]["count"], 1)
        self.assertEqual(summary["stages_ms"]["read"]["missing_count"], 2)
        self.assertEqual(summary["stages_ms"]["rectify"]["missing_count"], 3)
        self.assertIsNone(summary["stages_ms"]["rectify"]["mean"])
        self.assertEqual(summary["metadata"], metadata)
        self.assertTrue(summary["m0_only"])
        self.assertEqual(records, original_records)
        self.assertEqual(metadata, original_metadata)
        metadata["scope_limitations"].append("later edit")
        self.assertEqual(summary["metadata"], original_metadata)

    def test_invalid_measurements_are_rejected(self):
        for duration in (-1, float("nan"), float("inf"), True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                summarize_run([], duration, {})
        with self.assertRaises(ValueError):
            summarize_run(make_records([0]), 0, {})
        bad_fields = [
            ("completed_s", -0.1), ("completed_s", 1.01),
            ("latency_ms", float("nan")), ("latency_ms", -1),
            ("camera_latency_ms", float("inf")),
            ("stages_ms", {"read": -1}), ("stages_ms", []),
            ("json_bytes", 1.5), ("target_count", -1), ("object_count", True),
        ]
        for field, value in bad_fields:
            with self.subTest(field=field, value=value):
                records = make_records([0.5])
                records[0][field] = value
                with self.assertRaises(ValueError):
                    summarize_run(records, 1, {})
        records = make_records([0.5])
        del records[0]["frame_id"]
        with self.assertRaises(ValueError):
            summarize_run(records, 1, {})


class WriteReportTests(unittest.TestCase):
    def test_utf8_report_preserves_all_metrics_and_scope_without_writing_other_files(self):
        metadata = {
            "hardware": {"cpu": "测试 CPU", "gpu": "存在但未使用"},
            "input": {"data_type": "合成双目数据，不是真实摄像头"},
            "implementation": {"rectify": "CPU", "inference": "CPU 占位实现，无 GPU 推理"},
            "scope_limitations": [
                "未完成 1800 秒持续运行验收", "真实摄像头标定尚未验收", "三目标尚未验收",
            ],
            "acceptance": {"duration_1800s": False, "calibration": False, "three_targets": False},
            "expected_stages": ["gpu_inference"],
            "custom_fact": "必须原样保留",
        }
        summary = summarize_run(make_records([0.25, 0.75]), 1, metadata)
        original = deepcopy(summary)
        directory = Path(__file__).resolve().parent / "unused_mock_output"
        # Mock both filesystem operations to respect the two-source-file scope.
        with patch.object(Path, "mkdir", autospec=True) as mkdir, patch.object(
            Path, "write_text", autospec=True
        ) as write_text:
            self.assertIsNone(write_report(summary, directory))
        mkdir.assert_called_once_with(directory, parents=True, exist_ok=True)
        self.assertEqual(write_text.call_count, 2)
        written = {}
        for call in write_text.call_args_list:
            path, text = call.args
            self.assertEqual(path.parent, directory)
            self.assertEqual(call.kwargs["encoding"], "utf-8")
            self.assertEqual(text.encode("utf-8").decode("utf-8"), text)
            written[path.name] = text
        self.assertEqual(set(written), {"summary.json", "report.txt"})
        self.assertEqual(json.loads(written["summary.json"]), summary)
        report = written["report.txt"]
        for label in ("【硬件", "【输入", "【实际实现 CPU/GPU", "【全部范围限制", "仅 M0"):
            self.assertIn(label, report)
        for limitation in metadata["scope_limitations"]:
            self.assertIn(limitation, report)
        self.assertIn("CPU 占位实现，无 GPU 推理", report)
        self.assertIn("必须原样保留", report)
        self.assertIn("不等于文档要求的 1800 秒", report)
        self.assertIn(json.dumps(metadata, ensure_ascii=False, indent=2), report)
        for key in summary:
            if key != "metadata":
                self.assertIn(f'"{key}":', report)
        self.assertEqual(summary, original)

    def test_missing_hardware_or_execution_is_not_invented(self):
        summary = summarize_run([], 0, {})
        directory = Path(__file__).resolve().parent / "unused_mock_output"
        with patch.object(Path, "mkdir", autospec=True), patch.object(
            Path, "write_text", autospec=True
        ) as write_text:
            write_report(summary, directory)
        report = write_text.call_args_list[1].args[1]
        self.assertIn("未提供，不能推断或视为已验证", report)
        self.assertIn("未提供限制声明，不表示没有限制", report)
        self.assertIn('"worst_1s_count": null', report)


if __name__ == "__main__":
    unittest.main()
