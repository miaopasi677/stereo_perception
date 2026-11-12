"""Run `python benchmark.py --help`. All measurements exclude warmup/model loading."""
import argparse
from collections import deque
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/".runtime/ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def gpu_query():
    fields = ["name", "driver_version", "memory.total", "memory.used", "utilization.gpu", "temperature.gpu", "clocks.sm", "power.draw"]
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            return {"error": result.stderr.strip()}
        return {"gpus": [dict(zip(fields, line.split(", "))) for line in result.stdout.strip().splitlines()]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


class Telemetry:
    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        import psutil
        process = psutil.Process()
        start = time.perf_counter()
        while not self.stop_event.is_set():
            self.samples.append({"elapsed_s": time.perf_counter()-start,
                                 "rss_mib": process.memory_info().rss/1024**2,
                                 "available_ram_mib": psutil.virtual_memory().available/1024**2,
                                 "system_cpu_percent": psutil.cpu_percent(interval=None),
                                 "process_cpu_percent": process.cpu_percent(interval=None),
                                 **gpu_query()})
            self.stop_event.wait(1)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)


def parser():
    p = argparse.ArgumentParser(description="Windows stereo M0 benchmark: actual processing FPS, not model-only estimates")
    p.add_argument("--config", type=Path, default=ROOT/"config/benchmark.json")
    p.add_argument("--source", choices=["images", "videos", "synthetic", "camera"], default="images")
    p.add_argument("--left", default=str(ROOT/"data/teddy_left.png"))
    p.add_argument("--right", default=str(ROOT/"data/teddy_right.png"))
    p.add_argument("--synthetic-pair", action="store_true", help="Label supplied image pair as artificially generated, not real stereo")
    p.add_argument("--input-note", default=None, help="Additional explicit dataset/fixture description in report")
    p.add_argument("--mode", choices=["detection", "perception", "ranging"], default="ranging")
    p.add_argument("--model", type=Path)
    p.add_argument("--detector", choices=["yoloe", "yoloworld"], default="yoloe",
                   help="yoloe=single-model det+seg; yoloworld=YOLO-World + MobileSAM (plan B)")
    p.add_argument("--world-model", type=Path, help="Offline YOLO-World checkpoint (yoloworld backend)")
    p.add_argument("--sam-model", type=Path, help="MobileSAM weights (yoloworld backend)")
    p.add_argument("--calibration", type=Path, help="NPZ: K1,D1,K2,D2,R,T (meters),image_size [W,H]")
    p.add_argument("--imgsz", type=int, choices=[512, 640], default=640)
    p.add_argument("--device", default="0", help="CUDA index or cpu (explicit CPU only)")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--no-detector", action="store_true", help="Geometry smoke test only; not a full perception benchmark")
    p.add_argument("--parallel-stereo", action="store_true", help="Overlap CPU coarse SGBM with GPU model")
    p.add_argument("--stress-three", action="store_true", help="Three synthetic ROI partitions, NOT three recognized physical targets")
    p.add_argument("--mask-format", choices=["rle", "none"])
    p.add_argument("--mask-processing", choices=["bounded", "full"], help="Bounded exact RLE/moments (default), or original full-image reference")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--frames", type=int, default=0, help="Optional maximum measured frames; 0=no cap")
    p.add_argument("--capture-fps", type=float, default=15)
    p.add_argument("--output", type=Path)
    p.add_argument("--pub", help="Optional local ZeroMQ PUB, e.g. tcp://127.0.0.1:5556")
    p.add_argument("--control", help="Optional local ZeroMQ REP, e.g. tcp://127.0.0.1:5557")
    p.add_argument("--probe", action="store_true", help="Check CUDA and dependency versions; do not load weights")
    return p


def environment():
    import cv2
    import numpy as np
    import psutil
    import torch
    import ultralytics
    info = {"platform": platform.platform(), "python": sys.version, "cpu": platform.processor(),
            "ram_gib": psutil.virtual_memory().total/1024**3, "gpu": gpu_query(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "opencv": cv2.__version__,
            "opencv_cuda_devices": cv2.cuda.getCudaEnabledDeviceCount(),
            "ultralytics": ultralytics.__version__, "numpy": np.__version__}
    battery = psutil.sensors_battery()
    info["power_plugged"] = battery.power_plugged if battery else None
    if torch.cuda.is_available():
        # Actual matmul, not just the CUDA availability flag.
        t = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        result = t @ t
        torch.cuda.synchronize()
        info["cuda_matmul_finite"] = bool(torch.isfinite(result).all().item())
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["compute_capability"] = list(torch.cuda.get_device_capability(0))
    return info


def run(args):
    from metrics import summarize_run, write_report
    from pipeline import Pipeline, Detector
    from sources import ImagePair, VideoPair, CameraPair
    import torch
    env = environment()
    if args.probe:
        print(json.dumps(env, ensure_ascii=False, indent=2))
        return
    if args.seconds <= 0 or args.warmup < 0 or args.frames < 0 or args.capture_fps <= 0:
        raise ValueError("seconds/capture-fps must be positive; warmup/frames cannot be negative")
    if args.stress_three and args.mode != "ranging":
        raise ValueError("--stress-three requires --mode ranging")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.mask_format:
        cfg["mask_format"] = args.mask_format
    cfg["mask_processing"] = args.mask_processing or cfg.get("mask_processing", "bounded")
    source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                     for name in ("benchmark.py", "pipeline.py", "sources.py", "metrics.py", "world_sam.py")}
    out = args.output or ROOT/"results"/datetime.now().strftime("%Y%m%d_%H%M%S")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be empty to preserve prior measurements: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out/"environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    sam_path = sam_hash = None
    if args.detector == "yoloworld":
        model_path = args.world_model or ROOT/cfg.get("world_model", "models/yolov8s-worldv2-coco80.pt")
        sam_path = args.sam_model or ROOT/cfg.get("sam_model", "models/mobile_sam.pt")
        if not args.no_detector:
            for path in (model_path, sam_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Download/prepare model first: {path}")
            sam_hash = hashlib.sha256(sam_path.read_bytes()).hexdigest()
        from world_sam import WorldSamDetector
        detector = WorldSamDetector(cfg, model_path, sam_path, args.imgsz, args.device,
                                    not args.fp32, not args.no_detector)
    else:
        model_path = args.model or ROOT/cfg["model"]
        if not args.no_detector and not model_path.is_file():
            raise FileNotFoundError(f"Download model first: {model_path}")
        detector = Detector(cfg, model_path, args.imgsz, args.device, not args.fp32, not args.no_detector)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest() if not args.no_detector else None
    source = None
    pipe = None
    telemetry = Telemetry()
    context = pub = control = None
    records = []
    error = None
    interrupted = False
    elapsed = 0
    start = None
    output_times = deque()
    mode_changes = []
    try:
        if args.source in ("images", "synthetic"):
            source = ImagePair(args.left, args.right, args.source == "synthetic")
            if args.synthetic_pair:
                source.metadata["type"] = "synthetic_image_pair"
            if args.input_note:
                source.metadata["note"] = args.input_note
        elif args.source == "videos":
            source = VideoPair(args.left, args.right)
        else:
            source = CameraPair(args.left, args.right, cfg["width"], cfg["height"], args.capture_fps)
        pipe = Pipeline(cfg, detector, args.calibration, args.parallel_stereo, args.mode != "detection")
        pipe.set_ranging(args.mode == "ranging")
        if args.pub or args.control:
            import zmq
            context = zmq.Context()
            for endpoint in (args.pub, args.control):
                if endpoint and not endpoint.startswith("tcp://127.0.0.1:"):
                    raise ValueError("M0 networking is restricted to explicit localhost endpoints")
            if args.pub:
                pub = context.socket(zmq.PUB)
                pub.setsockopt(zmq.SNDHWM, 3)
                pub.bind(args.pub)
            if args.control:
                control = context.socket(zmq.REP)
                control.bind(args.control)
        print(f"Warmup {args.warmup} frames; model={model_path.name}; mode={args.mode}; input={args.imgsz}; parallel={args.parallel_stereo}", flush=True)
        for i in range(args.warmup):
            l, r, _ = source.read()
            pipe.process(l, r, -i-1, args.stress_three)
        pipe.set_ranging(False)
        pipe.set_ranging(args.mode == "ranging")
        if args.device != "cpu" and not args.no_detector:
            torch.cuda.reset_peak_memory_stats()
        telemetry.start()
        last_log = 0
        with (out/"frames.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            start = time.perf_counter()
            while time.perf_counter()-start < args.seconds and (not args.frames or len(records) < args.frames):
                frame_start = time.perf_counter()
                if control is not None and control.poll(0):
                    try:
                        message = control.recv_json()
                        if message.get("cmd") == "ranging" and message.get("action") in ("start", "stop"):
                            if args.mode == "detection" and message["action"] == "start":
                                raise ValueError("Ranging requires --mode perception or ranging")
                            enabled = message["action"] == "start"
                            if enabled != pipe.ranging_enabled:
                                mode_changes.append({"elapsed_s": time.perf_counter()-start, "ranging_enabled": enabled})
                            pipe.set_ranging(enabled)
                        elif message.get("cmd") != "status":
                            raise ValueError("Unknown command")
                        control.send_json({"ok": True, "state": pipe.ranging_state})
                    except (ValueError, AttributeError) as e:
                        control.send_json({"ok": False, "error": str(e)})
                l, r, capture = source.read()
                capture_ms = (time.perf_counter()-frame_start)*1000
                packet, stages = pipe.process(l, r, len(records), args.stress_three)
                packet["sync_skew_ms"] = capture["sync_skew_ms"]
                fps_timestamp = time.perf_counter()
                while output_times and output_times[0] <= fps_timestamp-1:
                    output_times.popleft()
                packet["fps"] = len(output_times) if fps_timestamp-start >= 1 else None
                packet["fps_scope"] = "completed outputs in preceding 1s, measured before current serialization"
                packet["latency_ms"] = (time.perf_counter()-frame_start)*1000
                packet["latency_scope"] = "read_to_pre_serialization; complete latency in timings.csv"
                packet["input_kind"] = source.metadata["type"]
                packet["dropped_frames"] = capture.get("dropped_frames", 0)
                stages["source_read"] = capture_ms
                ts = time.perf_counter()
                payload = json.dumps(packet, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                stream.write(payload+"\n")
                stream.flush()
                if pub is not None:
                    pub.send_string(payload)
                completed = time.perf_counter()
                stages["json_serialize_flush_publish"] = (completed-ts)*1000
                record = {"frame_id": len(records), "completed_s": completed-start,
                          "latency_ms": (completed-frame_start)*1000, "stages_ms": stages,
                          "object_count": len(packet["objects"]), "target_count": len(packet["ranging"]["targets"]),
                          "ranging_enabled": pipe.ranging_enabled,
                          "json_bytes": len(payload.encode("utf-8"))}
                if capture["capture_monotonic"] is not None:
                    record["camera_latency_ms"] = (completed-capture["capture_monotonic"])*1000
                records.append(record)
                output_times.append(completed)
                while output_times and output_times[0] <= completed-1:
                    output_times.popleft()
                elapsed = completed-start
                if elapsed-last_log >= 10:
                    print(f"{elapsed:.1f}s | {len(records)/elapsed:.2f} FPS | last={record['latency_ms']:.1f}ms | objects={record['object_count']} targets={record['target_count']}", flush=True)
                    last_log = elapsed
            elapsed = time.perf_counter()-start
    except KeyboardInterrupt:
        interrupted = True
        elapsed = time.perf_counter()-start if start else 0
    except Exception:
        error = traceback.format_exc()
        elapsed = time.perf_counter()-start if start else 0
    finally:
        if telemetry.thread.ident is not None:
            telemetry.stop()
        if pipe:
            pipe.close()
        if source:
            source.close()
        for socket in (pub, control):
            if socket is not None:
                socket.close(linger=0)
        if context is not None:
            context.term()
    scope = [
        "M0 Python reference implementation, not final C++/CUDA/TensorRT pipeline.",
        "GPU: YOLOE detection+segmentation; CPU: resize/remap, SGBM, masks/tracking/ranging, JSON.",
        "ROI SGBM fixed search + built-in consistency/uniqueness; custom Census/CUDA SGM and +/-1 epipolar search not implemented.",
        "No YOLO-World+MobileSAM comparison, ROS2, precision calibration or production camera reconnect acceptance.",
        "Output flush reaches OS buffers, not fsync; PUB if enabled measures enqueue, not subscriber receipt.",
        "Stage durations overlap when parallel_stereo=true; never add them to estimate throughput.",
        "Measured completion FPS includes per-frame output; model load/text encoding/warmup/report generation excluded.",
        "30-minute load tests and real synchronized calibrated camera / three physical target acceptance remain unverified.",
        "Open vocabulary model uses checkpoint's offline vocabulary; custom text encoding is outside this benchmark.",
    ]
    if not args.calibration:
        scope.append("Nominal camera intrinsics/baseline only. All distances are illustrative, not physically calibrated measurements.")
    if args.source == "images" and not args.synthetic_pair:
        scope.append("Cached static public stereo pair resized to processing resolution; NOT native 1080p motion/camera/decode workload.")
    if args.source == "synthetic":
        scope.append("Synthetic shifted random texture; not representative of semantic detection accuracy or real scenes.")
    if args.synthetic_pair:
        scope.append("Input image pair is ARTIFICIALLY GENERATED: compute stress only, not physical stereo or distance accuracy evidence.")
    if args.stress_three:
        scope.append("Three target workloads are synthetic ROI partitions; NOT three recognized or physically verified targets.")
    if args.no_detector:
        scope.append("DETECTOR DISABLED: geometry-only smoke test; full pipeline FPS requirement not tested.")
    if args.detector == "yoloworld":
        scope.append("Plan B: YOLO-World detection + separate MobileSAM segmentation; both stages timed inside detect_segment_wall.")
    if args.mode == "detection":
        scope.append("Stereo/ranging disabled: detection+segmentation workload only.")
    if records and not any(r["object_count"] for r in records):
        scope.append("NO DETECTED OBJECTS during this run: object masks/fusion/output workload has NOT been demonstrated.")
    if mode_changes:
        scope.append("Control commands changed ranging during measurement: this is a MIXED workload, not a single-mode FPS result.")
    metadata = {"hardware": env, "input": source.metadata if source else {"type": args.source},
                "implementation": {"detector": detector.description, "detector_backend": args.detector,
                                   "stereo": "OpenCV CPU SGBM 3WAY",
                                   "rectification": "OpenCV CPU remap", "parallel_stereo": args.parallel_stereo,
                                   "mask_processing": cfg["mask_processing"]},
                "source_sha256": source_hashes,
                "config": cfg, "mode": "mixed_control_states" if mode_changes else args.mode,
                "initial_mode": args.mode, "mode_changes": mode_changes, "imgsz": args.imgsz,
                "nonempty_detection_frames": sum(r["object_count"] > 0 for r in records),
                "ranging_frames_measured": sum(r["ranging_enabled"] for r in records),
                "object_workload_observed": any(r["object_count"] > 0 for r in records),
                "model_input_shape": [1, 3, args.imgsz, args.imgsz], "model": str(model_path),
                "model_sha256": model_hash, "sam_model": str(sam_path) if sam_path else None,
                "sam_sha256": sam_hash, "vocabulary": getattr(detector, "vocabulary", None),
                "scope_limitations": scope, "warmup_frames": args.warmup,
                "requested_seconds": args.seconds, "stress_three": args.stress_three,
                "detector_enabled": not args.no_detector, "error": error, "interrupted": interrupted,
                "calibration": str(args.calibration) if args.calibration else None,
                "acceptance_eligible": False,
                "peak_torch_allocated_mib": torch.cuda.max_memory_allocated()/1024**2 if args.device != "cpu" and not args.no_detector else None}
    summary = summarize_run(records, elapsed, metadata)
    write_report(summary, out)
    (out/"telemetry.json").write_text(json.dumps(telemetry.samples, ensure_ascii=False, indent=2), encoding="utf-8")
    if records:
        fields = ["frame_id", "completed_s", "latency_ms", "object_count", "target_count", "json_bytes", "ranging_enabled"]
        stage_names = sorted({k for r in records for k in r["stages_ms"]})
        if any("camera_latency_ms" in r for r in records):
            fields.append("camera_latency_ms")
        with (out/"timings.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields+[k+"_ms" for k in stage_names])
            writer.writeheader()
            for r in records:
                writer.writerow({**{k: r.get(k) for k in fields}, **{k+"_ms": r["stages_ms"].get(k) for k in stage_names}})
    print(json.dumps({"output": str(out), "frames": len(records), "elapsed_s": elapsed,
                      "fps": summary["throughput_fps"], "worst_1s": summary["worst_1s_count"],
                      "error": error, "interrupted": interrupted}, ensure_ascii=False, indent=2), flush=True)
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    run(parser().parse_args())
