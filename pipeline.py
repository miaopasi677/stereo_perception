"""M0 reference pipeline: CUDA YOLOE + CPU remap/SGBM, NOT a TensorRT/C++ build."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import time
import cv2
import numpy as np


def center_roi(width, height, ratio, cx=None, cy=None):
    if not 0 < ratio <= 1:
        raise ValueError("ROI area ratio must be in (0, 1]")
    cx = width / 2 if cx is None else cx
    cy = height / 2 if cy is None else cy
    rw, rh = max(1, round(width * math.sqrt(ratio))), max(1, round(height * math.sqrt(ratio)))
    x0, y0 = min(max(0, round(cx - rw / 2)), width - rw), min(max(0, round(cy - rh / 2)), height - rh)
    return [x0, y0, x0 + rw, y0 + rh]


def rle_encode(mask):
    """COCO-like alternating counts but explicitly ROW-major, not compressed COCO."""
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="C")
    edges = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.concatenate(([0], edges, [flat.size]))).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"format": "rle_uncompressed", "order": "C", "size": list(mask.shape), "counts": counts}


def rle_encode_bounded(mask, bounds=None):
    """Exact full-image C-order RLE, scanning transitions only inside nonzero bounds.

    Bounds come from the actual mask, never the detector box. Adjacent row runs
    are merged so counts are identical to rle_encode, including edge pixels.
    The pipeline supplies 2-D binary uint8 masks.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(mask) if bounds is None else bounds
    if not w or not h:
        counts = [int(mask.size)]
    else:
        crop = mask[y:y+h, x:x+w]
        padded = np.pad(crop, ((0, 0), (1, 1)), mode="constant")
        rows, cols = np.nonzero(padded[:, 1:] != padded[:, :-1])
        edges = (rows + y) * mask.shape[1] + cols + x
        # An end followed by a start at the same full-image offset is no
        # transition (foreground continues across a row boundary).
        different = np.diff(edges) != 0
        edges = edges[np.r_[True, different] & np.r_[different, True]]
        edges = edges[edges < mask.size]
        counts = np.diff(np.concatenate(([0], edges, [mask.size]))).tolist()
    return {"format": "rle_uncompressed", "order": "C", "size": list(mask.shape), "counts": counts}


def bbox_iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = iw * ih
    return area / max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area)


class Tracker:
    def __init__(self):
        self.next_id, self.frame = 1, 0
        self.tracks = {}

    def update(self, detections):
        self.frame += 1
        candidates = sorted(((bbox_iou(d["bbox"], t["bbox"]), i, ident)
                             for i, d in enumerate(detections)
                             for ident, t in self.tracks.items() if d["cls"] == t["cls"]), reverse=True)
        taken, used = set(), set()
        for iou, i, ident in candidates:
            if iou >= 0.3 and i not in taken and ident not in used:
                detections[i]["id"] = ident
                taken.add(i)
                used.add(ident)
        for i, d in enumerate(detections):
            if i not in taken:
                d["id"] = self.next_id
                self.next_id += 1
            self.tracks[d["id"]] = {"bbox": d["bbox"], "cls": d["cls"], "seen": self.frame}
        self.tracks = {i: t for i, t in self.tracks.items() if self.frame - t["seen"] <= 5}
        return detections


class Geometry:
    def __init__(self, cfg, calibration=None):
        self.w, self.h = cfg["width"], cfg["height"]
        self.fx = self.fy = self.w / (2 * math.tan(math.radians(cfg["hfov_deg"]) / 2))
        self.cx, self.cy = self.w / 2, self.h / 2
        self.baseline = cfg["baseline_m"]
        self.calibrated = calibration is not None
        if calibration:
            with np.load(calibration, allow_pickle=False) as c:
                if tuple(c["image_size"].tolist()) != (self.w, self.h):
                    raise ValueError("Calibration image_size must equal configured processing resolution")
                r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
                    c["K1"], c["D1"], c["K2"], c["D2"], (self.w, self.h), c["R"], c["T"],
                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
                if abs(float(p2[1, 3])) > 1e-6:
                    raise ValueError("Only horizontal stereo is supported")
                self.maps = [cv2.initUndistortRectifyMap(c[k], c[d], r, p, (self.w, self.h), cv2.CV_32FC1)
                             for k, d, r, p in (("K1", "D1", r1, p1), ("K2", "D2", r2, p2))]
                self.fx, self.fy, self.cx, self.cy = map(float, (p1[0, 0], p1[1, 1], p1[0, 2], p1[1, 2]))
                self.baseline = abs(float(p2[0, 3] / p2[0, 0]))
        else:
            # Nominal parallel camera: still execute remap to charge the remapping cost.
            xx, yy = np.meshgrid(np.arange(self.w, dtype=np.float32), np.arange(self.h, dtype=np.float32))
            self.maps = [(xx, yy), (xx, yy)]
        if self.baseline <= 0 or self.fx <= 0:
            raise ValueError("Positive focal length and baseline required")
        self.roi = center_roi(self.w, self.h, cfg["roi_area_ratio"], self.cx, self.cy)

    def rectify(self, left, right):
        out = []
        for frame, (mx, my) in zip((left, right), self.maps):
            if frame.shape[1::-1] != (self.w, self.h):
                if self.calibrated:
                    raise ValueError("Calibrated camera frames must have original calibration resolution")
                frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            out.append(cv2.remap(frame, mx, my, cv2.INTER_LINEAR))
        return out


class Stereo:
    def __init__(self, cfg, geometry):
        self.cfg, self.g = cfg, geometry
        scale = cfg["coarse_scale"]
        if not 0 < scale <= 1:
            raise ValueError("coarse_scale must be in (0, 1]")
        self.size = (round(self.g.w * scale), round(self.g.h * scale))
        self.scale = self.size[0] / self.g.w
        max_phys = self.g.fx * self.g.baseline / cfg["min_focus_distance_m"]
        n = int(math.ceil(min(max_phys * self.scale, cfg["coarse_max_disparity"]) / 16) * 16)
        if n + 8 >= self.size[0]:
            raise ValueError("Coarse image too narrow for disparity search")
        self.coarse = self._matcher(n)
        self.roi_matcher = None
        self.roi_search = min(cfg["roi_max_disparity"], math.ceil(max_phys / 16) * 16,
                              int((self.g.roi[0] - 16) // 16) * 16)
        if self.roi_search < 16:
            raise ValueError("Center ROI has insufficient left search margin")

    @staticmethod
    def _matcher(n):
        return cv2.StereoSGBM_create(minDisparity=0, numDisparities=int(n), blockSize=5,
                                    P1=8*25, P2=32*25, disp12MaxDiff=1,
                                    uniquenessRatio=10, speckleWindowSize=50,
                                    speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def compute_coarse(self, left, right):
        t = time.perf_counter()
        l = cv2.cvtColor(cv2.resize(left, self.size), cv2.COLOR_BGR2GRAY)
        r = cv2.cvtColor(cv2.resize(right, self.size), cv2.COLOR_BGR2GRAY)
        disparity = self.coarse.compute(l, r).astype(np.float32) / 16 / self.scale
        # A <=1px low-resolution disparity is not reliable enough for coarse ordering.
        disparity[disparity * self.scale < 1] = np.nan
        return disparity, (time.perf_counter() - t) * 1000

    def compute_roi(self, left, right):
        if self.roi_matcher is None:
            self.roi_matcher = self._matcher(self.roi_search)
        x0, y0, x1, y1 = self.g.roi
        sx, sy, ex, ey = max(0, x0-self.roi_search-8), max(0, y0-8), min(self.g.w, x1+8), min(self.g.h, y1+8)
        l = cv2.cvtColor(left[sy:ey, sx:ex], cv2.COLOR_BGR2GRAY)
        r = cv2.cvtColor(right[sy:ey, sx:ex], cv2.COLOR_BGR2GRAY)
        d = self.roi_matcher.compute(l, r).astype(np.float32) / 16
        d = d[y0-sy:y1-sy, x0-sx:x1-sx]
        # SGBM uses -1 for invalid; 0 can be valid infinity, but cannot be trusted alone.
        d[d <= 0] = np.nan
        return d

    def release_roi(self):
        self.roi_matcher = None


class Detector:
    def __init__(self, cfg, model_path, imgsz=640, device="0", half=True, enabled=True):
        self.cfg, self.imgsz, self.device, self.half, self.enabled = cfg, imgsz, device, half, enabled
        self.description = "DISABLED: geometry smoke test only"
        if enabled:
            import torch
            from ultralytics import YOLOE
            self.torch = torch
            torch.set_num_threads(cfg["torch_threads"])
            if device != "cpu" and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable: refusing silent CPU fallback; use --device cpu explicitly")
            if device == "cpu" and half:
                raise ValueError("FP16 benchmark requires CUDA; use --fp32 with CPU")
            self.model = YOLOE(str(model_path))
            self.description = f"YOLOE / PyTorch / {'FP16' if half else 'FP32'} / device={device}"
            self.vocabulary = self.model.names
            values = list(self.vocabulary.values()) if isinstance(self.vocabulary, dict) else list(self.vocabulary)
            if not values or all(str(name).isdigit() for name in values):
                raise ValueError("Checkpoint has placeholder class names: run prepare_model.py before benchmarking")
            if not hasattr(self.model.model, "pe") and not getattr(self.model.model.model[-1], "is_fused", False):
                raise ValueError("Offline text embeddings missing: run prepare_model.py first")
            # No text encoder or set_classes is invoked inside the measured loop.

    def infer(self, image):
        if not self.enabled:
            return []
        if self.device != "cpu":
            self.torch.cuda.synchronize()
        results = self.model.predict(image, imgsz=self.imgsz, device=self.device, half=self.half,
                                     conf=self.cfg["confidence"], iou=self.cfg["iou"],
                                     max_det=self.cfg["max_detections"], retina_masks=True, rect=False,
                                     verbose=False, save=False, stream=False)[0]
        if results.masks is None:
            if results.boxes is not None and len(results.boxes):
                raise RuntimeError("Model produced detections without segmentation masks")
            if self.device != "cpu":
                self.torch.cuda.synchronize()
            return []
        boxes = results.boxes.data.detach().cpu().numpy()
        masks = results.masks.data.to(self.torch.uint8).cpu().numpy()
        if len(boxes) != len(masks) or tuple(masks.shape[1:]) != tuple(image.shape[:2]):
            raise RuntimeError("Box/mask count or full-resolution mask dimensions mismatch")
        if self.device != "cpu":
            self.torch.cuda.synchronize()
        return [{"bbox": box[:4].astype(float).tolist(), "score": float(box[4]),
                 "cls": str(results.names[int(box[5])]), "_mask": mask}
                for box, mask in zip(boxes, masks)]


class Pipeline:
    def __init__(self, cfg, detector, calibration=None, parallel=False, coarse_enabled=True):
        self.cfg, self.detector = cfg, detector
        self.coarse_enabled = coarse_enabled
        self.mask_processing = cfg.get("mask_processing", "bounded")
        if self.mask_processing not in ("full", "bounded"):
            raise ValueError("mask_processing must be full or bounded")
        cv2.setNumThreads(cfg["opencv_threads"])
        self.g = Geometry(cfg, calibration)
        self.stereo = Stereo(cfg, self.g)
        self.tracker = Tracker()
        self.pool = ThreadPoolExecutor(max_workers=1) if parallel else None
        self.ranging_enabled, self.ranging_frames = False, 0
        self.ranging_state = "disabled"
        self.history = {}

    def set_ranging(self, enabled):
        enabled = bool(enabled)
        if enabled != self.ranging_enabled:
            self.ranging_frames = 0
            self.history.clear()
            self.ranging_state = "warming_up" if enabled else "disabled"
        self.ranging_enabled = enabled
        if not enabled:
            self.stereo.release_roi()

    def _objects(self, detections, disparity):
        objects = self.tracker.update(detections)
        for obj in objects:
            mask = obj["_mask"]
            bounds = cv2.boundingRect(mask) if self.mask_processing == "bounded" else (0, 0, mask.shape[1], mask.shape[0])
            x, y, w, h = bounds
            m = cv2.moments(mask[y:y+h, x:x+w], binaryImage=True) if w and h else {"m00": 0}
            if m["m00"]:
                u, v = x + m["m10"]/m["m00"], y + m["m01"]/m["m00"]
            else:
                u, v = (obj["bbox"][0]+obj["bbox"][2])/2, (obj["bbox"][1]+obj["bbox"][3])/2
            obj["centroid_px"] = [u, v]
            obj["bearing"] = {"azimuth_deg": math.degrees(math.atan((u-self.g.cx)/self.g.fx)),
                              "elevation_deg": math.degrees(math.atan((self.g.cy-v)/self.g.fy))}
            coarse_mask = cv2.resize(mask, self.stereo.size, interpolation=cv2.INTER_NEAREST)
            coarse_mask = cv2.erode(coarse_mask, np.ones((3, 3), dtype=np.uint8)) > 0
            vals = disparity[coarse_mask & np.isfinite(disparity)]
            valid_fraction = vals.size / max(1, np.count_nonzero(coarse_mask))
            d = float(np.median(vals)) if vals.size >= 20 and valid_fraction >= 0.3 else None
            z = self.g.fx * self.g.baseline / d if d else None
            obj["dist_coarse_m"] = z
            obj["depth_valid_fraction"] = valid_fraction
            obj["dist_level"] = "unknown" if z is None else "near" if z < 2 else "mid" if z < 10 else "far"
            obj["distance_calibrated"] = self.g.calibrated
            obj["mask"] = (rle_encode_bounded(mask, bounds) if self.mask_processing == "bounded" else rle_encode(mask)) if self.cfg["mask_format"] == "rle" else None
        # Invalid stereo cannot distinguish far from low-texture: label unknown, not reliable far.
        valid = sorted([o for o in objects if o["dist_coarse_m"] is not None], key=lambda o: o["dist_coarse_m"])
        for rank, obj in enumerate(valid, 1):
            obj["rel_rank"] = rank
        for obj in objects:
            if obj["dist_coarse_m"] is None:
                obj["rel_rank"] = None
        return objects

    def _range(self, objects, disparity, stress_three):
        x0, y0, x1, y1 = self.g.roi
        self.ranging_frames += 1
        candidates = []
        if stress_three:
            # Explicit workload fixture, NOT three recognized physical targets.
            for i, xs in enumerate(np.array_split(np.arange(x1-x0), 3)):
                mask = np.zeros(disparity.shape, bool)
                mask[:, xs] = True
                candidates.append((-(i+1), "synthetic_roi_partition", mask))
        else:
            choices = [(int(np.count_nonzero(o["_mask"][y0:y1, x0:x1])), o) for o in objects]
            for count, obj in sorted(choices, key=lambda p: p[0], reverse=True):
                if count >= 32:
                    candidates.append((obj["id"], obj["cls"], obj["_mask"][y0:y1, x0:x1].astype(bool)))
                if len(candidates) >= self.cfg["max_targets"]:
                    break
            if not candidates:
                vals = disparity[np.isfinite(disparity)]
                if vals.size >= 32:
                    bins = np.arange(0, self.stereo.roi_search+5, 4)
                    hist, edges = np.histogram(vals, bins)
                    occupied = np.zeros(disparity.shape, bool)
                    for i in np.argsort(hist)[::-1]:
                        mask = np.isfinite(disparity) & (disparity >= edges[i]-2) & (disparity < edges[i+1]+2) & ~occupied
                        if np.count_nonzero(mask) >= max(32, disparity.size * 0.05):
                            candidates.append((-(len(candidates)+1), "unknown", mask))
                            occupied |= mask
                        if len(candidates) == self.cfg["max_targets"]:
                            break
        targets = []
        current = set()
        for ident, cls, mask in candidates:
            current.add(ident)
            values = disparity[mask & np.isfinite(disparity)]
            valid_fraction = values.size / max(1, np.count_nonzero(mask))
            target = {"object_id": ident, "cls": cls, "dist_m": None, "xyz_m": None,
                      "sigma_m": None, "status": "low_texture", "conf": valid_fraction,
                      "calibrated": self.g.calibrated, "benchmark_fixture": bool(stress_three)}
            if values.size >= 16 and valid_fraction >= 0.3:
                d = float(np.median(values))
                mad = float(np.median(np.abs(values-d)))
                target["disparity_px"] = d
                if d < self.cfg["min_valid_disparity_px"]:
                    target["status"] = "beyond_range"
                    target["dist_lower_bound_m"] = self.g.fx * self.g.baseline / self.cfg["min_valid_disparity_px"]
                elif mad <= max(1.5, 0.05*d):
                    z = self.g.fx * self.g.baseline / d
                    min_z = max(self.cfg["min_focus_distance_m"], self.g.fx*self.g.baseline/self.stereo.roi_search)
                    if z < min_z:
                        target["status"] = "too_near"
                    else:
                        history = self.history.setdefault(ident, deque(maxlen=self.cfg["filter_frames"]))
                        if history and abs(z-history[-1]) > max(0.25, 0.2*history[-1]):
                            history.clear()
                        history.append(z)
                        filtered = float(np.median(history))
                        yy, xx = np.nonzero(mask)
                        u, v = float(xx.mean()+x0), float(yy.mean()+y0)
                        # Conservative dispersion floor; not a calibrated uncertainty bound.
                        sigma_d = max(0.1, 1.4826*mad)
                        sigma = filtered**2/(self.g.fx*self.g.baseline)*sigma_d
                        target.update(status="ok", dist_m=filtered, sigma_m=sigma,
                                      xyz_m=[(u-self.g.cx)*filtered/self.g.fx,
                                             (v-self.g.cy)*filtered/self.g.fy, filtered],
                                      reliable_upper_m=self.cfg["target_rel_error"]*self.g.fx*self.g.baseline/sigma_d)
            targets.append(target)
        self.history = {k: v for k, v in self.history.items() if k in current}
        state = "warming_up" if self.ranging_frames < self.cfg["filter_frames"] else "ok" if targets else "no_target"
        return {"state": state, "roi_px": self.g.roi, "targets": targets}

    def process(self, left, right, frame_id, stress_three=False):
        stages = {}
        t = time.perf_counter()
        left, right = self.g.rectify(left, right)
        stages["rectify_resize_cpu"] = (time.perf_counter()-t)*1000
        future = self.pool.submit(self.stereo.compute_coarse, left, right) if self.pool and self.coarse_enabled else None
        t = time.perf_counter()
        detections = self.detector.infer(left)
        stages["detect_segment_wall"] = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        if self.coarse_enabled:
            disparity, coarse_ms = future.result() if future else self.stereo.compute_coarse(left, right)
        else:
            disparity = np.full(self.stereo.size[::-1], np.nan, dtype=np.float32)
            coarse_ms = 0.0
        stages["coarse_stereo_cpu"] = coarse_ms
        stages["coarse_join_wait"] = (time.perf_counter()-t)*1000 if future else 0.0
        t = time.perf_counter()
        objects = self._objects(detections, disparity)
        stages["fusion_tracking_masks_cpu"] = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        if self.ranging_enabled:
            roi_d = self.stereo.compute_roi(left, right)
            ranging = self._range(objects, roi_d, stress_three)
        else:
            ranging = {"state": "disabled", "roi_px": self.g.roi, "targets": []}
        stages["roi_stereo_range_cpu"] = (time.perf_counter()-t)*1000
        self.ranging_state = ranging["state"]
        for obj in objects:
            obj.pop("_mask", None)
        packet = {"schema_version": "m0-1.0", "frame_id": frame_id, "timestamp": time.time(),
                  "status": "ok", "mode": "M0_benchmark_not_precision_acceptance",
                  "distance_calibrated": self.g.calibrated, "objects": objects, "ranging": ranging}
        return packet, stages

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)
