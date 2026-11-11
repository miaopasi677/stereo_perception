"""Inputs for M0. Cached replay is not a camera capture latency measurement."""
from pathlib import Path
from collections import deque
import threading
import time
import cv2
import numpy as np


class ImagePair:
    def __init__(self, left, right, synthetic=False):
        self.synthetic = synthetic
        if synthetic:
            rng = np.random.default_rng(42)
            self.left = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
            self.right = np.zeros_like(self.left)
            self.right[:, :-24] = self.left[:, 24:]
        else:
            self.left = cv2.imread(str(left))
            self.right = cv2.imread(str(right))
            if self.left is None or self.right is None:
                raise ValueError(f"Cannot read stereo pair: {left}, {right}")
            if self.left.shape != self.right.shape:
                raise ValueError("Left and right image dimensions must match")
        self.metadata = {
            "type": "synthetic_shift" if synthetic else "cached_stereo_image_pair",
            "left": str(left), "right": str(right),
            "native_size": list(self.left.shape[1::-1]),
            "camera_capture_included": False,
            "image_decode_included": False,
            "repeat_static_scene": True,
        }

    def read(self):
        return self.left, self.right, {"sync_skew_ms": None, "capture_monotonic": None}

    def close(self):
        pass


class VideoPair:
    def __init__(self, left, right):
        self.caps = [cv2.VideoCapture(str(left)), cv2.VideoCapture(str(right))]
        if not all(cap.isOpened() for cap in self.caps):
            self.close()
            raise ValueError("Cannot open both replay videos")
        sizes = [(c.get(cv2.CAP_PROP_FRAME_WIDTH), c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in self.caps]
        rates = [c.get(cv2.CAP_PROP_FPS) for c in self.caps]
        if sizes[0] != sizes[1] or abs(rates[0] - rates[1]) > 0.01:
            self.close()
            raise ValueError("Replay videos require equal dimensions and frame rates")
        self.metadata = {"type": "video_replay", "left": str(left), "right": str(right),
                         "camera_capture_included": False, "image_decode_included": True,
                         "sync": "paired frame indices; hardware synchronization unverified"}

    def read(self):
        frames = [c.read() for c in self.caps]
        if not all(ok for ok, _ in frames):
            for c in self.caps:
                c.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames = [c.read() for c in self.caps]
        if not all(ok for ok, _ in frames):
            raise RuntimeError("Unable to replay stereo video pair")
        return frames[0][1], frames[1][1], {"sync_skew_ms": None, "capture_monotonic": None}

    def close(self):
        for c in self.caps:
            c.release()


class CameraPair:
    """Two DirectShow readers with bounded queues and host-arrival timestamp pairing.

    Host timestamps do not measure exposure synchronization. No precision claim.
    Camera access must be explicitly requested through --source camera.
    """
    def __init__(self, left, right, width, height, fps=15):
        if int(left) == int(right):
            raise ValueError("Two distinct camera indices required")
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.queues = [deque(maxlen=3), deque(maxlen=3)]
        self.errors = [None, None]
        self.dropped = [0, 0]
        self.fps = fps
        self.caps = []
        self.threads = []
        try:
            for index in (left, right):
                cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
                self.caps.append(cap)
                if not cap.isOpened():
                    raise RuntimeError(f"Camera {index} unavailable")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
            for i in range(2):
                t = threading.Thread(target=self._reader, args=(i,), daemon=True)
                self.threads.append(t)
                t.start()
        except Exception:
            self.close()
            raise
        self.metadata = {"type": "camera", "backend": "DirectShow",
                         "camera_capture_included": True, "image_decode_included": True,
                         "sync": "host-arrival timestamps, not exposure timestamps",
                         "negotiated": [{"width": c.get(cv2.CAP_PROP_FRAME_WIDTH),
                                         "height": c.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                         "fps": c.get(cv2.CAP_PROP_FPS)} for c in self.caps]}

    def _reader(self, i):
        try:
            while not self.stop.is_set():
                ok, frame = self.caps[i].read()
                stamp = time.perf_counter()
                with self.condition:
                    if not ok:
                        self.errors[i] = "camera_error"
                        self.condition.notify_all()
                        return
                    if len(self.queues[i]) == 3:
                        self.dropped[i] += 1
                    self.queues[i].append((stamp, frame))
                    self.condition.notify_all()
        finally:
            # Only the owning reader may release a handle potentially inside native read().
            self.caps[i].release()

    def read(self):
        deadline = time.perf_counter() + 5
        with self.condition:
            while True:
                if any(self.errors):
                    raise RuntimeError("camera_error: reader failed; reconnect is not implemented in M0")
                if all(self.queues):
                    # Newest admissible pair rather than accumulating old frames.
                    pairs = [(min(a[0], b[0]), ia, ib, abs(a[0] - b[0]))
                             for ia, a in enumerate(self.queues[0])
                             for ib, b in enumerate(self.queues[1])
                             if abs(a[0] - b[0]) <= 0.5 / self.fps]
                    if pairs:
                        _, ia, ib, skew = max(pairs)
                        a, b = self.queues[0][ia], self.queues[1][ib]
                        for i, idx in enumerate((ia, ib)):
                            self.dropped[i] += idx
                            for _ in range(idx + 1):
                                self.queues[i].popleft()
                        return a[1], b[1], {"sync_skew_ms": skew * 1000,
                                           "capture_monotonic": min(a[0], b[0]),
                                           "dropped_frames": sum(self.dropped)}
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise RuntimeError("camera_error: no synchronized pair within 5 seconds")
                self.condition.wait(min(remaining, 0.1))

    def close(self):
        import warnings
        self.stop.set()
        for t in self.threads:
            if t.ident is not None:
                t.join(timeout=2)
                if t.is_alive():
                    warnings.warn("Camera read is still blocked; handle remains owned by its reader until it returns")
        # Constructor may fail before any reader was started.
        for i, cap in enumerate(self.caps):
            if i >= len(self.threads) or self.threads[i].ident is None:
                cap.release()
