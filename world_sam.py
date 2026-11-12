"""YOLO-World (open-vocabulary detection) + MobileSAM (box-prompted segmentation) backend.

This is the development document's alternative route (plan B): YOLO-World supplies detection
boxes and class labels, MobileSAM supplies a full-resolution mask per box. It exposes the same
infer() contract as pipeline.Detector so the stereo/ranging/JSON/metrics stages are unchanged
and the two routes are directly comparable. Text encoding is done offline (prepare_world_model.py).
"""
import numpy as np


class WorldSamDetector:
    def __init__(self, cfg, world_path, sam_path, imgsz=640, device="0", half=True, enabled=True):
        self.cfg, self.imgsz, self.device, self.half, self.enabled = cfg, imgsz, device, half, enabled
        self.description = "DISABLED: geometry smoke test only"
        self.vocabulary = None
        if enabled:
            import torch
            from ultralytics import YOLOWorld, SAM
            self.torch = torch
            torch.set_num_threads(cfg["torch_threads"])
            if device != "cpu" and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable: refusing silent CPU fallback; use --device cpu explicitly")
            if device == "cpu" and half:
                raise ValueError("FP16 benchmark requires CUDA; use --fp32 with CPU")
            self.world = YOLOWorld(str(world_path))
            self.vocabulary = self.world.names
            values = list(self.vocabulary.values()) if isinstance(self.vocabulary, dict) else list(self.vocabulary)
            if not values or all(str(name).isdigit() for name in values):
                raise ValueError("YOLO-World checkpoint has placeholder class names: run prepare_world_model.py first")
            if not hasattr(self.world.model, "txt_feats"):
                raise ValueError("Offline YOLO-World text embeddings missing: run prepare_world_model.py first")
            self.sam = SAM(str(sam_path))
            # MobileSAM's TinyViT encoder is unstable in FP16 (Half/Float mismatch in attention);
            # run it in FP32 for correctness. YOLO-World detection still uses the requested precision.
            self.sam_half = False
            self.description = (f"YOLO-World v2 s ({'FP16' if half else 'FP32'}) + MobileSAM (FP32) / "
                                f"PyTorch / device={device}")

    def infer(self, image):
        if not self.enabled:
            return []
        if self.device != "cpu":
            self.torch.cuda.synchronize()
        det = self.world.predict(image, imgsz=self.imgsz, device=self.device, half=self.half,
                                 conf=self.cfg["confidence"], iou=self.cfg["iou"],
                                 max_det=self.cfg["max_detections"], rect=False,
                                 verbose=False, save=False, stream=False)[0]
        boxes = det.boxes.data.detach().cpu().numpy()
        if len(boxes) == 0:
            if self.device != "cpu":
                self.torch.cuda.synchronize()
            return []
        # conf=0.0 keeps every prompted mask so masks stay 1:1 with detection boxes.
        seg = self.sam.predict(image, bboxes=boxes[:, :4].tolist(), device=self.device, half=self.sam_half,
                               conf=0.0, imgsz=1024, verbose=False, save=False, stream=False)[0]
        if seg.masks is None:
            raise RuntimeError("MobileSAM produced no masks for non-empty detections")
        masks = seg.masks.data.to(self.torch.uint8).cpu().numpy()
        if len(masks) != len(boxes) or tuple(masks.shape[1:]) != tuple(image.shape[:2]):
            raise RuntimeError("Box/mask count or full-resolution mask dimensions mismatch")
        if self.device != "cpu":
            self.torch.cuda.synchronize()
        return [{"bbox": box[:4].astype(float).tolist(), "score": float(box[4]),
                 "cls": str(det.names[int(box[5])]), "_mask": mask}
                for box, mask in zip(boxes, masks)]