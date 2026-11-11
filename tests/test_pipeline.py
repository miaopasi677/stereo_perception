import json
from pathlib import Path
import unittest
import cv2
import numpy as np
from pipeline import center_roi, rle_encode, rle_encode_bounded, Tracker, Geometry, Stereo, Pipeline
from sources import ImagePair

ROOT = Path(__file__).resolve().parents[1]


class NoDetector:
    def infer(self, image):
        return []


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'config/benchmark.json').read_text(encoding='utf-8'))
        self.cfg.update(width=640, height=480, coarse_scale=0.5, roi_max_disparity=128)

    def test_one_percent_roi(self):
        self.assertEqual(center_roi(1920, 1080, 0.01), [864, 486, 1056, 594])
        with self.assertRaises(ValueError):
            center_roi(1920, 1080, 0)

    def test_roi_clamped(self):
        x0, y0, x1, y1 = center_roi(640, 480, .01, 1, 1)
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertEqual((x1-x0)*(y1-y0), 64*48)

    def test_rle_roundtrip(self):
        for mask in (np.zeros((5, 6), np.uint8), np.ones((5, 6), np.uint8),
                     np.random.default_rng(8).integers(0, 2, (5, 6), dtype=np.uint8)):
            data = rle_encode(mask)
            restored = np.repeat(np.arange(len(data['counts'])) % 2, data['counts']).reshape(data['size'])
            np.testing.assert_array_equal(restored, mask)
            self.assertEqual(sum(data['counts']), mask.size)

    def test_bounded_rle_exhaustive_small_binary_masks(self):
        for bits in range(1 << 12):
            mask = ((bits >> np.arange(12)) & 1).astype(np.uint8).reshape(3, 4)
            self.assertEqual(rle_encode_bounded(mask), rle_encode(mask))

    def test_bounded_rle_sparse_dense_strided_and_edges(self):
        rng = np.random.default_rng(23)
        for height, width in ((1, 1), (1, 50), (50, 1), (27, 31), (1080, 1920)):
            masks = [np.zeros((height, width), np.uint8), np.ones((height, width), np.uint8)]
            for probability in (.01, .5, .99):
                masks.append((rng.random((height, width)) < probability).astype(np.uint8))
            crop_mask = np.zeros((height, width), np.uint8)
            crop_mask[height//3:max(height//3+1, 2*height//3), width//3:max(width//3+1, 2*width//3)] = 1
            masks += [crop_mask, crop_mask[::-1, ::-1]]
            for mask in masks:
                result = rle_encode_bounded(mask)
                self.assertEqual(result, rle_encode(mask))
                restored = np.repeat(np.arange(len(result['counts'])) % 2, result['counts']).reshape(mask.shape)
                np.testing.assert_array_equal(restored, mask)

    def test_bounded_processing_preserves_objects_and_ranging(self):
        from types import SimpleNamespace
        import copy
        masks = [np.zeros((480, 640), np.uint8) for _ in range(4)]
        masks[0][220:260, 290:340] = 1
        masks[1][210:270, 325:365] = 1
        masks[2][0:240, 0:140] = 1
        detections = [{'cls': 'fixture', 'bbox': [10, 10, 50, 50], 'score': .9, '_mask': mask}
                      for mask in masks]  # Deliberately wrong boxes: bounds must use actual masks.
        detector = SimpleNamespace(infer=lambda image: copy.deepcopy(detections))
        source = ImagePair(None, None, synthetic=True)
        left, right, _ = source.read()
        outputs = []
        for mode in ('full', 'bounded'):
            pipe = Pipeline({**self.cfg, 'mask_processing': mode}, detector)
            try:
                pipe.set_ranging(True)
                for i in range(5):
                    packet, _ = pipe.process(left, right, i)
                outputs.append(packet)
            finally:
                pipe.close()
        for packet in outputs:
            packet.pop('timestamp')
        for a, b in zip(outputs[0]['objects'], outputs[1]['objects']):
            np.testing.assert_allclose(a.pop('centroid_px'), b.pop('centroid_px'), rtol=0, atol=1e-10)
            for key in ('azimuth_deg', 'elevation_deg'):
                self.assertAlmostEqual(a['bearing'][key], b['bearing'][key], places=10)
            a.pop('bearing')
            b.pop('bearing')
        self.assertEqual(outputs[0], outputs[1])

    def test_invalid_mask_processing_rejected(self):
        with self.assertRaisesRegex(ValueError, 'mask_processing'):
            Pipeline({**self.cfg, 'mask_processing': 'invalid'}, NoDetector())

    def test_nominal_remap(self):
        g = Geometry(self.cfg)
        x = np.random.default_rng(10).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        l, r = g.rectify(x, x)
        np.testing.assert_array_equal(l, x)
        np.testing.assert_array_equal(r, x)

    def test_synthetic_disparity(self):
        g = Geometry(self.cfg)
        s = Stereo(self.cfg, g)
        source = ImagePair(None, None, synthetic=True)
        l, r, _ = source.read()
        disparity = s.compute_roi(l, r)
        self.assertGreater(np.isfinite(disparity).mean(), .8)
        self.assertAlmostEqual(float(np.nanmedian(disparity)), 24, delta=.2)
        coarse, ms = s.compute_coarse(l, r)
        self.assertAlmostEqual(float(np.nanmedian(coarse)), 24, delta=.5)
        self.assertGreater(ms, 0)

    def test_low_texture_not_false_distance(self):
        s = Stereo(self.cfg, Geometry(self.cfg))
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        self.assertEqual(int(np.isfinite(s.compute_roi(image, image)).sum()), 0)

    def test_tracker_stable_and_one_to_one(self):
        tracker = Tracker()
        initial = tracker.update([{'cls': 'cup', 'bbox': [10, 10, 50, 50]}, {'cls': 'cup', 'bbox': [80, 10, 110, 50]}])
        ids = [x['id'] for x in initial]
        moved = tracker.update([{'cls': 'cup', 'bbox': [12, 10, 52, 50]}, {'cls': 'cup', 'bbox': [81, 10, 111, 50]}])
        self.assertEqual([x['id'] for x in moved], ids)
        duplicate = tracker.update([{'cls': 'cup', 'bbox': [12, 10, 52, 50]}, {'cls': 'cup', 'bbox': [12, 10, 52, 50]}])
        self.assertNotEqual(duplicate[0]['id'], duplicate[1]['id'])

    def test_start_stop_1000_times_and_json(self):
        pipe = Pipeline(self.cfg, NoDetector())
        try:
            for _ in range(1000):
                pipe.set_ranging(True)
                pipe.set_ranging(False)
                self.assertIsNone(pipe.stereo.roi_matcher)
                self.assertEqual(pipe.history, {})
            source = ImagePair(None, None, synthetic=True)
            l, r, _ = source.read()
            pipe.set_ranging(True)
            for i in range(5):
                packet, stages = pipe.process(l, r, i, stress_three=True)
            self.assertEqual(packet['ranging']['state'], 'ok')
            self.assertEqual(len(packet['ranging']['targets']), 3)
            for t in packet['ranging']['targets']:
                self.assertEqual(t['status'], 'ok')
                self.assertAlmostEqual(t['dist_m'], pipe.g.fx*pipe.g.baseline/24, delta=.01)
                self.assertTrue(t['benchmark_fixture'])
                self.assertFalse(t['calibrated'])
            json.dumps(packet, allow_nan=False)
            pipe.set_ranging(False)
            packet, stages = pipe.process(l, r, 6)
            self.assertEqual(packet['ranging']['state'], 'disabled')
            self.assertEqual(packet['ranging']['targets'], [])
        finally:
            pipe.close()

    def test_stereo_disabled_stage(self):
        pipe = Pipeline(self.cfg, NoDetector(), coarse_enabled=False)
        try:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            _, stages = pipe.process(image, image, 0)
            self.assertEqual(stages['coarse_stereo_cpu'], 0)
        finally:
            pipe.close()

    def test_parallel_output_matches_serial(self):
        source = ImagePair(None, None, synthetic=True)
        l, r, _ = source.read()
        outputs = []
        for parallel in (False, True):
            pipe = Pipeline(self.cfg, NoDetector(), parallel=parallel)
            try:
                pipe.set_ranging(True)
                outputs.append(pipe.process(l, r, 0, stress_three=True)[0]['ranging'])
            finally:
                pipe.close()
        self.assertEqual(outputs[0], outputs[1])


class CameraOwnershipTests(unittest.TestCase):
    def test_close_never_releases_handle_owned_by_blocked_reader(self):
        from sources import CameraPair
        from unittest.mock import Mock
        from types import SimpleNamespace
        camera = CameraPair.__new__(CameraPair)
        camera.stop = Mock()
        cap = Mock()
        thread = SimpleNamespace(ident=123, join=Mock(), is_alive=lambda: True)
        camera.caps = [cap]
        camera.threads = [thread]
        with self.assertWarns(UserWarning):
            camera.close()
        cap.release.assert_not_called()

    def test_constructor_failure_releases_unowned_handle(self):
        from sources import CameraPair
        from unittest.mock import Mock
        camera = CameraPair.__new__(CameraPair)
        camera.stop = Mock()
        cap = Mock()
        camera.caps, camera.threads = [cap], []
        camera.close()
        cap.release.assert_called_once()


class DetectorContractTests(unittest.TestCase):
    def make_detector(self, result):
        from types import SimpleNamespace
        import torch
        from pipeline import Detector
        detector = Detector.__new__(Detector)
        detector.enabled = True
        detector.device = 'cpu'
        detector.half = False
        detector.imgsz = 640
        detector.torch = torch
        detector.cfg = {'confidence': .25, 'iou': .5, 'max_detections': 100}
        detector.model = SimpleNamespace(predict=lambda *a, **kw: [result])
        return detector

    def test_boxes_without_masks_are_not_silent_empty_detection(self):
        from types import SimpleNamespace
        result = SimpleNamespace(masks=None, boxes=[1])
        with self.assertRaisesRegex(RuntimeError, 'without segmentation'):
            self.make_detector(result).infer(np.zeros((32,32,3), np.uint8))

    def test_empty_scene_is_allowed(self):
        from types import SimpleNamespace
        result = SimpleNamespace(masks=None, boxes=[])
        self.assertEqual(self.make_detector(result).infer(np.zeros((32,32,3), np.uint8)), [])

    def test_count_mismatch_is_rejected(self):
        import torch
        from types import SimpleNamespace
        result = SimpleNamespace(boxes=SimpleNamespace(data=torch.zeros((2,6))),
                                 masks=SimpleNamespace(data=torch.ones((1,32,32))))
        with self.assertRaisesRegex(RuntimeError, 'count or full-resolution'):
            self.make_detector(result).infer(np.zeros((32,32,3), np.uint8))

    def test_mask_resolution_mismatch_is_rejected(self):
        import torch
        from types import SimpleNamespace
        result = SimpleNamespace(boxes=SimpleNamespace(data=torch.zeros((1,6))),
                                 masks=SimpleNamespace(data=torch.ones((1,16,16))))
        with self.assertRaisesRegex(RuntimeError, 'count or full-resolution'):
            self.make_detector(result).infer(np.zeros((32,32,3), np.uint8))


if __name__ == '__main__':
    unittest.main()
