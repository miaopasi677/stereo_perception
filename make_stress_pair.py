"""Create an explicitly artificial multi-object compute fixture from the bundled bus photo."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/".runtime/ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import ultralytics


def main():
    source = Path(ultralytics.__file__).parent/'assets/bus.jpg'
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(source)
    # Preserve object aspect ratio, tile three panels to raise mask/object count.
    panel = cv2.resize(image, (640, round(image.shape[0]*640/image.shape[1])))
    left = np.full((1080, 1920, 3), 110, dtype=np.uint8)
    y = (1080-panel.shape[0])//2
    for x in (0, 640, 1280):
        left[y:y+panel.shape[0], x:x+640] = panel
    right = np.zeros_like(left)
    right[:, :-48] = left[:, 48:]
    for name, array in [('stress_left.png', left), ('stress_right.png', right)]:
        path = ROOT/'data'/name
        if path.exists():
            raise FileExistsError(path)
        if not cv2.imwrite(str(path), array):
            raise RuntimeError('Failed to save '+str(path))
    meta = {'kind': 'synthetic_three_panel_bus_scene', 'original_photo': str(source),
            'right_image': 'artificial horizontal 48px shift',
            'purpose': 'multi-object segmentation/mask/JSON load only; NOT a real stereo scene or accuracy validation'}
    (ROOT/'data/stress_source.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
