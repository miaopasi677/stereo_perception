"""Render an existing benchmark frame. Does not run inference or change timing results."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    args = p.parse_args()
    summary = json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    meta = summary['metadata']
    if meta['input']['type'] not in ('cached_stereo_image_pair', 'synthetic_image_pair'):
        raise ValueError('Preview supports cached image pairs only; frame source must be known')
    with (args.run/'frames.jsonl').open(encoding='utf-8') as f:
        packet = json.loads(next(f))
    image = cv2.imread(meta['input']['left'])
    if image is None:
        raise ValueError('Original input image missing')
    width, height = meta['config']['width'], meta['config']['height']
    image = cv2.resize(image, (width, height))
    colors = [(60, 220, 255), (255, 140, 80), (150, 255, 100), (220, 120, 255)]
    for index, obj in enumerate(packet['objects']):
        color = colors[index % len(colors)]
        data = obj.get('mask')
        if data and data['format'] == 'rle_uncompressed' and data['order'] == 'C':
            flat = np.repeat(np.arange(len(data['counts']), dtype=np.int64) % 2, data['counts'])
            if flat.size != width*height:
                raise ValueError('Invalid mask dimensions in output')
            mask = flat.reshape(data['size']).astype(bool)
            image[mask] = (image[mask]*0.65 + np.asarray(color)*0.35).astype(np.uint8)
        x0, y0, x1, y1 = map(round, obj['bbox'])
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 3)
        label = f"#{obj['id']} {obj['cls']} {obj['score']:.2f}"
        cv2.putText(image, label, (max(0, x0), max(90, y0-10)), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    x0, y0, x1, y1 = packet['ranging']['roi_px']
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 3)
    cv2.putText(image, '1% center ROI', (x0, y0-12), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
    cv2.rectangle(image, (0, 0), (width, 76), (28, 28, 28), -1)
    text = f"M0 REPLAY | YOLOE | {summary['throughput_fps']:.2f} FPS | P95 {summary['latency_ms']['p95']:.1f} ms"
    cv2.putText(image, text, (20, 31), cv2.FONT_HERSHEY_SIMPLEX, .85, (240, 240, 240), 2)
    note = ('SYNTHETIC tiled/shifted pair - compute stress only - NO accuracy claim'
            if meta['input']['type'] == 'synthetic_image_pair'
            else 'Resized public stereo sample - nominal calibration - NO accuracy claim')
    cv2.putText(image, note, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, .7, (80, 200, 255), 2)
    output = args.run/'preview.jpg'
    if not cv2.imwrite(str(output), cv2.resize(image, (1280, 720))):
        raise RuntimeError('Preview save failed')
    print(output)


if __name__ == '__main__':
    main()
