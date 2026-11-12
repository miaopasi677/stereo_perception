"""Encode a vocabulary ONCE into an offline YOLO-World checkpoint (CLIP ViT-B/32 text features).

YOLO-World is detection-only; segmentation is supplied separately by MobileSAM at run time.
Changing vocabulary requires rerunning this tool; text encoding is not part of steady-state FPS.
The first run downloads the OpenAI CLIP ViT-B/32 weights (one-time); set HTTPS_PROXY if needed.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.runtime/ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab', type=Path, help='Optional UTF-8 file: one class per line; default COCO80')
    p.add_argument('--base', type=Path, default=ROOT/'models/yolov8s-worldv2.pt')
    p.add_argument('--output', type=Path, default=ROOT/'models/yolov8s-worldv2-coco80.pt')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if not args.base.is_file():
        raise FileNotFoundError(args.base)
    import torch
    import ultralytics
    from ultralytics import YOLOWorld
    from ultralytics.utils import YAML
    if args.vocab:
        names = [s.strip() for s in args.vocab.read_text(encoding='utf-8').splitlines() if s.strip()]
    else:
        data = YAML.load(Path(ultralytics.__file__).parent/'cfg/datasets/coco.yaml')
        names = list(data['names'].values())
    if not names or len(names) != len(set(names)):
        raise ValueError('Vocabulary must be nonempty and contain unique names')
    model = YOLOWorld(str(args.base))
    model.to(args.device)
    print(f'Encoding {len(names)} text classes once with CLIP ViT-B/32...', flush=True)
    # set_classes builds the CLIP text encoder and stores txt_feats on the WorldModel.
    model.set_classes(names)
    feats = model.model.txt_feats
    if not torch.isfinite(feats).all():
        raise RuntimeError('Nonfinite text features')
    if hasattr(model.model, 'clip_model'):
        model.model.clip_model = None  # do not bundle the CLIP encoder into the checkpoint
    model.to('cpu')
    # txt_feats is an ordinary tensor attribute, not a registered buffer; move it explicitly.
    model.model.txt_feats = model.model.txt_feats.cpu()
    model.save(str(args.output))
    print(json.dumps({'checkpoint': str(args.output), 'classes': len(names),
                      'features_shape': list(feats.shape)}, indent=2))


if __name__ == '__main__':
    main()