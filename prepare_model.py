"""Encode the standard COCO 80-class vocabulary ONCE and save an offline YOLOE checkpoint.

Changing vocabulary requires rerunning this tool; encoding is not part of steady-state FPS.
MobileCLIP-B(LT) is the exact encoder used to train the downloaded YOLOE checkpoint.
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
    p.add_argument('--output', type=Path, default=ROOT/'models/yoloe-v8s-coco80.pt')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    import torch
    import ultralytics
    from ultralytics import YOLOE
    from ultralytics.utils import YAML
    if args.vocab:
        names = [s.strip() for s in args.vocab.read_text(encoding='utf-8').splitlines() if s.strip()]
    else:
        data = YAML.load(Path(ultralytics.__file__).parent/'cfg/datasets/coco.yaml')
        names = list(data['names'].values())
    if not names or len(names) != len(set(names)):
        raise ValueError('Vocabulary must be nonempty and contain unique names')
    encoder = ROOT/'models/mobileclip_blt.pt'
    if not encoder.exists():
        raise FileNotFoundError(encoder)
    # Ultralytics resolves this encoder relative to cwd. No global cwd changes outside this process.
    os.chdir(ROOT/'models')
    model = YOLOE(str(ROOT/'models/yoloe-v8s-seg.pt'))
    model.to(args.device)
    print(f'Encoding {len(names)} text classes once...', flush=True)
    from ultralytics.nn.text_model import MobileCLIP
    # Explicitly use the native Apple BLT checkpoint already verified locally.
    # The generic build_text_model in this version unconditionally downloads a second .ts file.
    text_encoder = MobileCLIP('blt', torch.device(args.device))
    with torch.inference_mode():
        tokens = text_encoder.tokenize(names)
        raw_features = text_encoder.encode_text(tokens).reshape(1, len(names), -1)
        features = model.model.model[-1].get_tpe(raw_features)
    del text_encoder
    if not torch.isfinite(features).all():
        raise RuntimeError('Nonfinite text features')
    model.set_classes(names, features)
    if hasattr(model.model, 'clip_model'):
        model.model.clip_model = None
    model.to('cpu')
    # pe is an ordinary tensor attribute in this version, not a registered buffer.
    model.model.pe = model.model.pe.cpu()
    model.save(str(args.output))
    (ROOT/'config/vocab.txt').write_text('\n'.join(names)+'\n', encoding='utf-8')
    print(json.dumps({'checkpoint': str(args.output), 'classes': len(names), 'features_shape': list(features.shape)}, indent=2))


if __name__ == '__main__':
    main()
