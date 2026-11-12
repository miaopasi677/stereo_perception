"""Summarize existing actual runs. This script does not produce simulated FPS."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    rows = []
    for file in sorted(args.directory.glob('*/summary.json')):
        s = json.loads(file.read_text(encoding='utf-8'))
        m = s['metadata']
        row = {'scenario': file.parent.name, 'seconds': s['elapsed_s'], 'frames': s['frame_count'],
               'fps': s['throughput_fps'], 'p50_ms': s['latency_ms']['p50'],
               'p95_ms': s['latency_ms']['p95'], 'worst_1s': s['worst_1s_count'],
               'pass_5_fps': s['pass_5_fps'], 'pass_10_fps': s['pass_10_fps'],
               'objects_mean': s['object_count']['mean'], 'targets_mean': s['target_count']['mean'],
               'mode': m['mode'], 'imgsz': m['imgsz'], 'stress_three': m['stress_three'],
               'input_kind': m['input']['type'],
               'parallel': m['implementation']['parallel_stereo'],
               'detector_backend': m['implementation'].get('detector_backend', 'yoloe'),
               'detector': m['implementation']['detector'],
               'mask_processing': m['implementation'].get('mask_processing', 'full'),
               'mask_format': m['config']['mask_format'],
               'error': bool(m.get('error')), 'interrupted': m.get('interrupted', False)}
        row.update({name+'_mean_ms': data['mean'] for name, data in s['stages_ms'].items()})
        rows.append(row)
    if not rows:
        raise ValueError('No run summaries found')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.directory/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.directory/'comparison.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    detectors = sorted({r['detector'] for r in rows})
    detector_note = detectors[0] if len(detectors) == 1 else '多种检测后端（见各行 detector 列）'
    lines = ['双目视觉 Windows M0 实测对比', '',
             f'注意：静态公开双目图片回放；{detector_note} + CPU SGBM；不是 TensorRT/C++ 最终实现。',
             '1080p 是处理尺寸，默认Teddy原图为450×375；dense压力项为人为拼接/平移图，非真实双目。',
             '未验证真实双摄同步、测距精度和机器人散热。',
             'three_partitions 表示三个合成 ROI 分区的负载，不代表三个真实识别目标。',
             'full=原全图掩码处理；bounded=实际掩码边界区域处理，仍输出等价的全图RLE和质心。',
             '运行顺序、后台负载和频率变化会影响结果；不同图像/检测数量的成绩不能仅归因于分辨率。',
             '本表仅比较本机实测。短测不能替代每种模式各 30 分钟的正式验收。', '',
             '场景 | 平均FPS | P50毫秒 | P95毫秒 | 最差1秒完成数 | 秒数 | 平均物体/测距目标 | 掩码处理 | 检测后端']
    for r in rows:
        if r['p95_ms'] is None:
            lines.append(f"{r['scenario']} | 无有效帧 | error={r['error']}")
        else:
            lines.append(f"{r['scenario']} | {r['fps']:.2f} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['worst_1s']} | {r['seconds']:.1f} | {r['objects_mean']:.1f}/{r['targets_mean']:.1f} | {r['mask_processing']}/{r['mask_format']} | {r['detector_backend']}" + (' | 本次运行未完整成功' if r['error'] or r['interrupted'] else ''))
    lines += ['', '阶段平均耗时（毫秒；并行模式的阶段存在重叠，不能相加推算 FPS）：']
    for r in rows:
        timings = ', '.join(f'{k.removesuffix("_mean_ms")}={v:.2f}' for k, v in r.items() if k.endswith('_mean_ms') and v is not None)
        lines.append(r['scenario']+': '+timings)
    text = '\n'.join(lines)+'\n'
    (args.directory/'comparison.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
