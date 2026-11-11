"""Independently reconcile recorded CSV/JSONL with each benchmark summary."""
import argparse
import csv
import json
import math
from pathlib import Path
from metrics import summarize_run


def verify(run):
    s = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if s['metadata'].get('error') or s['metadata'].get('interrupted'):
        raise AssertionError(f'Unsuccessful run: {run}')
    records = []
    with (run/'timings.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            record = {'frame_id': int(row['frame_id']), 'completed_s': float(row['completed_s']),
                      'latency_ms': float(row['latency_ms']), 'object_count': int(row['object_count']),
                      'target_count': int(row['target_count']), 'json_bytes': int(row['json_bytes']),
                      'stages_ms': {k[:-3]: float(v) for k, v in row.items()
                                    if k.endswith('_ms') and k not in ('latency_ms','camera_latency_ms') and v}}
            if row.get('camera_latency_ms'):
                record['camera_latency_ms'] = float(row['camera_latency_ms'])
            records.append(record)
    calculated = summarize_run(records, s['elapsed_s'], s['metadata'])
    for key in ('frame_count', 'throughput_fps', 'worst_1s_count', 'pass_5_fps', 'pass_10_fps', 'latency_ms', 'camera_latency_ms', 'stages_ms', 'object_count', 'target_count', 'json_bytes'):
        if calculated[key] != s[key]:
            raise AssertionError(f'Summary mismatch: {run.name}/{key}')
    lines = 0
    with (run/'frames.jsonl').open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            p = json.loads(line)
            r = records[i]
            assert p['frame_id'] == i == r['frame_id']
            assert len(p['objects']) == r['object_count']
            assert len(p['ranging']['targets']) == r['target_count'] <= 3
            assert len(line.rstrip('\n').encode('utf-8')) == r['json_bytes']
            assert 0 <= p['latency_ms'] <= r['latency_ms']
            assert math.isfinite(r['latency_ms'])
            assert len({o['id'] for o in p['objects']}) == len(p['objects'])
            for o in p['objects']:
                assert '_mask' not in o
                mask = o.get('mask')
                if mask:
                    assert mask['order'] == 'C'
                    assert sum(mask['counts']) == math.prod(mask['size'])
                    assert all(isinstance(c, int) and c >= 0 for c in mask['counts'])
            for target in p['ranging']['targets']:
                if target['status'] == 'ok':
                    assert target['dist_m'] > 0 and math.isfinite(target['dist_m'])
                else:
                    assert target['dist_m'] is None
            lines += 1
    assert lines == len(records) == s['frame_count']
    assert records[-1]['completed_s'] <= s['elapsed_s']
    assert (run/'report.txt').is_file()
    assert (run/'environment.json').is_file()
    assert isinstance(json.loads((run/'telemetry.json').read_text(encoding='utf-8')), list)
    print(f'{run.name}: verified {lines} complete frames, {s["throughput_fps"]:.3f} FPS')
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    runs = sorted(file.parent for file in args.directory.glob('*/summary.json'))
    if not runs:
        raise ValueError('No run summaries found')
    total = sum(verify(run) for run in runs)
    print(f'PASS: {len(runs)} runs / {total} frames reconciled')


if __name__ == '__main__':
    main()
