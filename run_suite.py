"""Sequential tests to avoid GPU/CPU benchmark interference. Never run cases concurrently."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    output = args.output or ROOT/'results'/('suite_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    scenarios = [
        ('01_detect_segment_640', ['--mode', 'detection']),
        ('02_perception_640', ['--mode', 'perception']),
        ('03_ranging_640', ['--mode', 'ranging']),
        ('04_ranging_640_parallel', ['--mode', 'ranging', '--parallel-stereo']),
        ('05_ranging_512_parallel', ['--mode', 'ranging', '--parallel-stereo', '--imgsz', '512']),
        ('06_three_partitions_640_parallel', ['--mode', 'ranging', '--parallel-stereo', '--stress-three']),
    ]
    for name, switches in scenarios:
        print('\nCASE '+name, flush=True)
        subprocess.run([sys.executable, str(ROOT/'benchmark.py'), '--seconds', str(args.seconds),
                        '--output', str(output/name), *switches], check=True)
    subprocess.run([sys.executable, str(ROOT/'compare_results.py'), str(output)], check=True)


if __name__ == '__main__':
    main()
