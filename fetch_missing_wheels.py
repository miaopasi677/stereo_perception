"""Recover stalled large PyPI downloads with curl and verify PyPI SHA256 digests."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
PACKAGES = [('opencv-python', '4.10.0.84'), ('scipy', '1.15.3'), ('polars-runtime-32', '2.0.0')]


def fetch(spec):
    name, version = spec
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=30) as response:
        metadata = json.load(response)
    wheels = []
    for entry in metadata['urls']:
        file = entry['filename']
        if file.endswith('win_amd64.whl'):
            tags = file.split('-')
            py, abi = tags[-3:-1]
            if py == 'cp310' or (abi == 'abi3' and py in ('cp37','cp38','cp39')):
                wheels.append(entry)
    if len(wheels) != 1:
        raise ValueError((name, [w['filename'] for w in wheels]))
    entry = wheels[0]
    path = ROOT/'.wheels'/entry['filename']
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry['digests']['sha256']:
        print('Already verified '+entry['filename'], flush=True)
        return
    print('Fetching/resuming '+entry['filename'], flush=True)
    # Python/OpenSSL fallback avoids intermittent Windows schannel handshakes.
    offset = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(entry['url'], headers={'Range': f'bytes={offset}-'})
    with urllib.request.urlopen(request, timeout=90) as response:
        mode = 'ab' if response.status == 206 else 'wb'
        with path.open(mode) as output:
            while chunk := response.read(1024*1024):
                output.write(chunk)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry['digests']['sha256']:
        raise ValueError('SHA256 mismatch: '+str(path))
    print('Verified '+entry['filename']+' '+digest, flush=True)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, PACKAGES))
