"""Downloads declared public assets only. Existing files are never overwritten."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT/'data/sources.json').read_text(encoding='utf-8'))
    pair, weights = manifest['stereo_pair'], manifest['weights']
    assets = {
        ROOT/'data/teddy_left.png': pair['left_url'],
        ROOT/'data/teddy_right.png': pair['right_url'],
        ROOT/'models'/weights['filename']: weights['repository']+'/resolve/'+weights['revision']+'/'+weights['filename'],
        ROOT/'models/mobileclip_blt.pt': 'https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt',
    }
    hashes = {}
    for path, url in assets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.part')
            print('Downloading', url, flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
                while chunk := response.read(1024*1024):
                    out.write(chunk)
            if temporary.stat().st_size < 1024:
                raise RuntimeError('Unexpectedly small asset: '+str(temporary))
            temporary.replace(path)
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'data/asset_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    print(json.dumps(hashes, indent=2))


if __name__ == '__main__':
    main()
