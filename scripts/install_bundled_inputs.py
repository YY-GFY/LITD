#!/usr/bin/env python3
"""Verify and install included small inputs at the configured default locations."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--paper-checkpoint', action='store_true', help='Also install the saved biological-case checkpoint')
    args = p.parse_args()
    records = json.loads((ROOT/'resources/manifest.json').read_text())
    for row in records:
        source = ROOT/'resources'/row['path']
        if sha256(source) != row['sha256']:
            raise ValueError(f'Bundled input hash mismatch: {source}')
    if args.verify_only:
        print(f'Verified {len(records)} bundled inputs.'); return
    for row in records:
        if not row['install_to'] or (row['path'].startswith('checkpoints/') and not args.paper_checkpoint):
            continue
        source, target = ROOT/'resources'/row['path'], ROOT/row['install_to']
        if target.exists():
            if sha256(target) != row['installed_sha256']:
                raise FileExistsError(f'Refusing to replace a different existing input: {target}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix+'.partial')
        opener = gzip.open if source.suffix=='.gz' else open
        with opener(source,'rb') as src, tmp.open('wb') as dst:
            shutil.copyfileobj(src,dst)
        if sha256(tmp) != row['installed_sha256']:
            tmp.unlink(); raise ValueError(f'Decompressed hash mismatch: {source}')
        tmp.replace(target)
        print(target.relative_to(ROOT))
    print('Inputs installed. See docs/DATA.md for remaining large inputs and upstream dependencies.')


if __name__ == '__main__':
    main()
