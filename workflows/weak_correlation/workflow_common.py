"""Paths and provenance shared by weak-correlation entry points."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def load_config(path):
    return json.loads(resolve(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(paths, settings):
    return {'files': {str(Path(p).resolve()): sha256(p) for p in paths},
            'settings': settings}


def reusable(output, metadata, expected):
    if not output.is_file() or not metadata.is_file():
        return False
    try:
        saved = json.loads(metadata.read_text())
        return saved.get('inputs') == expected and saved.get('output_sha256') == sha256(output)
    except (ValueError, OSError):
        return False


def record(output, metadata, inputs):
    metadata.write_text(json.dumps({'inputs': inputs, 'output_sha256': sha256(output)}, indent=2))
