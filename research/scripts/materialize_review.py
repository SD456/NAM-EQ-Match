"""Restore and verify the published review records without downloading any assets."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def checked_path(relative):
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents:
        raise ValueError(f'Review path escapes the research directory: {relative}')
    return path


def main():
    manifest = json.loads((ROOT / 'review-snapshot-manifest.json').read_text(encoding='utf-8'))
    restored = 0
    for record in manifest['files']:
        source, target = checked_path(record['upload_path']), checked_path(record['path'])
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != record['upload_sha256']:
            raise ValueError(f'Published bytes changed: {record["upload_path"]}')
        if record['compressed']:
            data = gzip.decompress(data)
        if hashlib.sha256(data).hexdigest() != record['public_sha256']:
            raise ValueError(f'Expanded bytes changed: {record["path"]}')
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f'Refusing to overwrite changed file: {record["path"]}')
        if record['compressed']:
            target.write_bytes(data)
            restored += 1
    print(f'Verified {len(manifest["files"])} published records; materialized {restored} JSON files.')
    changed = [r['path'] for r in manifest['files'] if r['transformed_local_paths']]
    print(f'{len(changed)} records have explicitly transformed local paths; original hashes remain in the manifest.')
    print('Audio, weights, IR, annotation XML, and caches are not restored by this command.')


if __name__ == '__main__':
    main()
