"""Fetch the frozen IDMT single-note selection without downloading its full archive."""
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
import requests

URL = 'https://zenodo.org/api/records/7544110/files/IDMT-SMT-GUITAR_V2.zip/content'
SIZE = 1329918978
BASE = Path(__file__).resolve().parents[1] / 'assets' / 'stage6'
PREFIX = 'IDMT-SMT-GUITAR_V2/dataset1/'
FOLDERS = {'calibration': 'Ibanez Power Strat Clean Neck HU', 'heldout': 'Fender Strat Clean Neck SC'}


def get_range(start, end):
    for _ in range(6):
        r = requests.get(URL, headers={'Range': f'bytes={start}-{end}'}, timeout=60)
        if r.status_code == 429:
            time.sleep(min(int(r.headers.get('Retry-After', '30')), 60))
            continue
        r.raise_for_status()
        assert r.status_code == 206 and r.headers.get('Content-Range') == f'bytes {start}-{end}/{SIZE}'
        assert len(r.content) == end - start + 1
        return r.content
    raise RuntimeError('Range request rate limited')


def directory(data):
    p, entries = 0, []
    while data[p:p+4] == b'PK\x01\x02':
        flags, method = struct.unpack_from('<HH', data, p+8)
        crc, compressed, size = struct.unpack_from('<III', data, p+16)
        n, extra, comment = struct.unpack_from('<HHH', data, p+28)
        offset = struct.unpack_from('<I', data, p+42)[0]
        name = data[p+46:p+46+n].decode('utf-8')
        entries.append(dict(name=name, flags=flags, method=method, crc32=f'{crc:08x}', compressed_size=compressed, size=size, offset=offset))
        p += 46+n+extra+comment
    assert len(entries) == 4292 and data[p:p+4] == b'PK\x05\x06'
    return entries


def extract(entry):
    offset = entry['offset']
    raw = get_range(offset, entry['next_offset']-1)
    header = raw[:30]
    assert header[:4] == b'PK\x03\x04'
    flags, method = struct.unpack_from('<HH', header, 6)
    n, extra = struct.unpack_from('<HH', header, 26)
    assert flags == entry['flags'] and method == entry['method'] and not flags & 1
    assert raw[30:30+n].decode('utf-8') == entry['name']
    start = 30+n+extra
    payload = raw[start:start+entry['compressed_size']]
    assert len(payload) == entry['compressed_size']
    data = zlib.decompress(payload, -15) if method == 8 else payload
    assert method in (0, 8) and len(data) == entry['size']
    assert f'{zlib.crc32(data):08x}' == entry['crc32']
    return data, [[offset, entry['next_offset']-1]]


def main():
    entries = directory(get_range(1329215801, 1329918977))
    offsets = sorted(e['offset'] for e in entries) + [1329215801]
    assert len(offsets) == len(set(offsets))
    next_offsets = dict(zip(offsets, offsets[1:]))
    for entry in entries:
        entry['next_offset'] = next_offsets[entry['offset']]
    selection = []
    for split, folder in FOLDERS.items():
        candidates = [e for e in entries if e['name'].startswith(PREFIX+folder+'/') and e['name'].lower().endswith(('.wav', '.xml'))]
        assert len(candidates) == 156
        wav = {Path(e['name']).stem for e in candidates if e['name'].endswith('.wav')}
        xml = {Path(e['name']).stem for e in candidates if e['name'].endswith('.xml')}
        assert wav == xml and len(wav) == 78
        for e in candidates:
            e['split'] = split
            selection.append(e)
    assert len(selection) == 312
    BASE.mkdir(parents=True, exist_ok=True)
    manifest = {'source_url': URL, 'archive_size': SIZE, 'archive_md5_published': '06796e08731bccffaed6ae59361486e4', 'archive_md5_verified': False, 'archive_central_directory_range': [1329215801, 1329918977], 'license': 'CC-BY-NC-ND-4.0', 'selection': selection}
    (BASE/'data-provenance.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    def retrieve(e):
        e = e.copy()
        target = BASE/('di' if e['name'].endswith('.wav') else 'labels')/e['split']/Path(e['name']).name
        if target.exists():
            data = target.read_bytes()
            assert len(data) == e['size'] and f'{zlib.crc32(data):08x}' == e['crc32']
            ranges = [[e['offset'], e['next_offset']-1]]
        else:
            data, ranges = extract(e)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        e.update(output=str(target.relative_to(BASE)).replace(os.sep, '/'), sha256=hashlib.sha256(data).hexdigest(), ranges=ranges)
        return e
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, e in enumerate(pool.map(retrieve, selection), 1):
            selection[i-1] = e
            (BASE/'data-provenance.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
            print(f'{i}/{len(selection)} {e["name"]}', flush=True)
    print('Done; CRC checks are not authentication of the published ZIP MD5.')


if __name__ == '__main__':
    main()
