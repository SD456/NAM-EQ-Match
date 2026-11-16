"""Fetch the frozen stage 7 IDMT recording-level selection, without the full ZIP."""
import hashlib
import io
import json
import re
import shutil
import wave
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import prepare_stage6_di as source

BASE = Path(__file__).resolve().parents[1] / 'assets' / 'stage7'
DIRECTORY_START = 1329215801


def label_summary(data):
    root = ET.fromstring(data)
    global_parameters = {e.tag: e.text for e in root.find('globalParameter')}
    events = [{e.tag: e.text for e in event} for event in root.findall('./transcription/event')]
    events.sort(key=lambda e: Decimal(e['onsetSec']))
    assert all(e['excitationStyle'] == 'PK' and e['expressionStyle'] == 'NO' for e in events)
    pairs = [(Decimal(b['onsetSec']) - Decimal(a['onsetSec']), a['pitch'] == b['pitch']) for a, b in zip(events, events[1:])]
    assert all(gap >= Decimal('0.04') for gap, _ in pairs), 'simultaneous annotated onsets'
    bins = {'<=80ms': 0, '80-180ms': 0, '180-200ms': 0, '>200ms': 0}
    for gap, _ in pairs:
        bins['<=80ms' if gap <= Decimal('.08') else '80-180ms' if gap <= Decimal('.18') else '180-200ms' if gap <= Decimal('.2') else '>200ms'] += 1
    return {'global_parameters': global_parameters, 'events': events, 'event_count': len(events),
            'fast_pitch_changes_le_200ms': sum(gap <= Decimal('.2') and not same for gap, same in pairs),
            'same_pitch_repeats': sum(same for _, same in pairs),
            'fast_same_pitch_repeats_le_200ms': sum(gap <= Decimal('.2') and same for gap, same in pairs), 'ioi_bins': bins}


def main():
    entries = source.directory(source.get_range(DIRECTORY_START, source.SIZE - 1))
    offsets = sorted(e['offset'] for e in entries) + [DIRECTORY_START]
    assert len(offsets) == len(set(offsets))
    next_offsets = dict(zip(offsets, offsets[1:]))
    for e in entries:
        e['next_offset'] = next_offsets[e['offset']]
    pattern = re.compile(r'^IDMT-SMT-GUITAR_V2/dataset2/(audio|annotation)/(AR|FS|LP)_Lick([1-6])_KN\.(wav|xml)$')
    selected = []
    for e in entries:
        match = pattern.fullmatch(e['name'])
        if match:
            folder, guitar, lick, ext = match.groups()
            assert (folder, ext) in [('audio', 'wav'), ('annotation', 'xml')]
            selected.append((e, guitar, int(lick), ext))
    assert len(selected) == 36
    assert len({(g, l, ext) for _, g, l, ext in selected}) == 36
    BASE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASE.parent / 'stage6' / 'source-description.pdf', BASE / 'source-description.pdf')

    def retrieve(item):
        e, guitar, lick, ext = item
        split = 'calibration' if lick <= 3 else 'heldout'
        target = BASE / ('di' if ext == 'wav' else 'labels') / split / Path(e['name']).name
        data, ranges = source.extract(e)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        record = dict(e, guitar=guitar, lick=lick, split=split, output=target.relative_to(BASE).as_posix(),
                      sha256=hashlib.sha256(data).hexdigest(), ranges=ranges)
        if ext == 'xml':
            record['labels'] = label_summary(data)
        else:
            with wave.open(io.BytesIO(data), 'rb') as wav:
                record['wav_header'] = {'channels': wav.getnchannels(), 'sample_rate': wav.getframerate(),
                                        'sample_width_bytes': wav.getsampwidth(), 'frames': wav.getnframes()}
        print(target.name, flush=True)
        return record

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(retrieve, selected))
    manifest = {'source_url': source.URL, 'record_url': 'https://zenodo.org/records/7544110',
                'author_page': 'https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html',
                'license': 'CC-BY-NC-ND-4.0', 'archive_size': source.SIZE,
                'archive_md5_published': '06796e08731bccffaed6ae59361486e4', 'archive_md5_verified': False,
                'archive_central_directory_range': [DIRECTORY_START, source.SIZE - 1],
                'selection_rule': 'dataset2 AR/FS/LP picked normal KN licks 1-6; 1-3 calibration, 4-6 heldout',
                'fast_definition': 'successive labelled onset interval <= 0.2 seconds, different pitches',
                'records': records}
    (BASE / 'data-provenance.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
