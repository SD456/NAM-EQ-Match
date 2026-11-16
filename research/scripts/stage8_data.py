"""Frozen input inventory and leave-instrument-group-out splits for stage 8."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def _json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def _verified(path, expected):
    data = (ROOT / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return data


def records():
    """Return JSON-serializable records, verifying each input and label against provenance."""
    result = []
    for stage, key in ((6, 'selection'), (7, 'records')):
        entries = _json(f'assets/stage{stage}/data-provenance.json')[key]
        # Dataset 1 repeats stems across guitars, so pair on split as well.
        labels = {(e['split'], Path(e['name']).stem): e for e in entries if e['name'].endswith('.xml')}
        for e in entries:
            if not e['name'].endswith('.wav'):
                continue
            path = f'assets/stage{stage}/{e["output"]}'
            _verified(path, e['sha256'])
            label = labels[(e['split'], Path(e['name']).stem)]
            root = ET.fromstring(_verified(f'assets/stage{stage}/{label["output"]}', label['sha256']))
            if root.findtext('./globalParameter/audioFileName').lstrip('\\/') != Path(path).name:
                raise ValueError(f'Label does not match {path}')
            if stage == 6:
                folder = e['name'].split('/')[2]
                guitars = {'Ibanez Power Strat Clean Neck HU': 'Ibanez', 'Fender Strat Clean Neck SC': 'Fender'}
                guitar = guitars[folder]
                events = root.findall('./transcription/event')
                if len(events) != 1:
                    raise ValueError(f'Expected one note in {path}')
                event = events[0]
                pitch, string, fret = (int(event.findtext(k)) for k in ('pitch', 'stringNumber', 'fretNumber'))
                kind, role = 'isolated', 'Q' if fret % 2 == 0 else 'H'
                content_key = f'note:{string}:{fret}:{pitch}'
            else:
                guitars = {'AR': ('Aristides', 'Aristides 010'), 'FS': ('Fender', 'Fender Stratocaster'),
                           'LP': ('LesPaul', 'Gibson Les Paul')}
                guitar, model = guitars[e['guitar']]
                if (root.findtext('./globalParameter/instrumentModel') != model or
                        Path(path).name != f'{e["guitar"]}_Lick{e["lick"]}_KN.wav' or e['lick'] not in range(1, 7)):
                    raise ValueError(f'Unexpected lick label: {path}')
                pitch = string = fret = None
                kind, role = 'lick', 'Q' if e['lick'] <= 3 else 'H'
                content_key = f'lick:{e["lick"]}'
            result.append(dict(id=f'stage{stage}:{Path(path).stem}', path=path, parent=e['name'],
                               guitar=guitar, type=kind, q_or_h=role, source_hash=e['sha256'],
                               parent_hash=e['sha256'], content_key=content_key, pitch_midi=pitch,
                               string=string, fret=fret, session=None))

    for stage in (2, 3, 4):
        data = _json(f'assets/stage{stage}/di-provenance.json')
        parents = {p['id']: p for p in data['recordings']}
        for e in data['segments']:
            parent = parents[e['recording_id']]
            _verified(e['path'], e['sha256'])
            _verified(parent['path'], parent['sha256'])
            parent_id = 'directinput_01' if stage in (2, 3) else e['recording_id']
            if stage == 4 and e['id'] in ('T0', 'T1', 'H0', 'H1'):
                expected_parent = 'directinput_04' if e['id'].startswith('T') else 'directinput_05'
                if parent_id != expected_parent:
                    raise ValueError(f'Unexpected parent for {e["path"]}')
            role = ('Q' if stage == 4 and e['id'] in ('T0', 'T1') else
                    'H' if stage == 4 and e['id'] in ('H0', 'H1') else None)
            result.append(dict(id=f'stage{stage}:{e["id"]}', path=e['path'], parent=parent_id,
                               guitar='Sire', type='excerpt', q_or_h=role, source_hash=e['sha256'],
                               parent_hash=parent['sha256'], content_key=None, pitch_midi=None,
                               string=None, fret=None, session=None))
    result.sort(key=lambda r: r['id'])
    if len(result) != 192 or len({r['id'] for r in result}) != 192:
        raise ValueError('Expected 192 uniquely identified inputs')
    return result


def folds(items):
    """Return fixed training, query and holdout input IDs by held-out instrument group."""
    groups = ('Ibanez', 'Fender', 'Aristides', 'LesPaul', 'Sire')
    return {guitar: {
        'train': [r['id'] for r in items if r['guitar'] != guitar],
        'query': [r['id'] for r in items if r['guitar'] == guitar and r['q_or_h'] == 'Q'],
        'holdout': [r['id'] for r in items if r['guitar'] == guitar and r['q_or_h'] == 'H'],
    } for guitar in groups}


if __name__ == '__main__':
    items = records()
    print(json.dumps({'records': items, 'folds': folds(items)}, indent=2))
