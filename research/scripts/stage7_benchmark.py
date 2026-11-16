"""Freeze, calibrate, then evaluate audio-only onset peak merging."""
from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.metadata
import inspect
from pathlib import Path
import platform
import sys
import time
import xml.etree.ElementTree as ET

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage2_benchmark import read_json, sha, write_json
from scripts.onset_merging import (onset_peaks, merge_peaks, match_onset_indices,
                                   near_label_unmatched_fp_count, safety_metrics)

CLASSES = ['fast_change', 'repeat', 'fast_repeat']
METHODS = ['raw64', 'raw128'] + [f'{m}_r{r}' for m in ('first', 'strongest') for r in range(2, 6)]
FIXED = dict(analysis={'sr':16000,'hop':256,'raw_fft':1024,'diagnostic_fft':2048,'matching_grid_us':1},
             methods=['first','strongest'], radii_frames=[2,3,4,5], tolerances_seconds=[.05,.1],
             fast_ioi_seconds=.2, min_clips_per_role=3, min_pairs_per_class=20,
             held_macro_f1_min=.8, protected_classes=CLASSES, ioi_bin_edges_ms=[80,180,200],
             split_licks={'calibration':[1,2,3],'heldout':[4,5,6]}, guitar_ids=['AR','FS','LP'],
             historical_replay=True, frozen_before_audio_experiments=True,
             selection_rule='calibration only: macro F1 strictly higher, near-label FP strictly lower, zero losses of baseline-recovered protected later events or fully recovered pairs; sort macro F1 desc, FP asc, radius asc, first before strongest',
             support_policy='Prototype selection and descriptive heldout evaluation allowed with sparse labels; confirmation requires 20 pairs in each protected class in each role. Missing evidence takes precedence over a numerical pass.',
             scope='Merge raw64 peaks only; keep original retained peak frames and provenance. No pitch-based or label-based inference, new EQ, wet retrieval, Free Bird or deployment. Keep stage6 128 ms/.8 pitch candidate and all prior failures unchanged. Historical replay cannot participate in selection or become new holdout.')


def validate_config(cfg):
    for key, value in FIXED.items():
        if cfg.get(key) != value:
            raise ValueError(f'Unsupported stage7 profile: {key}')


def label_seconds(value):
    number = Decimal(value)
    if not number.is_finite() or number < 0 or number.as_tuple().exponent < -6:
        raise ValueError('Labels must be nonnegative exact integer microseconds')
    return int(number * 1000000) / 1000000


def recording_pairs(manifest, cfg):
    entries, grouped, assets = manifest['records'], {}, {}
    if len(entries) != 36:
        raise ValueError('Require eighteen original WAV/XML pairs')
    for entry in entries:
        path = 'assets/stage7/' + entry['output']
        if sha(ROOT/path) != entry['sha256']:
            raise ValueError(f'Changed downloaded source: {path}')
        assets[path] = entry['sha256']
        key = (entry['split'], entry['guitar'], entry['lick'])
        if key[1] not in cfg['guitar_ids'] or key[2] not in cfg['split_licks'][key[0]]:
            raise ValueError('Score or guitar split mismatch')
        suffix = Path(path).suffix.lower()
        if suffix in grouped.setdefault(key, {}):
            raise ValueError('Duplicate source entry')
        grouped[key][suffix] = (path, entry)
    records, hashes = [], set()
    for (split, guitar, lick), files in sorted(grouped.items()):
        if set(files) != {'.wav','.xml'}:
            raise ValueError('Unpaired sources')
        wav, wav_entry = files['.wav']
        xml_path, xml_entry = files['.xml']
        info, header = sf.info(ROOT/wav), wav_entry['wav_header']
        if (info.samplerate != 44100 or info.channels != 1 or info.subtype != 'PCM_24'
                or header != dict(channels=1,sample_rate=44100,sample_width_bytes=3,frames=info.frames)):
            raise ValueError('Require original mono 44.1 kHz PCM24 and matching header')
        xml = ET.parse(ROOT/xml_path).getroot()
        global_params = {child.tag:child.text for child in xml.find('./globalParameter')}
        events_xml = [{child.tag:child.text for child in event} for event in xml.findall('./transcription/event')]
        labels = xml_entry['labels']
        if global_params != labels['global_parameters'] or events_xml != labels['events']:
            raise ValueError('Manifest labels differ from actual XML')
        if (global_params['instrument'] != 'EGUI'
                or global_params['audioFileName'].replace('\\','/').split('/')[-1] != Path(wav).name):
            raise ValueError('Require paired electric guitar labels')
        events = []
        for event in events_xml:
            onset, offset = (label_seconds(event[k]) for k in ('onsetSec','offsetSec'))
            pitch, string, fret = (int(event[k]) for k in ('pitch','stringNumber','fretNumber'))
            if (event['excitationStyle'] != 'PK' or event['expressionStyle'] != 'NO'
                    or not 1 <= string <= 6 or not 0 <= fret <= 24
                    or pitch != [40,45,50,55,59,64][string-1] + fret
                    or not 0 <= onset < offset <= info.frames/info.samplerate + .00005
                    or (events and round((onset-events[-1]['onset'])*1e6) < 40000)):
                raise ValueError('Require valid ordered ordinary individual picks')
            events.append(dict(onset=onset,offset=offset,pitch=pitch,string=string,fret=fret))
        if not events or wav_entry['sha256'] in hashes:
            raise ValueError('Empty annotation or duplicate audio')
        hashes.add(wav_entry['sha256'])
        records.append(dict(id=Path(wav).stem,split=split,guitar=guitar,lick=lick,wav=wav,xml=xml_path,
                            sample_rate=info.samplerate,frames=info.frames,events=events))
    if len(records) != 18 or any(len([r for r in records if r['split']==s]) != 9 for s in cfg['split_licks']):
        raise ValueError('Incomplete three-guitar six-lick matrix')
    return records, assets


def plan(cfg):
    validate_config(cfg)
    protected = read_json(ROOT/cfg['baseline_snapshot'])
    for path, digest in protected.items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Protected baseline changed: {path}')
    records, assets = recording_pairs(read_json(ROOT/cfg['data_manifest']), cfg)
    historical_assets = read_json(ROOT/cfg['historical_results'])['plan']['assets']
    for path, digest in historical_assets.items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Historical source changed: {path}')
    sources = ['experiments/stage7.json','docs/stage7-protocol.md','scripts/stage7_benchmark.py',
               'scripts/onset_merging.py','scripts/frontend_calibration.py','scripts/stage2_benchmark.py',
               'tests/test_stage7_benchmark.py','tests/test_onset_merging.py',cfg['data_manifest'],
               cfg['baseline_snapshot'],cfg['historical_results'],'scripts/prepare_stage7_di.py',
               'docs/stage7-recording-spec.md','docs/stage7-preflight-review.md',
               'docs/stage7-data-idmt.md','docs/stage7-data-alternatives.md','assets/stage7/source-description.pdf']
    return dict(config=cfg,records=records,assets=assets,historical_assets=historical_assets,protected_baselines=protected,
                source_hashes={p:sha(ROOT/p) for p in sorted(set(sources))},
                environment=dict(python=platform.python_version(),platform=platform.platform(),
                    versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','librosa','soundfile')},
                    onset_source_sha256=sha(inspect.getsourcefile(librosa.onset.onset_detect))))


def summarize(rows):
    result = {}
    for method in METHODS:
        result[method] = {}
        for tol in ('0.05','0.1'):
            scores = [r['methods'][method]['scores'][tol] for r in rows]
            metrics = [s['matches']['metrics'] for s in scores]
            tp, fp, fn = (sum(m[k] for m in metrics) for k in ('tp','fp','fn'))
            errors = [e for m in metrics for e in m['errors_seconds']]
            protected = {}
            for name in CLASSES:
                group = [s['safety'][name] for s in scores]
                support = sum(g['support'] for g in group)
                item = dict(support=support,clips=sum(g['support']>0 for g in group))
                for k in ('raw_later_hits','merged_later_hits','raw_both_pair_hits','merged_both_pair_hits'):
                    item[k] = sum(g[k] for g in group)
                for k in ('baseline_recovered_later_lost','baseline_both_pair_lost'):
                    item[k] = [{'id':r['id'],'truth':v} for r,g in zip(rows,group) for v in g[k]]
                    item[k+'_count'] = len(item[k])
                for prefix in ('raw','merged'):
                    item[prefix+'_later_recall'] = item[prefix+'_later_hits']/support if support else None
                    item[prefix+'_both_pair_rate'] = item[prefix+'_both_pair_hits']/support if support else None
                protected[name] = item
            result[method][tol] = dict(tp=tp,fp=fp,fn=fn,near_fp=sum(s['near_fp'] for s in scores),
                other_fp=sum(s['other_fp'] for s in scores),n_clips=len(rows),
                macro_f1=float(np.mean([m['f1'] for m in metrics])) if metrics else 0,
                micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,
                median_signed_ms=float(np.median(errors)*1000) if errors else None,
                median_absolute_ms=float(np.median(np.abs(errors))*1000) if errors else None,
                protected=protected)
    return result


def eligible(candidate, baseline):
    return (candidate['macro_f1'] > baseline['macro_f1'] and candidate['near_fp'] < baseline['near_fp']
            and all(not p['baseline_recovered_later_lost_count'] and not p['baseline_both_pair_lost_count']
                    for p in candidate['protected'].values()))


def choose_setting(summary):
    feasible = [m for m in METHODS[2:] if eligible(summary[m]['0.05'], summary['raw64']['0.05'])]
    return min(feasible,key=lambda m:(-summary[m]['0.05']['macro_f1'],summary[m]['0.05']['fp'],
                                     int(m[-1]),m.startswith('strongest'))) if feasible else None


def gate(calibration, heldout, selected, cfg):
    support = {role:{name:dict(support=p['support'],clips=p['clips'],
                passed=p['support']>=cfg['min_pairs_per_class'] and p['clips']>=cfg['min_clips_per_role'])
                for name,p in summary['raw64']['0.05']['protected'].items()}
                for role,summary in (('calibration',calibration),('heldout',heldout))}
    supported = all(p['passed'] for role in support.values() for p in role.values())
    performance = bool(selected and eligible(heldout[selected]['0.05'],heldout['raw64']['0.05'])
                       and heldout[selected]['0.05']['macro_f1'] >= cfg['held_macro_f1_min'])
    status = 'insufficient_evidence' if not supported else ('passed' if performance else 'retain_baseline')
    return dict(status=status,support=support,support_passed=supported,performance_passed=performance,
                promote=supported and performance)


def loss_attribution(raw_matches, matches, frames, merged):
    """Explain assignment-relative losses without claiming physical causation."""
    own_truth = {g for _,g in matches['pairs']}
    assigned = {merged['kept_indices'][p]:g for p,g in matches['pairs']}
    result = []
    for p,g in raw_matches['pairs']:
        if g in own_truth:
            continue
        suppressor = merged['suppressed_by'].get(p)
        result.append(dict(truth_index=g,raw_peak_index=p,raw_frame=frames[p],
            cause='suppressed' if suppressor is not None else 'retained_reassigned',
            retained_peak_matched_truth=assigned.get(p),suppressor_index=suppressor,
            suppressor_frame=frames[suppressor] if suppressor is not None else None,
            distance_frames=abs(frames[p]-frames[suppressor]) if suppressor is not None else None,
            suppressor_matched_truth=assigned.get(suppressor)))
    return result


def evaluate(records, cfg, historical_rows=None):
    rows = []
    for record in records:
        audio, sr = sf.read(ROOT/record['wav'],dtype='float64')
        if audio.ndim != 1 or sr != record['sample_rate'] or len(audio) != record['frames'] or not np.isfinite(audio).all():
            raise ValueError('Audio differs from locked metadata')
        silent, methods = not np.any(audio), {}
        for name, window in (('raw64',1024),('raw128',2048)):
            peaks = dict(frames=np.array([],dtype=int),strengths=np.array([])) if silent else onset_peaks(audio,sr,window)
            frames = peaks['frames'].tolist()
            methods[name] = dict(frames=frames,strengths=peaks['strengths'].tolist(),predicted_onsets=[f*.016 for f in frames])
            if historical_rows is not None:
                old = historical_rows[record['id']]['predicted_onsets'][str(window)]
                if methods[name]['predicted_onsets'] != old:
                    raise ValueError(f'Historical raw peak mismatch: {record["id"]} {name}: {frames}')
        raw = methods['raw64']
        for method in METHODS[2:]:
            algorithm, radius = method.split('_r')
            merged = merge_peaks(np.array(raw['frames'],dtype=int),raw['strengths'],algorithm,int(radius))
            methods[method] = dict(**merged,predicted_onsets=[raw['predicted_onsets'][i] for i in merged['kept_indices']])
        truth = [e['onset'] for e in record['events']]
        for tol in cfg['tolerances_seconds']:
            raw_matches = match_onset_indices(raw['predicted_onsets'],truth,tol)
            for method in methods.values():
                matches = match_onset_indices(method['predicted_onsets'],truth,tol)
                near = near_label_unmatched_fp_count(matches,method['predicted_onsets'],truth,.05)
                method.setdefault('scores',{})[str(tol)] = dict(matches=matches,near_fp=near,
                    other_fp=matches['metrics']['fp']-near,safety=safety_metrics(raw_matches,matches,record['events']))
                if 'kept_indices' in method:
                    method['scores'][str(tol)]['loss_attribution'] = loss_attribution(raw_matches,matches,raw['frames'],method)
        if historical_rows is not None:
            for name, window in (('raw64',1024),('raw128',2048)):
                for old in historical_rows[record['id']]['onsets']:
                    if old['window'] == window:
                        score = methods[name]['scores'][str(old['tolerance'])]
                        score['stage6_original_metrics'] = old['metrics']
                        score['score_delta'] = {k:score['matches']['metrics'][k]-old['metrics'][k]
                                                for k in ('tp','fp','fn','precision','recall','f1')}
        rows.append(dict(**record,silent=silent,methods=methods))
        print(f'{record["split"]}: {record["id"]}',flush=True)
    return dict(rows=rows,summary=summarize(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='experiments/stage7.json')
    parser.add_argument('--freeze',action='store_true')
    args = parser.parse_args()
    started = time.time()
    cfg = read_json(ROOT/args.config)
    output = ROOT/cfg['output']
    lock_path = output/'locked-plan.json'
    if (output/'results.json').exists():
        raise SystemExit('Completed results exist; refusing overwrite')
    current = plan(cfg)
    if args.freeze:
        if lock_path.exists():
            raise SystemExit('Freeze already exists; refusing overwrite')
        output.mkdir(parents=True,exist_ok=True)
        write_json(lock_path,current)
        print(f'Frozen {len(current["records"])} recordings; no audio features computed')
        return
    if not lock_path.exists() or read_json(lock_path) != current:
        raise SystemExit('Missing or changed freeze; no analysis permitted')
    lock_hash = sha(lock_path)
    cal_path, selected_path = output/'calibration.json', output/'selected-setting.json'
    if cal_path.exists() or selected_path.exists():
        raise SystemExit('Partial analysis exists; refusing to repeat held-out evaluation')
    calibration = dict(lock_sha256=lock_hash,**evaluate([r for r in current['records'] if r['split']=='calibration'],cfg))
    write_json(cal_path,calibration)
    selected = dict(method=choose_setting(calibration['summary']),lock_sha256=lock_hash,
                    calibration_sha256=sha(cal_path),heldout_used=False)
    write_json(selected_path,selected)
    selection_hash = sha(selected_path)
    if read_json(selected_path) != selected or sha(cal_path) != selected['calibration_sha256']:
        raise ValueError('Selection persistence failed')
    heldout = evaluate([r for r in current['records'] if r['split']=='heldout'],cfg)
    historical, old = {}, read_json(ROOT/cfg['historical_results'])
    for split in ('calibration','heldout'):
        records = [dict(r,events=[r['event']]) for r in old['plan']['records'] if r['split']==split]
        historical[split] = evaluate(records,cfg,{r['id']:r for r in old[split]['rows']})
    if (plan(cfg) != current or sha(lock_path) != lock_hash or sha(selected_path) != selection_hash
            or sha(cal_path) != selected['calibration_sha256']):
        raise ValueError('Frozen sources changed during analysis')
    result = dict(plan=current,lock_sha256=lock_hash,calibration=calibration,selected=selected,heldout=heldout,
                  historical_replay=historical,gate=gate(calibration['summary'],heldout['summary'],selected['method'],cfg),
                  seconds=time.time()-started,scope='Research onset merging only; historical replay is not new validation')
    write_json(output/'results.json',result)
    print(result['gate'])


if __name__ == '__main__':
    main()
