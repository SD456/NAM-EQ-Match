"""Frozen leave-instrument-out low-dimensional covariance retrieval experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import subspace_angles
from scipy.signal import lfilter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage2_benchmark import read_json, sha, write_json as _write_json, paired_quality
from scripts.stage8_data import records, folds
from scripts.stage8_metrics import erb_feature, fit_model, score_model
from scripts.stage8_quality import quality_table
from scripts.multidi_scoring import half_octave_ltas, centered_spectral_distance
from tonematch.audio import db_to_lin, load_audio, rms
from tonematch.nam_backend import NamCapture
from tonematch.tone_stack import biquad_coeffs


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    _write_json(temporary, value)
    temporary.replace(path)


def environment():
    return {p: importlib.metadata.version(p) for p in
            ('numpy', 'scipy', 'scikit-learn', 'torch', 'neural-amp-modeler')}


def protect_snapshot():
    paths = []
    for folder in ('assets', 'artifacts', 'WAV_Data', 'scripts', 'tests', 'vendor', 'docs', 'experiments'):
        for p in (ROOT / folder).rglob('*'):
            rel = p.relative_to(ROOT).as_posix()
            if (p.is_file() and '__pycache__' not in p.parts and
                    not rel.startswith('artifacts/stage8/') and
                    not p.name.startswith(('stage8', 'test_stage8')) and
                    p.name not in ('decisions-tone-extraction.md', 'di-coverage.md')):
                paths.append(p)
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def validate_hashes(hashes):
    changed = [p for p, h in hashes.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    if changed:
        raise ValueError(f'Frozen/protected files changed: {changed[:5]}')


def make_plan():
    cfg = read_json(ROOT / 'experiments/stage8.json')
    items = records()
    previous = read_json(ROOT / 'artifacts/stage3/real/locked-plan.json')
    library = read_json(ROOT / 'assets/stage2/models-ir-provenance.json')
    for e in library['models'] + library['irs']:
        validate_hashes({e['path']: e['sha256']})
    source_names = ['experiments/stage8.json', 'docs/stage8-protocol.md', 'docs/stage8-preflight-review.md',
                    'assets/stage2/models-ir-provenance.json',
                    'artifacts/stage3/real/locked-plan.json']
    source_names += [p.relative_to(ROOT).as_posix() for folder, pattern in
                     [('scripts', 'stage8*.py'), ('tests', 'test_stage8*.py')]
                     for p in sorted((ROOT / folder).glob(pattern)) if p.name != 'stage8_analyze.py']
    return dict(config=cfg, records=items, folds=folds(items), rigs=previous['rigs'],
                library=library, source_hashes={p: sha(ROOT / p) for p in source_names},
                protected=protect_snapshot(), environment=environment(),
                device='cuda', precision='float32_no_tf32')


def color_function(cfg):
    p = cfg['post_eq']
    b, a = biquad_coeffs(p['kind'], cfg['sample_rate'], p['frequency_hz'], p['q'], p['gain_db'])
    return lambda x: lfilter(b, a, np.pad(x, (0, 256))).astype(np.float32)


def npz_write(path, **values):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def cache_paths(output, index):
    return output / 'features' / f'{index:03d}.npz', output / 'quality' / f'{index:03d}.npz'


def render_all(plan, output):
    cfg, items = plan['config'], plan['records']
    sr, gains = cfg['sample_rate'], cfg['gains_db']
    color = color_function(cfg)
    teachers = [i * len(gains) + g for i in cfg['teacher_indices'] for g in range(len(gains))]
    models = {m['id']: m for m in plan['library']['models']}
    irs = {m['id']: m for m in plan['library']['irs']}
    torch.set_num_threads(4)
    captures = []
    for rig in plan['rigs']:
        model, cab = models[rig['model_id']], rig['cab_id']
        captures.append(NamCapture(str(ROOT / model['path']), device='cuda', rig_type=model['rig_type'],
                                  cab_ir_path=str(ROOT / irs[cab]['path']) if cab else None))
    for sub in ('features', 'quality'):
        (output / sub).mkdir(exist_ok=True)
    audit_path = output / 'cache-audit.json'
    audit = read_json(audit_path) if audit_path.exists() else {'plan_sha256': canonical_hash(plan), 'items': {}}
    if audit['plan_sha256'] != canonical_hash(plan):
        raise ValueError('Cache plan changed')
    started = time.perf_counter()
    for index, item in enumerate(items):
        fp, qp = cache_paths(output, index)
        if str(index) in audit['items']:
            cached = audit['items'][str(index)]
            for path in [fp] + ([qp] if item['q_or_h'] == 'H' else []):
                if sha(path) != cached[path.parent.name + '_sha256']:
                    raise ValueError(f'Changed cache: {path}')
            continue
        dry, _ = load_audio(str(ROOT / item['path']), sr)
        x = dry.astype(np.float64) * db_to_lin(cfg['calibration_db'])
        wet = [cap.process(x * db_to_lin(g)) for cap in captures for g in gains]
        erb = np.array([erb_feature(y, sr) for y in wet])
        curves = [half_octave_ltas(y, sr) for y in wet]
        grid = curves[0][0]
        spectrum = np.array([a[1] for a in curves])
        values = dict(erb=erb, spectrum=spectrum, grid=grid)
        if item['q_or_h'] == 'Q':
            refs = [[wet[t], color(wet[t])] for t in teachers]
            values['query_erb'] = np.array([[erb_feature(y, sr) for y in pair] for pair in refs])
            values['query_spectrum'] = np.array([[half_octave_ltas(y, sr)[1] for y in pair] for pair in refs])
        if item['q_or_h'] == 'H':
            table = quality_table(wet, teachers, color)
            if not all(np.isfinite(v).all() for v in table.values()):
                raise ValueError('Nonfinite quality table')
            for t, flat in enumerate(teachers):
                if abs(table['sc'][t, 0, flat]) > 1e-7:
                    raise ValueError('E0 known setting must have zero quality error')
            if 'real_quality_fidelity' not in audit:
                for ti, eq, ci in ((0, 0, 6), (3, 1, 1)):
                    target = color(wet[teachers[ti]]) if eq else wet[teachers[ti]]
                    legacy = paired_quality(wet[ci], target)
                    np.testing.assert_allclose(table['sc'][ti, eq, ci], legacy['mrstft_sc'], atol=1e-7, rtol=0)
                    np.testing.assert_allclose(table['ltas'][ti, eq, ci], legacy['ltas_mae_db'], atol=1e-7, rtol=0)
                audit['real_quality_fidelity'] = dict(input=item['id'], comparisons=2, atol=1e-7)
            npz_write(qp, **table)
        npz_write(fp, **values)
        audit['items'][str(index)] = {
            'id': item['id'], 'features_sha256': sha(fp),
            **({'quality_sha256': sha(qp)} if item['q_or_h'] == 'H' else {}),
            'raw_rms': rms(dry), 'calibrated_rms': rms(x),
            'calibrated_peak': float(np.max(np.abs(x))), 'frames': len(x),
            'wet_peaks': [float(np.max(np.abs(y))) for y in wet]}
        write_json(audit_path, audit)
        print(f'rendered {index + 1}/{len(items)} {item["id"]} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    return audit


def grouped_mean(rows, key):
    """Q parent -> type -> instrument -> global, preserving equal target/gain weights."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r['guitar'], r['type'], r['parent'], r['teacher'], r['true_gi'])].append(r[key])
    parent = {k: float(np.mean(v)) for k, v in groups.items()}
    types = defaultdict(list)
    for (guitar, kind, _, teacher, gi), value in parent.items():
        types[(guitar, kind, teacher, gi)].append(value)
    instruments = defaultdict(list)
    for (guitar, kind, teacher, gi), value in types.items():
        instruments[guitar].append(float(np.mean(value)))
    per_fold = {g: float(np.mean(v)) for g, v in instruments.items()}
    return dict(macro=float(np.mean(list(per_fold.values()))), folds=per_fold)


def select_all(plan, output):
    """Never open H quality tables here; selections depend on train features and Q only."""
    cfg, items = plan['config'], plan['records']
    lookup = {r['id']: i for i, r in enumerate(items)}
    data = []
    for i in range(len(items)):
        with np.load(cache_paths(output, i)[0]) as z:
            data.append({k: z[k] for k in z.files})
    C, G = len(plan['rigs']), len(cfg['gains_db'])
    X = np.stack([d['erb'] for d in data], axis=1).reshape(C, G, len(items), -1)
    spectra = np.stack([d['spectrum'] for d in data], axis=1).reshape(C, G, len(items), -1)
    result, diagnostics = [], {}
    for guitar, fold in plan['folds'].items():
        train = [lookup[i] for i in fold['train']]
        models = {m: fit_model(X[:, :, train], m) for m in cfg['methods'] if m != 'spectrum160'}
        diagnostics[guitar] = {m: model['diagnostics'] for m, model in models.items()}
        queries, probe_distances = [], []
        for qid in fold['query']:
            qi, item = lookup[qid], items[lookup[qid]]
            for ti, teacher in enumerate(cfg['teacher_indices']):
                for gi, gain in enumerate(cfg['gains_db']):
                    for eq in range(2):
                        query_e = data[qi]['query_erb'][ti * G + gi, eq]
                        query_s = data[qi]['query_spectrum'][ti * G + gi, eq]
                        dist = centered_spectral_distance(spectra[:, :, train], query_s, data[qi]['grid'])
                        scores = {'spectrum160': np.median(dist, axis=-1)}
                        scores.update({m: score_model(query_e, model) for m, model in models.items()})
                        common = dict(guitar=guitar, type=item['type'], parent=item['parent'], query=qid,
                                      teacher=teacher, teacher_pos=ti * G + gi, true_gi=gi, eq=eq)
                        queries.append(dict(**common, erb=query_e))
                        probe_distances.append(dist)
                        for method, score in scores.items():
                            if score.shape != (C, G) or not np.isfinite(score).all():
                                raise ValueError('Invalid selection scores')
                            for mode in ('unknown', 'known'):
                                flat = int(np.argmin(score)) if mode == 'unknown' else int(np.argmin(score[:, gi])) * G + gi
                                result.append(dict(**common, method=method, gain_mode=mode, selected=flat,
                                                   top1=(flat // G == teacher), exact=(flat == teacher * G + gi)))
        # Cache only training distances for bootstrap; no held-out candidate enters its score.
        npz_write(output / f'bootstrap-input-{guitar}.npz', train_indices=np.array(train),
                  X=X[:, :, train], query_erb=np.array([q['erb'] for q in queries]),
                  probe_distances=np.array(probe_distances))
        write_json(output / f'bootstrap-queries-{guitar}.json',
                   [{k: v for k, v in q.items() if k != 'erb'} for q in queries])
        print(f'selected {guitar}: train={len(train)} Q={len(fold["query"])}', flush=True)
    selection = dict(plan_sha256=canonical_hash(plan), rows=result, covariance=diagnostics,
                     boundary='Selection completed before reading any H quality table')
    write_json(output / 'selection-locked.json', selection)
    return selection


def holdout_tables(plan, output):
    collected = defaultdict(list)
    for i, item in enumerate(plan['records']):
        if item['q_or_h'] == 'H':
            with np.load(cache_paths(output, i)[1]) as z:
                collected[(item['guitar'], item['type'], item['parent'])].append({k: z[k] for k in z.files})
    parents = {k: {metric: np.mean([v[metric] for v in values], axis=0) for metric in ('sc', 'ltas')}
               for k, values in collected.items()}
    grouped = defaultdict(list)
    for (guitar, kind, _), table in parents.items():
        grouped[(guitar, kind)].append(table)
    return {k: {metric: np.mean([v[metric] for v in values], axis=0) for metric in ('sc', 'ltas')}
            for k, values in grouped.items()}


def attach_quality(rows, tables, G):
    result = []
    for row in rows:
        table = tables[(row['guitar'], row['type'])]
        costs = table['sc'][row['teacher_pos'], row['eq']]
        allowed = np.arange(len(costs)) if row['gain_mode'] == 'unknown' else np.arange(row['true_gi'], len(costs), G)
        oracle = int(allowed[np.argmin(costs[allowed])])
        sc = float(costs[row['selected']])
        result.append(dict(**row, sc=sc, ltas=float(table['ltas'][row['teacher_pos'], row['eq'], row['selected']]),
                           oracle=oracle, oracle_sc=float(costs[oracle]), regret=max(0.0, sc-float(costs[oracle]))))
    return result


def bootstrap_indices(items, train, rng):
    by_guitar = defaultdict(lambda: defaultdict(list))
    for local, global_index in enumerate(train):
        r = items[int(global_index)]
        by_guitar[r['guitar']][r['parent']].append(local)
    groups = sorted(by_guitar)
    selected = []
    for guitar in rng.choice(groups, size=len(groups), replace=True):
        parents = sorted(by_guitar[guitar])
        for parent in rng.choice(parents, size=len(parents), replace=True):
            selected.extend(by_guitar[guitar][parent])
    return np.array(selected, dtype=int)


def bootstrap(plan, output, tables):
    cfg = plan['config']
    rng = np.random.default_rng(cfg['bootstrap']['seed'])
    inputs = {}
    for guitar in plan['folds']:
        with np.load(output / f'bootstrap-input-{guitar}.npz') as z:
            inputs[guitar] = {k: z[k] for k in z.files}
        inputs[guitar]['queries'] = read_json(output / f'bootstrap-queries-{guitar}.json')
    C, G, records_out = len(plan['rigs']), len(cfg['gains_db']), []
    for repeat in range(cfg['bootstrap']['repeats']):
        rows, stable, angles, conditions = [], True, [], []
        for guitar, d in inputs.items():
            sampled = bootstrap_indices(plan['records'], d['train_indices'], rng)
            try:
                model = fit_model(d['X'][:, :, sampled], 'ledoit_wolf')
                original = fit_model(d['X'], 'ledoit_wolf')
                for gi in range(G):
                    cov = model['covariance'][gi]
                    conditions.append(float(np.linalg.cond(cov)))
                    k = cfg['bootstrap']['top_subspace_dimensions']
                    _, b1 = np.linalg.eigh(cov)
                    _, b0 = np.linalg.eigh(original['covariance'][gi])
                    angles.append(float(np.degrees(np.max(subspace_angles(b1[:, -k:], b0[:, -k:])))))
                if max(conditions) > cfg['gate']['max_covariance_condition']:
                    stable = False
                for qi, q in enumerate(d['queries']):
                    if q['eq'] != 1:
                        continue
                    scores = {'spectrum160': np.median(d['probe_distances'][qi][:, :, sampled], axis=-1),
                              'ledoit_wolf': score_model(d['query_erb'][qi], model)}
                    for m, s in scores.items():
                        rows.append(dict(**q, method=m, gain_mode='unknown', selected=int(np.argmin(s))))
            except (ValueError, np.linalg.LinAlgError) as exc:
                stable = False
                print(f'bootstrap {repeat} {guitar} failed: {exc}', flush=True)
                break
        delta = None
        if stable:
            scored = attach_quality(rows, tables, G)
            means = {m: grouped_mean([r for r in scored if r['method'] == m], 'regret')['macro']
                     for m in cfg['bootstrap']['methods']}
            delta = means['spectrum160'] - means['ledoit_wolf']
        records_out.append(dict(repeat=repeat, valid=stable, e1_regret_improvement=delta,
                                improved=bool(stable and delta > 0),
                                max_condition=max(conditions) if conditions else None,
                                median_subspace_angle_deg=float(np.median(angles)) if angles else None))
        print(f'bootstrap {repeat+1}/{cfg["bootstrap"]["repeats"]} valid={stable} delta={delta}', flush=True)
    return records_out


def summarize(rows, cfg):
    summary = {}
    for mode in ('unknown', 'known'):
        summary[mode] = {}
        for eq in range(2):
            summary[mode][f'E{eq}'] = {}
            for method in cfg['methods']:
                part = [r for r in rows if r['method'] == method and r['eq'] == eq and r['gain_mode'] == mode]
                summary[mode][f'E{eq}'][method] = {k: grouped_mean(part, k) for k in ('regret', 'sc', 'ltas', 'top1', 'exact')}
    return summary


def decision(summary, rows, diagnostics, boots, cfg):
    e0, e1 = summary['unknown']['E0'], summary['unknown']['E1']
    B, L = (e1[m]['regret']['macro'] for m in ('spectrum160', 'ledoit_wolf'))
    fold_delta = {g: e1['spectrum160']['regret']['folds'][g]-v for g, v in e1['ledoit_wolf']['regret']['folds'].items()}
    keys = lambda r: (r['guitar'], r['query'], r['teacher'], r['true_gi'], r['eq'])
    base = {keys(r): r for r in rows if r['method'] == 'spectrum160' and r['eq'] == 1 and r['gain_mode'] == 'unknown'}
    harms = defaultdict(list)
    for r in rows:
        if r['method'] == 'ledoit_wolf' and r['eq'] == 1 and r['gain_mode'] == 'unknown':
            harms[r['guitar'] + '/' + r['type']].append(r['regret']-base[keys(r)]['regret'])
    p90 = {g: float(np.quantile(v, .9)) for g, v in harms.items()}
    # Diagnostics are public JSON; use the exact matrix diagnostics supplied by the estimator.
    conditions = [d['condition'] for group in diagnostics.values() for d in group['ledoit_wolf']]
    gate = cfg['gate']
    checks = dict(e1_mean=(B-L >= max(gate['relative_e1_regret_improvement']*B, gate['absolute_e1_regret_improvement'])),
                  improved_folds=sum(v > 0 for v in fold_delta.values()) >= gate['min_improved_folds'],
                  e0_no_material_harm=e0['ledoit_wolf']['regret']['macro']-e0['spectrum160']['regret']['macro'] <= gate['max_e0_regret_increase'],
                  tail=max(p90.values()) <= gate['max_fold_e1_p90_harm'],
                  conditioning=max(conditions) <= gate['max_covariance_condition'],
                  bootstrap=sum(b['improved'] for b in boots) >= gate['min_bootstrap_improved'])
    return dict(checks=checks, passed=all(checks.values()), e1_improvement=B-L,
                required_improvement=max(gate['relative_e1_regret_improvement']*B, gate['absolute_e1_regret_improvement']),
                fold_improvements=fold_delta, fold_type_p90_harm=p90,
                covariance_attribution={m: e1[m]['regret']['macro']-L for m in ('euclidean', 'diagonal')},
                production_replaced=False,
                interpretation='Internal engineering screen, not a confidence bound or proof of wet-only identification')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='artifacts/stage8/real')
    ap.add_argument('--freeze', action='store_true')
    args = ap.parse_args()
    output = (ROOT / args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / 'locked-plan.json'
    if args.freeze:
        plan = make_plan()
        if lock_path.exists() and read_json(lock_path) != plan:
            raise ValueError('Existing lock differs; use a new run directory')
        write_json(lock_path, plan)
        print(f'Frozen {len(plan["records"])} inputs, 5 folds, {canonical_hash(plan)}')
        return
    if (output / 'results.json').exists():
        raise FileExistsError('Completed results exist; refusing overwrite')
    plan = read_json(lock_path)
    if plan['environment'] != environment():
        raise ValueError('Frozen package versions differ from current environment')
    validate_hashes(plan['source_hashes'])
    validate_hashes(plan['protected'])
    started = time.perf_counter()
    audit = render_all(plan, output)
    selection = select_all(plan, output)
    tables = holdout_tables(plan, output)
    rows = attach_quality(selection['rows'], tables, len(plan['config']['gains_db']))
    summary = summarize(rows, plan['config'])
    boots = bootstrap(plan, output, tables)
    result = dict(plan_sha256=canonical_hash(plan), selection_sha256=sha(output / 'selection-locked.json'),
                  rows=rows, summary=summary, covariance=selection['covariance'], bootstrap=boots,
                  decision=decision(summary, rows, selection['covariance'], boots, plan['config']),
                  render_combinations=len(audit['items'])*len(plan['rigs'])*len(plan['config']['gains_db']),
                  seconds=time.perf_counter()-started)
    validate_hashes(plan['source_hashes'])
    validate_hashes(plan['protected'])
    result['protected_unchanged'] = True
    write_json(output / 'results.json', result)
    print(json.dumps(result['decision'], indent=2), flush=True)


if __name__ == '__main__':
    main()
