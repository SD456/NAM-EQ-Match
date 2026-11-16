"""Recompute the frozen stage-3 decisions and write descriptive, paired results.

No audio is rendered or modified. Run with the project's .venv Python.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/NAM-EQ-Match'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def near_set(curve, gains, cfg):
    span = np.ptp(curve)
    chosen = gains if span < cfg['flat_span'] else gains[curve <= curve.min() + cfg['relative_span'] * span]
    intervals = []
    for gain in chosen:
        if intervals and gain == intervals[-1][1] + 1:
            intervals[-1][1] = float(gain)
        else:
            intervals.append([float(gain), float(gain)])
    return dict(gains=chosen.tolist(), intervals=intervals, count=len(chosen),
                span_db=float(chosen[-1] - chosen[0]), flat=bool(span < cfg['flat_span']))


def passed(value, gate):
    return value['mrstft_sc'] <= gate['sc'] and value['ltas_mae_db'] <= gate['ltas_db']


def verify(result, cache=None, targets=None):
    plan = result['plan']
    cfg, rigs, teachers = plan['config'], plan['rigs'], plan['teachers']
    gains = np.asarray(cfg['gains_db'], float)
    assert len(rigs) == 12 and len(teachers) == 4 and len(gains) == 25
    expected = {(t, g, r, e) for t in teachers for g in cfg['truth_gains_db']
                for r in cfg['references'] for e in (False, True)}
    assert len(expected) == 32
    rows = result['retrieval']
    assert len(rows) == 128 and len(result['holdout']) == 128 and len(result['oracles']) == 32
    keys = {(r['case'], r['method']) for r in rows}
    assert len(keys) == 128
    assert {(r['truth'], r['truth_gain_db'], r['reference'], r['eq_on']) for r in rows} == expected
    by_case = {}
    for row in rows:
        by_case.setdefault(row['case'], []).append(row)
        assert row['case'] == f"{row['truth']}_{row['truth_gain_db']}_{row['reference']}_E{int(row['eq_on'])}"
        scores = np.asarray(row['all_gain_scores'], float)
        assert scores.shape == (12, 25) and np.isfinite(scores).all()
        best = scores.argmin(axis=1)
        order = sorted(range(12), key=lambda i: scores[i, best[i]])
        winner, truth = order[0], row['truth']
        assert row['winner'] == winner and row['gain_db'] == gains[best[winner]]
        assert row['top1'] == (winner == truth) and row['top3'] == (truth in order[:3])
        assert row['forced_truth_gain_db'] == gains[best[truth]]
        assert row['forced_truth_nominal_gain_error_db'] == abs(gains[best[truth]] - row['truth_gain_db'])
        assert row['boundary'] == (row['gain_db'] in (gains[0], gains[-1]))
        assert len(row['ranking']) == 12
        for rank, i in zip(row['ranking'], order):
            assert rank['rig'] == i and rank['gain'] == gains[best[i]]
            close(rank['score'], scores[i, best[i]])
        for name, i in [('forced_truth_gain_set', truth), ('selected_gain_set', winner)]:
            assert row[name] == near_set(scores[i], gains, cfg['gain_set'])
        assert row['true_gain_in_set'] == (row['truth_gain_db'] in row['forced_truth_gain_set']['gains'])
    for values in by_case.values():
        assert {r['method'] for r in values} == set(cfg['methods'])
    holdout = {(r['case'], r['method']): r for r in result['holdout']}
    assert set(holdout) == keys
    oracle = {r['case']: r for r in result['oracles']}
    assert set(oracle) == set(by_case)
    shared = {}
    for row in rows:
        h = holdout[row['case'], row['method']]
        for field in ('eq_on', 'winner', 'gain_db', 'reference'):
            assert h[field] == row[field]
        quality = h['forced_truth_estimated_gain_bypass']
        assert np.isfinite(list(quality.values())).all()
        key = row['case'], row['winner'], row['gain_db']
        if key in shared:
            assert shared[key] == h['variants']
        shared[key] = h['variants']
    for h in result['holdout'] + result['oracles']:
        assert set(h['variants']) == set(cfg['eq_variants'])
        bypass = h['variants']['bypass']
        for name, q in h['variants'].items():
            assert np.isfinite([q['mrstft_sc'], q['ltas_mae_db'], q['actual_max_abs_db'], q['band_mean_abs_db']]).all()
            assert min(q['mrstft_sc'], q['ltas_mae_db'], q['actual_max_abs_db'], q['band_mean_abs_db']) >= 0
            assert q['actual_max_abs_db'] <= cfg['correction']['max_gain_db'] + 1e-3
            assert q['taps'] == (1 if name == 'bypass' else cfg['correction']['n_taps'])
            ds = q['mrstft_sc'] - bypass['mrstft_sc']
            dl = q['ltas_mae_db'] - bypass['ltas_mae_db']
            close([q['delta_sc'], q['delta_ltas_db']], [ds, dl])
            assert q['passed'] == passed(q, cfg['gate'])
            assert q['harmful'] == (ds > cfg['gate']['harm_sc_delta'] or dl > cfg['gate']['harm_ltas_delta_db'])
            assert q['lost_pass'] == (passed(bypass, cfg['gate']) and not passed(q, cfg['gate']))
            assert q['sc_improved'] == (ds < -cfg['gate']['harm_sc_delta'])
            assert q['ltas_improved'] == (dl < -cfg['gate']['harm_ltas_delta_db'])
    for case, h in oracle.items():
        row = by_case[case][0]
        assert h['truth'] == row['truth'] and h['eq_on'] == row['eq_on']
        if not h['eq_on']:
            close([h['variants']['bypass'][k] for k in ('mrstft_sc', 'ltas_mae_db')], [0, 0])
    assert len(result['score_fidelity']) == 4 and all(r['passed'] for r in result['score_fidelity'])
    assert {(r['rig'], r['gain']) for r in result['score_fidelity']} == {(t, g) for t in teachers[:2] for g in (0, 6)}
    aggregated = 'not_recomputed: target features absent'
    if cache is not None:
        expected_cache = {f'{i}|{p}|{g:g}' for i in range(12) for p in cfg['probes'] for g in gains}
        assert set(cache['items']) == expected_cache and len(expected_cache) == 900
        assert cache['identity'] == plan['source_hashes']
        # Features can be independently rescored without re-rendering a single WAV.
        if targets:
            assert set(targets) == set(by_case)
            from tonematch.features import ToneFingerprint, eq_penalty, nonlinear_distance
            def fp(value):
                return ToneFingerprint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in value.items()})
            for case, values in by_case.items():
                target = targets[case]
                grid = np.asarray(target['grid'])
                band = (grid >= 80) & (grid <= 8000)
                current, spectrum = np.zeros((3, 12, 25)), np.zeros((3, 12, 25))
                tfp = fp(target['fingerprint'])
                for pi, probe in enumerate(cfg['probes']):
                    for i in range(12):
                        for gi, gain in enumerate(gains):
                            item = cache['items'][f'{i}|{probe}|{gain:g}']
                            close(item['grid'], grid)
                            cfp = fp(item['fingerprint'])
                            current[pi, i, gi] = nonlinear_distance(cfp, tfp)['total'] + .25 * eq_penalty(cfp, tfp)
                            delta = np.asarray(item['spectrum'])[band] - np.asarray(target['spectrum'])[band]
                            spectrum[pi, i, gi] = np.mean(np.abs(delta - np.median(delta)))
                expected_scores = dict(current_single=current[0], spectrum_single=spectrum[0],
                                       current_multi=np.median(current, axis=0), spectrum_multi=np.median(spectrum, axis=0))
                for row in values:
                    close(row['all_gain_scores'], expected_scores[row['method']])
            aggregated = 'recomputed_all_128_grids_from_900_probe_features_and_32_targets'
    return {'retrieval_rows_verified': 128, 'holdout_rows_verified': 128, 'oracle_rows_verified': 32,
            'score_fidelity_checks': 4, 'aggregated_scores': aggregated,
            'audio_metrics': 'stored SC/LTAS checked for finite values and derived flags; audio not rerendered'}


def median(values):
    return float(np.median(values)) if len(values) else None


def quality_summary(values):
    return {'n': len(values), 'sc_median': median([q['mrstft_sc'] for q in values]),
            'ltas_median_db': median([q['ltas_mae_db'] for q in values]),
            'pass_count': sum(q['passed'] for q in values),
            'harm_count': sum(q['harmful'] for q in values),
            'lost_pass_count': sum(q['lost_pass'] for q in values),
            'improved_count': sum(q['sc_improved'] or q['ltas_improved'] for q in values),
            'mixed_direction_count': sum(q['harmful'] and (q['sc_improved'] or q['ltas_improved']) for q in values),
            'band_mean_abs_db_median': median([q['band_mean_abs_db'] for q in values]),
            'actual_max_abs_db_max': max(q['actual_max_abs_db'] for q in values)}


def summarize(result):
    cfg = result['plan']['config']
    holdout = {(h['case'], h['method']): h for h in result['holdout']}
    groups = {}
    for method in cfg['methods']:
        groups[method] = {}
        for eq in (False, True):
            rows = [r for r in result['retrieval'] if r['method'] == method and r['eq_on'] == eq]
            assert len(rows) == 16
            hs = [holdout[r['case'], method] for r in rows]
            correct = [r for r in rows if r['top1']]
            stratified = {}
            for key in ('reference', 'truth'):
                stratified[key] = {}
                for value in sorted({r[key] for r in rows}):
                    rs = [r for r in rows if r[key] == value]
                    stratified[key][str(value)] = dict(n=len(rs), top1=sum(r['top1'] for r in rs),
                                                      top3=sum(r['top3'] for r in rs),
                                                      gain_error_median_db=median([r['forced_truth_nominal_gain_error_db'] for r in rs]),
                                                      truth_gain_coverage=sum(r['true_gain_in_set'] for r in rs),
                                                      selected_boundary_count=sum(r['boundary'] for r in rs),
                                                      forced_boundary_count=sum(r['forced_truth_gain_db'] in (-12, 12) for r in rs),
                                                      variants={v: quality_summary([holdout[r['case'], method]['variants'][v] for r in rs]) for v in cfg['eq_variants']})
            groups[method][f'E{int(eq)}'] = dict(
                n=16, top1=sum(r['top1'] for r in rows), top3=sum(r['top3'] for r in rows),
                forced_gain_error_median_db=median([r['forced_truth_nominal_gain_error_db'] for r in rows]),
                correct_chain_subset=dict(n=len(correct), gain_error_median_db=median([abs(r['gain_db'] - r['truth_gain_db']) for r in correct])),
                truth_gain_coverage=sum(r['true_gain_in_set'] for r in rows),
                gain_set_size_median=median([r['forced_truth_gain_set']['count'] for r in rows]),
                gain_set_span_median_db=median([r['forced_truth_gain_set']['span_db'] for r in rows]),
                flat_curve_count=sum(r['forced_truth_gain_set']['flat'] for r in rows),
                selected_boundary_count=sum(r['boundary'] for r in rows),
                forced_boundary_count=sum(r['forced_truth_gain_db'] in (-12, 12) for r in rows),
                stratified=stratified,
                variants={v: quality_summary([h['variants'][v] for h in hs]) for v in cfg['eq_variants']},
                forced_truth_estimated_gain_quality=dict(
                    sc_median=median([h['forced_truth_estimated_gain_bypass']['mrstft_sc'] for h in hs]),
                    ltas_median_db=median([h['forced_truth_estimated_gain_bypass']['ltas_mae_db'] for h in hs]),
                    pass_count=sum(passed(h['forced_truth_estimated_gain_bypass'], cfg['gate']) for h in hs)))
    paired = {}
    for family in ('current', 'spectrum'):
        single, multi = groups[family + '_single'], groups[family + '_multi']
        single_rows = {r['case']: r for r in result['retrieval'] if r['method'] == family + '_single'}
        multi_rows = {r['case']: r for r in result['retrieval'] if r['method'] == family + '_multi'}
        per_eq = {}
        for eq in ('E0', 'E1'):
            s, m = single[eq], multi[eq]
            per_eq[eq] = dict(top1_delta=m['top1'] - s['top1'], top3_delta=m['top3'] - s['top3'],
                gain_median_reduction_db=s['forced_gain_error_median_db'] - m['forced_gain_error_median_db'],
                bypass_pass_delta=m['variants']['bypass']['pass_count'] - s['variants']['bypass']['pass_count'],
                retrieval_gate=m['top1'] >= cfg['gate']['top1_count_per_eq'] and m['top3'] >= cfg['gate']['top3_count_per_eq'])
            paired_rows = [(r, multi_rows[case]) for case, r in single_rows.items() if f"E{int(r['eq_on'])}" == eq]
            per_eq[eq]['paired_cases'] = [dict(case=a['case'],
                top1_before=a['top1'], top1_after=b['top1'], top3_before=a['top3'], top3_after=b['top3'],
                gain_error_delta_db=b['forced_truth_nominal_gain_error_db'] - a['forced_truth_nominal_gain_error_db'],
                bypass_pass_before=holdout[a['case'], a['method']]['variants']['bypass']['passed'],
                bypass_pass_after=holdout[b['case'], b['method']]['variants']['bypass']['passed']) for a, b in paired_rows]
        paired[family] = dict(per_eq=per_eq, joint_improvement=all(
            q['retrieval_gate'] and q['gain_median_reduction_db'] >= cfg['gate']['gain_median_improvement_db']
            and q['bypass_pass_delta'] >= 0 for q in per_eq.values()))
    drift = {}
    for method in cfg['methods']:
        index = {(r['truth'], r['truth_gain_db'], r['reference'], r['eq_on']): r for r in result['retrieval'] if r['method'] == method}
        ref_pairs, eq_pairs = [], []
        for (t, g, ref, e), r in index.items():
            if ref == 'R0':
                other = index[t, g, 'R1', e]
                ref_pairs.append(dict(truth=t, truth_gain_db=g, eq_on=e, delta_db=other['forced_truth_gain_db'] - r['forced_truth_gain_db']))
            if not e:
                other = index[t, g, ref, True]
                eq_pairs.append(dict(truth=t, truth_gain_db=g, reference=ref, delta_db=other['forced_truth_gain_db'] - r['forced_truth_gain_db']))
        drift[method] = dict(reference_R1_minus_R0=ref_pairs, EQ_E1_minus_E0=eq_pairs,
            reference_abs_drift_median_db=median([abs(x['delta_db']) for x in ref_pairs]),
            EQ_abs_drift_median_db=median([abs(x['delta_db']) for x in eq_pairs]))
    oracle = {f'E{int(e)}': {v: quality_summary([r['variants'][v] for r in result['oracles'] if r['eq_on'] == e])
                            for v in cfg['eq_variants']} for e in (False, True)}
    guard = {}
    for e in (False, True):
        hs = [h for h in result['oracles'] if h['eq_on'] == e]
        guard[f'E{int(e)}'] = dict(n=len(hs),
            median_improves_guard_does_not=sum((h['variants']['median_fit']['sc_improved'] or h['variants']['median_fit']['ltas_improved']) and not (h['variants']['guard_fit']['sc_improved'] or h['variants']['guard_fit']['ltas_improved']) for h in hs),
            median_pass_guard_fail=sum(h['variants']['median_fit']['passed'] and not h['variants']['guard_fit']['passed'] for h in hs),
            guard_higher_sc_count=sum(h['variants']['guard_fit']['mrstft_sc'] > h['variants']['median_fit']['mrstft_sc'] for h in hs),
            guard_higher_ltas_count=sum(h['variants']['guard_fit']['ltas_mae_db'] > h['variants']['median_fit']['ltas_mae_db'] for h in hs))
    return dict(groups=groups, multi_vs_single=paired, forced_gain_drift=drift, oracle=oracle, oracle_guard_tradeoff=guard)


def draw(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = list(summary['groups'])
    labels = ['Current\nsingle', 'Spectrum\nsingle', 'Current\nmulti', 'Spectrum\nmulti']
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    x = np.arange(4)
    for eq, offset, color in [('E0', -.18, '#3274a1'), ('E1', .18, '#df8f44')]:
        values = [summary['groups'][m][eq] for m in methods]
        axes[0].bar(x + offset, [v['top1'] for v in values], .32, color=color, label=eq + ' Top1')
        axes[0].scatter(x + offset, [v['top3'] for v in values], marker='_', s=170, color='black', label='Top3' if eq == 'E0' else None, zorder=4)
        axes[1].bar(x + offset, [v['forced_gain_error_median_db'] for v in values], .32, color=color, label=eq)
    axes[0].axhline(12, linestyle=':', color='gray', linewidth=1)
    axes[0].axhline(14, linestyle='--', color='gray', linewidth=1)
    axes[0].set(title='Exact chain retrieval (16 cases / EQ)', ylabel='Cases', ylim=(0, 17))
    axes[1].set(title='Forced true-chain gain error', ylabel='Median absolute error (dB)', ylim=(0, 13))
    for ax in axes[:2]:
        ax.set_xticks(x, labels)
        ax.legend(fontsize=8)
    variants = ['single_fit', 'median_fit', 'guard_fit']
    for eq, offset, color in [('E0', -.18, '#3274a1'), ('E1', .18, '#df8f44')]:
        values = [summary['oracle'][eq][v] for v in variants]
        axes[2].bar(np.arange(3) + offset, [v['harm_count'] for v in values], .32, color=color, label=eq)
    axes[2].set(title='Oracle chain + gain: EQ harms H0', ylabel='Harmful cases / 16', ylim=(0, 17))
    axes[2].set_xticks(np.arange(3), ['P0 fit', 'Median fit', 'Guard fit'])
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Frozen same-recording study: 32 related targets, one holdout phrase', fontsize=13)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def report(result, summary, checks, digest):
    c0 = summary['groups']['current_single']['E0']
    cm0 = summary['groups']['current_multi']['E0']
    s1 = summary['groups']['spectrum_single']['E1']
    sm1 = summary['groups']['spectrum_multi']['E1']
    oe0 = summary['oracle']['E0']
    lines = ['# 第三阶段结果：多探针、增益漂移与错误 EQ', '',
        '本轮为同一次录音的 32 个相关目标（E0/E1 各 16），四种方法在同一案例上配对；仅 H0 一个新留出乐句。P0–P2、R0/R1、H0 与旧 A/B/C 均无样本重叠，但不能据此声称跨录音、吉他或乐手泛化。Free Bird 仍暂停，生产参数与第二阶段基线未改。', '',
        f"本轮两类评分的多探针均未达到联合改进门槛。既有评分在 E0 的强制真链增益误差中位数由 {c0['forced_gain_error_median_db']:.1f} 降至 {cm0['forced_gain_error_median_db']:.1f} dB；平滑谱在 E1 的 Top1 由 {s1['top1']}/16 升至 {sm1['top1']}/16、直通过筛由 {s1['variants']['bypass']['pass_count']}/16 升至 {sm1['variants']['bypass']['pass_count']}/16，但增益误差中位数由 {s1['forced_gain_error_median_db']:.1f} 变为 {sm1['forced_gain_error_median_db']:.1f} dB。链身份、驱动档与留出误差需要分别评价。", '',
        f"错误 EQ 仍未解决：E0 已知真链、真增益时，单段／中位数／门控校正分别使 {oe0['single_fit']['harm_count']}/16、{oe0['median_fit']['harm_count']}/16、{oe0['guard_fit']['harm_count']}/16 例实质恶化；后两者却都保持 {oe0['median_fit']['pass_count']}/16、{oe0['guard_fit']['pass_count']}/16 过筛。门控减小错误校正幅度，不能据过筛数认定无害。", '',
        '## 检索与数字增益', '',
        '|方法|EQ|Top1|Top3|强制真链增益绝对误差中位数 dB|近优集合覆盖真档|集合大小/宽度中位数|胜者/真链边界数|',
        '|---|---|---|---|---|---|---|---|']
    for method, eqs in summary['groups'].items():
        for eq, g in eqs.items():
            lines.append(f"|{method}|{eq}|{g['top1']}/16|{g['top3']}/16|{g['forced_gain_error_median_db']:.2f}|{g['truth_gain_coverage']}/16|{g['gain_set_size_median']:g} 档 / {g['gain_set_span_median_db']:g} dB|{g['selected_boundary_count']}/16 / {g['forced_boundary_count']}/16|")
    lines += ['', '近优集合按最低分加全曲线跨度的 5% 保留，平坦曲线全保留；允许不连通集合，它不是概率置信区间。错链增益只具名义意义，增益准确度以强制真链的全部 16 例为主。', '']
    for family, pair in summary['multi_vs_single'].items():
        lines.append(f"- {family} 多探针联合改进：{'达到' if pair['joint_improvement'] else '未达到'}冻结门槛。")
        for eq, p in pair['per_eq'].items():
            lines.append(f"  {eq}：Top1 变化 {p['top1_delta']:+d}/16，Top3 {p['top3_delta']:+d}/16；增益误差中位数降低 {p['gain_median_reduction_db']:.2f} dB；H0 直通过筛数变化 {p['bypass_pass_delta']:+d}/16。检索绝对门槛{'通过' if p['retrieval_gate'] else '未通过'}。")
    lines += ['', '联合门槛要求 E0/E1 均 Top1 ≥12/16、Top3 ≥14/16；各自增益误差中位数比同类单探针降低至少 1 dB，且 H0 直通过筛数均不下降。分项变化不等于联合获胜。', '',
              '## H0 校正：与同例直通配对', '',
              '|方法|EQ|方案|SC中位数|LTAS中位数 dB|过筛|恶化|改善|失守|双向越界|', '|---|---|---|---|---|---|---|---|---|---|']
    for method, eqs in summary['groups'].items():
        for eq, g in eqs.items():
            for variant, q in g['variants'].items():
                lines.append(f"|{method}|{eq}|{variant}|{q['sc_median']:.4f}|{q['ltas_median_db']:.3f}|{q['pass_count']}/16|{q['harm_count']}/16|{q['improved_count']}/16|{q['lost_pass_count']}/16|{q['mixed_direction_count']}/16|")
    lines += ['', '过筛为 SC ≤0.20 且 LTAS ≤1.5 dB；恶化为相对直通 ΔSC >0.02 或 ΔLTAS >0.25 dB，改善按对称负阈值判定。两指标同时反向越界时可同时计改善与恶化，双向越界单列。失守指直通过筛而校正后失败。', '',
              '## 已知真链与真增益：凭空校正与真实 EQ 拒绝', '',
              '|EQ|方案|SC中位数|LTAS中位数 dB|过筛|恶化|改善|实际80–8000Hz平均绝对EQ中位数 dB|', '|---|---|---|---|---|---|---|---|']
    for eq, variants in summary['oracle'].items():
        for variant, q in variants.items():
            lines.append(f"|{eq}|{variant}|{q['sc_median']:.4f}|{q['ltas_median_db']:.3f}|{q['pass_count']}/16|{q['harm_count']}/16|{q['improved_count']}/16|{q['band_mean_abs_db_median']:.3f}|")
    lines += ['', 'E0 已知真链、真增益的直通两项误差经断言为数值零附近，因而仅看校正后是否过筛会漏掉凭空引入的误差。上表同时报告恶化比例与实际 FIR 频响强度。']
    for eq, q in summary['oracle_guard_tradeoff'].items():
        lines.append(f"{eq}：中位数方案改善但 guard 无实质改善 {q['median_improves_guard_does_not']}/16；中位数过筛但 guard 失败 {q['median_pass_guard_fail']}/16；guard 的 SC/LTAS 高于中位数方案分别 {q['guard_higher_sc_count']}/16、{q['guard_higher_ltas_count']}/16。E1 的这些量用于检查保守门控是否丢失真实 EQ 的校正机会；它们不等同于逐频真 EQ 识别率。")
    lines += ['', 'median_fit 和 guard_fit 使用相同额外半倍频程平滑，仅门控不同；single_fit 沿用旧 P0 算法，因此与 median_fit 的差异包含探针数和额外平滑，不能只归因探针数。', '',
              '## 参考乐句、教师分层与漂移', '',
              '|方法|EQ|分层|Top1|Top3|强制真链增益误差中位数 dB|', '|---|---|---|---|---|---|']
    rigs = result['plan']['rigs']
    for method, eqs in summary['groups'].items():
        for eq, g in eqs.items():
            for key, strata in g['stratified'].items():
                for value, q in strata.items():
                    label = rigs[int(value)]['id'] if key == 'truth' else value
                    lines.append(f"|{method}|{eq}|{label}|{q['top1']}/{q['n']}|{q['top3']}/{q['n']}|{q['gain_error_median_db']:.2f}|")
    for method, d in summary['forced_gain_drift'].items():
        lines.append(f"\n{method}：R0→R1 真链最佳档绝对变化中位数 {d['reference_abs_drift_median_db']:.2f} dB（16 对）；E0→E1 为 {d['EQ_abs_drift_median_db']:.2f} dB（16 对）。逐对有符号差保存在 analysis.json。")
    lines += ['', '## 复核、产物与限制', '',
              f"结果 SHA-256：`{digest}`。检索 128 行的全部 12×25 分数、排名、档位、Top1/Top3、名义误差、边界与近优集合均独立复算；H0 与 oracle 的派生阈值、同设置同结果和 E0 自对照均验证。评分直连保真检查 4/4。聚合分数复核：`{checks['aggregated_scores']}`。", '',
              '基线与冻结源文件 SHA-256 逐项比对见 analysis.json。候选缓存必须恰为 900 个 P 探针条目，不含 R/H。原始结果保留逐例 SC/LTAS、增量与实际滤波器强度；此后处理不重新渲染音频，因此两项音频指标仅复核派生计算与自一致性。', '',
              '第二阶段与本轮参考／探针数据不同，不能把跨阶段数字差异归因算法。本轮结论限于四方法同案例配对；单一录音、封闭候选库、共享教师与 H0，禁止将 32 例视为独立抽样或推断生产可用性。', '',
              '![Stage 3 descriptive results](../artifacts/stage3/stage3-results.png)', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', default='artifacts/stage3/real/results.json')
    ap.add_argument('--self-check', action='store_true', help='also ensure corrupted decisions/metrics are rejected')
    args = ap.parse_args()
    path = ROOT / args.results
    result = read(path)
    cache = read(path.with_name('probe-fingerprints.json'))
    reference_path = path.with_name('reference-features.json')
    references = read(reference_path)
    if references:
        assert references['results_sha256'] == sha(path)
        assert references['probe_cache_sha256'] == sha(path.with_name('probe-fingerprints.json'))
        assert references['audit_script_sha256'] == sha(ROOT / 'scripts/audit_stage3_scores.py')
        assert references['passed'] and not references['holdout_rendered']
        assert references['score_matrices_checked'] == 128 and references['score_values_checked'] == 38400
        assert len(references['references']) == 32
    checks = verify(result, cache, references['references'] if references else None)
    plan = result['plan']
    checks['baseline_hashes'] = {p: sha(ROOT / p) == digest for p, digest in plan['baseline_hashes'].items()}
    checks['source_hashes'] = {p: sha(ROOT / p) == digest for p, digest in plan['source_hashes'].items()}
    assert all(checks['baseline_hashes'].values()) and all(checks['source_hashes'].values())
    assert result['baseline_preserved']
    segments = plan['di']['segments']
    assert {s['id']: s['role'] for s in segments} == dict(P0='probe', P1='probe', P2='probe', R0='reference', R1='reference', H0='holdout')
    old_segments = read(ROOT / 'assets/stage2/di-provenance.json')['segments']
    for i, segment in enumerate(segments):
        assert sha(ROOT / segment['path']) == segment['sha256']
        assert segment['frames'] == 8 * 48000 and segment['end_frame'] - segment['start_frame'] == segment['frames']
        assert segment['sample_rate'] == 48000 and segment['channels'] == 1
        for other in segments[:i] + old_segments:
            assert segment['recording_id'] != other['recording_id'] or max(segment['start_frame'], other['start_frame']) >= min(segment['end_frame'], other['end_frame'])
    checks['roles_hashes_and_nonoverlap'] = True
    if references:
        checks['reference_features_sha256'] = sha(reference_path)
    if args.self_check:
        for corruption in ('winner', 'score', 'quality', 'all_gain_scores'):
            bad = copy.deepcopy(result)
            if corruption == 'winner':
                bad['retrieval'][0]['winner'] = (bad['retrieval'][0]['winner'] + 1) % 12
            elif corruption == 'score':
                bad['retrieval'][0]['ranking'][0]['score'] += 1
            elif corruption == 'quality':
                bad['holdout'][0]['variants']['guard_fit']['delta_sc'] += 1
            else:
                row = bad['retrieval'][0]
                gi = list(result['plan']['config']['gains_db']).index(row['gain_db'])
                row['all_gain_scores'][row['winner']][gi] += 1000
            try:
                verify(bad)
            except AssertionError:
                continue
            raise AssertionError(f'Corruption was not rejected: {corruption}')
        checks['negative_corruption_checks'] = ['winner', 'ranking_score', 'delta_sc', 'all_gain_scores']
    summary = summarize(result)
    digest = sha(path)
    output = ROOT / 'artifacts/stage3'
    output.mkdir(parents=True, exist_ok=True)
    analysis = dict(source_results=str(path.relative_to(ROOT)), source_results_sha256=digest,
                    analyzer_sha256=sha(__file__), baseline_sha256=plan['baseline_hashes'], checks=checks, **summary)
    (output / 'analysis.json').write_text(json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    draw(summary, output / 'stage3-results.png')
    (ROOT / 'docs/stage3-findings.md').write_text(report(result, summary, checks, digest), encoding='utf-8')
    print(json.dumps({'verified': checks, 'joint_improvement': {k: v['joint_improvement'] for k, v in summary['multi_vs_single'].items()}}, indent=2))


if __name__ == '__main__':
    main()
