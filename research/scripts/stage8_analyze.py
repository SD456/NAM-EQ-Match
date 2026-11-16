"""Read frozen stage 8 results and cached features; never render or reselect."""
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage2_benchmark import read_json, sha, write_json
from scripts.stage8_benchmark import canonical_hash, grouped_mean


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    return dict(n=len(x), pearson=float(np.corrcoef(x, y)[0, 1])
                if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0 else None)


def diagnostics(plan, audit, features):
    items = plan['records']
    indices = {r['id']: i for i, r in enumerate(items)}
    count, gains = len(plan['rigs']), len(plan['config']['gains_db'])
    X = np.stack(features, axis=1).reshape(count, gains, len(items), -1)
    output = {}
    for group, fold in plan['folds'].items():
        train = [indices[i] for i in fold['train']]
        train_items = [items[i] for i in train]
        data = X[:, :, train]
        residual = data - data.mean(axis=2, keepdims=True)
        fold_result = []
        for gi, gain in enumerate(plan['config']['gains_db']):
            v = residual[:, gi]
            covariance = np.einsum('cni,cnj->ij', v, v) / (len(train) * count)
            values, vectors = np.linalg.eigh(covariance)
            vectors = vectors[:, -3:][:, ::-1]
            projections = (v @ vectors).mean(axis=0)
            pitch = [r['pitch_midi'] if r['pitch_midi'] is not None else np.nan for r in train_items]
            rms = [audit['items'][str(i)]['raw_rms'] for i in train]
            crest = [audit['items'][str(i)]['calibrated_peak'] / audit['items'][str(i)]['calibrated_rms']
                     if audit['items'][str(i)]['calibrated_rms'] > 0 else np.nan for i in train]
            directions = []
            for axis in range(3):
                p = projections[:, axis]
                groups = defaultdict(list)
                for r, value in zip(train_items, p):
                    groups[(r['guitar'], r['type'])].append(float(value))
                means = [{'guitar': g, 'type': t, 'n': len(vs), 'mean': float(np.mean(vs))}
                         for (g, t), vs in sorted(groups.items())]
                differences = [dict(a=f'{a["guitar"]}/{a["type"]}', b=f'{b["guitar"]}/{b["type"]}',
                                    mean_a_minus_b=a['mean'] - b['mean'])
                               for a, b in combinations(means, 2)
                               if a['guitar'] == b['guitar'] or a['type'] == b['type']]
                directions.append(dict(direction=axis + 1, pitch=correlation(p, pitch),
                                       input_rms=correlation(p, rms), crest=correlation(p, crest),
                                       means_by_guitar_type=means, mean_differences=differences,
                                       vector=vectors[:, axis].tolist()))
            contents = defaultdict(dict)
            for j, item in enumerate(train_items):
                if item['content_key'] is not None:
                    contents[(item['type'], item['content_key'])][item['guitar']] = j
            pairs = defaultdict(list)
            for (kind, _), matched in contents.items():
                for a, b in combinations(sorted(matched), 2):
                    # Same chain and gain, with no subtraction of a fitted guitar effect.
                    difference = data[:, gi, matched[a]] - data[:, gi, matched[b]]
                    pairs[(kind, a, b)].append(float(np.linalg.norm(difference, axis=1).mean()))
            matched = [dict(type=t, guitar_a=a, guitar_b=b, n_content=len(d),
                            mean_23d_distance=float(np.mean(d))) for (t, a, b), d in sorted(pairs.items())]
            fold_result.append(dict(gain_db=gain, sample_eigenvalues=values[::-1].tolist(),
                                    directions=directions, matched_content=matched))
        output[group] = fold_result
    return output


def summaries(rows, methods, gain_count):
    result = {}
    for mode in ('unknown', 'known'):
        result[mode] = {}
        for eq in (0, 1):
            result[mode][f'E{eq}'] = {}
            for method in methods:
                part = [r for r in rows if r['gain_mode'] == mode and r['eq'] == eq and r['method'] == method]
                if not part:
                    raise ValueError(f'Missing {mode} E{eq} {method} rows')
                if mode == 'unknown':
                    part = [dict(r, gain_correct=(r['selected'] % gain_count == r['true_gi'])) for r in part]
                metrics = ('regret', 'sc', 'top1', 'exact') + (('gain_correct',) if mode == 'unknown' else ())
                result[mode][f'E{eq}'][method] = {k: grouped_mean(part, k) for k in metrics}
    return result


def detailed_results(rows, methods, gains):
    gain_count = len(gains)
    detailed, errors = {}, {}
    for mode in ('unknown', 'known'):
        detailed[mode] = {}
        for eq in (0, 1):
            detailed[mode][f'E{eq}'] = {}
            for method in methods:
                part = [dict(r, gain_correct=(r['selected'] % gain_count == r['true_gi'])) for r in rows
                        if r['gain_mode'] == mode and r['eq'] == eq and r['method'] == method]
                groups = sorted({(r['guitar'], r['type']) for r in part})
                detailed[mode][f'E{eq}'][method] = [
                    dict(guitar=g, type=t, query_count=len({r['query'] for r in part if r['guitar'] == g and r['type'] == t}),
                         **{k: grouped_mean([r for r in part if r['guitar'] == g and r['type'] == t], k)['macro']
                            for k in ('regret', 'sc', 'top1', 'exact', 'gain_correct')}) for g, t in groups]
                if mode == 'unknown':
                    errors.setdefault(f'E{eq}', {})[method] = []
                    for gi, gain in enumerate(gains):
                        conditioned = [dict(r, gain_error=not r['gain_correct']) for r in part if r['true_gi'] == gi]
                        errors[f'E{eq}'][method].append(dict(true_gain_db=gain,
                            selected_wrong_gain_db=gains[1 - gi], rows=len(conditioned),
                            wrong_rows=sum(r['gain_error'] for r in conditioned),
                            grouped_wrong_rate=grouped_mean(conditioned, 'gain_error')))
    return detailed, errors


def input_levels(plan, audit):
    values = defaultdict(list)
    for i, record in enumerate(plan['records']):
        values[record['type']].append(audit['items'][str(i)]['raw_rms'])
    def stats(v):
        return dict(n=len(v), min=float(np.min(v)), median=float(np.median(v)), max=float(np.max(v)))
    return dict(raw_rms_all=stats([v for group in values.values() for v in group]),
                raw_rms_by_type={kind: stats(v) for kind, v in sorted(values.items())},
                common_calibration_db=plan['config']['calibration_db'])


def tail_audit(rows, plan):
    methods = ('spectrum160', 'ledoit_wolf')
    key = lambda r: (r['guitar'], r['type'], r['query'], r['teacher'], r['true_gi'])
    indexed = {}
    for row in rows:
        if row['eq'] == 1 and row['method'] in methods:
            k = (row['gain_mode'], row['method'], key(row))
            if k in indexed:
                raise ValueError(f'Duplicate result row: {k}')
            indexed[k] = row
    keys = {k[2] for k in indexed}
    if len(indexed) != len(keys) * 4:
        raise ValueError('Missing paired unknown/known rows')
    items = {r['id']: r for r in plan['records']}
    gains, rigs = plan['config']['gains_db'], plan['rigs']
    def selected(row):
        chain, gi = divmod(row['selected'], len(gains))
        return dict(chain=rigs[chain]['id'], gain_db=gains[gi], regret=row['regret'], sc=row['sc'])
    pairs = []
    for k in sorted(keys):
        unknown_base, unknown_lw = (indexed[('unknown', method, k)] for method in methods)
        known_base, known_lw = (indexed[('known', method, k)] for method in methods)
        item = items[k[2]]
        pairs.append(dict(guitar=k[0], type=k[1], query=k[2], path=item['path'], parent=item['parent'],
                          pitch_midi=item['pitch_midi'], string=item['string'], fret=item['fret'],
                          content_key=item['content_key'], teacher=rigs[k[3]]['id'], true_gain_db=gains[k[4]],
                          unknown_harm=unknown_lw['regret'] - unknown_base['regret'],
                          known_harm=known_lw['regret'] - known_base['regret'],
                          same_chain=unknown_base['selected'] // len(gains) == unknown_lw['selected'] // len(gains),
                          unknown_baseline=selected(unknown_base), unknown_lw=selected(unknown_lw),
                          known_baseline=selected(known_base), known_lw=selected(known_lw)))
    def breakdown(fields):
        groups = defaultdict(list)
        for p in pairs:
            groups[tuple(p[f] for f in fields)].append(p)
        out = []
        for group, values in sorted(groups.items(), key=lambda x: str(x[0])):
            harms = [p for p in values if p['unknown_harm'] > .05]
            out.append(dict(zip(fields, group), rows=len(values), unknown_harm_over_005=len(harms),
                            same_chain=sum(p['same_chain'] for p in harms),
                            different_chain=sum(not p['same_chain'] for p in harms),
                            known_harm_over_005=sum(p['known_harm'] > .05 for p in values),
                            unknown_p90=float(np.quantile([p['unknown_harm'] for p in values], .9)),
                            known_p90=float(np.quantile([p['known_harm'] for p in values], .9))))
        return out
    harmful = [p for p in pairs if p['unknown_harm'] > .05]
    return dict(threshold=.05, rows=len(pairs), unknown_harm_over_005=len(harmful),
                same_chain=sum(p['same_chain'] for p in harmful),
                different_chain=sum(not p['same_chain'] for p in harmful),
                known_harm_over_005=sum(p['known_harm'] > .05 for p in pairs),
                unknown_harm_still_known=sum(p['known_harm'] > .05 for p in harmful),
                by_guitar_type=breakdown(('guitar', 'type')), by_teacher=breakdown(('teacher',)),
                by_pitch=breakdown(('pitch_midi',)), worst_four=sorted(pairs, key=lambda p: p['unknown_harm'], reverse=True)[:4])


def fmt(value):
    return '—' if value is None else f'{value:.4f}'


def write_report(analysis, path):
    decision = analysis['decision']
    e1 = analysis['summary']['unknown']['E1']
    tail = decision['fold_type_p90_harm']
    worst = max(tail, key=tail.get)
    boot = analysis['bootstrap']
    angles = [b['median_subspace_angle_deg'] for b in boot['replicates'] if b['median_subspace_angle_deg'] is not None]
    boot_conditions = [b['max_condition'] for b in boot['replicates'] if b['max_condition'] is not None]
    conditions = analysis['lw_conditions']
    lines = ['# 阶段 8：协方差检索内部验证', '',
             '这是已观察录音上的留器材组内部开发验证；允许其他琴的相同内容进入训练，不能视为独立正式确认。',
             'Q/H 使用不同父原件；H 的质量指标按类型和父原件汇总。未知增益与已知增益分别报告。',
             '', f"未知 trim 的 E1 平均 regret：原频谱基线 {fmt(e1['spectrum160']['regret']['macro'])}，LW {fmt(e1['ledoit_wolf']['regret']['macro'])}，改善 {fmt(decision['e1_improvement'])}；五个琴组均改善，{boot['improved']}/{boot['total']} 次重抽样改善。",
             f"但整体预设门槛{'通过' if decision['passed'] else '未通过'}：尾部损害检查失败，最大琴组/类型 90 分位损害为 {worst} 的 {fmt(tail[worst])}，上限 {fmt(analysis['gate']['max_fold_e1_p90_harm'])}。平均改善不等于所有门槛通过；本结果不晋级，未替换生产方法。",
             f"检查项：`{json.dumps(decision['checks'], ensure_ascii=False)}`；完整分琴/类型尾部值见 analysis.json。",
             '', '以下数值按 Q 父源、类型、器材组逐层等权；top1 为链身份命中率，exact 同时要求增益命中。', '']
    for mode in ('unknown', 'known'):
        for eq in ('E0', 'E1'):
            lines += [f'## {mode} trim / {eq}', '', '| 方法 | regret | SC | top1 | exact |', '| --- | ---: | ---: | ---: | ---: |']
            values = analysis['summary'][mode][eq]
            for method, metrics in values.items():
                lines.append('| ' + ' | '.join([method] + [fmt(metrics[k]['macro']) for k in ('regret', 'sc', 'top1', 'exact')]) + ' |')
            lines += ['', '| 琴组 | 方法 | regret | SC | top1 | exact |', '| --- | --- | ---: | ---: | ---: | ---: |']
            for guitar in analysis['folds']:
                for method, metrics in values.items():
                    lines.append('| ' + ' | '.join([guitar, method] + [fmt(metrics[k]['folds'][guitar]) for k in ('regret', 'sc', 'top1', 'exact')]) + ' |')
            lines.append('')
            lines += ['| 琴组 | 类型 | Q 文件数 | 方法 | regret | SC | top1 | exact | gain correct |',
                      '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
            for method, entries in analysis['by_guitar_type'][mode][eq].items():
                for entry in entries:
                    lines.append('| ' + ' | '.join([entry['guitar'], entry['type'], str(entry['query_count']), method]
                                 + [fmt(entry[k]) for k in ('regret', 'sc', 'top1', 'exact', 'gain_correct')]) + ' |')
            lines.append('')
    lines += ['## 同特征对照与稳定性', '',
              '下面的差值为对照 regret 减 LW regret，正值表示 LW 改善；Euclidean、diagonal 与 LW 使用相同 23 维特征和类中心。',
              '| 模式 | 扰动 | Euclidean − LW | diagonal − LW |', '| --- | --- | ---: | ---: |']
    for mode in ('unknown', 'known'):
        for eq in ('E0', 'E1'):
            d = analysis['deltas'][mode][eq]
            lines.append(f'| {mode} | {eq} | {fmt(d["euclidean_minus_lw"])} | {fmt(d["diagonal_minus_lw"])} |')
    lines += ['', '## 数字 trim 与源电平', '',
              '两档是数字输入 trim，不是物理演奏力度。下表 SC 差为未知 trim 减已知 trim；两者各自在允许的候选范围中选择，比较绝对质量 SC。已知 trim 下链排序也重新进行，因此 SC 差不能全部归为纯增益效应。',
              '| 扰动 | 方法 | 未知减已知 SC | 未知 trim gain correct |', '| --- | --- | ---: | ---: |']
    for eq in ('E0', 'E1'):
        for method, v in analysis['summary']['unknown'][eq].items():
            difference = v['sc']['macro'] - analysis['summary']['known'][eq][method]['sc']['macro']
            lines.append(f'| {eq} | {method} | {fmt(difference)} | {fmt(v["gain_correct"]["macro"])} |')
    lines += ['', '| 扰动 | 方法 | 真 trim | 错选 trim | 错选行数/总行数 | 分组错选率 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for eq, methods in analysis['gain_errors'].items():
        for method, entries in methods.items():
            for entry in entries:
                lines.append(f'| {eq} | {method} | {entry["true_gain_db"]} dB | {entry["selected_wrong_gain_db"]} dB | {entry["wrong_rows"]}/{entry["rows"]} | {fmt(entry["grouped_wrong_rate"]["macro"])} |')
    levels = analysis['input_levels']
    lines += ['', '原始输入 RMS（线性幅度，未应用公共校准）：', '',
              '| 来源类型 | 文件数 | 最小 | 中位数 | 最大 |', '| --- | ---: | ---: | ---: | ---: |']
    for kind, value in [('全部', levels['raw_rms_all'])] + list(levels['raw_rms_by_type'].items()):
        lines.append(f'| {kind} | {value["n"]} | {fmt(value["min"])} | {fmt(value["median"])} | {fmt(value["max"])} |')
    lines += ['', f"统一输入校准为 {levels['common_calibration_db']:.3f} dB；共同缩放不消除来源录音增益差，RMS 也不是力度标签。"]
    tail_review = analysis['tail_audit']
    lines += ['', '## E1 尾部损害事后审计', '',
              '以下是冻结选择后的描述，不用于调整阈值或重选。逐个相同 Q、目标链和真实 trim 配对，用 LW regret 减 160 频谱基线 regret；超过 0.05 记为损害。已知 trim 比较在受限候选集上重新排序链和重新计算 oracle，增益不是唯一原因。',
              f"未知 trim：{tail_review['unknown_harm_over_005']}/{tail_review['rows']} 行损害超过 0.05，其中 {tail_review['same_chain']} 行选同一链但 trim 不同，{tail_review['different_chain']} 行选不同链。已知 trim 全部 {tail_review['rows']} 行中 {tail_review['known_harm_over_005']} 行超过 0.05；上述未知 trim 损害行中 {tail_review['unknown_harm_still_known']} 行在已知 trim 对照中仍超过 0.05。",
              '', '| 琴组 | 类型 | 配对行数 | 未知损害行数 | 已知损害行数 | 未知 P90 | 已知 P90 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for entry in tail_review['by_guitar_type']:
        lines.append(f'| {entry["guitar"]} | {entry["type"]} | {entry["rows"]} | {entry["unknown_harm_over_005"]} | {entry["known_harm_over_005"]} | {fmt(entry["unknown_p90"])} | {fmt(entry["known_p90"])} |')
    for title, entries, field in [('目标链', tail_review['by_teacher'], 'teacher'), ('MIDI 音高', tail_review['by_pitch'], 'pitch_midi')]:
        lines += ['', f'按{title}分组（无音高标签记为 —）：', '',
                  f'| {title} | 配对行数 | 未知损害行数 | 同链/不同链 | 已知损害行数 |', '| --- | ---: | ---: | ---: | ---: |']
        for entry in entries:
            value = entry[field] if entry[field] is not None else '—'
            lines.append(f'| {value} | {entry["rows"]} | {entry["unknown_harm_over_005"]} | {entry["same_chain"]}/{entry["different_chain"]} | {entry["known_harm_over_005"]} |')
    lines += ['', '损害最大的四个配对：', '']
    for i, entry in enumerate(tail_review['worst_four'], 1):
        lines += [f'{i}. `{entry["query"]}`（{entry["guitar"]}/{entry["type"]}，pitch={entry["pitch_midi"]}，string={entry["string"]}，fret={entry["fret"]}；`{entry["path"]}`）；目标 `{entry["teacher"]}`，真实 trim {entry["true_gain_db"]} dB；未知/已知损害 {fmt(entry["unknown_harm"])}/{fmt(entry["known_harm"])}。',
                  '']
        for name, label in [('unknown_baseline', '未知基线'), ('unknown_lw', '未知 LW'), ('known_baseline', '已知基线'), ('known_lw', '已知 LW')]:
            selection = entry[name]
            lines.append(f'   - {label}：`{selection["chain"]}` / {selection["gain_db"]} dB，regret {fmt(selection["regret"])}。')
        lines.append('')
    bs = analysis['bootstrap']
    lines += ['', f"预定分组重抽样：{bs['valid']}/{bs['total']} 次有效，{bs['improved']} 次 E1 regret 改善；每次跨折和增益的前三维子空间角度中位数为 {min(angles):.2f}–{max(angles):.2f} 度。",
              f"LW 原始各折/增益条件数为 {min(conditions):.2f}–{max(conditions):.2f}；各次重抽样的最大条件数为 {min(boot_conditions):.2f}–{max(boot_conditions):.2f}，上限 {analysis['gate']['max_covariance_condition']:.0e}。这些重抽样指标不是独立确认或置信界。",
              '各次改变量与子空间角度见 analysis.json；E0、E1 的 oracle 不同，regret 应在各自扰动内比较。',
              '', '## 训练折内描述', '',
              '原始共享类内协方差每条链分别中心化后等权平均。前三方向投影先跨链平均再与输入标签做 Pearson 相关，方向符号任意，相关不代表因果；pitch 仅用有标签的 IDMT 单音。RMS 与峰均比是数字输入统计，不能解释为真实演奏力度。',
              '各折/增益的特征值、方向相关、按琴和类型的投影均值、同内容跨琴 23 维距离及样本数见 analysis.json；曲线见 stage8-results.png。',
              '跨琴匹配仅在训练组内，比较同弦品音高单音或同编号乐句，不同拾音器、接口、录音条件及乐手等混杂，不能称为纯吉他效应。',
              '原有 4/16 数据和本实验任务不同，不直接归因比较；不重选模型或参数。', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def plot(analysis, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    methods = list(analysis['summary']['unknown']['E0'])
    x = np.arange(len(methods))
    for j, eq in enumerate(('E0', 'E1')):
        v = analysis['summary']['unknown'][eq]
        axes[0, 0].bar(x + (j - .5) * .36, [v[m]['regret']['macro'] for m in methods], .36, label=eq)
    axes[0, 0].set(title='Unknown trim: grouped regret', xticks=x, xticklabels=methods, ylabel='Regret')
    axes[0, 0].tick_params(axis='x', labelrotation=20)
    axes[0, 0].legend()
    for group, gains in analysis['training_diagnostics'].items():
        for gain in gains:
            axes[0, 1].semilogy(range(1, 24), np.maximum(gain['sample_eigenvalues'], 1e-12),
                                label=f'{group} {gain["gain_db"]} dB')
    axes[0, 1].set(title='Training raw shared covariance', xlabel='Eigenvalue rank', ylabel='Eigenvalue (log)')
    axes[0, 1].legend(fontsize=6, ncol=2)
    replicates = analysis['bootstrap']['replicates']
    for ax, key, title, label in ((axes[1, 0], 'e1_regret_improvement', 'Bootstrap E1 benefit', 'Baseline − LW regret'),
                                   (axes[1, 1], 'median_subspace_angle_deg', 'Bootstrap subspace stability', 'Median angle (degrees)')):
        points = [(b['repeat'] + 1, b[key]) for b in replicates if b[key] is not None]
        if points:
            ax.scatter(*zip(*points), s=24)
        ax.set(title=title, xlabel='Resample', ylabel=label)
        if key == 'e1_regret_improvement':
            ax.axhline(0, color='gray', linestyle='--')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    folder = ROOT / 'artifacts/stage8/real'
    result, plan, audit = (read_json(folder / name) for name in ('results.json', 'locked-plan.json', 'cache-audit.json'))
    plan_hash = canonical_hash(plan)
    if result['plan_sha256'] != plan_hash or audit['plan_sha256'] != plan_hash:
        raise ValueError('Result, cache and plan do not match')
    if sha(folder / 'selection-locked.json') != result['selection_sha256']:
        raise ValueError('Locked selections changed')
    features = []
    for i, record in enumerate(plan['records']):
        entry = audit['items'][str(i)]
        path = folder / 'features' / f'{i:03d}.npz'
        if entry['id'] != record['id'] or sha(path) != entry['features_sha256']:
            raise ValueError(f'Feature cache mismatch: {path}')
        with np.load(path, allow_pickle=False) as z:
            features.append(z['erb'])
    gains = plan['config']['gains_db']
    if len(gains) != 2:
        raise ValueError('This report requires the frozen two-trim design')
    summary = summaries(result['rows'], plan['config']['methods'], len(gains))
    by_guitar_type, gain_errors = detailed_results(result['rows'], plan['config']['methods'], gains)
    deltas = {mode: {eq: {f'{m}_minus_lw': vals[m]['regret']['macro'] - vals['ledoit_wolf']['regret']['macro']
                           for m in ('euclidean', 'diagonal')}
                     for eq, vals in by_eq.items()} for mode, by_eq in summary.items()}
    sc_trim_delta = {eq: {method: {
        'macro': value['sc']['macro'] - summary['known'][eq][method]['sc']['macro'],
        'folds': {g: score - summary['known'][eq][method]['sc']['folds'][g]
                  for g, score in value['sc']['folds'].items()}}
        for method, value in summary['unknown'][eq].items()} for eq in ('E0', 'E1')}
    boots = result['bootstrap']
    analysis = dict(status='post_result_descriptive_no_reselection', result_sha256=sha(folder / 'results.json'),
                    plan_sha256=plan_hash, cache_audit_sha256=sha(folder / 'cache-audit.json'),
                    script_sha256=sha(__file__), folds=plan['folds'], summary=summary, deltas=deltas,
                    by_guitar_type=by_guitar_type, gain_errors=gain_errors, unknown_minus_known_sc=sc_trim_delta,
                    tail_audit=tail_audit(result['rows'], plan),
                    input_levels=input_levels(plan, audit),
                    decision=result['decision'], gate=plan['config']['gate'],
                    lw_conditions=[d['condition'] for g in result['covariance'].values() for d in g['ledoit_wolf']],
                    training_diagnostics=diagnostics(plan, audit, features),
                    bootstrap=dict(total=len(boots), valid=sum(b['valid'] for b in boots),
                                   improved=sum(b['improved'] for b in boots), replicates=boots))
    write_json(ROOT / 'artifacts/stage8/analysis.json', analysis)
    plot(analysis, ROOT / 'artifacts/stage8/stage8-results.png')
    write_report(analysis, ROOT / 'docs/stage8-findings.md')
    print('Wrote analysis.json, stage8-results.png and stage8-findings.md')


if __name__ == '__main__':
    main()
