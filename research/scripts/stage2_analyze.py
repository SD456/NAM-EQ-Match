"""Recompute the frozen Stage 2 results without audio rendering or model loading."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/NAM-EQ-Match"))
from tonematch.features import ToneFingerprint, distortion_index, eq_penalty, nonlinear_distance

GROUPS = ("same_noeq", "same_eq", "cross_noeq", "cross_eq")
METRICS = ("top1", "top3", "coarse_top5", "exhaustive_top1", "exhaustive_top3", "nam_top1", "family_top1")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def group_key(row):
    return ("same" if row["same_di"] else "cross") + ("_eq" if row["eq_on"] else "_noeq")


def distribution(values):
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values))} if values else {"n": 0}


def summarize(data):
    cfg, rows = data["plan"]["config"], data["cases"]
    rigs = data["plan"]["rigs"]
    assert len(rows) == 64 * 6
    assert len({r["case"] for r in rows}) == 64
    assert len({(r["case"], r["method"]) for r in rows}) == len(rows)
    assert set(r["method"] for r in rows) == set(cfg["methods"])
    assert len(data["holdout"]) == 64
    assert {r["case"] for r in data["holdout"]} == {r["case"] for r in rows}
    assert len(data["production_fidelity"]) == 4
    assert len({r["case"] for r in data["production_fidelity"]}) == 4
    assert all(r["matches_production"] for r in data["production_fidelity"])
    for row in rows:
        truth, winner = row["truth"], row["winner"]
        refined, exhaustive = row["refined_ranking"], row["exhaustive_ranking"]
        assert len(refined) == 5 and len(exhaustive) == 12
        assert row["top1"] == (winner == truth)
        assert row["top3"] == (truth in [r["rig"] for r in refined[:3]])
        assert row["coarse_top5"] == (truth in [r["rig"] for r in refined])
        assert row["exhaustive_top1"] == (exhaustive[0]["rig"] == truth)
        assert row["exhaustive_top3"] == (truth in [r["rig"] for r in exhaustive[:3]])
        assert row["nam_top1"] == (rigs[truth]["model_id"] == rigs[winner]["model_id"])
        assert row["family_top1"] == (rigs[truth]["family"] == rigs[winner]["family"])
        assert row["same_di"] == (row["reference_di"] == row["search_di"])
        assert winner == refined[0]["rig"] and row["gain_db"] == refined[0]["gain"]
        assert row["forced_truth_nominal_gain_error_db"] == abs(row["forced_truth_best_gain_db"] - row["truth_gain_db"])
    summary, gain = {}, {}
    for method in cfg["methods"]:
        summary[method], gain[method] = {}, {}
        for key in GROUPS:
            group = [r for r in rows if r["method"] == method and group_key(r) == key]
            assert len(group) == 16
            result = {"n": len(group), **{metric: sum(r[metric] for r in group) for metric in METRICS},
                      "median_forced_truth_nominal_gain_error_db": float(np.median([r["forced_truth_nominal_gain_error_db"] for r in group]))}
            if key.startswith("cross"):
                result["internal_gate_passed"] = all(result[k] >= 16 * cfg["gate"][g] for k, g in
                    (("top1", "cross_top1_fraction"), ("top3", "cross_top3_fraction"), ("coarse_top5", "cross_coarse_top5_fraction")))
            summary[method][key] = result
            gain[method][key] = {
                "exact_rig_selected_nominal_error_db": distribution([r["nominal_gain_error_if_exact_rig"] for r in group if r["top1"]]),
                "forced_true_rig_nominal_error_db": distribution([r["forced_truth_nominal_gain_error_db"] for r in group]),
                "forced_true_rig_error_at_most_1db": sum(r["forced_truth_nominal_gain_error_db"] <= 1 for r in group),
                "exact_rig_selected_error_at_most_1db": sum(r["top1"] and r["nominal_gain_error_if_exact_rig"] <= 1 for r in group)}
    assert summary == data["summary"], "Independent retrieval aggregation differs from primary"
    quality, primary_quality = {}, {}
    for key in GROUPS:
        group = [r for r in data["holdout"] if group_key(r) == key]
        assert len(group) == 16
        quality[key], primary_quality[key] = {}, {}
        for variant in group[0]["metrics"]:
            sc = [r["metrics"][variant]["mrstft_sc"] for r in group]
            ltas = [r["metrics"][variant]["ltas_mae_db"] for r in group]
            assert np.isfinite(sc + ltas).all()
            count = sum(s <= cfg["gate"]["holdout_sc"] and l <= cfg["gate"]["holdout_ltas_db"] for s, l in zip(sc, ltas))
            primary_quality[key][variant] = {"median_sc": float(np.median(sc)), "median_ltas_db": float(np.median(ltas)), "within_screening_threshold": count}
            quality[key][variant] = {**primary_quality[key][variant], "n": 16, "sc": distribution(sc), "ltas_db": distribution(ltas)}
        before, after = "selected_before_eq", "selected_after_eq"
        quality[key]["paired_change_after_minus_before"] = {
            metric: distribution([r["metrics"][after][metric] - r["metrics"][before][metric] for r in group])
            for metric in ("mrstft_sc", "ltas_mae_db")}
        quality[key]["paired_improved_counts"] = {
            metric: sum(r["metrics"][after][metric] < r["metrics"][before][metric] for r in group)
            for metric in ("mrstft_sc", "ltas_mae_db")}
    assert primary_quality == data["holdout_summary"], "Independent holdout aggregation differs from primary"
    return summary, gain, quality


def feature_diagnostics(cache, plan):
    items = {key: ToneFingerprint(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in fp.items()})
             for key, fp in cache["items"].items()}
    assert cache["identity"] == plan["identity"]
    assert len(items) == 12 * 2 * 25
    result = {"definition": "Fixed gain 0/+6 dB; both candidate/target directions included. Weighted components, plus 0.25 EQ penalty. C excluded.", "by_gain": {}}
    for gain in (0, 6):
        same_rig, other_rig = [], []
        for i in range(12):
            for di, other in (("A", "B"), ("B", "A")):
                cand = items[f"{i}|{di}|{gain}"]
                target = items[f"{i}|{other}|{gain}"]
                same_rig.append({**nonlinear_distance(cand, target), "weighted_eq": 0.25 * eq_penalty(cand, target)})
                for j in range(12):
                    if i != j:
                        target = items[f"{j}|{di}|{gain}"]
                        other_rig.append({**nonlinear_distance(cand, target), "weighted_eq": 0.25 * eq_penalty(cand, target)})
        for collection in (same_rig, other_rig):
            for row in collection:
                row["baseline_total"] = row["total"] + row["weighted_eq"]
        result["by_gain"][str(gain)] = {
            label: {"n_directed_pairs": len(collection), "component_means": {k: float(np.mean([r[k] for r in collection])) for k in collection[0]}}
            for label, collection in (("same_rig_different_di", same_rig), ("different_rig_same_di", other_rig))}
    selected = [items[f"{i}|{di}|{gain}"] for i in range(12) for di in ("A", "B") for gain in (0, 6)]
    raw = {"crest": [(22 - fp.crest_db) / 12 for fp in selected], "kurtosis": [(10 - fp.kurtosis) / 7.5 for fp in selected],
           "peak_density": [fp.peak_density / .20 for fp in selected], "hf_ratio": [fp.hf_ratio / .30 for fp in selected]}
    result["distortion_index"] = {"n": len(selected), "distribution": distribution([distortion_index(fp) for fp in selected]),
        "components": {key: {"low_clipped": sum(v <= 0 for v in values), "high_clipped": sum(v >= 1 for v in values),
                             "high_clipped_fraction": sum(v >= 1 for v in values) / len(values), "n": len(values)} for key, values in raw.items()}}
    return result


def failure_cases(data):
    rigs = data["plan"]["rigs"]
    errors = []
    for row in data["cases"]:
        if row["method"] != "baseline" or row["top1"]:
            continue
        cause = ("scoring_prefers_other_rig" if not row["exhaustive_top1"] else
                 "coarse_pruning" if not row["coarse_top5"] else "local_gain_refinement")
        parts = row["winner_parts"]
        errors.append({"case": row["case"], "group": group_key(row), "truth": rigs[row["truth"]]["id"],
            "winner": rigs[row["winner"]]["id"], "true_gain_db": row["truth_gain_db"], "winner_gain_db": row["gain_db"],
            "forced_truth_best_gain_db": row["forced_truth_best_gain_db"], "coarse_top5": row["coarse_top5"],
            "exhaustive_top1": row["exhaustive_top1"], "diagnostic": cause,
            "winner_minus_truth_best_parts": {k: parts[k] - row["truth_best_parts"][k] for k in parts},
            "winner_minus_truth_at_true_gain_parts": {k: parts[k] - row["truth_at_true_gain_parts"][k] for k in parts},
            "refined_top2_score_gap": row["refined_ranking"][1]["score"] - row["refined_ranking"][0]["score"]})
    return errors


def plot(data, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    labels = ["Same / E0", "Same / E1", "Cross / E0", "Cross / E1"]
    x = np.arange(4)
    for offset, metric, label in ((-.24, "top1", "Strict Top1"), (0, "top3", "Strict Top3"), (.24, "coarse_top5", "Coarse Top5")):
        counts = [summary["baseline"][key][metric] for key in GROUPS]
        bars = axes[0].bar(x + offset, counts, width=.23, label=label)
        axes[0].bar_label(bars, fontsize=8, padding=2)
    axes[0].set(ylabel="Correct / 16 cases", ylim=(0, 19), title="Frozen baseline: exact rig retrieval")
    axes[0].set_yticks((0, 4, 8, 12, 16))
    axes[0].legend(fontsize=8, ncol=3, loc="upper center")
    for ax, metric, title, threshold in ((axes[1], "mrstft_sc", "Held-out C: paired MRSTFT SC", .2),
                                          (axes[2], "ltas_mae_db", "Held-out C: centered LTAS MAE (dB)", 1.5)):
        for offset, variant, label, color in ((-.22, "selected_before_eq", "Before EQ", "#4575b4"),
            (0, "selected_after_eq", "After EQ", "#d73027"), (.22, "true_rig_true_gain_fitted_eq", "True rig/gain + fitted EQ", "#1a9850")):
            values = [[r["metrics"][variant][metric] for r in data["holdout"] if group_key(r) == key] for key in GROUPS]
            boxes = ax.boxplot(values, positions=x + offset, widths=.18, patch_artist=True, manage_ticks=False,
                               medianprops={"color": "black"}, flierprops={"markersize": 2})
            for box in boxes["boxes"]:
                box.set_facecolor(color)
            ax.plot([], [], color=color, lw=6, label=label)
        ax.axhline(threshold, color="gray", ls="--", lw=1, label=f"Screen: {threshold}")
        ax.set(title=title, ylim=(0, None))
        ax.legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("12 real NAM/cab rigs; 16 dependent cases per group; descriptive results only", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def report(analysis, data, path):
    summary, gain, quality = analysis["summary"], analysis["gain"], analysis["holdout"]
    base = summary["baseline"]
    lines = ["# 第二阶段真实 NAM 检索结果", "", "本报告由 `scripts/stage2_analyze.py` 从冻结正式运行独立复算。"
             "仅使用 `artifacts/stage2/real_fp64`；早期 float32 校准运行在生产一致性检查处失败，其 partial cache 不进入统计。", "",
             "12 条真实链（10 个 NAM，其中两个另配一条箱体），4 条目标链 × 2 档真增益 × A/B 两个目标乐句 × 同/跨 DI × E0/E1 = 64 例。"
             "六种预声明评分共 384 行；4 例生产搜索直接核对全部通过。A/B/C 来自同一吉他、同一录音 session，各 8 秒；每格 16 例相互依赖，不能当作 16 个独立设备或独立演奏者。", "",
             "## 冻结基线", "", "| 分组 | Exact Top1 | Exact Top3 | Coarse Top5 | 全扫 Top1 | 全扫 Top3 | NAM Top1 | Family Top1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in GROUPS:
        lines.append("| " + key + " | " + " | ".join(f"{base[key][k]}/16" for k in METRICS) + " |")
    gates = analysis["gates"]
    lines += ["", f"同 DI/E0 工程门槛 16/16：{'通过' if gates['same_noeq_passed'] else '未通过'}。"
              f"跨 DI/E0 内部门槛：{'通过' if gates['cross_noeq_passed'] else '未通过'}；跨 DI/E1：{'通过' if gates['cross_eq_passed'] else '未通过'}。"
              "跨 DI 每格门槛固定为 Top1 ≥12/16、Top3 ≥14/16、coarse Top5 =16/16，未因结果放宽。随机排列参照分别为 1/12、3/12、5/12。", "",
              "## 数字增益", "", "误差均是公共 A 校准参考系下的 nominal dB，不能解释成物理驱动量恢复准确率。", "",
              "| 分组 | 选对完整链的分母 | 选对链时增益绝对误差中位数 dB | 强制真链误差中位数 dB（n=16） | 强制真链误差≤1 dB |", "|---|---:|---:|---:|---:|"]
    for key in GROUPS:
        g = gain["baseline"][key]
        selected = g["exact_rig_selected_nominal_error_db"]
        lines.append(f"| {key} | {selected['n']}/16 | {selected.get('median', '无')} | {g['forced_true_rig_nominal_error_db']['median']:.2f} | {g['forced_true_rig_error_at_most_1db']}/16 |")
    lines += ["", "## 留出 C 的同输入配对评价", "", "C 完全排除在检索、特征缓存与 EQ 拟合之外。每例把同一段 C 分别送入选中链和真链，尾长补零至两者较长长度；只为评价以整段 RMS 标量对齐候选输出。"
              "MRSTFT SC 用 512/2048/8192 点窗、75% overlap 的三个幅度谱收敛度平均；LTAS 在 80–8000 Hz 对数频点上去掉差值中位数后计算 MAE。"
              "下表连续值为每格16例的中位数；筛选计数要求 SC≤0.20 且 LTAS≤1.5 dB。这是客观内部筛选，不是听感等价证明。", "",
              "| 分组 | 版本 | SC 中位数 | LTAS MAE dB | 过筛选 |", "|---|---|---:|---:|---:|"]
    variants = {"selected_before_eq": "选中链，校正前", "selected_after_eq": "选中链，校正后", "true_rig_true_gain_before_eq": "已知真链真增益，校正前", "true_rig_true_gain_fitted_eq": "已知真链真增益，拟合EQ后"}
    for key in GROUPS:
        for variant, label in variants.items():
            q = quality[key][variant]
            lines.append(f"| {key} | {label} | {q['median_sc']:.4f} | {q['median_ltas_db']:.3f} | {q['within_screening_threshold']}/16 |")
    lines += ["", "已知真链真增益的拟合 EQ 参照也使用各例相同的目标/搜索演奏，在跨 DI 时仍面对跨演奏拟合的混杂；它不是把已知 E1 直接倒置。"
              "因此链身份失败和 C 上功能不接近需分别判断；真链参照若经拟合也恶化，说明误差不能全部归咎于链检索。每例前后差与分布保留在 analysis.json。", "",
              "## 固定增益特征诊断", "", "使用缓存中全部12条链 × A/B × 0/+6 dB共48个指纹。固定增益比较同链换DI（每档24个有向配对）和同DI换链（每档264个有向配对）；两方向都计入，避免把按目标范数归一化的MFCC距离误当作对称距离。下表为分项已加权总分的均值（含0.25 EQ惩罚），并非增益搜索后的最小分。", "",
              "| 增益 dB | 同链换DI | 同DI换链 |", "|---|---:|---:|"]
    feature = analysis["features"]
    for g, f in feature["by_gain"].items():
        lines.append(f"| {g} | {f['same_rig_different_di']['component_means']['baseline_total']:.4f} | {f['different_rig_same_di']['component_means']['baseline_total']:.4f} |")
    lines += ["", "各评分分项的已加权均值如下；有向距离只作描述，不把大量共享指纹的配对当作独立样本。", "",
              "| 分项 | g0 同链换DI | g0 同DI换链 | g6 同链换DI | g6 同DI换链 |", "|---|---:|---:|---:|---:|"]
    for component in feature["by_gain"]["0"]["same_rig_different_di"]["component_means"]:
        if component in ("total", "baseline_total"):
            continue
        values = [feature["by_gain"][g][label]["component_means"][component] for g in ("0", "6")
                  for label in ("same_rig_different_di", "different_rig_same_di")]
        lines.append("| " + component + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    sat = feature["distortion_index"]["components"]
    lines += ["", "distortion_index 各输入分量达到上限clip的比例：" + "、".join(f"{k} {v['high_clipped']}/48 ({v['high_clipped_fraction']:.1%})" for k, v in sat.items()) + "。"
              "分项均值、低端clip计数及index分布见 analysis.json；局部分量饱和不等于整个index恒定，也不证明所有指纹失去辨别力。", "",
              "## 失败与预声明消融", "", "| 方法 | same/E0 Top1 | same/E1 Top1 | cross/E0 Top1 | cross/E1 Top1 |", "|---|---:|---:|---:|---:|"]
    for method, groups in summary.items():
        lines.append("| " + method + " | " + " | ".join(f"{groups[key]['top1']}/16" for key in GROUPS) + " |")
    causes = Counter(r["diagnostic"] for r in analysis["baseline_errors"])
    lines += ["", "基线错误的诊断计数：" + ("、".join(f"{k}={v}" for k, v in causes.items()) or "无") + "。"
              "全扫仍偏爱错链归入scoring；全扫找回但粗排漏掉归入coarse_pruning；全扫找回、粗排入围但最终错选归入local_gain_refinement。"
              "最后一类也可能是竞争链局部区间改变相对名次；分类只定位两阶段与全扫的差异，不证明单一因果。", "",
              "逐例winner/truth、真档/强制真链最佳档、相对两种真链档位的分项差及Top2间距保存在 analysis.json。下表列前8个基线错误（按冻结运行顺序），不是按严重度筛选。", "",
              "| case | 真链 | winner | 真增益 / winner / 强制真链最佳 dB | 诊断 |", "|---|---|---|---|---|"]
    for e in analysis["baseline_errors"][:8]:
        lines.append(f"| {e['case']} | {e['truth']} | {e['winner']} | {e['true_gain_db']} / {e['winner_gain_db']} / {e['forced_truth_best_gain_db']} | {e['diagnostic']} |")
    lines += ["", "no_drive_gap去掉方向性驱动项；no_hf_terms仅去掉显式hf_ratio/spec_crest_hf，保留的drive_gap仍含HF；"
              "no_dynamic_texture去掉dyn_range/sustain/flux/mfcc_texture/mfcc_delta；no_eq_penalty去掉0.25 EQ项；eq_only仅保留该EQ项。"
              "五项消融均重新排序候选和增益，仅供解释，未据此改权重或自动替换生产基线。", "",
              "## 与旧合成反事实的关系及范围", "", "旧2×2 MockAmp诊断独立保存在toy-ablation.json：", ""]
    for cell in analysis["toy"]["cells"]:
        lines.append(f"- {cell['candidate_di']} / E{int(cell['target_has_eq'])}：有效drive误差 {cell['absolute_error_db']:.4f} dB，分数 {cell['score']:.6f}。")
    lines += ["", "旧四格是在不同名义模型可通过输入增益互相等效的tanh族上观察演奏/EQ混杂，不混入真实NAM分母。真实库没有把不同NAM的名义增益直接相加。", "",
              "本次只覆盖一把吉他同次录音、四条目标链、两个真增益与一个固定后级高架EQ；C也来自同次录音。"
              "未测试开放库未知目标、其他乐手/拾音器/硬件输入增益、歌曲分离、Free Bird或插件/板端回放。任何格通过内部阈值都不能证明通用音色识别成功；"
              "最终听感仍需独立主观评价。未根据本次结果调参。", "", "![Stage 2 scientific summary](../artifacts/stage2/stage2-results.png)", "",
              "复算命令：`.venv/Scripts/python.exe scripts/stage2_analyze.py`。独立汇总断言、384行/64case/64配对/4一致性断言及缓存身份断言均通过后才输出本报告。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/stage2/real_fp64")
    args = parser.parse_args()
    source = ROOT / args.input
    data, cache, plan = (read(source / name) for name in ("results.json", "fingerprints.json", "locked-plan.json"))
    assert data["plan"] == plan
    assert len(plan["rigs"]) == 12
    summary, gain, quality = summarize(data)
    toy = read(ROOT / "artifacts/stage2/toy-ablation.json")
    analysis = {"source": str(source.relative_to(ROOT)), "source_results_sha256": hashlib.sha256((source / "results.json").read_bytes()).hexdigest(),
        "checks": {"retrieval_rows": 384, "cases": 64, "holdout_pairs": 64, "production_fidelity": 4, "independent_summaries_equal_primary": True},
        "summary": summary, "gain": gain, "holdout": quality,
        "gates": {"same_noeq_passed": summary["baseline"]["same_noeq"]["top1"] == 16,
                  **{key + "_passed": summary["baseline"][key]["internal_gate_passed"] for key in ("cross_noeq", "cross_eq")}},
        "features": feature_diagnostics(cache, plan), "baseline_errors": failure_cases(data),
        "toy": {"purpose": toy["purpose"], "cells": [{"candidate_di": c["candidate_di"], "target_has_eq": c["target_has_eq"],
                "absolute_error_db": c["calibrated"]["absolute_error_db"], "score": c["calibrated"]["total_score"]} for c in toy["cells"]]}}
    analysis["baseline_strata"] = {
        group: {field: {
            str(value): {"n": len(subset), **{metric: sum(r[metric] for r in subset) for metric in METRICS},
                         "forced_gain_error_db": distribution([r["forced_truth_nominal_gain_error_db"] for r in subset])}
            for value in sorted({r[field] for r in data["cases"] if r["method"] == "baseline"})
            if (subset := [r for r in data["cases"] if r["method"] == "baseline" and group_key(r) == group and r[field] == value])}
            for field in ("truth_model", "truth_gain_db", "reference_di")}
        for group in GROUPS}
    dest = ROOT / "artifacts/stage2"
    (dest / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot(data, summary, dest / "stage2-results.png")
    report(analysis, data, ROOT / "docs/stage2-findings.md")
    print(json.dumps({"checks": analysis["checks"], "gates": analysis["gates"]}, indent=2))


if __name__ == "__main__":
    main()
