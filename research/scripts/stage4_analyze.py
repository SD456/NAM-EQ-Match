"""Audit the frozen stage-4 JSON and probe spectra on CPU; never render audio."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def near_set(curve, gains):
    flat = bool(np.ptp(curve) < 1e-9)
    chosen = gains if flat else gains[curve <= curve.min() + .05 * np.ptp(curve)]
    intervals = []
    for g in chosen:
        if intervals and g == intervals[-1][1] + 1:
            intervals[-1][1] = float(g)
        else:
            intervals.append([float(g), float(g)])
    return dict(gains=chosen.tolist(), intervals=intervals, count=len(chosen),
                span_db=float(chosen[-1] - chosen[0]), flat=flat)


def compatible(effective, gains, bounds):
    return [float(g) for g in gains if any(bounds[0] <= e-g <= bounds[1] for e in effective)]


def quality(value, bypass, gate):
    sc, lt = value["mrstft_sc"], value["ltas_mae_db"]
    assert np.isfinite([sc, lt]).all() and min(sc, lt) >= 0
    ds, dl = sc-bypass["mrstft_sc"], lt-bypass["ltas_mae_db"]
    passed = sc <= gate["sc"] and lt <= gate["ltas_db"]
    passed_before = bypass["mrstft_sc"] <= gate["sc"] and bypass["ltas_mae_db"] <= gate["ltas_db"]
    expected = dict(delta_sc=ds, delta_ltas_db=dl, passed=passed,
                    harmful=ds > gate["harm_sc_delta"] or dl > gate["harm_ltas_delta_db"],
                    lost_pass=passed_before and not passed,
                    sc_improved=ds < -gate["harm_sc_delta"],
                    ltas_improved=dl < -gate["harm_ltas_delta_db"])
    for key, v in expected.items():
        if isinstance(v, bool):
            assert value[key] == v, key
        else:
            close(value[key], v)
    return expected


def heldouts(values, gate):
    assert set(values) == {"H0", "H1"}
    for variants in values.values():
        for v in variants.values():
            quality(v, variants["bypass"], gate)


def verify(result, cache):
    plan = result["plan"]
    cfg, grid = plan["config"], np.asarray(result["grid"], float)
    gains = np.asarray(cfg["search_gains_db"], float)
    band = (grid >= 80) & (grid <= 8000)
    assert len(plan["rigs"]) == 12 and len(plan["teachers"]) == 4 and len(gains) == 25
    assert plan["calibration_rigs"] == [i for i in range(12) if i not in plan["teachers"]]
    assert cache["identity"] == plan["source_hashes"]
    expected_keys = {f"{i}|{p}|{g}" for i in range(12) for p in cfg["probes"] for g in cfg["search_gains_db"]}
    assert len(expected_keys) == 900 and set(cache["items"]) == expected_keys
    spectra = np.empty((12, 25, 3, len(grid)))
    for i in range(12):
        for gi, g in enumerate(cfg["search_gains_db"]):
            for pi, p in enumerate(cfg["probes"]):
                item = cache["items"][f"{i}|{p}|{g}"]
                close(item["grid"], grid)
                spectra[i, gi, pi] = item["spectrum"]
    assert np.isfinite(spectra).all()
    null = result["null_rows"]
    assert len(null) == 32
    assert {(r["rig"], r["gain"], r["clip"]) for r in null} == {
        (i, g, c) for i in plan["calibration_rigs"] for g in cfg["null_calibration"]["gains_db"]
        for c in cfg["calibration_references"]}
    curves = np.asarray([r["curve"] for r in null], float)
    assert curves.shape == (32, len(grid)) and np.isfinite(curves).all()
    q = float(np.max(np.abs(curves[:, band])))
    qi, qj = np.unravel_index(np.where(band, np.abs(curves), -1).argmax(), curves.shape)
    close(q, result["q"])
    expected_cases = {(i, g, e) for i in plan["teachers"] for g in cfg["truth_gains_db"] for e in (False, True)}
    oracle = result["oracle"]
    assert len(oracle) == 16
    assert {(r["truth"], r["truth_gain_db"], r["eq_on"]) for r in oracle} == expected_cases
    for row in oracle:
        assert row["case"] == f"{row['truth']}_{row['truth_gain_db']}_E{int(row['eq_on'])}"
        raw = np.asarray(row["raw_test_curves"], float)
        assert raw.shape == (2, len(grid)) and np.isfinite(raw).all()
        # Independently express the frozen two-reference rule, before clipping.
        mask = band & (np.sign(raw[0]) == np.sign(raw[1])) & (np.min(np.abs(raw), axis=0) > max(q, 1)) & (np.ptp(raw, axis=0) <= 3)
        decision = row["decision"]
        assert decision["apply"] == bool(mask.any())
        assert decision["supported_bins"] == int(mask.sum())
        close(decision["q"], q)
        assert set(row["irs"]) == {"bypass", "stage3_guard", "null_gate"}
        assert row["irs"]["bypass"] == [1.0]
        if not mask.any():
            assert row["irs"]["null_gate"] == [1.0]
            close(decision["actual_db"], np.zeros_like(grid))
        # FFT of stored float32 coefficients; no audio or new EQ design.
        for name, ir in row["irs"].items():
            h = np.asarray(ir, np.float32)
            assert h.ndim == 1 and np.isfinite(h).all()
            n = 1 << int(np.ceil(np.log2(max(65536, 64*len(h)))))
            response = 20*np.log10(np.maximum(np.abs(np.fft.rfft(h, n)), 1e-15))
            assert response.min() >= -4.5-1e-5 and response.max() <= 4.5+1e-5
            if name == "null_gate":
                close(decision["actual_db"], np.interp(grid, np.fft.rfftfreq(n, 1/48000), response))
        heldouts(row["holdouts"], cfg["gate"])
        for variants in row["holdouts"].values():
            assert set(variants) == {"bypass", "stage3_guard", "null_gate"}
            if not row["eq_on"]:
                assert variants["bypass"]["mrstft_sc"] < 1e-7 and variants["bypass"]["ltas_mae_db"] < 1e-7
            if not decision["apply"]:
                close([variants["null_gate"][k] for k in ("mrstft_sc", "ltas_mae_db")],
                      [variants["bypass"][k] for k in ("mrstft_sc", "ltas_mae_db")])
    rows = result["retrieval"]
    assert len(rows) == 32
    assert {(r["truth"], r["truth_gain_db"], r["eq_on"], r["reference"]) for r in rows} == {
        (*c, ref) for c in expected_cases for ref in cfg["test_references"]}
    for row in rows:
        target = np.asarray(row["reference_spectrum"], float)
        assert target.shape == grid.shape and np.isfinite(target).all()
        delta = spectra[..., band] - target[band]
        delta -= np.median(delta, axis=-1, keepdims=True)
        scores = np.median(np.mean(np.abs(delta), axis=-1), axis=-1)
        close(row["all_gain_scores"], scores)
        best = scores.argmin(axis=1)
        ranking = sorted(range(12), key=lambda i: scores[i, best[i]])
        truth, winner = row["truth"], ranking[0]
        assert row["winner"] == winner and row["gain_db"] == gains[best[winner]]
        assert row["forced_truth_gain_db"] == gains[best[truth]]
        close(row["nominal_forced_truth_error_db"], abs(gains[best[truth]]-row["truth_gain_db"]))
        assert row["top1"] == (truth == winner) and row["top3"] == (truth in ranking[:3])
        assert len(row["ranking"]) == 12
        for rank, i in zip(row["ranking"], ranking):
            assert rank["rig"] == i and rank["gain"] == gains[best[i]]
            close(rank["score"], scores[i, best[i]])
        near = near_set(scores[truth], gains)
        assert row["near_effective_gains"] == near
        assert row["compatible_original_gains_under_level_bounds"] == compatible(near["gains"], gains, cfg["gain_witness"]["level_bounds_db"])
        heldouts(row["holdout_bypass"], cfg["gate"])
    groups = oracle_summary(result)
    a, b, limits = groups["E0"]["null_gate"], groups["E1"]["null_gate"], cfg["gate"]
    gate = dict(e0_n=8, e1_n=8, false_apply=a["applied"], harmed_cases=a["any_harm"],
                e1_applied=b["applied"], e1_beneficial_cases=b["both_improved_without_harm"])
    gate["passed"] = bool(gate["false_apply"] <= limits["e0_max_false_apply"] and gate["harmed_cases"] <= limits["e0_max_harmed_cases"] and gate["e1_applied"] >= limits["e1_min_apply"] and gate["e1_beneficial_cases"] >= limits["e1_min_beneficial_cases"])
    assert result["gate"] == gate
    if not gate["passed"]:
        assert result["selected_chain_eq"] == []
        assert result["selected_chain_eq_status"] == "not_validated_oracle_gate_failed"
    else:
        assert len(result["selected_chain_eq"]) == 32 and result["selected_chain_eq_status"] == "evaluated"
        for r in result["selected_chain_eq"]:
            heldouts(r["holdouts"], cfg["gate"])
    return dict(raw_null_curves=32, raw_oracle_decisions=16, independent_score_matrices=32,
                independent_score_values=9600, probe_only_cache_entries=900, gate=gate,
                q_recomputed_db=q,
                q_maximum_source=dict(rig=null[qi]["rig"], gain=null[qi]["gain"], clip=null[qi]["clip"],
                                      frequency_hz=float(grid[qj]), raw_correction_db=float(curves[qi, qj])),
                null_band_maxima=[dict(rig=r["rig"], gain=r["gain"], clip=r["clip"],
                                      max_abs_db=float(np.max(np.abs(c[band])))) for r, c in zip(null, curves)],
                paired_quality_derivations=True, stored_fir_fullband_cap=True)


def oracle_summary(result):
    out = {}
    for eq in (False, True):
        rows = [r for r in result["oracle"] if r["eq_on"] == eq]
        variants = {}
        for name in ("bypass", "stage3_guard", "null_gate"):
            pairs = [[r["holdouts"][h][name] for h in ("H0", "H1")] for r in rows]
            applied = [bool(r["decision"]["apply"]) if name == "null_gate" else r["irs"][name] != [1.0] for r in rows]
            variants[name] = dict(n=8, applied=sum(applied), rejected=8-sum(applied),
                both_pass=sum(all(h["passed"] for h in pair) for pair in pairs),
                any_harm=sum(any(h["harmful"] for h in pair) for pair in pairs),
                both_improved_without_harm=sum(all(not h["harmful"] and (h["sc_improved"] or h["ltas_improved"]) for h in pair) for pair in pairs),
                sc_median=float(np.median([h["mrstft_sc"] for pair in pairs for h in pair])),
                ltas_median_db=float(np.median([h["ltas_mae_db"] for pair in pairs for h in pair])),
                cases=[dict(case=r["case"], applied=app, holdouts=r["holdouts"]) for r, app in zip(rows, applied)])
        out[f"E{int(eq)}"] = variants
    return out


def summarize(result, witness):
    groups = {}
    for eq in (False, True):
        rows = [r for r in result["retrieval"] if r["eq_on"] == eq]
        groups[f"E{int(eq)}"] = dict(n=16, top1=sum(r["top1"] for r in rows), top3=sum(r["top3"] for r in rows),
            nominal_gain_error_median_db=float(np.median([r["nominal_forced_truth_error_db"] for r in rows])),
            forced_gain_estimates_db=[r["forced_truth_gain_db"] for r in rows],
            near_coverage=sum(r["truth_gain_db"] in r["near_effective_gains"]["gains"] for r in rows),
            near_width_median_db=float(np.median([r["near_effective_gains"]["span_db"] for r in rows])),
            compatible_coverage=sum(r["truth_gain_db"] in r["compatible_original_gains_under_level_bounds"] for r in rows),
            compatible_width_median_db=float(np.median([np.ptp(r["compatible_original_gains_under_level_bounds"]) for r in rows])),
            both_holdout_bypass_pass=sum(all(h["bypass"]["passed"] for h in r["holdout_bypass"].values()) for r in rows),
            rows=rows)
        for name, key in (("forced_true_chain_single_effective_gain", "forced_truth_gain_db"),
                          ("selected_chain_single_effective_gain", "gain_db")):
            sets = [compatible([r[key]], result["plan"]["config"]["search_gains_db"], [-6, 6]) for r in rows]
            groups[f"E{int(eq)}"][name] = dict(
                n=16, coverage=sum(r["truth_gain_db"] in values for r, values in zip(rows, sets)),
                count_median=float(np.median([len(v) for v in sets])),
                span_median_db=float(np.median([np.ptp(v) for v in sets])),
                rows=[dict(case=r["case"], reference=r["reference"], effective_gain_db=r[key],
                           compatible_original_gains=v, count=len(v), span_db=float(np.ptp(v)),
                           nominal_truth_covered=r["truth_gain_db"] in v) for r, v in zip(rows, sets)])
    levels = []
    for s in result["plan"]["di"]["segments"]:
        db = float(20*np.log10(s["rms"]))
        levels.append(dict(id=s["id"], role=s["role"], raw_rms_dbfs=db,
                           calibrated_rms_dbfs=db+result["plan"]["config"]["calibration_db"],
                           peak=s["peak"], crest_db=float(20*np.log10(s["peak"]/s["rms"]))))
    drift = []
    for r in result["retrieval"]:
        if r["reference"] == "T0":
            other = next(x for x in result["retrieval"] if x["case"] == r["case"] and x["reference"] == "T1")
            drift.append(dict(case=r["case"], t1_minus_t0_db=other["forced_truth_gain_db"]-r["forced_truth_gain_db"]))
    return dict(retrieval=groups, oracle=oracle_summary(result), input_levels=levels, reference_gain_drift=drift,
                gain_witness=dict(pairs=16, observations=48, max_abs_wave_error=witness["max_absolute_wave_error"],
                    bitwise_equal=sum(p["bitwise_equal"] for p in witness["pairs"]),
                    original_gain_coverage=sum(o["device_gain"] in o["compatible_device_gains"] for o in witness["observations"]),
                    compatible_width_median_db=float(np.median([np.ptp(o["compatible_device_gains"]) for o in witness["observations"]]))))


def verify_files(result, witness):
    plan = result["plan"]
    hashes = {name: {p: sha(ROOT/p) == digest for p, digest in values.items()} for name, values in
              [("source_hashes", plan["source_hashes"]), ("protected_baselines", plan["protected_baselines"]),
               ("gain_witness_sources", witness["source_hashes"]) ]}
    assert all(all(v.values()) for v in hashes.values())
    assert result["baseline_preserved"] and witness["baseline_preserved"]
    assert plan["config"] == read(ROOT/"experiments/stage4.json")
    meta = plan["di"]
    roles = {"probe": {"P0", "P1", "P2"}, "calibration_reference": {"C0", "C1"}, "test_reference": {"T0", "T1"}, "holdout": {"H0", "H1"}}
    assert len(meta["segments"]) == 9 and len(meta["recordings"]) == 4
    recording_ids = set()
    for role, ids in roles.items():
        group = [s for s in meta["segments"] if s["role"] == role]
        assert {s["id"] for s in group} == ids
        recs = {s["recording_id"] for s in group}
        assert len(recs) == 1 and not recs & recording_ids
        recording_ids |= recs
        for i, s in enumerate(group):
            assert s["frames"] == s["end_frame"]-s["start_frame"] == 8*48000
            assert s["sample_rate"] == 48000 and s["channels"] == 1
            for earlier in group[:i]:
                assert max(s["start_frame"], earlier["start_frame"]) >= min(s["end_frame"], earlier["end_frame"])
    assert "directinput_01" not in recording_ids
    library = read(ROOT/"assets/stage2/models-ir-provenance.json")
    assets = library["models"]+library["irs"]+meta["segments"]+meta["recordings"]
    hashes["asset_hashes"] = {s["path"]: sha(ROOT/s["path"]) == s["sha256"] for s in assets}
    assert all(hashes["asset_hashes"].values())
    assert len(witness["pairs"]) == 16 and len(witness["observations"]) == 48
    assert all(p["passed"] and p["max_abs_error"] == 0 and p["bitwise_equal"] for p in witness["pairs"])
    assert witness["max_absolute_wave_error"] == 0
    gains = plan["config"]["search_gains_db"]
    for o in witness["observations"]:
        assert o["effective_gain"] == o["device_gain"]+o["digital_level"]
        assert o["known_level_oracle_gain"] == o["device_gain"]
        assert o["compatible_device_gains"] == compatible([o["effective_gain"]], gains, [-6, 6])
    return hashes


def draw(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for eq, color in [("E0", "#3274a1"), ("E1", "#df8f44")]:
        vals = summary["retrieval"][eq]["forced_gain_estimates_db"]
        axes[0].hist(vals, bins=np.arange(-12.5, 13.5, 1), alpha=.55, color=color, label=eq+" (16 references)")
    axes[0].set(title="Nominal gain estimates: forced true chain", xlabel="Gain (dB)", ylabel="Reference count")
    axes[0].text(.02, .97, "Equal total input: 16/16 bitwise equal\nMax wave error = 0; gain/level not separable", transform=axes[0].transAxes, va="top", fontsize=8)
    names = ["bypass", "stage3_guard", "null_gate"]
    x = np.arange(3)
    for eq, offset, color in [("E0", -.18, "#3274a1"), ("E1", .18, "#df8f44")]:
        vals = summary["oracle"][eq]
        axes[1].bar(x+offset, [vals[n]["applied"] for n in names], .34, color=color, label=eq)
        axes[2].bar(x+offset, [vals[n]["any_harm"] for n in names], .34, color=color, label=eq)
    axes[1].set(title="EQ application: false E0 / detected E1", ylabel="Applied tasks / 8", ylim=(0, 9))
    axes[2].set(title="Either holdout harmed (oracle chain + gain)", ylabel="Tasks / 8 (H0/H1 paired)", ylim=(0, 9))
    for ax in axes[1:]:
        ax.set_xticks(x, ["Always bypass", "Stage-3 guard", "Null gate"])
    for ax in axes:
        ax.legend(fontsize=8, loc="lower right" if ax == axes[0] else "upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Four recording files, shared player/guitar; descriptive frozen study")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def report(result, summary, checks, digest):
    gate = checks["gate"]
    lines = ["# 第四阶段结果：增益不可辨识与跨录音乐句的 EQ 判定", "",
        f"冻结研究门槛{'通过' if gate['passed'] else '未通过'}。经验零假设阈值 Q = {result['q']:.6f} dB，来自非教师 8 条候选、2 档增益、C0/C1 的 32 条不限幅校准曲线。E0 错加 EQ {gate['false_apply']}/8、任一留出受损 {gate['harmed_cases']}/8；E1 启用 {gate['e1_applied']}/8、两段均改善且无害 {gate['e1_beneficial_cases']}/8。冻结条件为依次 ≤1、≤1、≥6、≥6；没有根据结果重选参数。", "",
        f"所选链 EQ 状态：`{result['selected_chain_eq_status']}`。" + ("门槛失败，部署组未验证；本批真实 E1 检出率为 0/8，全部拒绝。零误加来自全旁路，不能据此认定 EQ 问题解决。" if not gate['passed'] else "部署组已按冻结计划评价，明细见原始结果。"), "",
        "## 增益与原始输入电平", "",
        "48 次实际两处数字缩放渲染组成 16 对等总输入反例：16/16 逐样本完全相等，最大波形误差 0。在固定 NAM 之前的两个数字输入 trim 位置，湿声只能约束有效增益 g+l，原始 NAM 输入增益与数字输入电平结构上不可辨识；这不证明所有实体音箱 Gain 旋钮都等价于该数字缩放。真实拨弦强弱还会改变频谱、起音与包络，并非纯电平缩放；没有力度标签，本轮未验证真实拨弦力度分离。", "",
        f"反事实在数字电平 ±6 dB 假设下的原始档集合覆盖 48/48，集合跨度中位数 {summary['gain_witness']['compatible_width_median_db']:g} dB；已知数字电平的减法 oracle 只验证代数。以下跨录音结果对合成器名义档评分，集合没有概率覆盖保证。", "",
        "|条件|Top1|Top3|强制真链误差中位数 dB|近优覆盖|近优跨度中位数 dB|近优并集兼容覆盖|并集跨度中位数 dB|两 H 直通过筛|",
        "|---|---|---|---|---|---|---|---|---|"]
    for eq, v in summary["retrieval"].items():
        lines.append(f"|{eq}|{v['top1']}/16|{v['top3']}/16|{v['nominal_gain_error_median_db']:g}|{v['near_coverage']}/16|{v['near_width_median_db']:g}|{v['compatible_coverage']}/16|{v['compatible_width_median_db']:g}|{v['both_holdout_bypass_pass']}/16|")
    lines += ["", "近优集合沿用分数跨度的 5%，跨度 <10⁻⁹ 时全保留；上表兼容集合为每个近优有效档对应 ±6 dB 原始档集合的并集，再与 −12…+12 dB 格点相交，可能比单点 e±6 更宽。两种集合均非置信区间。32 行完整 12×25 分数均从 900 项 P-only 谱缓存及保存的目标谱独立重算，逐探针居中频带 MAE 后取三探针中位数。", "",
              "单点估计的兼容集合另列如下：只用一个估计有效档 e，计算 [e−6,e+6] 与搜索格点的交集。优先解释强制正确链；winner 若选中错误 NAM，其增益档不能与真 NAM 跨模型比较物理参数，该列覆盖只作合成器名义诊断。", "",
              "|条件|单点来源|真名义档覆盖|集合档数中位数|跨度中位数 dB|", "|---|---|---|---|---|"]
    for eq, v in summary["retrieval"].items():
        for name, label in (("forced_true_chain_single_effective_gain", "强制正确链"), ("selected_chain_single_effective_gain", "检索 winner（仅名义诊断）")):
            s = v[name]
            lines.append(f"|{eq}|{label}|{s['coverage']}/16|{s['count_median']:g}|{s['span_median_db']:g}|")
    null = np.asarray([r["curve"] for r in result["null_rows"]])
    grid = np.asarray(result["grid"])
    absolute = np.where((grid >= 80) & (grid <= 8000), np.abs(null), -1)
    ri, fi = np.unravel_index(absolute.argmax(), absolute.shape)
    source = result["null_rows"][ri]
    lines += ["", f"Q 的最大值来自 rig {source['rig']}、g={source['gain']} dB、{source['clip']} 在 {grid[fi]:.3f} Hz 的原始校准修正 {null[ri, fi]:.6f} dB。这是跨所有校准曲线的最坏单频点幅度，不代表整个频带或全部曲线存在 22.8 dB 偏差。", "",
              "## 真链与真增益的 EQ 对照", "", "|条件|方法|启用|拒绝|两 H 均过筛|任一 H 受损|两 H 各有改善且无害|SC 中位数|LTAS 中位数 dB|", "|---|---|---|---|---|---|---|---|---|"]
    for eq, variants in summary["oracle"].items():
        for name, v in variants.items():
            lines.append(f"|{eq}|{name}|{v['applied']}/8|{v['rejected']}/8|{v['both_pass']}/8|{v['any_harm']}/8|{v['both_improved_without_harm']}/8|{v['sc_median']:.5f}|{v['ltas_median_db']:.4f}|")
    lines += ["", "每条件 8 个两参考任务，H0/H1 是同一任务的两段留出，不能算 16 个独立实验。过筛为 SC ≤0.20 且居中 LTAS ≤1.5 dB；受损为同例 ΔSC >0.02 或 ΔLTAS >0.25 dB；实质改善要求至少一项严格低于对应负阈值，且另一项无受损。始终旁路可无害，却不能改善 E1，故不算成功。两 H 过筛与改善无害是不同统计。", "",
        "旧 guard 沿用 T0 单参考；新规则同时要求 T0/T1 原始曲线同号、最小绝对值 >max(Q,1 dB)、极差 ≤3 dB，支持后才平滑并限制 FIR 响应。Q 不含 T/H；本分析从保存的原始曲线复核判决，未拿旧限幅曲线代替。", "",
        "## 数字电平与参考漂移", "", "|片段|角色|原始 RMS dBFS|统一校准后 RMS dBFS|峰均比 dB|", "|---|---|---|---|---|"]
    for v in summary["input_levels"]:
        lines.append(f"|{v['id']}|{v['role']}|{v['raw_rms_dbfs']:.3f}|{v['calibrated_rms_dbfs']:.3f}|{v['crest_db']:.3f}|")
    levels = summary["input_levels"]
    pmean = float(np.mean([x["raw_rms_dbfs"] for x in levels if x["role"] == "probe"]))
    for role in ("calibration_reference", "test_reference", "holdout"):
        delta = float(np.mean([x["raw_rms_dbfs"] for x in levels if x["role"] == role]))-pmean
        lines.append(f"\n{role} 平均片段 RMS dBFS 相对 P 组为 {delta:+.3f} dB。")
    lines += [f"\nT0→T1 强制真链估计档的绝对变化中位数为 {np.median([abs(d['t1_minus_t0_db']) for d in summary['reference_gain_drift']]):g} dB（16 对相关条件，逐对值存 analysis.json）。录音的电平、频谱及音符内容差异可解释候选增益漂移的可能来源，单凭 RMS 不能识别原因或换算真实力度。", "",
        "## 复核与适用范围", "",
        f"原始独立结果 SHA-256：`{digest}`。分析代码 SHA-256、缓存及反事实结果 SHA-256、逐项源文件/全部素材/受保护基线哈希、32 条 null 的 Q、16 次 raw 判决、9600 个检索分数、排名/近优集合与停止门槛均保存在 analysis.json。破坏性自检拒绝项：`{', '.join(checks.get('negative_corruption_checks', []))}`。", "",
        "本后处理只读 JSON 与已存探针谱，未重新渲染湿声；留出 SC/LTAS 只复核保存值的派生阈值和旁路配对一致性，保存的 FIR 另用 FFT 复核全频 ±4.5 dB 限幅。逐例检索和 oracle/H0/H1 指标保存在 analysis.json 及原始 results.json。", "",
        "四个新文件分别分配探针、校准、目标与留出，支持跨文件乐句检查；同一乐手、Sire 吉他与接口，不能据此证明跨采集 session、乐手、设备或模型家族泛化。教师与非教师候选可能共享 NAM/箱体，目标和留出也被重复用于多个相关任务。源端 ADC/采集增益及真实力度标签未知；RMS 仅描述数字电平，没有实现原始力度/增益分离。", "",
        "官方目录与严格 TLS 复核、ZIP CRC 证据见 assets/stage4/di/transport-verification.json；原始下载曾降级证书验证，CRC 不提供密码学完整性，官方全包 MD5 未验证。来源和方法限制均保留。生产控制器、阈值与阶段二/三结果保持原样，Free Bird 未开始。", "",
        "![Stage 4 results](../artifacts/stage4/stage4-results.png)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    path = ROOT/"artifacts/stage4/independent/results.json"
    result = read(path)
    cache_path = path.with_name("probe-spectra.json")
    cache = read(cache_path)
    witness_path = ROOT/"artifacts/stage4/gain_witness/results.json"
    witness = read(witness_path)
    checks = verify(result, cache)
    checks.update(verify_files(result, witness))
    if args.self_check:
        corruptions = ("q", "nullcurve", "rawcurve", "rawdecision", "rank", "gate", "score", "paired_delta")
        for name in corruptions:
            bad = copy.deepcopy(result)
            if name == "q":
                bad["q"] += 1
            elif name == "nullcurve":
                bad["null_rows"][0]["curve"] = [1000.0]*len(bad["grid"])
            elif name == "rawcurve":
                bad["oracle"][0]["raw_test_curves"] = [[1000.0]*len(bad["grid"])]*2
            elif name == "rawdecision":
                bad["oracle"][0]["decision"]["apply"] = not bad["oracle"][0]["decision"]["apply"]
            elif name == "rank":
                bad["retrieval"][0]["ranking"][0]["score"] += 1
            elif name == "gate":
                bad["gate"]["passed"] = not bad["gate"]["passed"]
            elif name == "score":
                bad["retrieval"][0]["all_gain_scores"][0][0] += 1
            else:
                bad["oracle"][0]["holdouts"]["H0"]["null_gate"]["delta_sc"] += 1
            try:
                verify(bad, cache)
            except AssertionError:
                continue
            raise AssertionError(f"Corruption accepted: {name}")
        checks["negative_corruption_checks"] = list(corruptions)
    summary = summarize(result, witness)
    output = ROOT/"artifacts/stage4"
    analysis = dict(source_results_sha256=sha(path), analyzer_sha256=sha(__file__),
                    probe_cache_sha256=sha(cache_path), gain_witness_sha256=sha(witness_path), checks=checks, **summary)
    (output/"analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    draw(summary, output/"stage4-results.png")
    (ROOT/"docs/stage4-findings.md").write_text(report(result, summary, checks, sha(path)), encoding="utf-8")
    print(json.dumps(dict(verified=True, gate=checks["gate"], output="artifacts/stage4/analysis.json"), indent=2))


if __name__ == "__main__":
    main()
