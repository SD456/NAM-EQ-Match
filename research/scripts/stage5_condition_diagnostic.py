"""Post-result gate diagnosis only: no parameter changes or retrieval ranking."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from scipy.signal import lfilter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import stage5_benchmark as replay
from scripts.conditioned_scoring import audio_conditions


def diagnose(audio):
    """Observe the frozen helper's actual pYIN call, without replacing its result."""
    import librosa

    original = librosa.pyin
    observed = {}

    def observe(y, **kwargs):
        expected = dict(sr=16000, fmin=82.4, fmax=1319.0, frame_length=1024,
                        hop_length=256, resolution=0.5, center=False)
        assert kwargs == expected, "diagnostic parameters must equal the frozen helper"
        f0, flag, probability = original(y, **kwargs)
        rms = librosa.feature.rms(y=y, frame_length=1024,
                                  hop_length=256, center=False)[0]
        observed.update(f0=f0, flag=flag, probability=probability, rms=rms)
        return f0, flag, probability

    with patch.object(librosa, "pyin", observe):
        conditions = audio_conditions(audio, replay.SR)
    f0, flag, probability, rms = (observed[k] for k in ("f0", "flag", "probability", "rms"))
    times = conditions["times"]
    assert all(x.shape == times.shape for x in (f0, flag, probability, rms))
    assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
    finite = np.isfinite(f0)
    confident = probability >= 0.8
    rms_pass = rms > np.percentile(rms, 95) * 0.01
    edge_pass = (times >= 0.032) & (times < 8 - 0.032)
    final = flag & finite & confident & rms_pass & edge_pass
    np.testing.assert_array_equal(final, conditions["voiced"])
    assert np.array_equal(finite, flag), "default unvoiced pYIN F0 must be NaN"
    midi = np.rint(librosa.hz_to_midi(f0[flag])).astype(int)
    values, counts = np.unique(midi, return_counts=True)
    return {
        "frame_count": len(times),
        "raw_voiced_flag_frames": int(flag.sum()),
        "raw_finite_f0_frames": int(finite.sum()),
        "voiced_probability": dict(zip(("min", "median", "p95", "max"),
                                         map(float, np.percentile(probability, (0, 50, 95, 100))))),
        "probability_at_least_0_8_frames": int(confident.sum()),
        "rms_pass_frames": int(rms_pass.sum()),
        "edge_pass_frames": int(edge_pass.sum()),
        "raw_flag_and_probability_frames": int((flag & confident).sum()),
        "raw_flag_and_rms_and_edge_frames": int((flag & rms_pass & edge_pass).sum()),
        "final_intersection_frames": int(final.sum()),
        "raw_flag_midi_histogram": {str(k): int(n) for k, n in zip(values, counts)},
        "raw_flag_f0_median_hz": float(np.median(f0[flag])) if flag.any() else None,
    }


def main():
    import librosa
    import librosa.core.pitch as pitch

    output = ROOT / "artifacts/stage5/conditioning-diagnostic.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite completed diagnostic")
    cfg = replay.read_json(ROOT / "experiments/stage5.json")
    folder = ROOT / cfg["output"]
    locked = replay.read_json(folder / "locked-plan.json")
    completed = replay.read_json(folder / "results.json")
    identity = replay.sha(folder / "locked-plan.json")
    torch.set_num_threads(4)
    assert replay.plan(cfg) == locked
    assert completed["lock_sha256"] == identity
    inputs = {
        item["id"]: replay.load_audio(str(ROOT / item["path"]), replay.SR)[0].astype(np.float64)
        * replay.db_to_lin(cfg["calibration_db"])
        for item in locked["di"]["segments"]
        if item["id"] in cfg["probes"] + cfg["test_references"]
    }
    dry = {}
    for probe in cfg["probes"]:
        dry[probe] = diagnose(inputs[probe])
        assert dry[probe]["final_intersection_frames"] == completed["dry_probe_conditions"][probe]["voiced_frames"]
        print(f"diagnostic dry {probe}: {dry[probe]['final_intersection_frames']} final frames", flush=True)

    models, irs = ({x["id"]: x for x in locked["library"][key]} for key in ("models", "irs"))
    eq = cfg["post_eq"]
    b, a = replay.biquad_coeffs(eq["kind"], replay.SR, eq["frequency_hz"], eq["q"], eq["gain_db"])
    targets = []
    for index, row in enumerate(completed["retrieval"]):
        rig = locked["rigs"][row["truth"]]
        model = models[rig["model_id"]]
        nam = replay.NamCapture(
            str(ROOT / model["path"]), device="cuda", rig_type=model["rig_type"],
            cab_ir_path=str(ROOT / irs[rig["cab_id"]]["path"]) if rig["cab_id"] else None,
        )
        wet = nam.process(inputs[row["reference"]] * replay.db_to_lin(row["truth_gain_db"]))
        del nam
        if row["eq_on"]:
            wet = lfilter(b, a, np.pad(wet, (0, 256))).astype(np.float32)
        diagnostics = diagnose(wet)
        assert diagnostics["final_intersection_frames"] == row["query"]["voiced_frames"]
        targets.append({**{k: row[k] for k in ("case", "reference", "truth", "truth_gain_db", "eq_on")},
                        **diagnostics})
        print(f"diagnostic target {index + 1}/32: {diagnostics['final_intersection_frames']} final frames", flush=True)
    assert len(dry) == 3 and len(targets) == 32
    assert replay.plan(cfg) == locked and replay.sha(folder / "locked-plan.json") == identity
    source = Path(inspect.getsourcefile(librosa.pyin))
    all_rows = list(dry.values()) + targets
    summary = {
        "recordings": len(all_rows),
        "all_final_counts_equal_completed_development": True,
        "recordings_with_raw_voiced_flag": sum(r["raw_voiced_flag_frames"] > 0 for r in all_rows),
        "recordings_with_probability_at_least_0_8": sum(r["probability_at_least_0_8_frames"] > 0 for r in all_rows),
        "recordings_with_final_voiced_frames": sum(r["final_intersection_frames"] > 0 for r in all_rows),
    }
    replay.write_json(output, {
        "status": "post_result_fixed_parameter_diagnosis_not_new_validation",
        "lock_sha256": identity,
        "completed_results_sha256": replay.sha(folder / "results.json"),
        "diagnostic_script_sha256": replay.sha(Path(__file__)),
        "analysis_parameters": cfg["analysis"],
        "implementation": {
            "librosa_version": librosa.__version__, "pitch_source_path": str(source),
            "pitch_source_sha256": replay.sha(source),
            "pyin_signature": str(inspect.signature(librosa.pyin)),
            "pyin_helper_source_sha256": __import__("hashlib").sha256(
                inspect.getsource(pitch.__pyin_helper).encode()).hexdigest(),
            "inspection": [
                "voiced_prob is clipped summed voiced observation probability from YIN candidates",
                "voiced_flag is the separately Viterbi-decoded state sequence",
                "default fill_na=np.nan makes unvoiced decoded F0 non-finite",
                "installed transition_min_prob default is 0.0001; unchanged in frozen and diagnostic calls",
                "diagnostic observes the exact frozen helper call and returns pYIN outputs unchanged",
            ],
        },
        "summary": summary, "dry_probes": dry, "wet_queries": targets,
        "limits": ["No threshold, frame length or pitch algorithm changed; no parameter sweep",
                   "No new retrieval ranking, candidate score, wave or MIDI array persisted",
                   "Raw decoded pitch is not ground truth and cannot establish monophony",
                   "Synthetic success alone cannot distinguish all library issues from real-data inapplicability"],
    })
    print(summary, flush=True)


if __name__ == "__main__":
    main()
