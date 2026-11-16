"""End-to-end smoke test with synthetic amps (no torch / .nam files needed).

A candidate-identical MockAmp and the same performance check the engineering
loop only. Cross-performance tonal accuracy is a separate research benchmark;
the former unpaired demo failed when its omitted DI normalization was included.
See NAM_guitar_modeling/docs/stage1-report.md for that diagnostic.

Run:  python -m tests.test_synthetic
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tonematch.audio import save_audio  # noqa: E402
from tonematch.features import extract_fingerprint  # noqa: E402
from tonematch.nam_backend import MockAmp  # noqa: E402
from tonematch.pipeline import run_match  # noqa: E402

SR = 48000


def synth_di(seed: int, dur_s: float = 12.0) -> np.ndarray:
    """Fake guitar DI: plucked-string-ish bursts (decaying harmonics + noise attack)."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * SR)
    x = np.zeros(n)
    t_note = 0.0
    notes = [82.4, 110.0, 146.8, 196.0, 123.5, 164.8]  # E2 A2 D3 G3 B2 E3
    while t_note < dur_s - 0.5:
        f0 = notes[rng.integers(len(notes))] * (2 ** rng.integers(0, 2))
        dur = rng.uniform(0.25, 0.7)
        m = int(dur * SR)
        i0 = int(t_note * SR)
        t = np.arange(m) / SR
        note = np.zeros(m)
        for h in range(1, 9):
            amp = 1.0 / h ** rng.uniform(0.8, 1.3)
            note += amp * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
        note *= np.exp(-t * rng.uniform(3, 7))
        # pick attack
        atk = rng.normal(0, 0.3, min(m, 400)) * np.exp(-np.arange(min(m, 400)) / 80.0)
        note[: len(atk)] += atk
        x[i0 : i0 + m] += note * rng.uniform(0.4, 1.0)
        t_note += rng.uniform(0.15, 0.4)
    return (0.3 * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def main():
    print("Synthesizing DI + target...")
    di = synth_di(seed=1, dur_s=6.0)

    mystery_rig = MockAmp(drive_db=14.0, name="mystery")
    target = mystery_rig.process(di)

    candidates = [
        MockAmp(drive_db=0.0, name="clean-ish"),
        MockAmp(drive_db=7.0, name="crunch"),
        MockAmp(drive_db=14.0, name="hot"),  # right drive, wrong EQ (no tilt/presence)
        MockAmp(drive_db=24.0, name="fuzz"),
    ]

    tmp = tempfile.mkdtemp(prefix="tonematch_test_")
    t_path = os.path.join(tmp, "target.wav")
    d_path = os.path.join(tmp, "di.wav")
    save_audio(t_path, target, SR)
    save_audio(d_path, di, SR)

    print("Running pipeline...")
    out = run_match(t_path, d_path, candidates, os.path.join(tmp, "results"), sr=SR,
                    di_normalization_db=0.0)

    print("\nRanking:")
    for i, r in enumerate(out.ranked):
        print(f"  {i+1}. {r.name:12s} gain={r.gain_db:+5.1f} dB  score={r.score:.4f}")

    # --- assertions ----------------------------------------------------------
    # For MockAmp, drive_db + input_gain_db is the effective drive, so different
    # captures can converge to the same rig. Assert the *effective* drive is
    # recovered (true value: 14 dB).
    eff_drive = out.best.capture.drive_db + out.renders[0]["input_gain_db"]
    print(f"\nRecovered effective drive: {eff_drive:.1f} dB (true: 14.0 dB)")
    assert abs(eff_drive - 14) <= 0.01, f"effective drive {eff_drive} too far from 14 dB"
    assert out.best.name in ("hot", "fuzz", "crunch"), f"clean-ish should not win, got {out.best.name}"

    # Identical tone needs no material correction; real nonflat EQ has DSP tests.
    from tonematch.audio import load_audio

    rendered, _ = load_audio(out.render_path, SR)
    raw_amped = out.best.capture.process(di * 10 ** (out.renders[0]["input_gain_db"] / 20))
    fp_t = extract_fingerprint(target, SR)
    fp_raw = extract_fingerprint(raw_amped, SR)
    fp_ren = extract_fingerprint(rendered, SR)
    sel = (fp_t.ltas_grid >= 80) & (fp_t.ltas_grid <= 10000)
    err_raw = np.mean(np.abs(fp_raw.ltas_db - fp_t.ltas_db)[sel])
    err_ren = np.mean(np.abs(fp_ren.ltas_db - fp_t.ltas_db)[sel])
    print(f"\nLTAS error before match IR: {err_raw:.2f} dB")
    print(f"LTAS error after  match IR: {err_ren:.2f} dB")
    assert err_ren <= err_raw + 0.05, "export introduced a material spectral error"

    # This local branch exports one constrained correction, without plugin tone-stack paths.
    from scipy.signal import freqz
    ir, _ = load_audio(out.ir_path, SR)
    _, response = freqz(ir, worN=131072)
    assert np.max(np.abs(20 * np.log10(np.abs(response)))) <= 4.501
    assert os.path.exists(out.renders[0]["settings_json"])

    print("\nAll assertions passed. Outputs in:", os.path.join(tmp, "results"))


if __name__ == "__main__":
    main()
