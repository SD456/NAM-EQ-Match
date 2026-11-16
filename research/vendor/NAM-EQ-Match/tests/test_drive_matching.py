"""Regression test: a driven target must not be matched with a clean capture.

Reported bug: with a folder of mixed captures, clean amps were sometimes picked
for overdriven/high-gain targets. Two root causes were found in features.py:

  1. `_rel()` was unbounded, so a near-zero feature (e.g. spectral flatness of
     tonal material) could produce a distance of 9-10 and outvote every real
     distortion cue.
  2. `flatness_fizz` measured spectral flatness of the 2-8 kHz band including
     silent/quiet frames. A clean signal has almost no HF content there, so the
     band was just noise floor - which is maximally flat. Clean captures
     therefore scored *better* on "fizziness" than genuinely distorted ones.

Run:  python -m tests.test_drive_matching
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal import butter, lfilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tonematch.features import (  # noqa: E402
    distortion_index,
    eq_penalty,
    extract_fingerprint,
    nonlinear_distance,
)

SR = 48000


class _Amp:
    """Pre-emphasis -> asymmetric soft clip (drive) -> cab lowpass."""

    def __init__(self, drive_db: float, name: str, tilt: float = 0.0):
        self.drive_db = drive_db
        self.name = name
        self.path = name
        self.sample_rate = SR
        self.tilt = tilt

    def process(self, x):
        y = lfilter([1.0, -0.8], [1.0], x)
        g = 10 ** (self.drive_db / 20)
        y = np.tanh(g * y) + 0.12 * np.tanh(g * y) ** 2  # even harmonics too
        b, a = butter(4, 6000 / (SR / 2), "low")
        y = lfilter(b, a, y)
        if self.tilt:
            b2, a2 = butter(1, 800 / (SR / 2), "low")
            y = y + self.tilt * lfilter(b2, a2, y)
        return (y / (np.max(np.abs(y)) + 1e-9) * 0.5).astype(np.float32)


def _di(seed: int, dur_s: float = 8.0, sustain: bool = False):
    rng = np.random.default_rng(seed)
    n = int(dur_s * SR)
    x = np.zeros(n)
    t0 = 0.0
    while t0 < dur_s - 2.0:
        f0 = [82.4, 110.0, 146.8, 196.0, 246.9][rng.integers(5)] * (2 ** rng.integers(0, 2))
        length = rng.uniform(1.2, 2.0) if sustain else rng.uniform(0.2, 0.5)
        m = min(int(length * SR), n - int(t0 * SR))
        if m <= 0:
            break
        i0 = int(t0 * SR)
        t = np.arange(m) / SR
        note = sum(
            (1 / h**1.1) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6))
            for h in range(1, 15)
        )
        note *= np.exp(-t * (0.6 if sustain else rng.uniform(3, 7)))
        if not sustain:
            k = min(m, 300)
            note[:k] += rng.normal(0, 0.4, k) * np.exp(-np.arange(k) / 50)
        x[i0 : i0 + m] += note * rng.uniform(0.4, 1.0)
        t0 += length * 0.9 if sustain else rng.uniform(0.2, 0.5)
    return (0.3 * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def _rank(target, di, candidates):
    tfp = extract_fingerprint(target, SR)
    scored = []
    for amp in candidates:
        best = None
        for g in np.linspace(-12, 12, 7):
            fp = extract_fingerprint(amp.process(di * 10 ** (g / 20)), SR)
            s = nonlinear_distance(fp, tfp)["total"] + 0.25 * eq_penalty(fp, tfp)
            if best is None or s < best[0]:
                best = (s, g)
        scored.append((best[0], amp, best[1]))
    scored.sort(key=lambda r: r[0])
    return scored, tfp


def main():
    candidates = [_Amp(d, f"drive{d:+03d}") for d in (0, 6, 12, 18, 24, 30, 36)]
    di_spiky = _di(1)
    di_sustained = _di(3, sustain=True)

    print("Feature sanity: distortion index must rise with drive")
    prev = -1.0
    for d in (0, 12, 24, 36):
        fp = extract_fingerprint(_Amp(d, "x").process(di_spiky), SR)
        idx = distortion_index(fp)
        print(f"  drive {d:+3d} dB -> D-index {idx:.3f}  crest {fp.crest_db:5.2f}  hf {fp.hf_ratio:.3f}")
        assert idx >= prev - 0.02, "distortion index is not monotonic in drive"
        prev = idx

    print("\nDriven targets must not be matched with clean captures:")
    failures = []
    for label, target_drive, di in [
        ("high gain, matched playing", 36, di_spiky),
        ("high gain, DI played differently", 36, di_sustained),
        ("hi-gain w/ dark cab", 30, di_spiky),
        ("mid gain", 24, di_spiky),
    ]:
        target = _Amp(target_drive, "target", tilt=0.6).process(_di(2, sustain=True))
        scored, tfp = _rank(target, di, candidates)
        winner = scored[0][1]
        # the winning rig's own drive must be in the driven half of the ladder
        ok = winner.drive_db >= 12
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label:34s} target D={distortion_index(tfp):.2f} "
              f"-> winner {winner.name} (score {scored[0][0]:.2f}, "
              f"clean drive+000 scored {[s for s, a, _ in scored if a.drive_db == 0][0]:.2f})")
        if not ok:
            failures.append(label)

    print("\nClean targets must not be matched with the most driven capture:")
    for label, di in [("clean target, spiky DI", di_spiky), ("clean target, sustained DI", di_sustained)]:
        target = _Amp(0, "target", tilt=0.6).process(_di(2))
        scored, tfp = _rank(target, di, candidates)
        winner = scored[0][1]
        ok = winner.drive_db <= 24  # must not pick the extreme end of the ladder
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label:34s} target D={distortion_index(tfp):.2f} "
              f"-> winner {winner.name}")
        if not ok:
            failures.append(label)

    if failures:
        print("\nFAILURES:", failures)
        sys.exit(1)
    print("\nAll drive-matching assertions passed.")


if __name__ == "__main__":
    main()
