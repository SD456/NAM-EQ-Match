"""Reintroduce two former bugs temporarily; their regression tests must fail."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("tonematch/search.py", "di_coarse = active_segment(di_refine, sr, coarse_seg_s)",
     "di_coarse = di_refine[: int(coarse_seg_s * sr)]",
     "tests/test_search_backend.py::test_silent_prefix_does_not_hide_the_active_di"),
    ("tonematch/match_eq.py", "    return y\n", "    return y[: len(x)]\n",
     "tests/test_dsp.py::test_end_impulse_preserves_complete_convolution_tail"),
]
results = []
for relative, fixed, broken, test in CASES:
    path = ROOT / "vendor/NAM-EQ-Match" / relative
    original = path.read_text(encoding="utf-8")
    assert original.count(fixed) == 1, f"Unexpected source; refusing mutation: {path}"
    command = [sys.executable, "-m", "pytest", "-q", test]
    try:
        path.write_text(original.replace(fixed, broken), encoding="utf-8")
        bad = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    finally:
        path.write_text(original, encoding="utf-8")
    assert bad.returncode == 1 and "1 failed" in bad.stdout, bad.stdout + bad.stderr
    good = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert good.returncode == 0, good.stdout + good.stderr
    results.append({"test": test, "former_bug_rejected": True, "fixed_version_passed": True,
                    "failure_evidence": bad.stdout})
output = ROOT / "artifacts/stage1"
output.mkdir(parents=True, exist_ok=True)
(output / "regression-guards.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("2 former bugs rejected; 2 restored checks passed; source restored")
