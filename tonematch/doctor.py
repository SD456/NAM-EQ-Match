"""Environment + .nam file diagnostics.

Run:
    python -m tonematch.doctor                 # check the environment
    python -m tonematch.doctor model.nam       # ...and try loading a capture
    python -m tonematch.doctor path/to/folder  # ...or every .nam in a folder
"""

from __future__ import annotations

import json
import os
import sys
import traceback

OK = "  [ok] "
BAD = "  [FAIL] "
WARN = "  [warn] "


def _check_import(name: str, min_hint: str | None = None) -> bool:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        print(f"{OK}{name} {ver}")
        return True
    except Exception as e:  # noqa: BLE001
        hint = f" -> pip install {min_hint or name}" if min_hint != "" else ""
        print(f"{BAD}{name}: {type(e).__name__}: {e}{hint}")
        return False


def check_environment() -> bool:
    print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
    print("Core dependencies:")
    ok = True
    ok &= _check_import("numpy")
    ok &= _check_import("scipy")
    ok &= _check_import("soundfile")

    print("\nNAM backend:")
    torch_ok = _check_import("torch")
    ok &= torch_ok
    if torch_ok:
        import torch

        print(f"{OK}torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    nam_ok = _check_import("nam", min_hint="neural-amp-modeler")
    ok &= nam_ok
    if nam_ok:
        try:
            from nam.models import init_from_nam  # noqa: F401

            print(f"{OK}nam.models.init_from_nam available")
        except ImportError:
            try:
                from nam.models._from_nam import init_from_nam  # noqa: F401

                print(f"{OK}nam.models._from_nam.init_from_nam available")
            except ImportError:
                print(
                    f"{BAD}init_from_nam not found - your neural-amp-modeler is too old.\n"
                    "         -> pip install -U neural-amp-modeler   (needs >= 0.13)"
                )
                ok = False

    print("\nOptional:")
    _check_import("gradio")
    _check_import("matplotlib")
    _check_import("demucs")
    _check_import("requests")
    return bool(ok)


def check_nam_file(path: str) -> bool:
    print(f"\nChecking {path}")
    if not os.path.exists(path):
        print(f"{BAD}file does not exist")
        return False
    try:
        with open(path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
    except Exception as e:  # noqa: BLE001
        print(f"{BAD}not readable as JSON ({type(e).__name__}: {e}) - corrupt download?")
        return False

    arch = config.get("architecture")
    sr = config.get("sample_rate")
    meta = config.get("metadata") or {}
    print(f"{OK}JSON ok - architecture={arch}, sample_rate={sr}, name={meta.get('name')}")
    if arch == "SlimmableContainer":
        try:
            from .nam_backend import unwrap_container

            inner = unwrap_container(config)
            subs = config.get("config", {}).get("submodels") or []
            print(
                f"{OK}NAM A2 container: {len(subs)} submodel(s); "
                f"using full-size '{inner.get('architecture')}'"
            )
            arch = inner.get("architecture")
        except Exception as e:  # noqa: BLE001
            print(f"{BAD}could not unwrap SlimmableContainer: {e}")
            return False
    if arch not in ("WaveNet", "LSTM", "Linear"):
        print(f"{WARN}unusual architecture {arch!r} - may not be supported by init_from_nam")

    try:
        from .nam_backend import NamCapture

        cap = NamCapture(path)
        print(f"{OK}model loads - receptive field {cap.model.receptive_field}, sr {cap.sample_rate}")
    except Exception:  # noqa: BLE001
        print(f"{BAD}model failed to load - full traceback:")
        traceback.print_exc()
        return False

    try:
        import numpy as np

        y = cap.process(np.zeros(int(0.5 * cap.sample_rate), dtype=np.float32))
        print(f"{OK}processes audio ({len(y)} samples out)")
        return True
    except Exception:  # noqa: BLE001
        print(f"{BAD}model loaded but failed to process audio - full traceback:")
        traceback.print_exc()
        return False


def main():
    args = sys.argv[1:]
    env_ok = check_environment()

    targets = []
    for a in args:
        if os.path.isdir(a):
            targets.extend(
                os.path.join(root, f)
                for root, _, files in os.walk(a)
                for f in files
                if f.lower().endswith(".nam")
            )
        else:
            targets.append(a)

    file_ok = True
    for t in targets:
        file_ok &= check_nam_file(t)

    print()
    if env_ok and (file_ok or not targets):
        print("All good." if targets else "Environment looks good. Pass a .nam file/folder to test loading.")
    else:
        print("Problems found - see [FAIL] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
