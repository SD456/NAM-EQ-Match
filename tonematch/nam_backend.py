"""Backends that turn a DI signal into an amped signal.

`NamCapture` wraps a .nam file via the neural-amp-modeler package (PyTorch).
`MockAmp` is a lightweight numpy stand-in used by the tests.
Both expose: .name, .sample_rate, .process(x) -> y
"""

from __future__ import annotations

import json
import os

import numpy as np

from .audio import EPS

DEFAULT_SR = 48000


def _import_init_from_nam():
    try:
        from nam.models import init_from_nam  # >= 0.13

        return init_from_nam
    except ImportError:
        pass
    try:
        from nam.models._from_nam import init_from_nam

        return init_from_nam
    except ImportError as e:
        raise ImportError(
            "Could not import `init_from_nam` from the neural-amp-modeler package. "
            "Install/upgrade it with: pip install -U neural-amp-modeler"
        ) from e


def unwrap_container(config: dict) -> dict:
    """Unwrap NAM A2 'SlimmableContainer' files.

    A SlimmableContainer holds several complete, independent sub-models at
    different sizes ("max_value" ascending; the last one is full quality).
    Each sub-model is itself a full .nam document, so we simply select the
    largest and load that.
    """
    outer_meta = config.get("metadata") or {}
    outer_sr = config.get("sample_rate")
    while config.get("architecture") == "SlimmableContainer":
        subs = config.get("config", {}).get("submodels") or []
        if not subs:
            raise ValueError("SlimmableContainer has no submodels")
        config = max(subs, key=lambda s: s.get("max_value", 0.0))["model"]
    # carry outer metadata/sample_rate down if the submodel lacks them
    if outer_meta and not (config.get("metadata") or {}).get("name"):
        config.setdefault("metadata", {})
        config["metadata"].setdefault("name", outer_meta.get("name"))
    if outer_sr and not config.get("sample_rate"):
        config["sample_rate"] = outer_sr
    return config


def resolve_device(device: str | None = None) -> str:
    """Resolve 'auto'/None to the best available torch device."""
    import torch

    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available - falling back to CPU. "
              "Install a CUDA build of PyTorch (see pytorch.org) to use your GPU.")
        return "cpu"
    return device


class NamCapture:
    """A .nam capture loaded as a PyTorch model.

    device: 'auto' (default) uses CUDA when available, else CPU.
    """

    def __init__(self, path: str, device: str | None = None):
        import torch

        self._torch = torch
        self.device = resolve_device(device)
        init_from_nam = _import_init_from_nam()
        with open(path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
        config = unwrap_container(config)
        self.model = init_from_nam(config)
        self.model.eval()
        self.model.to(self.device)
        sr = config.get("sample_rate")
        self.sample_rate = int(sr) if sr else DEFAULT_SR
        meta = config.get("metadata") or {}
        self.name = meta.get("name") or os.path.splitext(os.path.basename(path))[0]
        self.path = path

    def process(self, x: np.ndarray, chunk_s: float = 10.0) -> np.ndarray:
        """Run audio through the capture (chunked, receptive-field aware)."""
        torch = self._torch
        rf = int(self.model.receptive_field)
        chunk = int(chunk_s * self.sample_rate)
        out = np.empty(len(x), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(x), chunk):
                s = max(0, i - (rf - 1))
                seg = torch.from_numpy(
                    np.ascontiguousarray(x[s : i + chunk], dtype=np.float32)
                ).to(self.device)
                y = self.model(seg, pad_start=(s == 0))
                out[i : i + chunk] = y.cpu().numpy()[-(min(chunk, len(x) - i)) :]
        return out


class MockAmp:
    """Simple nonlinear amp sim (pre-emphasis -> tanh drive -> tilt EQ).

    Used for tests and as a no-torch fallback demo.
    """

    def __init__(
        self,
        drive_db: float = 12.0,
        tilt_db_per_oct: float = 0.0,
        presence_db: float = 0.0,
        sample_rate: int = DEFAULT_SR,
        name: str | None = None,
    ):
        self.drive_db = drive_db
        self.tilt = tilt_db_per_oct
        self.presence_db = presence_db
        self.sample_rate = sample_rate
        self.name = name or f"MockAmp(drive={drive_db:+.0f}dB)"
        self.path = self.name

    def process(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter

        sr = self.sample_rate
        # pre-emphasis (bright cap feel)
        b = [1.0, -0.85]
        y = lfilter(b, [1.0], x)
        # drive
        g = 10.0 ** (self.drive_db / 20.0)
        y = np.tanh(g * y) / np.tanh(g * 0.5 + EPS) * 0.5
        # crude tilt EQ via first-order shelves
        if abs(self.tilt) > 1e-6 or abs(self.presence_db) > 1e-6:
            y = _tilt_filter(y, sr, self.tilt, self.presence_db)
        return y.astype(np.float32)


def _tilt_filter(x: np.ndarray, sr: int, tilt_db_per_oct: float, presence_db: float) -> np.ndarray:
    """Apply an FFT-domain tilt/presence EQ (offline, fine for a mock)."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    f = np.maximum(f, 20.0)
    g_db = tilt_db_per_oct * np.log2(f / 1000.0)
    # presence bump around 4 kHz
    g_db += presence_db * np.exp(-0.5 * ((np.log2(f / 4000.0)) / 0.5) ** 2)
    X *= 10.0 ** (g_db / 20.0)
    return np.fft.irfft(X, n=n)


def load_captures(
    paths_or_dir,
    limit: int | None = None,
    errors_out: list | None = None,
    device: str | None = None,
) -> list:
    """Load NamCapture objects from a directory or a list of .nam paths.

    Load failures are printed and, if `errors_out` is given, appended to it
    as (path, error_message) tuples so callers (e.g. the GUI) can show them.
    """
    if isinstance(paths_or_dir, str):
        if os.path.isdir(paths_or_dir):
            paths = sorted(
                os.path.join(root, f)
                for root, _, files in os.walk(paths_or_dir)
                for f in files
                if f.lower().endswith(".nam")
            )
        else:
            paths = [paths_or_dir]
    else:
        paths = list(paths_or_dir)
    if limit:
        paths = paths[:limit]
    captures = []
    for p in paths:
        try:
            captures.append(NamCapture(p, device=device))
        except Exception as e:  # noqa: BLE001 - report and continue
            msg = f"{type(e).__name__}: {e}"
            print(f"[warn] failed to load {p}: {msg}")
            if errors_out is not None:
                errors_out.append((p, msg))
    if not paths and errors_out is not None:
        errors_out.append((str(paths_or_dir), "no .nam files found at this location"))
    return captures
