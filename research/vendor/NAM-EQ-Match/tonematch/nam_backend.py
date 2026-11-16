"""Backends that turn a DI signal into an amped signal.

`NamCapture` wraps a .nam file via the neural-amp-modeler package (PyTorch).
`MockAmp` is a lightweight numpy stand-in used by the tests.
Both expose: .name, .sample_rate, .process(x) -> y
"""

from __future__ import annotations

import json
import os
import hashlib
from copy import deepcopy

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
    Used only by library/doctor inspection. NamCapture rejects containers;
    selecting a submodel for actual inference is outside stage 1.
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
        raise RuntimeError("CUDA was explicitly requested but is unavailable; select cpu/auto or install CUDA PyTorch")
    return device


class NamCapture:
    """A .nam capture loaded as a PyTorch model.

    device: 'auto' (default) uses CUDA when available, else CPU.
    """

    def __init__(self, path: str, device: str | None = None,
                 cab_ir_path: str | None = None, rig_type: str = "unknown"):
        import torch

        self._torch = torch
        self.device = resolve_device(device)
        # Match CPU/plugin float32 behavior; CUDA TF32 otherwise changes nonlinear outputs.
        if self.device.startswith("cuda"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        self.precision = "float32_no_tf32"
        init_from_nam = _import_init_from_nam()
        with open(path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
        self.metadata = config.get("metadata") or {}
        self.architecture = config.get("architecture")
        if self.architecture != "WaveNet":
            raise ValueError(f"Unsupported NAM architecture {self.architecture!r}; stage 1 supports native WaveNet only")
        if config.get("sample_rate") != DEFAULT_SR:
            raise ValueError("NAM must declare native sample_rate 48000; missing/other rates are unsupported")
        if rig_type not in ("unknown", "full-rig", "amp-only"):
            raise ValueError("rig_type must be unknown, full-rig, or amp-only")
        if (rig_type == "amp-only") != (cab_ir_path is not None):
            raise ValueError("amp-only requires a cabinet IR; cabinet IR is only valid for amp-only")
        self.rig_type = rig_type
        self.cab_ir_path = os.path.abspath(cab_ir_path) if cab_ir_path is not None else None
        self.cab_ir = None
        self.cab_ir_sha256 = None
        self.cab_ir_sample_rate = None
        if self.cab_ir_path is not None:
            import soundfile as sf
            self.cab_ir, self.cab_ir_sample_rate = sf.read(self.cab_ir_path, dtype="float32")
            if self.cab_ir_sample_rate != DEFAULT_SR:
                raise ValueError("Cabinet IR must be native 48000 Hz")
            self._validate_audio(self.cab_ir, "Cabinet IR")
            if not np.any(self.cab_ir):
                raise ValueError("Cabinet IR must not be silent")
            with open(self.cab_ir_path, "rb") as fp:
                self.cab_ir_sha256 = hashlib.sha256(fp.read()).hexdigest()
        input_level = self.metadata.get("input_level_dbu")
        loudness = self.metadata.get("loudness")
        self.lv2_input_adjustment_db = 12.0 - float(12.0 if input_level is None else input_level)
        self.lv2_output_adjustment_db = -18.0 - float(-18.0 if loudness is None else loudness)
        if not np.isfinite([self.lv2_input_adjustment_db, self.lv2_output_adjustment_db]).all():
            raise ValueError("NAM calibration metadata must be finite")
        load_config = {**config, "config": deepcopy(config["config"])}
        self.config_adapter = []
        for layer in load_config["config"]["layers"]:
            if layer.pop("gated", False):
                raise ValueError("Legacy gated WaveNet is unsupported in stage 1")
            if "head" not in layer:
                # NAM 0.13 renamed the legacy 1x1 head; weight shapes/order are unchanged.
                layer["head"] = {"out_channels": layer.pop("head_size"),
                                 "kernel_size": 1, "bias": layer.pop("head_bias")}
                self.config_adapter.append("legacy_wavenet_head_to_nam_0_13")
        self.model = init_from_nam(load_config)
        weights = torch.tensor(config["weights"], dtype=torch.float32)
        if weights.ndim != 1 or not weights.numel() or not torch.isfinite(weights).all():
            raise ValueError("NAM weights must be a finite, nonempty vector")
        # The official loader ignores the import count; reject unconsumed trailing weights.
        if self.model.import_weights(weights) != len(weights):
            raise ValueError("NAM weight count does not match its WaveNet configuration")
        self.model.eval()
        self.model.to(self.device)
        self.sample_rate = DEFAULT_SR
        meta = config.get("metadata") or {}
        self.name = meta.get("name") or os.path.splitext(os.path.basename(path))[0]
        self.path = os.path.abspath(path)
        with open(path, "rb") as fp:
            self.sha256 = hashlib.sha256(fp.read()).hexdigest()

    @staticmethod
    def _validate_audio(x, label):
        if x.ndim != 1 or not x.size or not np.isfinite(x).all():
            raise ValueError(f"{label} must be nonempty, finite mono audio")

    def process(self, x: np.ndarray, chunk_s: float = 10.0) -> np.ndarray:
        """Render the complete candidate chain, preserving the cabinet tail."""
        y = self.process_raw(x, chunk_s)
        if self.cab_ir is not None:
            from scipy.signal import fftconvolve
            y = fftconvolve(y, self.cab_ir, mode="full").astype(np.float32)
        self._validate_audio(y, "Candidate output")
        return y

    def process_raw(self, x: np.ndarray, chunk_s: float = 10.0) -> np.ndarray:
        """Run audio through the capture (chunked, receptive-field aware)."""
        torch = self._torch
        x = np.asarray(x)
        self._validate_audio(x, "NAM input")
        if not np.isfinite(chunk_s) or chunk_s <= 0 or int(chunk_s * self.sample_rate) < 1:
            raise ValueError("chunk_s must be finite and contain at least one sample")
        rf = int(self.model.receptive_field)
        if rf < 1:
            raise ValueError("NAM receptive field must be positive")
        chunk = int(chunk_s * self.sample_rate)
        out = np.empty(len(x), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(x), chunk):
                s = max(0, i - (rf - 1))
                seg = torch.from_numpy(
                    np.ascontiguousarray(x[s : i + chunk], dtype=np.float32)
                ).to(self.device)
                y = self.model(seg, pad_start=(s == 0))
                y = y.cpu().numpy()
                expected = min(chunk, len(x) - i)
                self._validate_audio(y, "NAM output")
                expected_output = len(seg) if s == 0 else len(seg) - rf + 1
                if len(y) != expected_output:
                    raise ValueError("NAM output length does not match the receptive-field contract")
                out[i : i + chunk] = y[-expected:]
        self._validate_audio(out, "NAM output")
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
            # Recursive: pick up .nam files in subfolders too (community PR #3)
            paths = sorted(
                os.path.join(root, p)
                for root, _dirs, files in os.walk(paths_or_dir)
                for p in files
                if p.lower().endswith(".nam")
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
