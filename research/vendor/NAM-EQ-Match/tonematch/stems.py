"""Guitar stem extraction from full mixes using Demucs.

Uses `htdemucs_6s` (6 stems: drums, bass, other, vocals, guitar, piano) so the
target can be a finished mix instead of an isolated guitar track.

Optional dependency:  pip install demucs
"""

from __future__ import annotations

import os

STEM_CHOICES = ("guitar", "other", "guitar+other")


def extract_stem(
    path: str,
    out_dir: str,
    stem: str = "guitar",
    model_name: str | None = None,
    device: str | None = None,
    progress_cb=None,
) -> str:
    """Separate `path` and return the path of the extracted stem wav.

    stem: 'guitar' (best for most songs), 'other' (4-stem fallback bucket),
          or 'guitar+other' (when guitar bleeds into 'other').
    """
    try:
        from demucs.api import Separator, save_audio
    except ImportError as e:
        raise ImportError(
            "Demucs is not installed. Install it with:  pip install demucs"
        ) from e
    import torch

    if stem not in STEM_CHOICES:
        raise ValueError(f"stem must be one of {STEM_CHOICES}, got {stem!r}")

    model_name = model_name or "htdemucs_6s"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if progress_cb:
        progress_cb(0.05, f"loading Demucs model '{model_name}' ({device})")
    sep = Separator(model=model_name, device=device)

    if progress_cb:
        progress_cb(0.15, "separating stems (this can take a while on CPU)")
    _, separated = sep.separate_audio_file(path)

    if stem == "guitar+other":
        missing = [s for s in ("guitar", "other") if s not in separated]
        if missing:
            raise ValueError(f"Model '{model_name}' has no stem(s): {missing}")
        audio = separated["guitar"] + separated["other"]
    elif stem in separated:
        audio = separated[stem]
    elif "other" in separated:
        print(f"[warn] model '{model_name}' has no '{stem}' stem; using 'other'")
        stem, audio = "other", separated["other"]
    else:
        raise ValueError(f"Model '{model_name}' has stems {list(separated)}, none usable.")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{base}_{stem.replace('+', '_')}.wav")
    save_audio(audio, out_path, samplerate=sep.samplerate)
    if progress_cb:
        progress_cb(1.0, f"stem saved: {out_path}")
    return out_path
