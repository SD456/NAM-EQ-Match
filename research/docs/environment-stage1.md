# Stage 1 Windows environment

Environment location: `NAM_guitar_modeling/.venv`, isolated from the global Python installation.

Verified versions: Python 3.12.7; NumPy 2.5.4; SciPy 1.18.1; SoundFile 0.14.0; Matplotlib 3.11.2; PyTorch 2.10.0+cu128; neural-amp-modeler 0.13.0; pytest 9.2.0. `requirements-lock-win.txt` freezes all 95 installed distributions, including transitive dependencies and packaging tools. `pip check` reports no broken requirements.

Run from the project directory in PowerShell:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup-env.ps1
```

The installer uses Python 3.12, installs the official `torch==2.10.0+cu128` Windows wheel with its published SHA256, then installs the complete Windows dependency lock. It checks dependency consistency, executes a GPU matrix multiplication, and loads the actual bundled NAM capture on both CPU and CUDA. RTK is required by this workspace's command policy.

The direct requirements are NumPy, SciPy, SoundFile, Matplotlib, PyTorch, neural-amp-modeler 0.13.0, and pytest. NAM itself requires the training stack (Lightning, transformers, librosa, TensorBoard, etc.). Requests may appear transitively through librosa/pooch; no application cloud requests are enabled. Gradio and Demucs are excluded.

PyTorch selection is based on the official [Blackwell/CUDA 12.8 release announcement](https://pytorch.org/blog/pytorch-2-7/) and the [official release support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md). The selected official wheel is `torch-2.10.0+cu128-cp312-cp312-win_amd64.whl`, installed from `https://download.pytorch.org/whl/cu128`. The NVIDIA driver is 610.88 and hardware is an RTX 5060 with 8151 MiB reported memory. The driver reports CUDA 13.3 compatibility; PyTorch's own bundled CUDA runtime is 12.8.

The first index installation downloaded through the official index's `download-r2.pytorch.org` mirror but pip rejected the wheel's checksum. Installation was retried from the official `download.pytorch.org` origin, preserving the expected SHA256 `fbde8f6a9ec8c76979a0d14df21c10b9e5cab6f0d106a73ca73e2179bc597cae`; checksum verification was never disabled. The setup script uses that origin URL and hash.

## Real model fixture

`assets/test_models/Lum_Marshall_2204_1982_AMP_Channel3_A1.nam` is a real public Marshall 2204 1982 amp-only capture by Lum / Steffen Dangmann, not a randomly initialized network. It is WaveNet format 0.5.4 at 48000 Hz with 13802 weights. It comes from [VoLum commit a0b2f08b6be3fa2f7fdb58543f579d32e0ebe2f3](https://github.com/guitarlum/VoLum/blob/a0b2f08b6be3fa2f7fdb58543f579d32e0ebe2f3/rigs/Marshall%202204%201982/AMP-2204-3.nam), rediscovered through the existing WaveYard audition provenance and downloaded again from that pinned URL.

- Model SHA256: `a9579e9a136b9dd0a360ecd64235bedaa58f4468e7c58dc8e734ca2b19fd042f`.
- License: repository MIT license, copied as `assets/test_models/VoLum-LICENSE.txt`; no separate model restriction was found. Provenance records the exact URL, author, format, metadata, and license scope.
- License SHA256: `c6cbc47bb4ccb43771b97c4c1dba1fd35d4f50cbde3ef92fd32d0b13a6b33534`.
- Metadata reports validation ESR 0.005843426197022801. Input/output dBu calibration metadata is absent; this fixture does not establish calibrated physical amplifier equivalence.

Recheck real inference with:

```powershell
rtk .venv/Scripts/python.exe assets/test_models/verify_model.py
```

This asserts the pinned model hash, 48 kHz sample rate, model parameter device, finite/nonzero 48000-sample output, and CPU/CUDA agreement (`atol=1e-5`, `rtol=1e-4`). The sine input is a diagnostic signal; the model weights are the actual captured public model. No user WAV is opened or modified by this verification.

The application backend explicitly disables cuDNN and CUDA matmul TF32 when loading a CUDA capture, so actual processing uses full float32. The verifier asserts this application policy rather than setting a special test-only policy. The initial default TF32 run showed maximum absolute difference around 0.000157, while full float32 passes the stricter assertion.

Observed real-model result: 48000 input/output samples, CUDA peak 0.1191292256, CPU/CUDA maximum absolute difference 0.00000201166. The compiled CUDA architecture list includes `sm_120`, matching this RTX 5060. Evidence is saved in `artifacts/stage1/real-nam-environment.json`. A separate GPU matrix multiply also completed (64x64 ones -> value 64). First-run timing includes initialization and is not a throughput benchmark.

NAM 0.13 requires a layer `head` object whereas this genuine legacy 0.5.4 capture stores `head_size`/`head_bias`. The patched backend converts those structural fields in memory and verifies that re-exported float32 weights exactly equal all original weights. The on-disk model stays byte-identical. Legacy gated models are explicitly rejected in stage 1 rather than silently interpreted with different activations.
