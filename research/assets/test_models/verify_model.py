"""Assert real NAM CPU/CUDA inference and print reproducible environment evidence."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vendor' / 'NAM-EQ-Match'))
from tonematch.nam_backend import NamCapture

path = Path(__file__).with_name('Lum_Marshall_2204_1982_AMP_Channel3_A1.nam')
assert hashlib.sha256(path.read_bytes()).hexdigest() == 'a9579e9a136b9dd0a360ecd64235bedaa58f4468e7c58dc8e734ca2b19fd042f'
assert torch.cuda.is_available(), 'CUDA is required for this environment validation'
torch.set_num_threads(4)
x = (0.08 * np.sin(2 * np.pi * 220 * np.arange(48000) / 48000)).astype(np.float32)
outputs = {}
timings = {}
for device in ('cpu', 'cuda'):
    capture = NamCapture(str(path), device=device)
    assert capture.sample_rate == 48000
    assert next(capture.model.parameters()).device.type == device
    if device == 'cuda':
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
    start = time.perf_counter()
    outputs[device] = capture.process(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    timings[device] = time.perf_counter() - start
    assert outputs[device].shape == x.shape
    assert np.isfinite(outputs[device]).all()
    assert np.max(np.abs(outputs[device])) > 1e-6
np.testing.assert_allclose(outputs['cuda'], outputs['cpu'], atol=1e-5, rtol=1e-4)
report = {
    'python': platform.python_version(),
    'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0),
    'capability': list(torch.cuda.get_device_capability(0)),
    'compiled_architectures': torch.cuda.get_arch_list(),
    'cuda_cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
    'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
    'neural_amp_modeler': importlib.metadata.version('neural-amp-modeler'),
    'model': path.name,
    'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'model_config_adapter': capture.config_adapter,
    'model_receptive_field': int(capture.model.receptive_field),
    'model_parameter_device': str(next(capture.model.parameters()).device),
    'sample_rate': 48000,
    'input_samples': len(x),
    'output_samples': len(outputs['cuda']),
    'cuda_peak': float(np.max(np.abs(outputs['cuda']))),
    'cpu_cuda_max_absolute_error': float(np.max(np.abs(outputs['cpu'] - outputs['cuda']))),
    'forward_seconds_first_run': timings,
    'passed': True,
}
print(json.dumps(report, indent=2))
