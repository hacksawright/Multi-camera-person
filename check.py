from __future__ import annotations
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
WORKSPACE = PROJECT.parent
ASSETS = PROJECT

print('=' * 58)
print(' PERSON IDENTITY TRACKING - CLEAN CHECK')
print('=' * 58)
print('Project :', PROJECT)
print('Venv    :', WORKSPACE / '.venv')
print('Cache   :', WORKSPACE / '.cache')
print('Models  :', PROJECT / 'models')
print('Data    :', PROJECT / 'data')

checks = {
    'AdaFace source': PROJECT / 'third_party' / 'AdaFace' / 'net.py',
    'deep-person-reid': PROJECT / 'third_party' / 'deep-person-reid' / 'torchreid' / '__init__.py',
    'AdaFace IR50': PROJECT / 'models' / 'adaface' / 'adaface_ir50_webface4m.ckpt',
    'OSNet x1.0': PROJECT / 'models' / 'osnet' / 'osnet_x1_0_msmt17.pth',
    'SCRFD 10G': PROJECT / 'models' / 'scrfd' / 'det_10g.onnx',
    'YOLO11s': PROJECT / 'models' / 'yolo' / 'yolo11s.pt',
    'cam0': PROJECT / 'data' / 'cam0.avi',
    'cam1': PROJECT / 'data' / 'cam1.avi',
    'cam2': PROJECT / 'data' / 'cam2.avi',
}
for name, path in checks.items():
    print(f'{name:19} {"OK" if path.exists() else "--":2}  {path}')

errors = 0
try:
    import torch
    print('PyTorch             ', torch.__version__)
    print('CUDA available      ', torch.cuda.is_available())
    if torch.cuda.is_available(): print('GPU                 ', torch.cuda.get_device_name(0))
except Exception as e:
    print('PyTorch ERROR        ', repr(e)); errors += 1

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print('ORT providers       ', providers)
    if 'CUDAExecutionProvider' not in providers: print('WARNING: CUDAExecutionProvider missing')
except Exception as e:
    print('ONNX Runtime ERROR   ', repr(e)); errors += 1

try:
    local_reid = PROJECT / 'third_party' / 'deep-person-reid'
    sys.path = [p for p in sys.path if 'deep-person-reid' not in p]
    sys.path.insert(0, str(local_reid))
    import torchreid
    from torchreid.utils import FeatureExtractor
    loaded = Path(torchreid.__file__).resolve()
    print('deep-person-reid OK ', loaded)
    if local_reid.resolve() not in loaded.parents:
        print('deep-person-reid ERR source is not project-local:', loaded)
        errors += 1
except Exception as e:
    print('deep-person-reid ERR', repr(e)); errors += 1

for key in ('AdaFace source','deep-person-reid','AdaFace IR50','OSNet x1.0','SCRFD 10G','YOLO11s'):
    if not checks[key].exists():
        errors += 1
        print('ERROR: missing', key)

if errors:
    print(f'CHECK FAILED: {errors} issue(s). Run setup.ps1.')
    sys.exit(2)
print('CHECK OK')
