from importlib import metadata, util
from packaging.requirements import Requirement
import pandas as pd

REQUESTED = {
    'tensorflow': 'tensorflow==2.19.1',
    'numpy': 'numpy>=1.26,<3',
    'pandas': 'pandas>=2.0,<3',
    'matplotlib': 'matplotlib>=3.8,<4',
    'PIL': 'pillow>=10',
    'sklearn': 'scikit-learn>=1.4,<2',
    'medmnist': 'medmnist>=3,<4',
    'kagglehub': 'kagglehub>=0.3,<1',
    'tensorflow_datasets': 'tensorflow-datasets>=4.9,<5',
}


def check_requested_libraries(strict=True):
    """Inspect versions/specs without initializing TensorFlow or reserving GPU memory."""
    rows=[]
    for module,text in REQUESTED.items():
        requirement=Requirement(text)
        try:
            version=metadata.version(requirement.name)
            found=util.find_spec(module) is not None
            passed=found and requirement.specifier.contains(version)
        except (metadata.PackageNotFoundError, ModuleNotFoundError):
            version=None;found=False;passed=False
        rows.append({'module':module,'requirement':text,'installed':version,'module_found':found,'passes':passed})
    frame=pd.DataFrame(rows)
    if strict and not frame.passes.all():
        raise RuntimeError('Requested dependency check failed: '+', '.join(frame.loc[~frame.passes,'requirement']))
    return frame


def runtime_manifest():
    import platform
    import sys
    import torch
    packages = {}
    for name in ['torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'matplotlib',
                 'tensorflow', 'transformers', 'protobuf', 'tensorflow-datasets']:
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    result = {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
              'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}
    if torch.cuda.is_available():
        result.update(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0))
    return result
