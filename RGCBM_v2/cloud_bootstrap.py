"""Bootstrap managed cloud runtimes; preserve a working GPU build when possible."""
from importlib import metadata
import subprocess
import sys


def prepare_cloud_runtime(requirements):
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError('This notebook supports Python 3.10–3.13. Select a supported cloud runtime before installation.')
    print('Python compatibility profile:', 'TensorFlow 2.20.0 (Python 3.13)' if sys.version_info[:2] >= (3, 13) else 'TensorFlow 2.19.1 (Python 3.10–3.12)')
    modules = {'numpy': 'numpy', 'pandas': 'pandas', 'matplotlib': 'matplotlib',
               'PIL': 'pillow', 'sklearn': 'scikit-learn', 'tensorflow': 'tensorflow',
               'google.protobuf': 'protobuf', 'transformers': 'transformers',
               'accelerate': 'accelerate', 'huggingface_hub': 'huggingface-hub',
               'tokenizers': 'tokenizers', 'safetensors': 'safetensors',
               'tensorflow_datasets': 'tensorflow-datasets', 'medmnist': 'medmnist',
               'kagglehub': 'kagglehub', 'google.genai': 'google-genai'}
    loaded = {}
    for module, distribution in modules.items():
        if module in sys.modules:
            version = getattr(sys.modules[module], '__version__', None)
            if version is None:
                try: version = metadata.version(distribution)
                except metadata.PackageNotFoundError: version = None
            loaded[distribution] = str(version)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade-strategy',
                           'only-if-needed', *requirements])
    changed = [name for name, version in loaded.items() if version != metadata.version(name)]
    if changed:
        raise RuntimeError('Restart the runtime/kernel, then rerun from the first cell. Already loaded packages changed: ' + ', '.join(changed))
    capability = None
    try:
        import torch
        import torchvision
        if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) < (2, 7):
            raise RuntimeError('PyTorch 2.7 or newer is required.')
        # This also detects incompatible torchvision compiled operators.
        torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5)
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            test = torch.ones(2, device='cuda')
            assert float((test + 1).sum()) == 4.
            torch.cuda.synchronize()
    except (ImportError, OSError, RuntimeError, AttributeError, AssertionError) as exc:
        # CUDA 12.6 keeps older cloud GPU architectures; Blackwell requires 12.8.
        flavor = 'cu128' if capability and capability[0] >= 10 else 'cu126'
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade',
                               'torch==2.7.1', 'torchvision==0.22.1', '--index-url',
                               'https://download.pytorch.org/whl/' + flavor])
        raise RuntimeError('Repaired the PyTorch/torchvision pair. Restart this runtime/kernel and rerun from the first cell. Original check: ' + str(exc)[:180]) from None
    print('Cloud stack checked:', torch.__version__, torchvision.__version__, 'CUDA', torch.version.cuda)
