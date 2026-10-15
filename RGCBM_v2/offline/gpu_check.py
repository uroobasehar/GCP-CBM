from pathlib import Path
import argparse
import json
import platform
import sys
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from rgcbm.core import training_config, RGCBM
from rgcbm.environment import check_requested_libraries

parser = argparse.ArgumentParser()
parser.add_argument('--require-gpu', action='store_true')
args = parser.parse_args()
print(check_requested_libraries().to_string(index=False))
info = {'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}
if torch.cuda.is_available():
    info.update(gpu=torch.cuda.get_device_name(0), compute_capability=torch.cuda.get_device_capability(0), compiled_architectures=torch.cuda.get_arch_list(), memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    x = torch.randn(256, 256, device='cuda')
    assert torch.isfinite(x @ x).all()
    model = RGCBM(200, training_config(), root / 'assets').cuda().eval()
    with torch.no_grad():
        output = model(torch.rand(2, 3, 240, 240, device='cuda'))
    assert output['probs'].shape == (2, 200) and torch.isfinite(output['probs']).all()
    torch.cuda.synchronize()
    info['gpu_functional_check'] = 'matrix multiplication and pretrained model inference passed'
elif args.require_gpu:
    print(json.dumps(info, indent=2))
    raise RuntimeError('CUDA is unavailable. Install a current NVIDIA RTX PRO Blackwell driver supporting CUDA 12.8 before going offline; do not silently run training on CPU.')
print(json.dumps(info, indent=2))
(root / 'offline' / 'machine_validation.json').write_text(json.dumps(info, indent=2))
