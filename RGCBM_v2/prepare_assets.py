"""Online preparation only. Run before transferring the bundle to an offline computer."""
from pathlib import Path
import argparse
import torch
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from rgcbm.data import prepare_data, sha256, atomic_json

parser = argparse.ArgumentParser()
parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
parser.add_argument('--datasets', nargs='+', default=['cub', 'pets', 'brain_mri'])
parser.add_argument('--llm', action='store_true')
args = parser.parse_args()
root = Path(args.root).resolve()
assets = root / 'assets'; assets.mkdir(parents=True, exist_ok=True)
for name in args.datasets: prepare_data(root, name, online=True)
backbone = assets / 'efficientnet_b1_imagenet.pth'
if not backbone.exists():
    torch.hub.set_dir(str(assets / 'torch-cache'))
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
    torch.save(model.state_dict(), backbone)
if args.llm:
    from huggingface_hub import snapshot_download, HfApi
    metadata = HfApi().model_info('Qwen/Qwen2.5-1.5B-Instruct', files_metadata=True)
    model_dir = assets / 'qwen2.5-1.5b-instruct'
    snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision=metadata.sha, local_dir=str(model_dir), allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'LICENSE*', 'README.md'])
    verified = []
    for item in metadata.siblings:
        if item.rfilename.endswith('.safetensors'):
            expected = item.lfs.sha256
            actual = sha256(model_dir / item.rfilename)
            if expected != actual: raise RuntimeError('Qwen weights failed publisher SHA-256 verification.')
            verified.append({'name': item.rfilename, 'sha256': actual})
    atomic_json(assets / 'llm_manifest.json', {'repo': 'Qwen/Qwen2.5-1.5B-Instruct', 'revision': metadata.sha, 'weights': verified})
atomic_json(assets / 'backbone_manifest.json', {'architecture': 'torchvision EfficientNet-B1', 'weights': 'IMAGENET1K_V2', 'sha256': sha256(backbone), 'note': 'Architecture unchanged; torchvision preprocessing/checkpoint differs from the original TensorFlow implementation. Re-run comparisons; do not reuse old metrics.'})
print('Preparation completed; transfer the full bundle, not just the notebook.')
