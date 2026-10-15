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
    weights = EfficientNet_B1_Weights.IMAGENET1K_V2
    import re
    prefix = re.search(r'-([0-9a-f]+)\.pth$', weights.url).group(1)
    torch.hub.download_url_to_file(weights.url, str(backbone), hash_prefix=prefix)
if args.llm:
    import json
    revision = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
    expected = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
    model_dir = assets / 'qwen2.5-1.5b-instruct'
    required = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'model.safetensors']
    if not all((model_dir / name).is_file() for name in required):
        from huggingface_hub import snapshot_download
        snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision=revision, local_dir=str(model_dir), allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'LICENSE*', 'README.md'])
    actual = sha256(model_dir / 'model.safetensors')
    if actual != expected: raise RuntimeError('Qwen weights failed the pinned publisher SHA-256 check. Preserve the error and replace the corrupt asset; no unverified model will be used.')
    atomic_json(assets / 'llm_manifest.json', {'repo': 'Qwen/Qwen2.5-1.5B-Instruct', 'revision': revision, 'weights': [{'name': 'model.safetensors', 'sha256': actual}], 'offline_reuse': 'complete local assets verified before any model download'})
atomic_json(assets / 'backbone_manifest.json', {'architecture': 'torchvision EfficientNet-B1', 'weights': 'IMAGENET1K_V2', 'sha256': sha256(backbone), 'note': 'Architecture unchanged; torchvision preprocessing/checkpoint differs from the original TensorFlow implementation. Re-run comparisons; do not reuse old metrics.'})
print('Preparation completed; transfer the full bundle, not just the notebook.')
