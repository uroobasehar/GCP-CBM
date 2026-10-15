"""Probe compiled libraries in fresh processes, including both TF/PyTorch import orders."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from importlib import metadata


def probe(order):
    if order == 'tensorflow-first':
        import tensorflow as tf
        import torch
    else:
        import torch
        import tensorflow as tf
    import numpy as np
    import pandas as pd
    import scipy.linalg
    import sklearn.preprocessing
    import cv2
    import h5py
    import pyarrow as pa
    import pyarrow.compute as pc
    from PIL import Image
    from torchvision.ops import nms
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from tensorflow_metadata.proto.v0 import schema_pb2
    from transformers import AutoModelForCausalLM, Qwen2Config

    assert float(tf.matmul(tf.constant([[1., 2.]]), tf.constant([[3.], [4.]])).numpy()[0, 0]) == 11.
    torch.set_num_threads(2)
    assert torch.isfinite(torch.ones(2, 2) @ torch.ones(2, 2)).all()
    assert nms(torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]]), torch.tensor([.9, .8]), .5).tolist() == [0, 1]
    np.testing.assert_allclose(scipy.linalg.solve([[2., 0.], [0., 4.]], [4., 8.]), [2., 2.])
    assert np.isfinite(sklearn.preprocessing.StandardScaler().fit_transform([[1., 2.], [3., 4.]])).all()
    frame = pd.DataFrame({'value': [1, 2, 3]})
    assert frame.value.sum() == 6
    image = np.asarray(Image.new('RGB', (8, 8), 'white'))
    assert cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).shape == (8, 8)
    with h5py.File('in-memory-validation.h5', 'w', driver='core', backing_store=False) as store:
        store['values'] = np.arange(3)
        assert store['values'][:].sum() == 3
    assert pc.sum(pa.array([1, 2, 3])).as_py() == 6
    schema_pb2.Schema()
    figure = Figure(figsize=(2, 2)); figure.subplots().plot([0, 1], [0, 1])
    FigureCanvasAgg(figure).draw()
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model = AutoModelForCausalLM.from_config(config).eval()
    with torch.inference_mode():
        assert model(torch.tensor([[1, 2, 3]])).logits.shape == (1, 3, 32)
    return {'status': 'passed', 'order': order, 'python': sys.version, 'tensorflow': tf.__version__, 'torch': torch.__version__, 'numpy': np.__version__, 'package_versions': {name: metadata.version(name) for name in ['pyarrow', 'scipy', 'scikit-learn', 'pandas', 'opencv-python-headless', 'Pillow', 'h5py', 'protobuf', 'matplotlib', 'transformers', 'torchvision']}, 'operations': ['TensorFlow CPU matmul', 'PyTorch CPU matmul', 'torchvision compiled NMS', 'SciPy solve', 'scikit-learn scaler', 'pandas aggregation', 'Pillow/OpenCV conversion', 'h5py in-memory roundtrip', 'PyArrow compute', 'protobuf schema', 'Matplotlib Agg render', 'Transformers tiny Qwen forward']}


def run_checks(output):
    import os
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'rgcbm-matplotlib-cache'))
    results = []
    for order in ['tensorflow-first', 'torch-first']:
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe', order], capture_output=True, text=True, timeout=180, env=environment)
        except subprocess.TimeoutExpired:
            results.append({'order': order, 'status': 'failed', 'error': 'Native probe timed out after 180 seconds.'})
            continue
        entry = {'order': order, 'returncode': process.returncode}
        if process.returncode == 0:
            entry.update(json.loads(process.stdout.strip().splitlines()[-1]))
        else:
            entry.update(status='failed', stdout=process.stdout[-8000:], stderr=process.stderr[-8000:])
        results.append(entry)
    result = {'status': 'passed' if all(r['status'] == 'passed' for r in results) else 'failed', 'platform': sys.platform, 'scope': 'Native CPU library and import-order checks on this host; GPU model execution is checked separately.', 'probes': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if result['status'] != 'passed':
        raise RuntimeError(f'Native library probe failed. Inspect {output}; do not suppress OpenMP/DLL errors.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', choices=['tensorflow-first', 'torch-first'])
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'runtime' / 'native_library_validation.json')
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe(args.probe)))
    else:
        run_checks(args.output)
        print('Both native import-order probes passed. Report:', args.output)
