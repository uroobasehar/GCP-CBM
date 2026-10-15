"""Regression checks for the ABI and undeclared NumPy/PyArrow incompatibilities."""
from pathlib import Path
import json
import sys
import tempfile
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'offline'))
from audit_dependencies import audit, windows_tags

cases = {
    'example-1.0-cp310-cp310-win_amd64.whl': False,
    'example-1.0-cp312-cp312-win_amd64.whl': False,
    'example-1.0-cp311-cp311-win32.whl': False,
    'example-1.0-cp311-cp311-manylinux_2_28_x86_64.whl': False,
    'example-1.0-cp310-abi3-win_amd64.whl': True,
    'example-1.0-cp311-cp311-win_amd64.whl': True,
    'example-1.0-py3-none-any.whl': True,
}
for name, expected in cases.items():
    assert bool(windows_tags().intersection(parse_wheel_filename(name)[3])) == expected, name
with tempfile.TemporaryDirectory() as folder:
    temporary = Path(folder)
    record = json.loads((ROOT / 'offline/wheel_validation.json').read_text())
    for package in record['packages']:
        if package['name'] == 'pyarrow':
            package['version'] = '26.0.0'
    (temporary / 'wheel_validation.json').write_text(json.dumps(record))
    (temporary / 'conda_manifest.json').write_bytes((ROOT / 'offline/conda_manifest.json').read_bytes())
    try:
        audit(temporary)
    except RuntimeError as exc:
        assert 'PyArrow >=26 requires NumPy >=2' in str(exc)
    else:
        raise AssertionError('The incompatible PyArrow/NumPy pair was accepted.')
print('Passed: seven target ABI fixtures and rejection of the original PyArrow/NumPy conflict.')
