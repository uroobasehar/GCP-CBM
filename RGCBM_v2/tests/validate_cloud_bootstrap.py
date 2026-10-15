"""Exercise restart protection and repair decisions without package downloads."""
from pathlib import Path
import sys
import types
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cloud_bootstrap as bootstrap
import torch

with patch.object(bootstrap.subprocess,'check_call') as install:
    bootstrap.prepare_cloud_runtime([])
    assert install.call_count==1

with patch.dict(sys.modules,{'numpy':types.SimpleNamespace(__version__='0.0')}):
    with patch.object(bootstrap.subprocess,'check_call'):
        try:bootstrap.prepare_cloud_runtime([])
        except RuntimeError as exc:assert 'Restart the runtime' in str(exc)
        else:raise AssertionError('Loaded obsolete NumPy was not rejected')

with patch.object(torch,'__version__','2.6.0'),patch.object(bootstrap.subprocess,'check_call') as install:
    try:bootstrap.prepare_cloud_runtime([])
    except RuntimeError as exc:assert 'Repaired' in str(exc)
    else:raise AssertionError('Old PyTorch was not repaired')
    command=install.call_args.args[0]
    assert 'torch==2.7.1' in command and 'torchvision==0.22.1' in command and command[-1].endswith('/cu126')

with patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'get_device_capability',return_value=(12,0)),patch.object(torch,'ones',side_effect=RuntimeError('unsupported GPU architecture')),patch.object(bootstrap.subprocess,'check_call') as install:
    try:bootstrap.prepare_cloud_runtime([])
    except RuntimeError as exc:assert 'Repaired' in str(exc)
    else:raise AssertionError('Unsupported Blackwell kernel did not request repair')
    assert install.call_args.args[0][-1].endswith('/cu128')

with patch.object(bootstrap.sys,'version_info',(3,13)):
    try:bootstrap.prepare_cloud_runtime([])
    except RuntimeError as exc:assert 'Python 3.10' in str(exc)
    else:raise AssertionError('Unsupported Python runtime accepted')
print('Passed: compatible stack, restart after loaded-library changes, paired legacy-GPU repair, Blackwell CUDA 12.8 repair, and unsupported Python rejection.')
