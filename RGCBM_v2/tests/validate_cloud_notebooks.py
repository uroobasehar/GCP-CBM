"""Execute real cloud notebook logic locally with explicit managed-service substitutions."""
from pathlib import Path
import argparse
import json
import os
import re
import sys
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

SOURCE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
args=parser.parse_args()
OUTPUT=Path(args.output).resolve();OUTPUT.mkdir(parents=True,exist_ok=True)
from importlib import metadata
from packaging.requirements import Requirement
for line in (SOURCE/'cloud_requirements.txt').read_text().splitlines():
    requirement=Requirement(line)
    if requirement.marker and not requirement.marker.evaluate():continue
    assert requirement.specifier.contains(metadata.version(requirement.name)),line
records=[]
for mode in ['Colab','Kaggle']:
    root=OUTPUT/mode.lower()
    root.mkdir(parents=True,exist_ok=True)
    notebook=nbformat.read(SOURCE/'notebooks'/f'RGCBM_{mode}.ipynb',as_version=4)
    notebook.cells[1].source=notebook.cells[1].source.replace('BATCH_SIZE = 24','BATCH_SIZE = 2').replace('SMOKE_TEST = False','SMOKE_TEST = True')
    notebook.cells[1].source+=f'''\nRUN_NAME += '_cloud_validation_{mode.lower()}'
import sys
sys.path.insert(0,{str(SOURCE/'tests')!r})
from drive_fixture import install_cloud_fixture
fixture=install_cloud_fixture({mode!r},{str(root)!r},{str(SOURCE)!r})
'''
    # Full evaluation/journal code remains unchanged. Turn optional audits on for this smoke run.
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        cell.source=cell.source.replace('RUN_CUB_ATTRIBUTE_AUDIT = False','RUN_CUB_ATTRIBUTE_AUDIT = True')
    completed=[];started=time.monotonic()
    def finished(cell,cell_index,**kwargs):
        completed.append(cell_index)
        print(mode,'completed code cell',cell_index+1,'/',len(notebook.cells),flush=True)
    os.environ.update(IPYTHONDIR=str(OUTPUT/'ipython'),JUPYTER_RUNTIME_DIR=str(OUTPUT/'runtime'),MPLCONFIGDIR=str(OUTPUT/'matplotlib'))
    for folder in ['ipython','runtime','matplotlib','ipc']:(OUTPUT/folder).mkdir(exist_ok=True)
    manager=KernelManager(kernel_name='python3',transport='ipc',ip=str(OUTPUT/'ipc'/mode.lower()))
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,km=manager,timeout=600,startup_timeout=90,store_widget_state=False,on_cell_executed=finished)
    error=None
    try:
        client.execute(cwd=str(root))
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        nbformat.write(notebook,OUTPUT/f'executed_{mode.lower()}_smoke.ipynb')
        record={'platform_notebook':mode,'status':'passed' if error is None else 'failed','cells':len(notebook.cells),'executed_code_cells':len(completed),'duration_seconds':time.monotonic()-started,'data':'real CUB images, original split metadata','runtime':f'local Linux CPU, Python {sys.version.split()[0]}, NumPy {metadata.version("numpy")}','substitutions':['pip calls simulated after installed requirement checks','Colab mount simulated' if mode=='Colab' else 'Google OAuth/Drive API simulated'],'executed':'embedded source preparation, actual data/asset helper subprocess, both training stages, checkpointing, evaluation, report generation, all journal code, storage cell logic','ablation_sweep':'disabled in this cloud smoke check; separately validated in offline notebook','live_cloud_authentication':'not tested; requires actual Google/Kaggle session','error':error}
        records.append(record)
        (OUTPUT/'cloud_notebook_validation.json').write_text(json.dumps(records,indent=2))
    if mode=='Colab':
        run=OUTPUT/'colab-mounted-drive/results'/f'cub_K16_seed42_smoke_cloud_validation_colab'
        assert (run/'last.pt').is_file() and (run/'figures/learning_curves.png').is_file()
    print(mode,'notebook smoke completed',flush=True)
print(json.dumps(records,indent=2),flush=True)
