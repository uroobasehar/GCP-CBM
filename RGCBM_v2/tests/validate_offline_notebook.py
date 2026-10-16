"""Execute all offline notebook cells on real images under an external network-denial guard."""
from pathlib import Path
import argparse,json,os,re,sys,time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
SOURCE=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
HERE=args.output.resolve();HERE.mkdir(parents=True);ROOT=HERE/'project';ROOT.mkdir()
for name in ['data','assets']:(ROOT/name).symlink_to(SOURCE/name,target_is_directory=True)
os.environ.update(RGCBM_ROOT=str(ROOT),IPYTHONDIR=str(HERE/'ipython'),JUPYTER_RUNTIME_DIR=str(HERE/'runtime'),MPLCONFIGDIR=str(HERE/'matplotlib'))
for folder in ['ipc','ipython','runtime','matplotlib']:(HERE/folder).mkdir()
nb=nbformat.read(SOURCE/'notebooks/RGCBM_Offline.ipynb',as_version=4)
nb.cells[1].source=nb.cells[1].source.replace('BATCH_SIZE = 24','BATCH_SIZE = 2').replace('SMOKE_TEST = False','SMOKE_TEST = True').replace("'cub': 64","'cub': 16").replace("LLM_BACKEND = 'local_qwen'","LLM_BACKEND = 'template'")
nb.cells[1].source+='\nRUN_TAG += "_offline_validation"\nRUN_NAME = f"{DATASET}_K{NUM_CONCEPTS}_seed{SEED}_{RUN_TAG}_smoke"\nimport socket\ntry: socket.socket(socket.AF_INET)\nexcept PermissionError: print("Confirmed: Internet sockets blocked inside kernel")\nelse: raise RuntimeError("Missing external no-network enforcement")\n'
for cell in nb.cells:
 if cell.cell_type!='code':continue
 cell.source=cell.source.replace('RUN_CUB_ATTRIBUTE_AUDIT = False','RUN_CUB_ATTRIBUTE_AUDIT = True').replace('RUN_ABLATIONS = False','RUN_ABLATIONS = True')
 cell.source=re.sub(r'ABLATION_VARIANTS = \[.*?\]',"ABLATION_VARIANTS = ['no_finetune','K32']",cell.source,flags=re.S)
 cell.source=cell.source.replace('ABLATION_SEEDS = [42, 43, 44]','ABLATION_SEEDS = [42]')
km=KernelManager(kernel_name='python3',transport='ipc',ip=str(HERE/'ipc/notebook'));km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
completed=[];started=time.monotonic()
def done(cell,cell_index,**kwargs):
 completed.append(cell_index);print('Completed code cell',cell_index+1,'/',len(nb.cells),flush=True)
error=None
try:
 NotebookClient(nb,km=km,timeout=600,startup_timeout=90,store_widget_state=False,on_cell_executed=done).execute(cwd=str(ROOT))
 status=json.loads(next((ROOT/'results').glob('dataset_suite_*/queue_status.json')).read_text());assert status['complete']
 for ds in ['cub','pets','brain_mri']:
  rows=list((ROOT/'results'/f'{ds}_ablation_summary_guided_v4').glob('*.csv'));assert rows,ds
except Exception as exc:
 error=f'{type(exc).__name__}: {exc}';raise
finally:
 nbformat.write(nb,HERE/'executed_offline_smoke.ipynb')
 record={'status':'passed' if error is None else 'failed','cells':len(nb.cells),'executed_code_cells':len(completed),'duration_seconds':time.monotonic()-started,'data':'real CUB, Pet and MRI images','runtime':f'Linux CPU Python {sys.version.split()[0]}; not Windows or GPU validation','network':'external inherited seccomp denies Internet sockets; Unix IPC permits Jupyter','test_parameters':{'smoke_only':True,'batch_size':2,'concepts':16,'head_epochs':1,'finetune_epochs':1,'max_batches':2,'ablations_per_dataset':['no_finetune','K32'],'seeds':[42],'LLM':'deterministic template; actual Qwen tested separately'},'error':error}
 (HERE/'notebook_validation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
