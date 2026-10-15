"""Run after offline setup, with the network disconnected, on the actual target."""
from pathlib import Path
import argparse
import json
import os
import platform
import sys
import time

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
parser=argparse.ArgumentParser()
parser.add_argument('--root',default=str(PACKAGE))
parser.add_argument('--dataset',choices=['cub','pets','brain_mri'],default='cub')
parser.add_argument('--require-gpu',action='store_true')
parser.add_argument('--kernel',default='rgcbm-offline')
parser.add_argument('--llm',action='store_true',help='Also generate one local Qwen report and record any validated fallback.')
args=parser.parse_args()
root=Path(args.root).resolve();runtime=root/'runtime';runtime.mkdir(parents=True,exist_ok=True)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',PIP_NO_INDEX='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(runtime/'matplotlib'),TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2')
report={'python':sys.version,'platform':platform.platform(),'checks':{},'research_accuracy':'not measured; two training batches only'}
output=runtime/f'offline_validation_{args.dataset}.json'
try:
    from rgcbm.environment import check_requested_libraries,runtime_manifest
    report['requested_libraries']=check_requested_libraries().to_dict(orient='records')
    import tensorflow as tf
    assert float(tf.matmul(tf.constant([[1.,2.]]),tf.constant([[3.],[4.]])).numpy()[0,0])==11.
    import medmnist,kagglehub,tensorflow_datasets as tfds
    assert 'pathmnist' in medmnist.INFO
    assert 'oxford_iiit_pet' in tfds.list_builders(with_community_datasets=False)
    from tensorflow_metadata.proto.v0 import schema_pb2
    schema_pb2.Schema()
    report['checks']['tensorflow_registry_protobuf']='passed'
    import torch
    torch.set_num_threads(4)
    report['environment']=runtime_manifest()
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Resolve the NVIDIA system driver before accepting this offline machine.')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        x=torch.randn(64,64,device=device);assert torch.isfinite(x@x).all();torch.cuda.synchronize()
        report['checks']['cuda_matrix']='passed'
    from rgcbm.core import Config
    from rgcbm.data import build_manifest,leakage_audit,ImageDataset
    from rgcbm.train import fit,load_model
    from rgcbm.evaluate import ground_concepts,evaluate
    from rgcbm.visualize import visualize_run
    from rgcbm.reports import generate_reports
    manifest=build_manifest(root,args.dataset)
    assert not leakage_audit(root,manifest)['cross_split_exact_duplicates']
    cfg=Config(dataset=args.dataset,batch_size=2,head_epochs=1,finetune_epochs=1,max_train_batches=1,max_validation_batches=1,checkpoint_every_batches=1)
    name=f'offline_acceptance_{args.dataset}_{time.time_ns()}'
    model,run,status=fit(root,manifest,cfg,device=device,run_name=name)
    assert status['complete']
    history=json.loads((run/'history.json').read_text())
    assert [row['stage'] for row in history]==['head','finetune']
    assert all(row['classifier_l2_update']>0 for row in history)
    grounding=ground_concepts(model,root,manifest,run,max_batches=1)
    scores=evaluate(model,root,manifest,run,grounding,max_batches=1)
    assert scores['n_test_evaluated']==2
    visualize_run(model,root,manifest,run,grounding,examples=1,prototype_batches=1)
    records=generate_reports(run,root/'assets',backend='local_qwen' if args.llm else 'template',max_cases=1)
    report['report_backend']={k:records[0][k] for k in ['backend_requested','backend_used','fallback_reason']}
    checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    assert {'optimizer','scheduler','scaler','rng','progress'}.issubset(checkpoint)
    restored,_=load_model(run,root/'assets',device=device)
    image=ImageDataset(root,manifest,'test',cfg)[0][0][None].to(device)
    with torch.no_grad():torch.testing.assert_close(model(image)['logits'],restored(image)['logits'])
    report['checks']['real_image_training_explanations_checkpoint']='passed'
    report['run_directory']=str(run)
    from jupyter_client import KernelManager
    options={'kernel_name':args.kernel}
    if os.name!='nt':options.update(transport='ipc',ip=str(runtime/'acceptance-ipc'))
    manager=KernelManager(**options)
    client=None
    try:
        if Path(manager.kernel_spec.argv[0]).resolve()!=Path(sys.executable).resolve():
            raise RuntimeError('The registered Jupyter kernel points at another Python. Register it again with this environment before acceptance.')
        manager.start_kernel(cwd=str(root));client=manager.client();client.start_channels();client.wait_for_ready(timeout=90)
        msg=client.execute('import sys, torch; assert torch.isfinite(torch.ones(2, 2) @ torch.ones(2, 2)).all(); print(sys.executable)')
        while True:
            answer=client.get_shell_msg(timeout=90)
            if answer.get('parent_header',{}).get('msg_id')==msg:break
        if answer['content']['status']!='ok':raise RuntimeError('Registered Jupyter kernel failed its PyTorch operation: '+str(answer['content']))
        report['checks']['registered_jupyter_kernel']='passed'
    finally:
        if client is not None:client.stop_channels()
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    report['status']='passed'
except Exception as exc:
    report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
    raise
finally:
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (runtime/'offline_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Acceptance report:',output)
