"""Drive persistence contract checks using the real storage implementation and fake transport."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drive_fixture import DriveFixture
from rgcbm.cloud import DriveStore


def validate(directory):
    base=Path(directory);base.mkdir(parents=True,exist_ok=True)
    service=DriveFixture(base/'remote')
    credentials=type('Credentials',(),{'valid':True})()
    with patch('googleapiclient.discovery.build',return_value=service):
        store=DriveStore(credentials=credentials)
    run=base/'first/results/example';run.mkdir(parents=True)
    for name in ['configuration.json','best.pt','last.pt']:(run/name).write_bytes(name.encode())
    store.sync_run(run)
    assert [event[1] for event in service.events if event[0]=='upload'][-1]=='last.pt'
    count=sum(event[0]=='upload' for event in service.events)
    store.sync_run(run)
    assert sum(event[0]=='upload' for event in service.events)==count,'Unchanged files uploaded again'
    restored=store.restore_run(base/'cold','example')
    assert (restored/'last.pt').read_bytes()==b'last.pt'
    (restored/'last.pt').write_bytes(b'newer-local-state')
    store.restore_run(base/'cold','example')
    assert (restored/'last.pt').read_bytes()==b'newer-local-state','Remote restore overwrote newer local progress'
    data=base/'first/data';data.mkdir()
    (data/'cub.zip').write_bytes(b'fixture-only-dataset')
    store.ensure_dataset(base/'first','cub')
    store.ensure_dataset(base/'cold','cub')
    assert (base/'cold/data/cub.zip').read_bytes()==b'fixture-only-dataset'
    assets=base/'first/assets';llm=assets/'qwen2.5-1.5b-instruct';llm.mkdir(parents=True)
    for name in ['efficientnet_b1_imagenet.pth','backbone_manifest.json','llm_manifest.json']:(assets/name).write_bytes(name.encode())
    for name in ['model.safetensors','config.json','tokenizer.json','LICENSE']:(llm/name).write_bytes(name.encode())
    store.sync_assets(base/'first',include_llm=True)
    store.restore_assets(base/'cold',include_llm=True)
    assert (base/'cold/assets/qwen2.5-1.5b-instruct/model.safetensors').read_bytes()==b'model.safetensors'
    assert not any('token' in str(path).lower() for path in run.iterdir())
    return {'status':'passed','checks':['resumable upload/download API contract','last checkpoint uploaded after companion state','unchanged upload skipped','cold-session checkpoint restore','newer local checkpoint preserved','existing local dataset persisted remotely','dataset restored','backbone and complete LLM asset restore'],'scope':'local fake Drive transport; no real Google authentication or Drive access'}


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='rgcbm-drive-',dir='/tmp') as directory:print(json.dumps(validate(directory),indent=2))
