"""Local Drive API simulation for contract tests; never claims live Google authorization."""
from pathlib import Path
import json
import os
import re
import shutil
import sys
import types


class Request:
    def __init__(self, action): self.action=action
    def execute(self): return self.action()
    def next_chunk(self, num_retries=0): return None,self.action()


class Transport:
    def __init__(self, service): self.service=service
    def request(self, uri, method='GET', body=None, headers=None, **kwargs):
        import httplib2
        path=self.service.directory/uri.split('://',1)[1]
        size=path.stat().st_size
        match=re.search(r'bytes=(\d+)-(\d+)',(headers or {}).get('range',''),re.I)
        begin,end=(int(match[1]),min(int(match[2]),size-1)) if match else (0,size-1)
        with path.open('rb') as f:f.seek(begin);data=f.read(end-begin+1)
        response=httplib2.Response({'status':'206','content-range':f'bytes {begin}-{end}/{size}','content-length':str(len(data))})
        return response,data


class DriveFixture:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.rows={};self.events=[];self.serial=0
    def files(self):return self
    def list(self,q,**kwargs):
        def action():
            name=re.search(r"name='((?:\\.|[^'])*)'",q)
            name=name[1].replace("\\'", "'").replace('\\\\','\\') if name else None
            parent=re.search(r"'([^']+)' in parents",q)
            folder="mimeType='application/vnd.google-apps.folder'" in q
            rows=[dict(r) for r in self.rows.values() if (name is None or r['name']==name) and (parent is None or parent[1] in r['parents']) and (not folder or r['mimeType']=='application/vnd.google-apps.folder')]
            return {'files':rows}
        return Request(action)
    def _save(self,body,media=None,key=None):
        if key is None:self.serial+=1;key=f'file{self.serial}'
        row=self.rows.get(key,{'id':key,'parents':['root'],'mimeType':'application/octet-stream'})
        row.update(body)
        if media is not None:
            target=self.directory/key
            if target.exists():target.unlink()
            os.link(media._filename,target)
            row['size']=str(target.stat().st_size)
        self.rows[key]=row;self.events.append(('upload' if media else 'folder',row.get('name'),key))
        return {'id':key}
    def create(self,body,fields=None,media_body=None):return Request(lambda:self._save(body,media_body))
    def update(self,fileId,media_body,fields=None):return Request(lambda:self._save({},media_body,fileId))
    def get(self,fileId,fields=None):return Request(lambda:dict(self.rows[fileId]))
    def get_media(self,fileId):
        self.events.append(('download',self.rows[fileId]['name'],fileId))
        return types.SimpleNamespace(http=Transport(self),uri='fixture://'+fileId,headers={})


def install_cloud_fixture(mode,root,source):
    """Replace only managed-host mount/auth and pip operations for local notebook execution."""
    root,source=Path(root),Path(source)
    for folder in ['data','assets']: (root/folder).mkdir(parents=True,exist_ok=True)
    for name in ['cub','pets','brain_mri']:
        (root/'data'/f'{name}.zip').symlink_to(source/'data'/f'{name}.zip')
        (root/'data'/name).symlink_to(source/'data'/name,target_is_directory=True)
    (root/'assets/efficientnet_b1_imagenet.pth').symlink_to(source/'assets/efficientnet_b1_imagenet.pth')
    os.environ['RGCBM_ROOT']=str(root)
    fixture=DriveFixture(root.parent/(mode.lower()+'-drive-api'))
    import subprocess
    original=subprocess.check_call
    def check(command,*args,**kwargs):
        if list(command[1:4])==['-m','pip','install']:
            fixture.events.append(('pip-simulated',None,None));return 0
        return original(command,*args,**kwargs)
    subprocess.check_call=check
    if mode=='Colab':
        drive_root=root.parent/'colab-mounted-drive';os.environ['RGCBM_DRIVE_ROOT']=str(drive_root)
        for folder in ['datasets','assets','results']:(drive_root/folder).mkdir(parents=True,exist_ok=True)
        (drive_root/'datasets/cub.zip').symlink_to(source/'data/cub.zip')
        colab=types.ModuleType('google.colab')
        colab.drive=types.SimpleNamespace(mount=lambda *a,**k:fixture.events.append(('mount-simulated',None,None)))
        sys.modules['google.colab']=colab
    else:
        import googleapiclient.discovery
        googleapiclient.discovery.build=lambda *a,**k:fixture
        os.environ['GDRIVE_TOKEN_JSON']=json.dumps({'token':'offline-fixture-token','refresh_token':'offline-fixture-refresh','token_uri':'https://oauth2.googleapis.com/token','client_id':'offline-fixture','client_secret':'offline-fixture','expiry':'2099-01-01T00:00:00Z'})
    return fixture
