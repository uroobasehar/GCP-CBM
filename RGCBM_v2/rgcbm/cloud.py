"""Explicit Google Drive persistence for Colab and Kaggle; never embeds credentials."""
from pathlib import Path
import io
import json
import os
from .data import atomic_json


class DriveStore:
    def __init__(self, folder_name='RGCBM_MultiDataset', token_info=None, credentials=None, folder_id=None):
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        if credentials is None:
            if token_info is None:
                token_info = os.getenv('GDRIVE_TOKEN_JSON')
                if token_info is None:
                    try:
                        from kaggle_secrets import UserSecretsClient
                        token_info = UserSecretsClient().get_secret('GDRIVE_TOKEN_JSON')
                    except Exception:
                        raise RuntimeError('Add the OAuth token JSON securely as Kaggle secret GDRIVE_TOKEN_JSON. See docs/CLOUD_STORAGE.md.') from None
            credentials = Credentials.from_authorized_user_info(json.loads(token_info) if isinstance(token_info, str) else token_info)
        if not credentials.valid:
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                raise RuntimeError('Google Drive credentials cannot refresh. Reauthorize using authorize_drive.py.')
        self.service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
        folder_id = folder_id or os.getenv('RGCBM_DRIVE_FOLDER_ID')
        if folder_id:
            info = self.service.files().get(fileId=folder_id, fields='id,mimeType,trashed').execute()
            if info.get('trashed') or info.get('mimeType') != 'application/vnd.google-apps.folder':
                raise RuntimeError('RGCBM_DRIVE_FOLDER_ID must identify an accessible, active Drive folder.')
            self.folder = info['id']
        else:
            self.folder = self._folder(folder_name, None)
        self.child_folders = {}
        self._uploaded = {}

    def _list(self, query):
        rows, token = [], None
        while True:
            result = self.service.files().list(q=query, spaces='drive', fields='nextPageToken,files(id,name,mimeType,modifiedTime,size)', pageToken=token, pageSize=1000).execute()
            rows.extend(result.get('files', []))
            token = result.get('nextPageToken')
            if not token: return rows

    @staticmethod
    def _quote(value):
        return str(value).replace('\\', '\\\\').replace("'", "\\'")

    def _folder(self, name, parent):
        query = f"name='{self._quote(name)}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent: query += f" and '{parent}' in parents"
        else: query += " and 'root' in parents"
        found = self._list(query)
        if len(found) > 1:
            raise RuntimeError(f'Multiple Drive folders named {name}; resolve the ambiguity in Drive.')
        if found: return found[0]['id']
        metadata = {'name': name, 'mimeType': 'application/vnd.google-apps.folder'}
        if parent: metadata['parents'] = [parent]
        return self.service.files().create(body=metadata, fields='id').execute()['id']

    def folder_for(self, relative):
        parent = self.folder
        accumulated = ''
        for part in Path(relative).parts:
            accumulated += '/' + part
            if accumulated not in self.child_folders:
                self.child_folders[accumulated] = self._folder(part, parent)
            parent = self.child_folders[accumulated]
        return parent

    def upload(self, path, relative_folder=''):
        from googleapiclient.http import MediaFileUpload
        path = Path(path)
        cache_key = (str(path.resolve()), str(relative_folder))
        stamp = (path.stat().st_size, path.stat().st_mtime_ns)
        previous = self._uploaded.get(cache_key)
        if previous and previous[:2] == stamp:
            return previous[2]
        parent = self.folder_for(relative_folder)
        found = self._list(f"name='{self._quote(path.name)}' and '{parent}' in parents and trashed=false")
        if len(found) > 1: raise RuntimeError('Ambiguous Drive file name.')
        media = MediaFileUpload(str(path), resumable=True, chunksize=8 * 1024 * 1024)
        if found:
            request = self.service.files().update(fileId=found[0]['id'], media_body=media, fields='id')
        else:
            request = self.service.files().create(body={'name': path.name, 'parents': [parent]}, media_body=media, fields='id')
        response = None
        while response is None:
            _, response = request.next_chunk(num_retries=3)
        if (path.stat().st_size, path.stat().st_mtime_ns) != stamp:
            raise RuntimeError('File changed while uploading; retry sync after the local save completes.')
        self._uploaded[cache_key] = (*stamp, response['id'])
        return response['id']

    def download(self, name, target, relative_folder=''):
        from googleapiclient.http import MediaIoBaseDownload
        parent = self.folder_for(relative_folder)
        found = self._list(f"name='{self._quote(name)}' and '{parent}' in parents and trashed=false")
        if not found: return False
        if len(found) > 1: raise RuntimeError('Ambiguous Drive file name.')
        target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + '.partial')
        with partial.open('wb') as output:
            downloader = MediaIoBaseDownload(output, self.service.files().get_media(fileId=found[0]['id']), chunksize=8 * 1024 * 1024)
            done = False
            while not done: _, done = downloader.next_chunk(num_retries=3)
        os.replace(partial, target)
        return True

    def sync_run(self, run):
        run = Path(run)
        # Upload the resume commit after all companion state, never before it.
        for path in sorted(run.rglob('*'), key=lambda p: (p.name == 'last.pt', str(p))):
            if path.is_file() and path.suffix != '.tmp':
                if path.name == 'remote_sync_error.json': continue
                self.upload(path, str(Path('results') / run.name / path.relative_to(run).parent))
        error = run / 'remote_sync_error.json'
        if error.exists(): error.unlink()

    def restore_run(self, root, run_name):
        # Resume training artifacts before fit; arrays/figures can be regenerated.
        run = Path(root) / 'results' / run_name
        if (run / 'last.pt').is_file():
            if not (run / 'configuration.json').is_file():
                raise RuntimeError('Local checkpoint has no configuration. Preserve it and recover its matching configuration before restoring.')
            # A failed remote sync must never roll newer local progress backwards.
            return run
        remote = str(Path('results') / run_name)
        for name in ['configuration.json', 'best.pt', 'best_any.pt', 'head_best.pt', 'finetune_best.pt', 'history.json', 'training_status.json', 'last.pt']:
            self.download(name, run / name, remote)
        return run

    def ensure_dataset(self, root, name):
        archive = Path(root) / 'data' / f'{name}.zip'
        parent = self.folder_for('datasets')
        found = self._list(f"name='{self._quote(archive.name)}' and '{parent}' in parents and trashed=false")
        if len(found) > 1: raise RuntimeError('Ambiguous dataset archive in Drive.')
        if not archive.exists() and found:
            self.download(archive.name, archive, 'datasets')
        if not archive.exists():
            from .data import prepare_data
            prepare_data(root, name, online=True)
        if not found:
            self.upload(archive, 'datasets')
        return archive

    def sync_assets(self, root, include_llm=False):
        assets = Path(root) / 'assets'
        for name in ['efficientnet_b1_imagenet.pth', 'backbone_manifest.json']:
            self.upload(assets / name, 'assets')
        if include_llm:
            self.upload(assets / 'llm_manifest.json', 'assets')
            for path in sorted((assets / 'qwen2.5-1.5b-instruct').iterdir()):
                if path.is_file() and (path.suffix in ['.json', '.txt', '.model', '.safetensors', '.md'] or path.name.startswith('LICENSE')):
                    self.upload(path, 'assets/qwen2.5-1.5b-instruct')

    def restore_assets(self, root, include_llm=False):
        assets = Path(root) / 'assets'
        for name in ['efficientnet_b1_imagenet.pth', 'backbone_manifest.json'] + (['llm_manifest.json'] if include_llm else []):
            if not (assets / name).is_file(): self.download(name, assets / name, 'assets')
        if include_llm:
            folder = 'assets/qwen2.5-1.5b-instruct'
            for item in self._list(f"'{self.folder_for(folder)}' in parents and trashed=false"):
                name = item['name']
                if Path(name).name != name or '/' in name or '\\' in name:
                    raise RuntimeError('Unsafe Drive asset filename.')
                if item.get('mimeType') == 'application/vnd.google-apps.folder': continue
                if Path(name).suffix not in ['.json', '.txt', '.model', '.safetensors', '.md'] and not name.startswith('LICENSE'): continue
                target = assets / 'qwen2.5-1.5b-instruct' / name
                if not target.is_file(): self.download(name, target, folder)
