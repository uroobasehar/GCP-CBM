"""Explicit Google Drive persistence for Colab and Kaggle; never embeds credentials."""
from pathlib import Path
import io
import json
import os
from .data import atomic_json


class DriveStore:
    def __init__(self, folder_name='RGCBM_MultiDataset', token_info=None, credentials=None):
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
        self.folder = self._folder(folder_name, None)
        self.child_folders = {}

    def _list(self, query):
        rows, token = [], None
        while True:
            result = self.service.files().list(q=query, spaces='drive', fields='nextPageToken,files(id,name,modifiedTime,size)', pageToken=token, pageSize=1000).execute()
            rows.extend(result.get('files', []))
            token = result.get('nextPageToken')
            if not token: return rows

    @staticmethod
    def _quote(value):
        return str(value).replace('\\', '\\\\').replace("'", "\\'")

    def _folder(self, name, parent):
        query = f"name='{self._quote(name)}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent: query += f" and '{parent}' in parents"
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
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix != '.tmp':
                self.upload(path, str(Path('results') / run.name / path.relative_to(run).parent))
        error = run / 'remote_sync_error.json'
        if error.exists(): error.unlink()

    def restore_run(self, root, run_name):
        # Resume training artifacts before fit; arrays/figures can be regenerated.
        run = Path(root) / 'results' / run_name
        remote = str(Path('results') / run_name)
        for name in ['configuration.json', 'last.pt', 'best.pt', 'head_best.pt', 'finetune_best.pt', 'history.json', 'training_status.json']:
            self.download(name, run / name, remote)
        return run

    def ensure_dataset(self, root, name):
        archive = Path(root) / 'data' / f'{name}.zip'
        if not archive.exists():
            self.download(archive.name, archive, 'datasets')
        if not archive.exists():
            from .data import prepare_data
            prepare_data(root, name, online=True)
            self.upload(archive, 'datasets')
        return archive
