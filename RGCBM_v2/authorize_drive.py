"""Run on an ONLINE computer with a browser; save token securely, never print it."""
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('client_json', help='Google Cloud OAuth Desktop application client JSON')
parser.add_argument('--output', default='private_drive_token.json')
parser.add_argument('--scope', choices=['file', 'drive'], default='file', help='Use drive to access the existing Colab-created project folder; file limits access to files created by this OAuth application.')
args = parser.parse_args()
target = Path(args.output)
if target.exists(): raise FileExistsError('Refusing to overwrite an existing OAuth token file.')
scope = 'https://www.googleapis.com/auth/drive.file' if args.scope == 'file' else 'https://www.googleapis.com/auth/drive'
flow = InstalledAppFlow.from_client_secrets_file(args.client_json, scopes=[scope])
credentials = flow.run_local_server(port=0, access_type='offline', prompt='consent')
target.write_text(credentials.to_json(), encoding='utf-8')
try: target.chmod(0o600)
except OSError: pass
print('Authorization saved locally. Put the JSON in Kaggle secret GDRIVE_TOKEN_JSON; never commit or share it.')
