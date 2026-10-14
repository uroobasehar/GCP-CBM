from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[1]
for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
    if not line.strip(): continue
    expected, relative = line.split('  ', 1)
    digest = hashlib.sha256()
    with (root / relative).open('rb') as source:
        while chunk := source.read(1024 * 1024): digest.update(chunk)
    if digest.hexdigest() != expected: raise RuntimeError('Checksum mismatch: ' + relative)
print('All included bundle files passed SHA-256 verification.')
