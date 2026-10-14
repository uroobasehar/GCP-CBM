"""Verify the Windows/Python-3.11 dependency closure and generate a hash-locked list."""
from pathlib import Path
import email
import hashlib
import json
import zipfile
from collections import defaultdict
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.specifiers import SpecifierSet
from packaging.version import Version

root = Path(__file__).resolve().parent
environment = {'implementation_name': 'cpython', 'implementation_version': '3.11.15', 'os_name': 'nt', 'platform_machine': 'AMD64', 'platform_release': '11', 'platform_system': 'Windows', 'platform_version': '10.0.26100', 'python_full_version': '3.11.15', 'platform_python_implementation': 'CPython', 'python_version': '3.11', 'sys_platform': 'win32', 'extra': ''}
runtime = json.loads((root / 'conda_manifest.json').read_text())
python_version = next(p['version'] for p in runtime['packages'] if p['name'] == 'python')
environment.update(implementation_version=python_version, python_full_version=python_version)
inventory = defaultdict(list)
for path in sorted((root / 'wheelhouse').glob('*.whl')):
    name, version, _, tags = parse_wheel_filename(path.name)
    with zipfile.ZipFile(path) as archive:
        metadata = email.message_from_bytes(archive.read(next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))))
        if archive.testzip() is not None: raise RuntimeError('Corrupt wheel: ' + path.name)
    if metadata['Requires-Python'] and not SpecifierSet(metadata['Requires-Python']).contains(python_version):
        continue
    if not any(t.platform in ['win_amd64', 'any'] and (t.interpreter.startswith('py3') or t.interpreter in ['cp311', 'cp310', 'cp39', 'cp38', 'cp37']) for t in tags):
        continue
    inventory[canonicalize_name(name)].append({'version': version, 'path': path, 'requirements': metadata.get_all('Requires-Dist') or []})
roots = [Requirement(line.strip()) for line in (root / 'requirements-win-py311.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
selected, extras = {}, defaultdict(set)
for _ in range(100):
    constraints = defaultdict(list)
    new_extras = defaultdict(set)
    for requirement in roots:
        name = canonicalize_name(requirement.name)
        constraints[name].append(requirement.specifier)
        new_extras[name].update(requirement.extras)
    for name, chosen in selected.items():
        for text in chosen['requirements']:
            requirement = Requirement(text)
            tests = extras[name] | {''}
            if requirement.marker and not any(requirement.marker.evaluate({**environment, 'extra': extra}) for extra in tests):
                continue
            target = canonicalize_name(requirement.name)
            constraints[target].append(requirement.specifier)
            new_extras[target].update(requirement.extras)
    new_selected = {}
    for name, specifications in constraints.items():
        allowed = [item for item in inventory[name] if all(spec.contains(item['version'], prereleases=True) for spec in specifications)]
        if not allowed: raise RuntimeError(f'Windows closure missing/incompatible: {name}, constraints={specifications}')
        new_selected[name] = max(allowed, key=lambda item: item['version'])
    stable = {k: v['version'] for k, v in selected.items()} == {k: v['version'] for k, v in new_selected.items()} and dict(extras) == dict(new_extras)
    selected, extras = new_selected, new_extras
    if stable: break
else: raise RuntimeError('Dependency selection did not converge.')
lines, records = [], []
for name, item in sorted(selected.items()):
    digest = hashlib.sha256()
    with item['path'].open('rb') as source:
        while chunk := source.read(1024 * 1024): digest.update(chunk)
    lines.append(f'{name}=={item["version"]} --hash=sha256:{digest.hexdigest()}')
    records.append({'name': name, 'version': str(item['version']), 'wheel': item['path'].name, 'sha256': digest.hexdigest(), 'extras_required': sorted(extras[name])})
(root / 'lock-win-py311.txt').write_text('\n'.join(lines) + '\n')
(root / 'wheel_validation.json').write_text(json.dumps({'status': 'passed', 'target': environment, 'selected_packages': len(records), 'packages': records, 'validation': 'wheel ZIP integrity, Python constraints, transitive dependencies and Windows markers checked; native Windows execution untested'}, indent=2))
print('Windows dependency closure passed:', len(records), 'hash-locked packages.')
