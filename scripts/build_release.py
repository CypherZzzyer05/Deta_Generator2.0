"""Build and verify the independent wheel without replacing installed v1."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

root = Path(__file__).resolve().parents[1]
artifacts = root/'artifacts'
artifacts.mkdir(exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                '.', '--wheel-dir', str(artifacts)], cwd=root, check=True)
wheel = artifacts/'deta_generator2-2.0.0-py3-none-any.whl'
installed = root/'tmp/wheel2'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--upgrade',
                '--target', str(installed), str(wheel)], check=True)
with zipfile.ZipFile(wheel) as archive:
    package_files = [name for name in archive.namelist()
                     if name.startswith('deta_generator2/') and not name.endswith('/')]
    for name in package_files:
        assert archive.read(name) == (installed/name).read_bytes(), name
        assert archive.read(name) == (root/'src'/name).read_bytes(), name
env = dict(os.environ, PYTHONPATH=str(installed), PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
def call(*arguments):
    output = subprocess.check_output([sys.executable, '-m', 'deta_generator2', *arguments],
                                      env=env, cwd=root, text=True, encoding='utf-8')
    print(output)
    return output
assert call('--version').strip() == '2.0.0'
job = root/'tmp/installed-wheel-job'
if not (job/'job.json').exists():
    call('init', '--job-dir', str(job), '--samples', '1', '--threads', '8')
config = json.loads((job/'job.json').read_text(encoding='utf-8-sig'))
assert config['simulation']['metal_position'] == 'above_substrate'
assert config['simulation']['metal_thickness_nm'] == 100
assert config['simulation']['metal_mesh_nm'] == 8
assert config['simulation']['mesh_policy'] == 'metal_only'
call('plan', '--config', str(job/'job.json'))
report = dict(version='2.0.0', distribution='deta-generator2', module='deta_generator2',
              wheel=wheel.name, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
              package_files_verified=len(package_files), installed_initialization_verified=True,
              physical_stack_bottom_to_top=['SiO2', 'Au', 'As2Se3 tetramer'],
              original_package_replaced=False)
(artifacts/'checksums.json').write_bytes((json.dumps(report, indent=2)+'\n').encode('utf-8'))
print(json.dumps(report, indent=2))
