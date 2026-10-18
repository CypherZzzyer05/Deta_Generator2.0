"""Windows launcher integration tests; simulated spectra are never research data."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which('powershell') or shutil.which('pwsh')


@unittest.skipUnless(sys.platform == 'win32' and POWERSHELL, 'Windows PowerShell integration')
class ParallelLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.job = self.root/'job with spaces'
        site = self.root/'custom_site'
        site.mkdir()
        self.trace = self.root/'trace'
        self.trace.mkdir()
        self.sample_trace = self.root/'sample_trace'
        self.sample_trace.mkdir()
        # Each child uses the actual CLI, packing and merge implementations.
        # Only the native simulator is replaced by the existing test fixture.
        (site/'sitecustomize.py').write_text('''
import json, os, pathlib, sys, time
import psutil
if os.environ.get('PARALLEL_TEST_CPUS') == '64':
    # Model the user's 64-CPU host only inside these isolated child fixtures.
    # The production CPU-budget check remains unchanged.
    psutil.cpu_count = lambda logical=True: 64
    import deta_generator2.resources as resources
    resources.cpu_inventory = lambda: dict(budget_cpus=64,psutil_logical_cpus=64,
        windows_total_logical_cpus=64,processor_groups=1,affinity_count=64,notes=[])
from test_generator import TestOnlySimulator
import deta_generator2.pipeline as pipeline
import deta_generator2.cli as cli
def test_native_check(config, preflight=False):
    # Test the entry sequencing, not native layout accuracy or FDTD physics.
    assert preflight and config.simulation.metal_mesh_nm == 8
    (pathlib.Path(os.environ['PARALLEL_TEST_TRACE']).parent/'preflight_called').write_text('test fixture only')
    return {'solver_run_performed':False,'test_fixture_only':True}
cli.native_check = test_native_check
class TimedTestSimulator(TestOnlySimulator):
    def __enter__(self):
        self.start = time.time()
        self.run_calls = 0
        return self
    def __exit__(self, *args):
        trace = pathlib.Path(os.environ['PARALLEL_TEST_TRACE'])/f'{os.getpid()}.json'
        trace.write_text(json.dumps({'started':self.start,'ended':time.time()}))
    def run(self, geometry, project):
        if os.environ.get('PARALLEL_TEST_FAIL_SHARD') and '--shard-index' in sys.argv:
            index = sys.argv[sys.argv.index('--shard-index')+1]
            if index == os.environ['PARALLEL_TEST_FAIL_SHARD']:
                raise RuntimeError('Deliberate test-only shard failure')
        self.run_calls += 1
        started = time.time()
        try:
            expected = int(os.environ.get('PARALLEL_TEST_BARRIER_JOBS', '0'))
            if expected and self.run_calls == 1:
                barrier = pathlib.Path(os.environ['PARALLEL_TEST_BARRIER_DIR'])
                (barrier/f'{os.getpid()}.ready').write_text('test fixture only')
                deadline = time.monotonic() + 45
                while len(list(barrier.glob('*.ready'))) < expected:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'Test barrier timed out: expected {expected} simultaneous sample.run calls')
                    time.sleep(.02)
            time.sleep(.02)
            return super().run(geometry, project)
        finally:
            trace = pathlib.Path(os.environ['PARALLEL_TEST_RUN_TRACE'])/f'{os.getpid()}_{self.run_calls}.json'
            trace.write_text(json.dumps({'started':started,'ended':time.time(),
                                         'pid':os.getpid(),'sample':pathlib.Path(project).stem,
                                         'test_fixture_only':True}))
pipeline.FDTDSimulator = TimedTestSimulator
''',encoding='utf-8')
        self.env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(site),str(ROOT/'src'),str(ROOT/'tests')]),
                        PARALLEL_TEST_TRACE=str(self.trace), PARALLEL_TEST_RUN_TRACE=str(self.sample_trace),
                        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        NUMEXPR_NUM_THREADS='1')

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, command, samples=100, env=None):
        return subprocess.run([POWERSHELL,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/run-parallel.ps1'),
            '-Command',command,'-Python',sys.executable,'-JobDir',str(self.job),'-Samples',str(samples),
            '-Workers','4','-Concurrency','2','-Threads','1'],env=env or self.env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=120)

    def test_prepare_100_then_parallel_run_merge_and_resume(self):
        prepared = self.invoke('prepare')
        self.assertEqual(prepared.returncode,0,prepared.stdout)
        self.assertFalse(list(self.trace.iterdir()))
        run = self.invoke('run')
        self.assertEqual(run.returncode,0,run.stdout)
        with np.load(self.job/'merged/dataset.npz') as data:
            self.assertEqual(data['spectra'].shape,(100,1000))
            self.assertEqual(data['geometry_values'].shape,(100,7))
            self.assertEqual(data['sample_ids'].tolist(),list(range(100)))
            self.assertEqual([len(data[name+'_indices']) for name in ('train','val','test')],[80,10,10])
        ids = []
        for index in range(4):
            manifest = json.loads((self.job/f'shard{index}/manifest.json').read_text())
            selected = [row['id'] for row in manifest['samples'] if row['state']=='success']
            self.assertEqual(selected,list(range(index,100,4)))
            ids += selected
        self.assertEqual(sorted(ids),list(range(100)))
        intervals = [json.loads(path.read_text()) for path in self.trace.iterdir()]
        self.assertEqual(len(intervals),4)
        events = sorted([(row['started'],1) for row in intervals]+[(row['ended'],-1) for row in intervals])
        active = peak = 0
        for _,change in events:
            active += change
            peak = max(peak,active)
        self.assertLessEqual(peak,2)
        with np.load(self.job/'merged/dataset.npz') as dataset:
            previous = {key:dataset[key].copy() for key in dataset.files if key!='metadata_json'}
        original_samples = {(index,path.name):path.read_bytes() for index in range(4)
                            for path in (self.job/f'shard{index}/samples').glob('*.npz')}
        resumed = self.invoke('run')
        self.assertEqual(resumed.returncode,0,resumed.stdout)
        self.assertEqual(len(list(self.trace.iterdir())),4)
        with np.load(self.job/'merged/dataset.npz') as dataset:
            for key,values in previous.items(): np.testing.assert_array_equal(dataset[key],values)
        for (index,name),original in original_samples.items():
            self.assertEqual((self.job/f'shard{index}/samples'/name).read_bytes(),original)

    def test_failed_worker_preserves_other_shards_then_can_resume(self):
        prepared = self.invoke('prepare',samples=8)
        self.assertEqual(prepared.returncode,0,prepared.stdout)
        failed = self.invoke('run',samples=8,env=dict(self.env,PARALLEL_TEST_FAIL_SHARD='1'))
        self.assertNotEqual(failed.returncode,0)
        self.assertFalse((self.job/'merged/dataset.npz').exists())
        self.assertTrue((self.job/'shard0/samples/sample_000000.npz').exists())
        resumed = self.invoke('run',samples=8)
        self.assertEqual(resumed.returncode,0,resumed.stdout)
        with np.load(self.job/'merged/dataset.npz') as data:
            self.assertEqual(data['sample_ids'].tolist(),list(range(8)))

    def test_existing_task_rejects_changed_sample_count(self):
        prepared = self.invoke('prepare',samples=8)
        self.assertEqual(prepared.returncode,0,prepared.stdout)
        original = (self.job/'job.json').read_bytes()
        changed = self.invoke('prepare',samples=100)
        self.assertNotEqual(changed.returncode,0)
        self.assertEqual((self.job/'job.json').read_bytes(),original)

    def test_pycharm_module_start_preflights_generates_and_merges(self):
        command = [sys.executable,'-m','deta_generator2','parallel','--job-dir',str(self.job),
                   '--samples','8','--workers','4','--concurrency','2','--threads','1']
        result = subprocess.run(command,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                text=True,encoding='utf-8',errors='replace',timeout=120)
        self.assertEqual(result.returncode,0,result.stdout)
        self.assertTrue((self.root/'preflight_called').exists())
        self.assertIn('Starting CPU dataset generation',result.stdout)
        self.assertIn('success=8/8',result.stdout)
        with np.load(self.job/'merged/dataset.npz') as data:
            self.assertEqual(data['sample_ids'].tolist(),list(range(8)))
        status = subprocess.run(command+['--parallel-action','status'],env=self.env,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=30)
        self.assertEqual(status.returncode,0,status.stdout)
        self.assertIn('success=8/8',status.stdout)

    def test_jobs_four_and_ten_reach_requested_sample_run_overlap(self):
        # A session-lifetime upper bound also passes a wholly serial launcher.
        # This barrier instead requires N actual sample.run calls to enter
        # before any of those test-only calls can finish.
        for jobs,threads in ((4,16),(10,6)):
            with self.subTest(jobs=jobs):
                job = self.root/f'jobs {jobs}'
                barrier = self.root/f'barrier_{jobs}'
                barrier.mkdir()
                run_trace = self.root/f'sample_trace_{jobs}'
                run_trace.mkdir()
                env = dict(self.env, PARALLEL_TEST_CPUS='64',
                           PARALLEL_TEST_BARRIER_JOBS=str(jobs),
                           PARALLEL_TEST_BARRIER_DIR=str(barrier),
                           PARALLEL_TEST_RUN_TRACE=str(run_trace))
                command = [sys.executable,'-m','deta_generator2','parallel','--job-dir',str(job),
                           '--samples',str(jobs),'--workers',str(jobs),'--jobs',str(jobs)]
                result = subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                        text=True,encoding='utf-8',errors='replace',timeout=120)
                failure_logs = '\n'.join(f'{path.name}:\n{path.read_text(encoding="utf-8",errors="replace")[-5000:]}'
                                         for path in (job/'parallel_logs').glob('*.stderr.log')) if result.returncode else ''
                self.assertEqual(result.returncode,0,result.stdout+'\n'+failure_logs)
                self.assertIn('native_engines=0',result.stdout,
                              'The fake simulator must not be reported as a native FDTD engine')
                intervals = [json.loads(path.read_text()) for path in run_trace.glob('*.json')]
                self.assertEqual(len(intervals),jobs,result.stdout)
                self.assertEqual(len({row['pid'] for row in intervals}),jobs)
                self.assertTrue(all(row['test_fixture_only'] for row in intervals))
                events = sorted([(row['started'],1) for row in intervals]+
                                [(row['ended'],-1) for row in intervals])
                active = peak = 0
                for _,change in events:
                    active += change
                    peak = max(peak,active)
                self.assertEqual(peak,jobs,'Requested sample.run concurrency was never reached')
                self.assertGreater(min(row['ended'] for row in intervals),
                                   max(row['started'] for row in intervals))
                config = json.loads((job/'job.json').read_text())
                self.assertEqual(config['simulation']['threads'],threads)
                self.assertEqual(config['simulation']['processes'],1)
                plan = json.loads((job/'parallel_plan.json').read_text(encoding='utf-8-sig'))
                self.assertEqual(plan['workers'],jobs)
                with np.load(job/'merged/dataset.npz') as data:
                    self.assertEqual(data['sample_ids'].tolist(),list(range(jobs)))


if __name__ == '__main__': unittest.main()
