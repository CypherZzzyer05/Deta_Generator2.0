"""Takeover integrity and dynamic single-sample scheduling; no native FDTD."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import textwrap
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import psutil

from deta_generator2.cli import initialize
from deta_generator2.config import load_config
from deta_generator2.pipeline import (atomic_json, checksum, directory_lock, generate_dataset,
                                     prepare_manifest)
from deta_generator2.tail import (controller_lock, inspect_remaining, live_workers, run_tail,
                                 stop_original, verified_controller)
from test_generator import TestOnlySimulator

ROOT = Path(__file__).resolve().parents[1]


class TailTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.job = self.root / 'job with spaces'
        initialize(self.job, 'au-metal', 12, 1, None, 8)
        self.config = load_config(self.job / 'job.json')
        atomic_json(self.job / 'parallel_plan.json', {'samples': 12, 'workers': 4})
        self.patch = patch('deta_generator2.pipeline.FDTDSimulator', TestOnlySimulator)
        TestOnlySimulator.fail = False
        self.patch.start()
        for i in range(4):
            # Four finished and eight remaining, all on the original fixed plan.
            generate_dataset(replace(self.config, output_dir=self.job / f'shard{i}'),
                             sample_ids=[i])

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def hashes(self):
        return {str(p.relative_to(self.job)): checksum(p) for p in self.job.glob('shard*/samples/*')}

    def child_environment(self, fail_id=None, barrier_jobs=0):
        site = self.root / 'site'
        site.mkdir(exist_ok=True)
        trace = self.root / 'trace'
        trace.mkdir(exist_ok=True)
        barrier = self.root / 'barrier'
        barrier.mkdir(exist_ok=True)
        fixture = '''
import json, os, pathlib, sys, time
import deta_generator2.pipeline as pipeline
from test_generator import TestOnlySimulator
class TimedSimulator(TestOnlySimulator):
    def run(self, geometry, project):
        sample = int(pathlib.Path(project).stem.split('_')[-1])
        start = time.time()
        if os.environ.get('TAIL_TEST_FAIL') == str(sample):
            raise RuntimeError('Test-only sample failure')
        expected = int(os.environ.get('TAIL_TEST_BARRIER','0'))
        if expected:
            barrier = pathlib.Path(os.environ['TAIL_TEST_TRACE']).parent/'barrier'
            (barrier/f'{os.getpid()}.ready').write_text('test-only')
            deadline = time.monotonic()+30
            while len(list(barrier.glob('*.ready'))) < expected:
                if time.monotonic()>deadline: raise RuntimeError('Test barrier timed out')
                time.sleep(.01)
        time.sleep(.08)
        result = super().run(geometry, project)
        out = pathlib.Path(os.environ['TAIL_TEST_TRACE'])/f'{sample}.json'
        out.write_text(json.dumps({'sample_id':sample,'start':start,'end':time.time(),
                                   'test_fixture_only':True}))
        return result
pipeline.FDTDSimulator = TimedSimulator
'''
        (site/'sitecustomize.py').write_text(
            'import os, traceback\ntry:\n'+textwrap.indent(fixture,'    ')+
            '\nexcept BaseException:\n    traceback.print_exc()\n    os._exit(97)\n',encoding='utf-8')
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([str(site), str(ROOT / 'src'), str(ROOT / 'tests')])
        env['TAIL_TEST_TRACE'] = str(trace)
        env['TAIL_TEST_FAIL'] = '' if fail_id is None else str(fail_id)
        env['TAIL_TEST_BARRIER'] = str(barrier_jobs)
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        return env, trace

    def test_preview_is_readonly_and_keeps_all_original_ids(self):
        before = {str(p): checksum(p) for p in self.job.rglob('*') if p.is_file()}
        result = inspect_remaining(self.job)
        self.assertEqual(result['manifest_success'], 4)
        self.assertEqual(result['remaining_ids'], list(range(4, 12)))
        after = {str(p): checksum(p) for p in self.job.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_refuses_running_workers_before_copy_or_launch(self):
        with patch('deta_generator2.tail.live_workers', return_value=[object()]), \
             patch('deta_generator2.tail.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'still alive'):
                run_tail(self.job, jobs=2, threads=1)
            spawn.assert_not_called()
        self.assertFalse((self.job / 'tail_resume').exists())

    def test_refuses_locked_source_shards(self):
        with directory_lock(self.job / 'shard2'):
            with self.assertRaisesRegex(RuntimeError, 'Another generator'):
                run_tail(self.job, jobs=2, threads=1)

    def test_refuses_corrupt_success_without_replacing_original(self):
        sample = self.job / 'shard0/samples/sample_000000.npz'
        sample.write_bytes(b'corrupt')
        with patch('deta_generator2.tail.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                run_tail(self.job, jobs=2, threads=1)
            spawn.assert_not_called()
        self.assertEqual(sample.read_bytes(), b'corrupt')

    def test_dynamic_four_processes_run_once_per_remaining_sample_and_merge(self):
        before = self.hashes()
        env, trace = self.child_environment(barrier_jobs=4)
        with patch.dict(os.environ, env, clear=True):
            result = run_tail(self.job, jobs=4, threads=1, poll_seconds=.03)
        self.assertEqual(result['success'], 12)
        rows = [json.loads(p.read_text()) for p in trace.glob('*.json')]
        self.assertEqual(sorted(row['sample_id'] for row in rows), list(range(4, 12)))
        events = [(r['start'], 1) for r in rows] + [(r['end'], -1) for r in rows]
        current = peak = 0
        for _, delta in sorted(events):
            current += delta
            peak = max(peak, current)
        self.assertEqual(peak, 4)  # Test simulator concurrency, never native FDTD.
        self.assertEqual(self.hashes(), before)
        with np.load(self.job / 'tail_resume/merged/dataset.npz', allow_pickle=False) as data:
            self.assertEqual(data['spectra'].shape, (12, 1000))
            self.assertEqual(data['geometry_values'].shape, (12, 7))
            np.testing.assert_array_equal(data['sample_ids'], np.arange(12))
        with patch('deta_generator2.tail.subprocess.Popen') as spawn:
            result = run_tail(self.job, jobs=4, threads=1)
            spawn.assert_not_called()
        self.assertEqual(result['success'], 12)

    def test_failed_single_sample_can_resume_without_source_or_plan_changes(self):
        before = self.hashes()
        env, trace = self.child_environment(fail_id=7)
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, '11/12'):
                run_tail(self.job, jobs=4, threads=1, poll_seconds=.03)
        result = inspect_remaining(self.job)
        self.assertEqual(result['remaining_ids'], [7])
        env, _ = self.child_environment()
        with patch.dict(os.environ, env, clear=True):
            summary = run_tail(self.job, jobs=4, threads=1, poll_seconds=.03)
        self.assertEqual(summary['success'], 12)
        self.assertEqual(self.hashes(), before)
        self.assertEqual(len(list(trace.glob('*.json'))), 8)

    def test_stale_running_manifest_does_not_count_as_saved_success(self):
        path = self.job / 'shard0/manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['samples'][4]['state'] = 'running'
        atomic_json(path, manifest)
        preview = inspect_remaining(self.job)
        self.assertIn(4, preview['remaining_ids'])
        self.assertEqual(preview['running_ids_without_saved_success'], [4])

    def test_stop_rejects_recycled_pid_and_never_terminates_it(self):
        atomic_json(self.job / 'parallel_workers.json', {'workers': [
            {'pid': 99, 'process_name': 'python', 'started_at_unix': 10}]})
        process = SimpleNamespace(pid=99, is_running=lambda: True, name=lambda: 'python.exe',
                                  create_time=lambda: 100, terminate=lambda: self.fail('Recycled PID terminated'))
        api = SimpleNamespace(Process=lambda pid: process, NoSuchProcess=psutil.NoSuchProcess,
                              AccessDenied=psutil.AccessDenied)
        self.assertEqual(live_workers(self.job, api), [])

    def test_controller_identity_requires_exact_module_and_task_directory(self):
        good = SimpleNamespace(name=lambda: 'powershell.exe', cmdline=lambda: [
            'powershell', '-File', 'C:/pkg/run-parallel.ps1', '-JobDir', str(self.job)])
        self.assertTrue(verified_controller(good, self.job))
        self.assertFalse(verified_controller(good, self.root / 'other'))
        python = SimpleNamespace(name=lambda: 'python.exe', cmdline=lambda: [
            'python', '-m', 'deta_generator2.tail', '--job-dir', str(self.job), '--action', 'run'])
        self.assertTrue(verified_controller(python, self.job))
        self.assertFalse(verified_controller(python, self.root / 'other'))

    @unittest.skipUnless(sys.platform == 'win32' and shutil.which('powershell'), 'Windows controller stop integration')
    def test_stop_only_exact_task_controller_and_children_preserves_unrelated_python(self):
        fixture = self.root / 'fake_worker.py'
        fixture.write_text('''
import json, os, pathlib, subprocess, sys, time, psutil
root=pathlib.Path(sys.argv[1])
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],creationflags=subprocess.CREATE_NO_WINDOW)
me=psutil.Process()
(root/'parallel_workers.json').write_text(json.dumps({'workers':[{'index':0,'pid':os.getpid(),
 'process_name':me.name(),'started_at_unix':me.create_time()}]}))
(root/'test_engine.json').write_text(json.dumps({'pid':child.pid}))
time.sleep(120)
''', encoding='utf-8')
        script = self.root / 'run-parallel.ps1'
        script.write_text('''param([string]$JobDir,[string]$Python,[string]$WorkerFile)
$line = '"' + $WorkerFile + '" "' + $JobDir + '"'
$child=Start-Process -FilePath $Python -ArgumentList $line -WindowStyle Hidden -PassThru
Wait-Process -Id $child.Id
''', encoding='utf-8')
        controller = subprocess.Popen(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
            '-File', str(script), '-JobDir', str(self.job), '-Python', sys.executable, '-WorkerFile', str(fixture)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        bystander = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(120)'],
                                    creationflags=subprocess.CREATE_NO_WINDOW)
        owned = []
        try:
            deadline = time.monotonic() + 20
            while not (self.job / 'test_engine.json').is_file():
                if time.monotonic() > deadline:
                    self.fail('Test worker did not start')
                time.sleep(.05)
            worker = json.loads((self.job / 'parallel_workers.json').read_text())['workers'][0]['pid']
            engine = json.loads((self.job / 'test_engine.json').read_text())['pid']
            owned = [worker, engine]
            result = stop_original(self.job)
            self.assertEqual(result['stopped_workers'], 1)
            self.assertEqual(result['stopped_controllers'], 1)
            controller.wait(timeout=10)
            self.assertIsNone(bystander.poll())
            self.assertEqual(live_workers(self.job), [])
            for pid in owned:
                self.assertFalse(psutil.pid_exists(pid))
        finally:
            for pid in [controller.pid, bystander.pid, *owned]:
                try:
                    psutil.Process(pid).kill()
                except psutil.NoSuchProcess:
                    pass
            bystander.wait(timeout=10)
            controller.wait(timeout=10)



if __name__ == '__main__':
    unittest.main()
