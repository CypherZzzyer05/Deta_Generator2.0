"""Permanent merge-path regressions, using test-only spectra and real read locks."""
from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from deta_generator2 import pipeline
from deta_generator2.cli import initialize
from deta_generator2.config import load_config
from deta_generator2.merge import merge_shards
from deta_generator2.pipeline import checksum, generate_dataset, prepare_manifest
from deta_generator2.storage import pipeline_write_retries, retry_io
from test_generator import TestOnlySimulator


class MergeStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.job = Path(self.temp.name)/'job with spaces'
        initialize(self.job, 'au-metal', 4, 1, None, metal_mesh_nm=8)
        self.config = load_config(self.job/'job.json')
        self.sources = [self.job/f'shard{i}' for i in range(2)]
        with patch('deta_generator2.pipeline.FDTDSimulator', TestOnlySimulator):
            for i, source in enumerate(self.sources):
                generate_dataset(replace(self.config, output_dir=source), sample_ids=range(i,4,2))
        self.target = replace(self.config, output_dir=self.job/'merged')

    def tearDown(self):
        self.temp.cleanup()

    def test_merge_preserves_sources_and_can_repeat_without_solver(self):
        before = {p:checksum(p) for source in self.sources for p in source.rglob('*') if p.is_file()}
        with patch('deta_generator2.pipeline.FDTDSimulator', side_effect=AssertionError('No solver')):
            self.assertEqual(merge_shards(self.target, self.sources)['success'],4)
            self.assertEqual(merge_shards(self.target, self.sources)['success'],4)
        self.assertEqual({p:checksum(p) for p in before}, before)
        with np.load(self.target.output_dir/'dataset.npz') as data:
            self.assertEqual(data['sample_ids'].tolist(), [0,1,2,3])

    def assert_locked_merge_succeeds(self, name):
        if name == 'dataset.npz':
            merge_shards(self.target, self.sources)
        else:
            self.target.output_dir.mkdir(parents=True, exist_ok=True)
            with pipeline_write_retries():
                prepare_manifest(self.target)
        path = self.target.output_dir/name
        before = {p:checksum(p) for source in self.sources for p in source.glob('samples/*')}
        handle = path.open('rb')
        handle.read(1)
        retries = []
        real_sleep = __import__('time').sleep
        def release_then_sleep(delay):
            retries.append(delay)
            handle.close()
            real_sleep(.005)
        try:
            with patch('deta_generator2.storage.time.sleep', side_effect=release_then_sleep):
                result = merge_shards(self.target, self.sources)
            self.assertTrue(retries, 'A real Windows replacement conflict was not exercised')
            self.assertEqual(result['success'],4)
            self.assertEqual({p:checksum(p) for p in before},before)
            with np.load(self.target.output_dir/'dataset.npz') as data:
                self.assertEqual(data['spectra'].shape,(4,1000))
        finally:
            handle.close()

    @unittest.skipUnless(sys.platform == 'win32', 'Windows destination read handle')
    def test_locked_final_manifest_retries_on_actual_merge_path(self):
        self.assert_locked_merge_succeeds('manifest.json')

    @unittest.skipUnless(sys.platform == 'win32', 'Windows destination read handle')
    def test_locked_final_dataset_retries_on_actual_merge_path(self):
        self.assert_locked_merge_succeeds('dataset.npz')

    def test_retry_is_bounded_and_nested_contexts_do_not_multiply_retries(self):
        denied = PermissionError('test denial')
        denied.winerror = 5
        writer = Mock(side_effect=denied)
        original_npz = pipeline.atomic_npz
        with patch('deta_generator2.storage.sys.platform','win32'), \
             patch('deta_generator2.storage.time.sleep'), \
             patch('deta_generator2.pipeline.atomic_json', writer):
            with self.assertRaises(PermissionError):
                with pipeline_write_retries():
                    once = pipeline.atomic_json
                    with pipeline_write_retries():
                        self.assertIs(pipeline.atomic_json,once)
                        pipeline.atomic_json('path',{})
            self.assertEqual(writer.call_count,101)
            self.assertIs(pipeline.atomic_json,writer)
        self.assertIs(pipeline.atomic_npz,original_npz)

    def test_transient_errors_retry_and_other_errors_propagate(self):
        for code in (5,32,33):
            error = PermissionError('test transient sharing')
            error.winerror = code
            writer = Mock(side_effect=[error,'ok'])
            with patch('deta_generator2.storage.sys.platform','win32'), \
                 patch('deta_generator2.storage.time.sleep'):
                self.assertEqual(retry_io(writer)('path'),'ok')
                self.assertEqual(writer.call_count,2)
        error = PermissionError('test other failure')
        error.winerror = 1314
        writer = Mock(side_effect=error)
        with self.assertRaises(PermissionError): retry_io(writer)('path')
        self.assertEqual(writer.call_count,1)
        writer = Mock(side_effect=PermissionError('plain'))
        with patch('deta_generator2.storage.sys.platform','linux'):
            with self.assertRaises(PermissionError): retry_io(writer)('path')
        self.assertEqual(writer.call_count,1)


if __name__ == '__main__':
    unittest.main()
