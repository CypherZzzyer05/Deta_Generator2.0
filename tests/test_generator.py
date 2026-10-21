"""Integrity and portable-job tests. Fake spectra exist only inside these tests."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from deta_generator2.cli import initialize, main
from deta_generator2.config import Geometry, SimulationSettings, load_config
from deta_generator2.lumerical import FDTDSimulator, mesh_grid_summary, mesh_audit, validate_spectrum, stack_interfaces_nm
from deta_generator2.pipeline import generate_dataset, prepare_manifest, directory_lock, checksum, atomic_npz, atomic_json
from deta_generator2.merge import merge_shards


class TestOnlySimulator:
    calls = 0
    version = 'test-only'
    fail = False
    def __init__(self, settings, lumapi_path):
        self.settings = settings
        self.lumapi = SimpleNamespace(__file__='test-only')
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def material_identity(self):
        return {'test-only': {'fitted_index_ri': [[1., 0.]], 'accuracy_certified': False}}
    def run(self, geometry, project):
        type(self).calls += 1
        if type(self).fail: raise RuntimeError('deliberate test failure')
        grids = {axis: np.array([0., 3., 6.])*1e-9 for axis in ('x', 'y', 'z')}
        if self.settings.mesh_policy == 'metal_only':
            top = stack_interfaces_nm(self.settings)['metal_max']
            bottom = top-self.settings.metal_thickness_nm
            cells = int(np.ceil(self.settings.metal_thickness_nm/self.settings.metal_mesh_nm))
            grids = {'x': np.array([0., 50., 100.])*1e-9, 'y': np.array([0., 50., 100.])*1e-9,
                     'z': np.r_[bottom-50, np.linspace(bottom, top, cells+1), top+50]*1e-9}
        return {'wavelength_nm': self.settings.wavelength_grid(), 'reflectance': np.full(1000, .2),
                'transmittance': np.full(1000, .7), 'absorptance': np.full(1000, .1),
                'status': 2, 'elapsed_seconds': .1, 'fdtd_version': self.version,
                'native_layout_verified': True,
                **{f'mesh_{axis}_nm': values*1e9 for axis, values in grids.items()},
                'mesh_audit_json': json.dumps(mesh_audit(grids, self.settings))}


class PortableGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        initialize(self.root/'job', 'au-4', 4, 8, None)
        self.config = load_config(self.root/'job/job.json')
        TestOnlySimulator.calls = 0
        TestOnlySimulator.fail = False
        self.patch = patch('deta_generator2.pipeline.FDTDSimulator', TestOnlySimulator)
        self.patch.start()
    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_runtime_changes_preserve_identity(self):
        alternate = replace(self.config, output_dir=self.root/'elsewhere', lumapi_path='X:/lumapi.py',
                            simulation=replace(self.config.simulation, threads=32, processes=2, save_projects=False))
        self.assertEqual(self.config.metadata(), alternate.metadata())
        self.assertNotIn('output_dir', self.config.metadata())
        self.assertNotIn('lumapi_path', self.config.metadata())

    def test_material_location_changes_preserve_identity_but_content_changes_do_not(self):
        csv = self.root/'copied.csv'
        csv.write_bytes(Path(self.config.simulation.resonator_nk_csv).read_bytes())
        alternate = replace(self.config, simulation=replace(self.config.simulation, resonator_nk_csv=str(csv)))
        self.assertEqual(self.config.metadata(), alternate.metadata())
        csv.write_bytes(csv.read_bytes()+b'\n')
        self.assertNotEqual(self.config.metadata(), alternate.metadata())

    def test_mesh_and_layer_change_identity(self):
        fine8 = replace(self.config, simulation=replace(self.config.simulation, mesh_nm=8, maximum_mesh_nm=8))
        bare = replace(self.config, simulation=replace(self.config.simulation, remove_metal=True))
        self.assertNotEqual(self.config.metadata(), fine8.metadata())
        self.assertNotEqual(self.config.metadata(), bare.metadata())

    def test_resume_on_changed_machine_reuses_samples(self):
        generate_dataset(self.config, limit=1)
        changed = replace(self.config, lumapi_path='X:/lumapi.py', simulation=replace(self.config.simulation, threads=32))
        result = generate_dataset(changed)
        self.assertEqual(result['success'], 4)
        self.assertEqual(TestOnlySimulator.calls, 4)
        generate_dataset(changed)
        self.assertEqual(TestOnlySimulator.calls, 4)

    def test_output_relocation_can_resume(self):
        import shutil
        generate_dataset(self.config, limit=1)
        copied = self.root/'new-data'
        shutil.copytree(self.config.output_dir, copied)
        result = generate_dataset(replace(self.config, output_dir=copied))
        self.assertEqual(result['success'], 4)
        self.assertEqual(TestOnlySimulator.calls, 4)

    def test_corrupt_success_is_recomputed(self):
        generate_dataset(self.config)
        path = self.config.output_dir/'samples/sample_000000.npz'
        path.write_bytes(b'corrupt')
        result = generate_dataset(self.config)
        self.assertEqual(result['success'], 4)
        self.assertEqual(TestOnlySimulator.calls, 5)

    def test_failed_sample_is_excluded_and_can_resume(self):
        TestOnlySimulator.fail = True
        result = generate_dataset(self.config, limit=1)
        self.assertEqual(result['success'], 0)
        self.assertEqual(result['failed'], 1)
        TestOnlySimulator.fail = False
        result = generate_dataset(self.config)
        self.assertEqual(result['success'], 4)

    def test_different_physics_cannot_resume(self):
        generate_dataset(self.config, limit=1)
        altered = replace(self.config, simulation=replace(self.config.simulation, substrate_depth_nm=400))
        with self.assertRaisesRegex(ValueError, 'different configuration'):
            generate_dataset(altered)

    def test_150nm_samples_cannot_resume_as_100nm(self):
        legacy = replace(self.config, simulation=replace(self.config.simulation, metal_thickness_nm=150))
        generate_dataset(legacy, limit=1)
        sample = self.config.output_dir/'samples/sample_000000.npz'
        previous_checksum = checksum(sample)
        with self.assertRaisesRegex(ValueError, 'different configuration'):
            generate_dataset(self.config)
        self.assertEqual(TestOnlySimulator.calls, 1)
        self.assertEqual(checksum(sample), previous_checksum)

    def test_fixed_plan_detects_modified_geometry(self):
        with directory_lock(self.config.output_dir):
            prepare_manifest(self.config)
        path = self.config.output_dir/'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['samples'][0]['geometry_values'][0] += 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Sampling plan'):
            generate_dataset(self.config)

    def test_solver_version_change_is_rejected(self):
        generate_dataset(self.config, limit=1)
        with patch.object(TestOnlySimulator, 'version', 'changed'):
            with self.assertRaisesRegex(ValueError, 'Solver version changed'):
                generate_dataset(self.config)

    def test_actual_fit_change_is_rejected(self):
        generate_dataset(self.config, limit=1)
        changed = {'test-only': {'fitted_index_ri': [[1.1, 0.]]}}
        with patch.object(TestOnlySimulator, 'material_identity', return_value=changed):
            with self.assertRaisesRegex(ValueError, 'Actual material fits changed'):
                generate_dataset(self.config)

    def test_shard_merge_preserves_all_ids_labels_and_splits(self):
        a = replace(self.config, output_dir=self.root/'a')
        b = replace(self.config, output_dir=self.root/'b', simulation=replace(self.config.simulation, threads=16))
        generate_dataset(a, sample_ids=(0, 2))
        generate_dataset(b, sample_ids=(1, 3))
        result = merge_shards(self.config, [a.output_dir, b.output_dir])
        self.assertEqual(result['success'], 4)
        self.assertEqual(TestOnlySimulator.calls, 4)
        with np.load(self.config.output_dir/'dataset.npz') as dataset:
            self.assertEqual(dataset['spectra'].shape, (4, 1000))
            self.assertEqual(dataset['geometry_values'].shape, (4, 7))
            self.assertEqual(dataset['parameter_units'].tolist(), ['nm']*5+['deg']*2)
            self.assertEqual(dataset['sample_ids'].tolist(), [0, 1, 2, 3])
        merge_shards(self.config, [a.output_dir, b.output_dir])
        self.assertTrue((a.output_dir/'samples/sample_000000.npz').is_file())

    def test_merge_rejects_mixed_mesh(self):
        a = replace(self.config, output_dir=self.root/'a')
        b = replace(self.config, output_dir=self.root/'b', simulation=replace(self.config.simulation, mesh_nm=8, maximum_mesh_nm=8))
        generate_dataset(a, sample_ids=(0,))
        generate_dataset(b, sample_ids=(1,))
        with self.assertRaisesRegex(ValueError, 'physics/sampling'):
            merge_shards(self.config, [a.output_dir, b.output_dir])

    def test_merge_rejects_100nm_and_150nm_gold(self):
        current = replace(self.config, output_dir=self.root/'gold100')
        legacy = replace(self.config, output_dir=self.root/'gold150',
                         simulation=replace(self.config.simulation, metal_thickness_nm=150))
        generate_dataset(current, sample_ids=(0,))
        generate_dataset(legacy, sample_ids=(1,))
        with self.assertRaisesRegex(ValueError, 'physics/sampling'):
            merge_shards(self.config, [current.output_dir, legacy.output_dir])
        self.assertTrue((legacy.output_dir/'samples/sample_000001.npz').is_file())

    def test_merge_rejects_changed_actual_grid_even_with_updated_checksum(self):
        a = replace(self.config, output_dir=self.root/'a')
        generate_dataset(a, sample_ids=(0,))
        path = a.output_dir/'samples/sample_000000.npz'
        with np.load(path) as z: values = {key: z[key] for key in z.files}
        values['mesh_z_nm'] = np.array([0., 10.])
        atomic_npz(path, **values)
        manifest = json.loads((a.output_dir/'manifest.json').read_text())
        manifest['samples'][0]['sha256'] = checksum(path)
        atomic_json(a.output_dir/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'mesh exceeds'):
            merge_shards(self.config, [a.output_dir])

    def test_merge_rejects_conflicting_duplicate_sample(self):
        a = replace(self.config, output_dir=self.root/'a')
        b = replace(self.config, output_dir=self.root/'b')
        generate_dataset(a, sample_ids=(0,))
        generate_dataset(b, sample_ids=(0,))
        path = b.output_dir/'samples/sample_000000.npz'
        with np.load(path) as z: values = {key: z[key] for key in z.files}
        values['elapsed_seconds'] = .2
        atomic_npz(path, **values)
        manifest = json.loads((b.output_dir/'manifest.json').read_text())
        manifest['samples'][0]['sha256'] = checksum(path)
        atomic_json(b.output_dir/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            merge_shards(self.config, [a.output_dir, b.output_dir])

    def test_merge_rejects_fit_and_version_changes(self):
        a = replace(self.config, output_dir=self.root/'a')
        b = replace(self.config, output_dir=self.root/'b')
        generate_dataset(a, sample_ids=(0,))
        generate_dataset(b, sample_ids=(1,))
        path = b.output_dir/'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['material_identity']['test-only']['fitted_index_ri'][0][0] = 1.1
        atomic_json(path, manifest)
        with self.assertRaisesRegex(ValueError, 'different actual material fits'):
            merge_shards(self.config, [a.output_dir, b.output_dir])
        manifest['fdtd_version'] = 'changed'
        atomic_json(path, manifest)
        with self.assertRaisesRegex(ValueError, 'different solver versions'):
            merge_shards(self.config, [a.output_dir, b.output_dir])

    def test_live_output_is_locked(self):
        with directory_lock(self.config.output_dir):
            with self.assertRaisesRegex(RuntimeError, 'Another generator'):
                generate_dataset(self.config)

    def test_init_never_overwrites(self):
        with self.assertRaisesRegex(ValueError, 'existing jobs'):
            initialize(self.root/'job', 'au-8', 4, 8, None)

    def test_bare_preset_keeps_reference_stack(self):
        initialize(self.root/'bare', 'bare-4', 2, 8, None)
        config = load_config(self.root/'bare/job.json')
        self.assertTrue(config.simulation.remove_metal)
        self.assertEqual(config.simulation.metal_material, 'Au (Gold) - JC passive NIR v1')
        self.assertEqual(config.simulation.substrate_depth_nm, 350)

    def test_invalid_shard_and_threads_are_errors(self):
        self.assertEqual(main(['init', '--job-dir', str(self.root/'bad'), '--threads', '0']), 1)
        self.assertEqual(main(['generate', '--config', str(self.root/'job/job.json'), '--shard-count', '2']), 1)

    def test_default_task_uses_metal_only_and_custom_step(self):
        self.assertEqual(main(['init', '--job-dir', str(self.root/'default'), '--samples', '2']), 0)
        settings = load_config(self.root/'default/job.json').simulation
        self.assertEqual(settings.mesh_policy, 'metal_only')
        self.assertIsNone(settings.maximum_mesh_nm)
        self.assertEqual(settings.metal_mesh_nm, 8)
        initialize(self.root/'metal8', 'au-metal', 2, 32, None, metal_mesh_nm=8)
        self.assertEqual(load_config(self.root/'metal8/job.json').simulation.metal_mesh_nm, 8)
        self.assertEqual(main(['init', '--job-dir', str(self.root/'invalid-metal'), '--metal-mesh-nm', '10']), 1)
        self.assertFalse((self.root/'invalid-metal').exists())

    def test_mesh_policy_change_cannot_resume_or_overwrite_old_sample(self):
        generate_dataset(self.config, limit=1)
        sample = self.config.output_dir/'samples/sample_000000.npz'
        original = checksum(sample)
        altered = replace(self.config, simulation=replace(self.config.simulation, mesh_policy='metal_only', maximum_mesh_nm=None))
        with self.assertRaisesRegex(ValueError, 'different configuration'):
            generate_dataset(altered)
        self.assertEqual(TestOnlySimulator.calls, 1)
        self.assertEqual(checksum(sample), original)

    def test_metal_only_coarse_air_can_pack_and_resume(self):
        initialize(self.root/'metal', 'au-metal', 4, 8, None)
        config = load_config(self.root/'metal/job.json')
        generate_dataset(config, limit=1)
        result = generate_dataset(config)
        self.assertEqual(result['success'], 4)
        self.assertEqual(TestOnlySimulator.calls, 4)
        with np.load(config.output_dir/'samples/sample_000000.npz') as sample:
            audit = json.loads(str(sample['mesh_audit_json'].item()))
            self.assertFalse(audit['whole_domain_limit_enforced'])
            self.assertFalse(audit['all_returned_cells_below_10_nm'])
            self.assertLessEqual(audit['metal_region']['maximum_step_nm'], 8)
            self.assertEqual(audit['metal_region']['cells'], 13)

    def test_merge_rejects_whole_domain_and_metal_only(self):
        a = replace(self.config, output_dir=self.root/'whole')
        b = replace(self.config, output_dir=self.root/'metal',
                    simulation=replace(self.config.simulation, mesh_policy='metal_only', maximum_mesh_nm=None))
        generate_dataset(a, sample_ids=(0,))
        generate_dataset(b, sample_ids=(1,))
        with self.assertRaisesRegex(ValueError, 'physics/sampling'):
            merge_shards(self.config, [a.output_dir, b.output_dir])

    def test_merge_rechecks_actual_metal_cells(self):
        initialize(self.root/'metal', 'au-metal', 2, 8, None)
        config = load_config(self.root/'metal/job.json')
        shard = replace(config, output_dir=self.root/'shard')
        generate_dataset(shard, sample_ids=(0,))
        path = shard.output_dir/'samples/sample_000000.npz'
        with np.load(path) as sample: values = {key: sample[key] for key in sample.files}
        values['mesh_z_nm'] = np.array([-500., -100., -90., 0., 100.])
        atomic_npz(path, **values)
        manifest = json.loads((shard.output_dir/'manifest.json').read_text())
        manifest['samples'][0]['sha256'] = checksum(path)
        atomic_json(shard.output_dir/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'metal z mesh'):
            merge_shards(config, [shard.output_dir])


class NativeBoundaryTests(unittest.TestCase):
    def test_metal_policy_rejects_global_limit_and_missing_metal(self):
        with self.assertRaises(ValueError):
            SimulationSettings(mesh_policy='metal_only', maximum_mesh_nm=8, mesh_nm=8)
        with self.assertRaises(ValueError):
            SimulationSettings(mesh_policy='metal_only')

    def test_metal_audit_checks_cells_crossing_interface(self):
        settings = SimulationSettings(mesh_policy='metal_only', metal_material='Au (Gold) - Palik',
                                      substrate_depth_nm=350, metal_thickness_nm=100, metal_mesh_nm=2.5)
        grids = {'x': np.array([0., 50.])*1e-9, 'y': np.array([0., 50.])*1e-9,
                 'z': np.r_[-500., np.arange(-100., 1., 2.5), 100.]*1e-9}
        self.assertFalse(mesh_audit(grids, settings)['all_returned_cells_below_10_nm'])
        # A cell extending across an interface must not escape the local check.
        changed = grids | {'z': np.r_[-500., -101., np.arange(-97.5, 1., 2.5), 100.]*1e-9}
        with self.assertRaisesRegex(ValueError, 'metal z mesh'):
            mesh_audit(changed, settings)

    def test_metal_only_build_creates_only_metal_override(self):
        settings = SimulationSettings(mesh_policy='metal_only', metal_material='Au (Gold) - Palik',
                                      mesh_nm=8, metal_mesh_nm=2.5, substrate_depth_nm=350)
        simulator = object.__new__(FDTDSimulator)
        simulator.settings = settings
        simulator.fdtd = MagicMock()
        simulator.validate_native_layout = MagicMock()
        simulator.build(Geometry(670, 165, 300, 110, 115, theta_odd_deg=-5, theta_even_deg=41))
        overrides = [call.kwargs for call in simulator.fdtd.addmesh.call_args_list]
        self.assertEqual([row['name'] for row in overrides], ['metal_mesh'])
        self.assertFalse(overrides[0]['override_x_mesh'])
        self.assertFalse(overrides[0]['override_y_mesh'])
        self.assertTrue(overrides[0]['override_z_mesh'])

    def test_bare_metal_only_pins_reference_xy_using_existing_override(self):
        settings = SimulationSettings(mesh_policy='metal_only', metal_material='Au (Gold) - Palik',
                                      remove_metal=True, mesh_nm=8, metal_mesh_nm=2.5, substrate_depth_nm=350)
        simulator = object.__new__(FDTDSimulator)
        simulator.settings = settings
        simulator.fdtd = MagicMock()
        simulator.fdtd.getresult.return_value = np.array([0., 7e-9, 14e-9])
        simulator.validate_native_layout = MagicMock()
        simulator.build(Geometry(670, 165, 300, 110, 115, theta_odd_deg=-5, theta_even_deg=41))
        self.assertEqual(simulator.fdtd.addmesh.call_count, 1)
        for axis in ('x', 'y'):
            simulator.fdtd.setnamed.assert_any_call('metal_mesh', f'override {axis} mesh', True)
            simulator.fdtd.setnamed.assert_any_call('metal_mesh', 'd'+axis, 7e-9)
        simulator.fdtd.select.assert_any_call('metal_film')

    def test_actual_mesh_in_any_axis_cannot_reach_10(self):
        grids = {axis: np.array([0., 8e-9, 16e-9]) for axis in ('x', 'y', 'z')}
        self.assertEqual(mesh_grid_summary(grids, 8)['z']['maximum_step_nm'], 8)
        for axis in grids:
            changed = grids | {axis: np.array([0., 10e-9])}
            with self.assertRaises(ValueError): mesh_grid_summary(changed, 9)

    def test_nonconvergence_and_bad_energy_are_rejected(self):
        from deta_generator2.config import SimulationSettings
        settings = SimulationSettings()
        grid = settings.wavelength_grid()
        with self.assertRaises(ValueError):
            validate_spectrum(grid, np.full(1000, .2), np.full(1000, .7), settings, 1)
        with self.assertRaises(ValueError):
            validate_spectrum(grid, np.full(1000, .7), np.full(1000, .7), settings, 2)

    def test_native_memory_gate_prevents_engine_start(self):
        with TemporaryDirectory() as d:
            initialize(Path(d)/'job', 'au-4', 1, 8, None)
            config = load_config(Path(d)/'job/job.json')
            simulator = object.__new__(FDTDSimulator)
            simulator.settings = config.simulation
            simulator.version = 'test-only'
            simulator.fdtd = MagicMock()
            simulator.build = MagicMock()
            simulator.material_identity = MagicMock()
            simulator.fdtd.runsystemcheck.return_value = {'Memory_Recommended_Bytes': 10e9}
            with patch('deta_generator2.lumerical.psutil.virtual_memory', return_value=SimpleNamespace(available=1e9)):
                with self.assertRaises(MemoryError):
                    simulator.run(config.sample_geometries()[0], Path(d)/'sample.fsp')
            simulator.fdtd.run.assert_not_called()
            self.assertTrue((Path(d)/'sample.resources.json').is_file())

    def test_quadrant_angles_are_independent(self):
        g = Geometry(670, 165, 300, 110, 115, theta_odd_deg=-5, theta_even_deg=41)
        centers = g.centers()
        angles = g.angles()
        for (x, y), angle in zip(centers, angles):
            self.assertEqual(angle, -5 if x*y > 0 else 41)

    def test_native_time_limit_retries_until_shutoff(self):
        with TemporaryDirectory() as d:
            initialize(Path(d)/'job', 'au-4', 1, 8, None)
            config = load_config(Path(d)/'job/job.json')
            simulator = object.__new__(FDTDSimulator)
            simulator.settings = config.simulation
            simulator.version = 'test-only'
            simulator.fdtd = MagicMock()
            simulator.build = MagicMock()
            simulator.material_identity = MagicMock()
            simulator.fdtd.runsystemcheck.return_value = {'Memory_Recommended_Bytes': 1e8}
            statuses = iter((1, 2))
            def native(name, key):
                if name == 'FDTD':
                    return next(statuses) if key == 'status' else np.array([0., 3e-9, 6e-9])
                return {'lambda': config.simulation.wavelength_grid()*1e-9,
                        'T': np.full(1000, .2 if name == 'R' else -.7)}
            simulator.fdtd.getresult.side_effect = native
            with patch('deta_generator2.lumerical.psutil.virtual_memory', return_value=SimpleNamespace(available=1e9)), \
                    patch('deta_generator2.lumerical.pml_monitor_audit', return_value={'all_planes_outside_pml': True}):
                result = simulator.run(config.sample_geometries()[0], Path(d)/'sample.fsp')
            self.assertEqual(result['status'], 2)
            self.assertEqual(result['run_attempts'], 2)
            self.assertEqual(result['simulation_time_ps'], 8)
            self.assertEqual(simulator.fdtd.run.call_count, 2)
            simulator.build.assert_called_once()


if __name__ == '__main__': unittest.main()
