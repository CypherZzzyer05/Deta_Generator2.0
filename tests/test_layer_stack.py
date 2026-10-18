from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from deta_generator2.config import SimulationSettings, load_config
from deta_generator2.cli import initialize
from deta_generator2.lumerical import mesh_audit, stack_interfaces_nm


class LayerStackTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimulationSettings(
            substrate_depth_nm=350, metal_thickness_nm=100,
            metal_material='Au (Gold) - Palik', metal_position='above_substrate',
            metal_mesh_nm=8, mesh_policy='metal_only')

    def test_direct_contact_and_contiguous_finite_layers(self):
        stack = stack_interfaces_nm(self.settings)
        self.assertEqual(stack, dict(substrate_min=-450, substrate_max=-100,
                                     metal_min=-100, metal_max=0, stack_min=-450))
        self.assertEqual(stack['substrate_max'], stack['metal_min'])
        self.assertEqual(stack['metal_max'], 0)  # Actual pillar base.

    def test_original_order_remains_explicitly_available(self):
        stack = stack_interfaces_nm(replace(self.settings, metal_position='below_substrate'))
        self.assertEqual(stack, dict(substrate_min=-350, substrate_max=0,
                                     metal_min=-450, metal_max=-350, stack_min=-450))

    def test_mesh_checks_the_new_metal_location(self):
        grids = {'x': np.array([-300, 0, 300])*1e-9,
                 'y': np.array([-300, 0, 300])*1e-9,
                 'z': np.r_[-500, -450, -100, np.linspace(-100, 0, 14)[1:], 100, 500]*1e-9}
        report = mesh_audit(grids, self.settings)
        self.assertAlmostEqual(report['metal_region']['maximum_step_nm'], 100/13)
        # An override left at the old position must not pass the new audit.
        old = dict(grids, z=np.r_[-500, np.linspace(-450, -350, 14), -100, 0, 500]*1e-9)
        with self.assertRaisesRegex(ValueError, 'metal z mesh'):
            mesh_audit(old, self.settings)

    def test_invalid_position_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'metal_position'):
            replace(self.settings, metal_position='ambiguous')

    def test_v2_jobs_explicitly_record_order_and_thickness(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)/'new-job'
            initialize(directory, 'au-metal', 4, 2, None, 8)
            config = load_config(directory/'job.json')
            self.assertEqual(config.simulation.metal_position, 'above_substrate')
            self.assertEqual(config.metadata()['simulation']['metal_position'], 'above_substrate')
            self.assertEqual(config.metadata()['simulation']['metal_thickness_nm'], 100)
            data = json.loads((directory/'job.json').read_text())
            for position in ('missing', 'below_substrate'):
                if position == 'missing':
                    data['simulation'].pop('metal_position')
                else:
                    data['simulation']['metal_position'] = position
                (directory/'job.json').write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'v2 requires explicit'):
                    load_config(directory/'job.json')


if __name__ == '__main__':
    unittest.main()
