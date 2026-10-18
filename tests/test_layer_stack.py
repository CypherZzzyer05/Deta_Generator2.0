from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

import numpy as np

from deta_generator2.config import Geometry, SimulationSettings, load_config
from deta_generator2.cli import initialize
from deta_generator2.lumerical import FDTDSimulator, mesh_audit, stack_interfaces_nm


class LayerStackTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimulationSettings(
            substrate_depth_nm=350, metal_thickness_nm=100,
            substrate_termination='finite', bottom_padding_nm=1200,
            metal_material='Au (Gold) - Palik', metal_position='above_substrate',
            metal_mesh_nm=8, mesh_policy='metal_only')

    def test_direct_contact_and_contiguous_finite_layers(self):
        stack = stack_interfaces_nm(self.settings)
        self.assertEqual(stack, dict(substrate_min=-450, substrate_max=-100,
                                     metal_min=-100, metal_max=0, stack_min=-450,
                                     domain_min=-1650, transmission_z=-1050))
        self.assertEqual(stack['substrate_max'], stack['metal_min'])
        self.assertEqual(stack['metal_max'], 0)  # Actual pillar base.

    def test_original_order_remains_explicitly_available(self):
        stack = stack_interfaces_nm(replace(self.settings, metal_position='below_substrate'))
        self.assertEqual(stack, dict(substrate_min=-350, substrate_max=0,
                                     metal_min=-450, metal_max=-350, stack_min=-450,
                                     domain_min=-1650, transmission_z=-1050))

    def test_thick_glass_covers_lower_pml_and_transmission_plane(self):
        settings = replace(self.settings, substrate_termination='pml')
        stack = stack_interfaces_nm(settings)
        self.assertEqual(stack['substrate_min'], -2850)
        self.assertLess(stack['substrate_min'], stack['domain_min'])
        self.assertLess(stack['domain_min'], stack['transmission_z'])
        self.assertLess(stack['transmission_z'], stack['substrate_max'])
        self.assertEqual(stack['substrate_max'], stack['metal_min'])
        simulator = object.__new__(FDTDSimulator)
        simulator.settings, simulator.fdtd = settings, MagicMock()
        simulator.validate_native_layout = MagicMock()
        simulator.build(Geometry(670, 165, 300, 110, 115))
        rectangles = {c.kwargs['name']: c.kwargs for c in simulator.fdtd.addrect.call_args_list}
        native_bottom = simulator.fdtd.addfdtd.call_args.kwargs['properties']['z min']
        self.assertAlmostEqual(rectangles['substrate']['z_min'], -2850e-9)
        self.assertLess(rectangles['substrate']['z_min'], native_bottom)
        self.assertEqual(rectangles['substrate']['z_max'], rectangles['metal_film']['z_min'])
        monitors = {c.kwargs['name']: c.kwargs for c in simulator.fdtd.addpower.call_args_list}
        self.assertGreater(monitors['T']['z'], native_bottom)
        self.assertLess(monitors['T']['z'], rectangles['substrate']['z_max'])

    def test_invalid_termination_and_bottom_metal_pml_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'substrate_termination'):
            replace(self.settings, substrate_termination='ambiguous')
        with self.assertRaisesRegex(ValueError, 'bottom-PML substrate'):
            replace(self.settings, substrate_termination='pml', metal_position='below_substrate')

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
            self.assertEqual(config.metadata()['simulation']['substrate_termination'], 'pml')
            data = json.loads((directory/'job.json').read_text())
            missing = json.loads(json.dumps(data))
            missing['simulation'].pop('substrate_termination')
            (directory/'job.json').write_text(json.dumps(missing))
            with self.assertRaisesRegex(ValueError, 'explicit substrate_termination'):
                load_config(directory/'job.json')
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
