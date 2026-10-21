from dataclasses import replace
import unittest
from unittest.mock import MagicMock

import numpy as np

from deta_generator2.config import SimulationSettings
from deta_generator2.lumerical import FDTDSimulator, pml_monitor_audit
from deta_generator2.materials import (JC_GOLD_MATERIAL, gold_model,
    gold_permittivity, metal_fit_metrics, metal_fit_errors)


class MaterialTests(unittest.TestCase):
    def test_archived_lossless_gold_fit_is_rejected(self):
        original = np.full(1000, .2+6j)
        lossless = np.full(1000, 1e-8+6j)
        errors = metal_fit_errors('Au (Gold) - Palik', metal_fit_metrics(original, lossless))
        self.assertIn('almost removes Ohmic loss', errors[0])

    def test_corrected_gold_limits_are_not_silently_relaxed(self):
        original = np.full(1000, .2+6j)
        values = metal_fit_metrics(original, original)
        self.assertEqual(metal_fit_errors(JC_GOLD_MATERIAL, values), [])
        for key, limit in gold_model()['fit_limits'].items():
            invalid = dict(values, **{key: limit*1.01})
            self.assertTrue(metal_fit_errors(JC_GOLD_MATERIAL, invalid), key)
        self.assertTrue(metal_fit_errors(JC_GOLD_MATERIAL, dict(values, passive_in_band=False)))

    def test_material_identity_blocks_solve_but_allows_doctor_evidence(self):
        simulator = object.__new__(FDTDSimulator)
        simulator.settings = SimulationSettings(wavelength_start_nm=700, wavelength_stop_nm=1700,
            metal_material='Au (Gold) - Palik')
        simulator.fdtd = MagicMock()
        simulator.fdtd.getindex.side_effect = lambda name, *args: np.full(1000,
            .2+6j if name == 'Au (Gold) - Palik' else 1.45+0j)
        simulator.fdtd.getfdtdindex.side_effect = lambda name, *args: np.full(1000,
            1e-8+6j if name == 'Au (Gold) - Palik' else 1.45+0j)
        with self.assertRaisesRegex(ValueError, 'Invalid metal material fit'):
            simulator.material_identity()
        evidence = simulator.material_identity(enforce=False)['Au (Gold) - Palik']
        self.assertFalse(evidence['material_fit_passed'])
        self.assertLess(evidence['fitted_imag_epsilon_range'][1], 1e-5)

    def test_positive_model_is_passive_across_extended_frequency_sample(self):
        values = gold_permittivity(np.geomspace(1e10, 1e18, 12000))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(values.imag > 0))

    def test_corrected_model_band_is_explicit(self):
        settings = SimulationSettings(wavelength_start_nm=700, wavelength_stop_nm=1700,
            metal_material=JC_GOLD_MATERIAL)
        with self.assertRaisesRegex(ValueError, '700-1700'):
            replace(settings, wavelength_stop_nm=1800)

    def test_source_and_monitor_inside_pml_are_rejected(self):
        native = MagicMock()
        planes = {'source': 1000e-9, 'R': 2000e-9, 'T': -1000e-9}
        native.getnamed.side_effect = lambda name, key: 12 if key == 'pml layers' else planes[name]
        grids = {'z': np.linspace(-2000, 4000, 121)*1e-9}
        self.assertTrue(pml_monitor_audit(native, grids)['all_planes_outside_pml'])
        planes['R'] = 3900e-9
        with self.assertRaisesRegex(ValueError, 'overlaps PML'):
            pml_monitor_audit(native, grids)


if __name__ == '__main__':
    unittest.main()
