"""Diagnose whether native fit restrictions suppress measured loss.

These candidates are not production materials. Record off-band gain and
require subsequent native FDTD stability/optical checks.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deta_generator2.lumerical import load_lumapi
from probe_metal_fit import metrics


def main():
    output = ROOT/'research/metal_loss_fit'
    wave = np.linspace(700., 1700., 1000)
    frequency = 299792458/(wave*1e-9)
    extended_frequency = np.geomspace(1e10, 1e18, 12000)
    rows, curves = [], {'wavelength_nm': wave, 'extended_frequency_hz': extended_frequency}
    with load_lumapi().FDTD(hide=True) as fdtd:
        for source, key in [('Au (Gold) - Palik', 'Palik'),
                            ('Au (Gold) - Johnson and Christy', 'JC')]:
            original = np.asarray(fdtd.getindex(source, frequency)).reshape(-1)
            for weight in (1., 10.):
                name = fdtd.copymaterial(source)
                parameters = {'tolerance': 0., 'max coefficients': 12, 'imaginary weight': weight,
                              'make fit passive': 0, 'improve numerical stability': 0}
                for prop, value in parameters.items():
                    fdtd.setmaterial(name, prop, value)
                fitted = np.asarray(fdtd.getfdtdindex(name, frequency,
                    frequency.min(), frequency.max())).reshape(-1)
                extended = np.asarray(fdtd.getfdtdindex(name, extended_frequency,
                    frequency.min(), frequency.max())).reshape(-1)
                values = metrics(original, fitted)
                loss = (extended**2).imag
                row = {'dataset': source, 'parameters': parameters, 'metrics': values,
                    'extended_frequency_hz': [1e10, 1e18], 'extended_minimum_Im_epsilon': float(loss.min()),
                    'extended_negative_loss_points': int(np.sum(loss < -1e-8))}
                rows.append(row)
                curves[f'{key}_weight{weight:g}'] = fitted
                curves[f'{key}_weight{weight:g}_extended'] = extended
                print(json.dumps(row), flush=True)
    (output/'unrestricted_probe.json').write_text(json.dumps({'solver_run_performed': False,
        'results': rows}, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(output/'unrestricted_probe.npz', **curves)


if __name__ == '__main__':
    main()
