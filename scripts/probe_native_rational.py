"""Check native MCM representation of smooth, passive rational reference curves."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deta_generator2.lumerical import load_lumapi
from fit_drude_lorentz import permittivity
from probe_metal_fit import metrics


def main():
    output = ROOT/'research/metal_loss_fit'
    wave = np.linspace(700., 1700., 1000)
    frequency = 299792458/(wave*1e-9)
    models = []
    for filename, dataset, poles in [('drude_lorentz_probe.json', 'CRC', 2),
                                    ('positive_poles_probe.json', 'JC', 3)]:
        row = next(row for row in json.loads((output/filename).read_text())['results']
                   if row['dataset'] == dataset and row['lorentz_count'] == poles)
        models.append(row)
    results, curves = [], {'wavelength_nm': wave}
    with load_lumapi().FDTD(hide=True) as fdtd:
        version = str(fdtd.version())
        for model in models:
            dataset = model['dataset']
            reference = np.sqrt(permittivity(2*np.pi*frequency/1e15, model['parameters']))
            base = 'Au (Gold) - CRC' if dataset == 'CRC' else 'Au (Gold) - Johnson and Christy'
            original = np.asarray(fdtd.getindex(base, frequency)).reshape(-1)
            for coefficients in (8, 12):
                material = fdtd.addmaterial('Sampled 3D data')
                fdtd.setmaterial(material, 'sampled 3d data', np.column_stack((frequency[::-1], reference[::-1]**2)))
                for name, value in [('tolerance', 0.), ('max coefficients', coefficients),
                                    ('imaginary weight', 10.), ('make fit passive', 1),
                                    ('improve numerical stability', 1)]:
                    fdtd.setmaterial(material, name, value)
                fitted = np.asarray(fdtd.getfdtdindex(material, frequency,
                    frequency.min(), frequency.max())).reshape(-1)
                row = {'dataset': dataset, 'max_coefficients': coefficients,
                    'analytic_model_metrics': metrics(reference, fitted),
                    'original_data_metrics': metrics(original, fitted),
                    'reference_parameters': model['parameters']}
                results.append(row)
                curves[f'{dataset}_mcm{coefficients}'] = fitted
                print(json.dumps(row), flush=True)
    (output/'native_rational_probe.json').write_text(json.dumps({'native_version': version,
        'solver_run_performed': False, 'results': results}, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(output/'native_rational_probe.npz', **curves)


if __name__ == '__main__':
    main()
