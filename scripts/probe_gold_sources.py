"""Compare available gold datasets without changing production materials."""
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
    rows, curves = [], {'wavelength_nm': wave}
    with load_lumapi().FDTD(hide=True) as fdtd:
        version = str(fdtd.version())
        for key, base in [('CRC', 'Au (Gold) - CRC'),
                          ('JC', 'Au (Gold) - Johnson and Christy')]:
            raw = np.asarray(fdtd.getindex(base, frequency)).reshape(-1)
            curves[key+'_measured'] = raw
            for label, parameters in [('default', {}),
                    ('loss_weighted', {'tolerance': 0., 'max coefficients': 8,
                                      'imaginary weight': 10.})]:
                name = fdtd.copymaterial(base)
                for prop, value in parameters.items():
                    fdtd.setmaterial(name, prop, value)
                fitted = np.asarray(fdtd.getfdtdindex(name, frequency,
                    frequency.min(), frequency.max())).reshape(-1)
                curves[key+'_'+label] = fitted
                row = {'dataset': base, 'label': label, 'parameters': parameters,
                       'metrics': metrics(raw, fitted)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    np.savez_compressed(output/'source_probe.npz', **curves)
    (output/'source_probe.json').write_text(json.dumps({'native_version': version,
        'solver_run_performed': False, 'results': rows}, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
