"""Check whether three positive Lorentz poles resolve remaining fit errors."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from fit_drude_lorentz import permittivity
from probe_metal_fit import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'research/metal_loss_fit'
    with np.load(output/'fit_probe.npz') as data:
        wave, palik = data['wavelength_nm'], data['Au_measured']
    with np.load(output/'source_probe.npz') as data:
        datasets = {'Palik': palik, 'JC': data['JC_measured'], 'CRC': data['CRC_measured']}
    previous = json.loads((output/'drude_lorentz_probe.json').read_text())['results']
    crc_initial = next(r['parameters'] for r in previous
                       if r['dataset'] == 'CRC' and r['lorentz_count'] == 2)
    starts = {'CRC': crc_initial+[.02, 2.3, .2],
              'JC': [5., 13., .1, .3, 1., .2, 3., 4., 2., .05, 2., .3],
              'Palik': [1.1, 12., .1, 20., .8, 1.2, .4, 2.78, .01, .1, 1.9, .1]}
    omega = 2*np.pi*299792458/(wave*1e-9)/1e15
    curves, rows = {'wavelength_nm': wave}, []
    for name, original in datasets.items():
        eps = original**2

        def residual(parameters):
            delta = permittivity(omega, parameters)-eps
            return np.r_[delta.real/abs(eps.real), delta.imag/eps.imag]

        lower = [1., 1., .001]+[0., .05, .001]*3
        upper = [30., 30., 2.]+[500., 30., 30.]*3
        fit = least_squares(residual, starts[name], bounds=(lower, upper),
            max_nfev=1000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        fitted = np.sqrt(permittivity(omega, fit.x))
        values = metrics(original, fitted)
        passes = (values['passive_in_band'] and values['max_n_error'] <= .02 and
            values['max_k_error'] <= .05 and values['max_relative_real_epsilon_error'] <= .02
            and values['max_relative_imag_epsilon_error'] <= .05)
        row = {'dataset': name, 'lorentz_count': 3, 'parameters_frequency_unit': '1e15 rad/s',
            'parameters': fit.x.tolist(), 'converged': bool(fit.success),
            'metrics': values, 'passes_provisional_material_checks': bool(passes)}
        rows.append(row)
        curves[name+'_D3L'] = fitted
        print(json.dumps(row), flush=True)
    (output/'positive_poles_probe.json').write_text(json.dumps({'solver_run_performed': False,
        'results': rows}, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(output/'positive_poles_probe.npz', **curves)


if __name__ == '__main__':
    main()
