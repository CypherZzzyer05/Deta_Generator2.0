"""Refine existing passive poles against n/k and epsilon error scales."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from fit_drude_lorentz import permittivity
from probe_metal_fit import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'research/metal_loss_fit'
    previous = json.loads((output/'positive_poles_probe.json').read_text())['results']
    with np.load(output/'source_probe.npz') as data:
        wave = data['wavelength_nm']
        datasets = {'JC': data['JC_measured'], 'CRC': data['CRC_measured']}
    omega = 2*np.pi*299792458/(wave*1e-9)/1e15
    curves, rows = {'wavelength_nm': wave}, []
    for name, original in datasets.items():
        eps = original**2

        def residual(parameters):
            fitted_eps = permittivity(omega, parameters)
            fitted = np.sqrt(fitted_eps)
            delta, delta_eps = fitted-original, fitted_eps-eps
            scaled = np.r_[delta.real/.02, delta.imag/.05,
                delta_eps.real/(abs(eps.real)*.02), delta_eps.imag/(eps.imag*.05)]
            return scaled*abs(scaled)

        initial = next(row['parameters'] for row in previous if row['dataset'] == name)
        fit = least_squares(residual, initial,
            bounds=([1., 1., .001]+[0., .05, .001]*3,
                    [30., 30., 2.]+[500., 30., 30.]*3),
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
    (output/'refined_poles_probe.json').write_text(json.dumps({'solver_run_performed': False,
        'results': rows}, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(output/'refined_poles_probe.npz', **curves)


if __name__ == '__main__':
    main()
