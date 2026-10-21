"""Reduce the maximum J&C n/k residual with a passive eight-pole model."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from fit_drude_lorentz import permittivity
from probe_metal_fit import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = ROOT/'research/metal_loss_fit'
    previous = json.loads((output/'jc_six_poles.json').read_text())
    initial = previous['parameters']+[.001, 1.17, .02, .001, 1.35, .02]
    if args.resume:
        initial = json.loads((output/'jc_candidate.json').read_text())['parameters']
    with np.load(output/'source_probe.npz') as data:
        wave, original = data['wavelength_nm'], data['JC_measured']
    omega = 2*np.pi*299792458/(wave*1e-9)/1e15
    eps = original**2

    def residual(parameters):
        fitted_eps = permittivity(omega, parameters)
        delta = np.sqrt(fitted_eps)-original
        delta_eps = fitted_eps-eps
        scaled = np.r_[delta.real/.02, delta.imag/.043,
            delta_eps.real/(abs(eps.real)*.02), delta_eps.imag/(eps.imag*.05)]
        return scaled**4

    fit = least_squares(residual, initial,
        bounds=([1., 1., .001]+[0., .05, .001]*8,
                [30., 30., 2.]+[500., 30., 30.]*8),
        max_nfev=4000, x_scale='jac', xtol=1e-11, ftol=1e-11, gtol=1e-11)
    fitted = np.sqrt(permittivity(omega, fit.x))
    values = metrics(original, fitted)
    passes = (values['passive_in_band'] and values['max_n_error'] <= .02 and
        values['max_k_error'] <= .05 and values['max_relative_real_epsilon_error'] <= .02
        and values['max_relative_imag_epsilon_error'] <= .05)
    row = {'dataset': 'JC', 'lorentz_count': 8, 'parameters_frequency_unit': '1e15 rad/s',
        'parameters': fit.x.tolist(), 'converged': bool(fit.success), 'nfev': fit.nfev,
        'initial_parameters': initial,
        'objective_k_error_scale': .043,
        'metrics': values, 'passes_provisional_material_checks': bool(passes),
        'solver_run_performed': False}
    (output/'jc_candidate.json').write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(output/'jc_candidate.npz', wavelength_nm=wave, fitted_index=fitted)
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
