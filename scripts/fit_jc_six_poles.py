"""Refine the J&C passive model at the remaining long-wave interpolation bends."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from fit_drude_lorentz import permittivity
from probe_metal_fit import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'research/metal_loss_fit'
    previous = json.loads((output/'jc_four_poles.json').read_text())
    initial = previous['parameters']+[.01, 1.13, .05, .01, 1.4, .08]
    with np.load(output/'source_probe.npz') as data:
        wave, original = data['wavelength_nm'], data['JC_measured']
    omega = 2*np.pi*299792458/(wave*1e-9)/1e15
    eps = original**2

    def residual(parameters):
        fitted_eps = permittivity(omega, parameters)
        delta = np.sqrt(fitted_eps)-original
        delta_eps = fitted_eps-eps
        scaled = np.r_[delta.real/.02, delta.imag/.05,
            delta_eps.real/(abs(eps.real)*.02), delta_eps.imag/(eps.imag*.05)]
        return scaled*abs(scaled)

    fit = least_squares(residual, initial,
        bounds=([1., 1., .001]+[0., .05, .001]*6,
                [30., 30., 2.]+[500., 30., 30.]*6),
        max_nfev=2000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    fitted = np.sqrt(permittivity(omega, fit.x))
    values = metrics(original, fitted)
    passes = (values['passive_in_band'] and values['max_n_error'] <= .02 and
        values['max_k_error'] <= .05 and values['max_relative_real_epsilon_error'] <= .02
        and values['max_relative_imag_epsilon_error'] <= .05)
    row = {'dataset': 'JC', 'lorentz_count': 6, 'parameters_frequency_unit': '1e15 rad/s',
        'parameters': fit.x.tolist(), 'converged': bool(fit.success), 'nfev': fit.nfev,
        'metrics': values, 'passes_provisional_material_checks': bool(passes),
        'solver_run_performed': False}
    (output/'jc_six_poles.json').write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(output/'jc_six_poles.npz', wavelength_nm=wave, fitted_index=fitted)
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
