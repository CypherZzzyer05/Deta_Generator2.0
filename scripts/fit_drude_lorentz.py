"""Fit passive Drude/Lorentz candidates; do not change generator defaults."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from probe_metal_fit import metrics

ROOT = Path(__file__).resolve().parents[1]


def permittivity(omega, parameters):
    eps_inf, plasma, collision, *oscillators = parameters
    result = eps_inf-plasma**2/(omega*(omega+1j*collision))
    for strength, resonance, damping in np.reshape(oscillators, (-1, 3)):
        result += strength*resonance**2/(resonance**2-omega**2-1j*damping*omega)
    return result


def main():
    output = ROOT/'research/metal_loss_fit'
    with np.load(output/'fit_probe.npz') as data:
        wave, palik = data['wavelength_nm'], data['Au_measured']
    with np.load(output/'source_probe.npz') as data:
        datasets = {'Palik': palik, 'JC': data['JC_measured'], 'CRC': data['CRC_measured']}
    omega = 2*np.pi*299792458/(wave*1e-9)/1e15
    rows, curves = [], {'wavelength_nm': wave}
    for name, original in datasets.items():
        eps = original**2

        def residual(parameters):
            delta = permittivity(omega, parameters)-eps
            return np.r_[delta.real/abs(eps.real), delta.imag/eps.imag]

        for count in (1, 2):
            initial = [5., 13.5, .1, 3., 4.5, 2.]
            lower = [1., 1., .001, 0., .05, .001]
            upper = [30., 30., 2., 500., 30., 30.]
            if count == 2:
                initial += [1., 2., .5]
                lower += [0., .05, .001]
                upper += [500., 30., 30.]
            fit = least_squares(residual, initial, bounds=(lower, upper),
                max_nfev=3000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
            fitted = np.sqrt(permittivity(omega, fit.x))
            row = {'dataset': name, 'lorentz_count': count,
                'parameters_frequency_unit': '1e15 rad/s',
                'parameters': fit.x.tolist(), 'converged': bool(fit.success),
                'metrics': metrics(original, fitted)}
            rows.append(row)
            curves[name+f'_D{count}L'] = fitted
            print(json.dumps(row), flush=True)
    (output/'drude_lorentz_probe.json').write_text(json.dumps({'solver_run_performed': False,
        'results': rows}, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(output/'drude_lorentz_probe.npz', **curves)


if __name__ == '__main__':
    main()
