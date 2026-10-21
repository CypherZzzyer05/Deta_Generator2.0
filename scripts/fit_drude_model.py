"""Test a positive-damping Drude model against the archived native Palik data."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from probe_metal_fit import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'research/metal_loss_fit'
    with np.load(output/'fit_probe.npz') as data:
        wave = data['wavelength_nm']
        original = data['Au_measured']
    omega = 2*np.pi*299792458/(wave*1e-9)/1e15
    eps = original**2

    def model(parameters):
        eps_inf, plasma, collision = parameters
        return eps_inf-plasma**2/(omega*(omega+1j*collision))

    def residual(parameters):
        delta = model(parameters)-eps
        return np.r_[delta.real/abs(eps.real), delta.imag/eps.imag]

    fit = least_squares(residual, [9., 13.7, .1],
        bounds=([1., 1., .001], [30., 30., 2.]), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    fitted = np.sqrt(model(fit.x))
    result = {'model': 'Drude', 'permittivity': float(fit.x[0]),
        'plasma_resonance_rad_s': float(fit.x[1]*1e15),
        'plasma_collision_rad_s': float(fit.x[2]*1e15),
        'metrics': metrics(original, fitted), 'solver_run_performed': False}
    (output/'drude_probe.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
