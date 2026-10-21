"""Audited near-IR gold dispersion and native-fit acceptance checks."""
import hashlib
import json
from pathlib import Path

import numpy as np

JC_GOLD_MATERIAL = 'Au (Gold) - JC passive NIR v1'
JC_GOLD_SOURCE = 'Au (Gold) - Johnson and Christy'
GOLD_MODEL_PATH = Path(__file__).parent/'assets/materials/au_jc_passive_nir_v1.json'


def gold_model():
    return json.loads(GOLD_MODEL_PATH.read_text(encoding='utf-8'))


def gold_model_sha256():
    return hashlib.sha256(GOLD_MODEL_PATH.read_bytes()).hexdigest()


def gold_permittivity(frequency_hz):
    parameters = gold_model()['parameters']
    eps_inf, plasma, collision, *oscillators = parameters
    omega = 2*np.pi*np.asarray(frequency_hz)/1e15
    result = eps_inf-plasma**2/(omega*(omega+1j*collision))
    for strength, resonance, damping in np.reshape(oscillators, (-1, 3)):
        result += strength*resonance**2/(resonance**2-omega**2-1j*damping*omega)
    return result


def install_gold_material(fdtd):
    model = gold_model()
    wave = np.linspace(*model['wavelength_range_nm'], 1000)
    frequency = 299792458/(wave*1e-9)
    material = fdtd.addmaterial('Sampled 3D data')
    fdtd.setmaterial(material, 'name', JC_GOLD_MATERIAL)
    fdtd.setmaterial(JC_GOLD_MATERIAL, 'sampled 3d data',
        np.column_stack((frequency[::-1], gold_permittivity(frequency[::-1]))))
    for key, value in model['native_fit_settings'].items():
        fdtd.setmaterial(JC_GOLD_MATERIAL, key, value)


def metal_fit_metrics(original, fitted):
    raw, native = np.asarray(original)**2, np.asarray(fitted)**2
    return {'max_n_error': float(np.max(abs(original.real-fitted.real))),
        'max_k_error': float(np.max(abs(original.imag-fitted.imag))),
        'max_relative_real_epsilon_error': float(np.max(abs((native.real-raw.real)/raw.real))),
        'max_relative_imag_epsilon_error': float(np.max(abs((native.imag-raw.imag)/raw.imag))),
        'raw_imag_epsilon_range': [float(raw.imag.min()), float(raw.imag.max())],
        'fitted_imag_epsilon_range': [float(native.imag.min()), float(native.imag.max())],
        'passive_in_band': bool(np.all(native.imag >= 0))}


def metal_fit_errors(name, metrics):
    """Reject the demonstrated bad Au fit; apply fixed limits to corrected gold."""
    if name == JC_GOLD_MATERIAL:
        limits = gold_model()['fit_limits']
        errors = [f'{key}={metrics[key]:.8g} exceeds {limit}'
                  for key, limit in limits.items() if metrics[key] > limit]
        if not metrics['passive_in_band']:
            errors.append('The native gold fit has gain in the source band')
        return errors
    if name == 'Au (Gold) - Palik':
        raw_loss = metrics['raw_imag_epsilon_range']
        fitted_loss = metrics['fitted_imag_epsilon_range']
        if fitted_loss[1] < .1*raw_loss[0]:
            return ['Legacy Au/Palik fit almost removes Ohmic loss. Create a new 2.2 job; preserve old results for comparison.']
    return []
