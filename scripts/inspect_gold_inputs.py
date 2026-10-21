"""Archive native gold input samples near the Palik near-IR change."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deta_generator2.lumerical import load_lumapi


def main():
    rows = []
    with load_lumapi().FDTD(hide=True) as fdtd:
        version = str(fdtd.version())
        for name in ['Au (Gold) - Palik', 'Au (Gold) - Johnson and Christy']:
            data = np.asarray(fdtd.getmaterial(name, 'sampled 3d data'))
            wave = 299792458/data[:, 0].real*1e9
            order = np.argsort(wave)
            selected = order[(wave[order] > 1150.) & (wave[order] < 1350.)]
            index = np.sqrt(data[:, 1])
            rows.append({'name': name, 'native_sample_count': len(data),
                'samples_near_1240nm': [{'wavelength_nm': float(wave[i]),
                    'n': float(index[i].real), 'k': float(index[i].imag)} for i in selected]})
    output = {'native_version': version, 'query': "getmaterial(name, 'sampled 3d data')",
              'solver_run_performed': False, 'results': rows}
    (ROOT/'research/metal_loss_fit/input_samples.json').write_text(
        json.dumps(output, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
