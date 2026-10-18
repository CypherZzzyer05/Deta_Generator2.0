"""Native single-geometry layer-order test; no data are added to old datasets."""

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import psutil

from deta_generator2.config import Geometry, SimulationSettings
from deta_generator2.lumerical import FDTDSimulator, mesh_audit, stack_interfaces_nm


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--position', choices=['above_substrate', 'below_substrate'],
                        default='above_substrate')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--dataset', type=Path,
                        default=Path('D:/')/'\u4eff\u771f'/'au100_metal8_100'/'dataset.npz')
    args = parser.parse_args()
    output = ROOT / 'jobs' / ('local_sample0_' + args.position)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            output/'run.log', encoding='utf-8')])
    original_job = json.loads((args.dataset.parent/'job.json').read_text(encoding='utf-8-sig'))
    parameters = dict(original_job['simulation'])
    parameters.update(metal_position=args.position, processes=1, threads=args.threads,
                      hide_fdtd=True, save_projects=True,
                      resonator_nk_csv=str(ROOT/'src/deta_generator2/assets/materials/as2se3_joseph_700nm.csv'))
    settings = SimulationSettings(**parameters)
    with np.load(args.dataset, allow_pickle=False) as data:
        row = int(np.flatnonzero(data['sample_ids'] == 0).item())
        values = data['geometry_values'][row].copy()
        baseline = {k: data[k][row].copy() for k in ('spectra', 'transmittance', 'absorptance')}
        baseline['wavelength_nm'] = data['wavelength_nm'].copy()
    geometry = Geometry(*values[:5], theta_odd_deg=values[5], theta_even_deg=values[6])
    write_json(output/'settings.json', asdict(settings))
    project = output/'sample0.fsp'
    started = time.time()
    with FDTDSimulator(settings, r'D:\Program Files\Lumerical\v241\api\python\lumapi.py') as simulator:
        simulator.build(geometry)
        simulator.fdtd.save(str(project))
        grids = {axis: np.asarray(simulator.fdtd.getresult('FDTD', axis)).reshape(-1)
                 for axis in ('x', 'y', 'z')}
        audit = mesh_audit(grids, settings)
        resources = simulator.fdtd.runsystemcheck()
        recommended = float(resources.get('Memory_Recommended_Bytes', resources.get('Memory_Recommended')))
        available = psutil.virtual_memory().available
        materials = simulator.material_identity()
        objects = {}
        for name in ('substrate', 'metal_film', 'si_0', 'si_1', 'si_2', 'si_3', 'T', 'R', 'source'):
            keys = ('z min', 'z max') if name not in ('T', 'R', 'source') else ('z',)
            objects[name] = {key: float(np.asarray(simulator.fdtd.getnamed(name, key)).item())*1e9
                             for key in keys}
        report = dict(solver_run_performed=False, native_layout_verified=True,
                      fdtd_version=simulator.version, geometry_values=values.tolist(),
                      metal_position=args.position, stack_interfaces_nm=stack_interfaces_nm(settings),
                      native_objects_nm=objects, mesh_audit=audit,
                      recommended_ram_gib=recommended/2**30, available_ram_gib=available/2**30,
                      material_fit_errors={name: {k: v for k, v in value.items()
                                                if k != 'fitted_index_ri'} for name, value in materials.items()},
                      source_dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest())
        write_json(output/'preflight.json', report)
        write_json(output/'material_identity.json', materials)
        logging.info('Preflight: %s', json.dumps(report, ensure_ascii=False))
        if not args.run:
            return
        # Leave headroom for the GUI and other applications after the native check.
        if available < recommended + 300*2**20:
            raise MemoryError(f'Need {recommended/2**30+.293:.2f} GiB available; '
                              f'currently {available/2**30:.2f} GiB')
        write_json(output/'state.json', dict(state='running', started_unix=started,
                                            pid=__import__('os').getpid(), project=str(project)))
        logging.info('Starting native FDTD: %s', args.position)
        result = simulator.run(geometry, project)
        result.update(geometry_values=values, settings_json=json.dumps(asdict(settings)),
                      source_dataset_sha256=report['source_dataset_sha256'])
        np.savez_compressed(output/'result.npz', **result)
        np.savez_compressed(output/'historical_server_sample0.npz', **baseline)
        write_json(output/'state.json', dict(state='complete', started_unix=started,
                                            elapsed_seconds=time.time()-started,
                                            status=int(result['status']), result=str(output/'result.npz')))
        logging.info('Completed status=%s, solver time=%.1fs', result['status'], result['elapsed_seconds'])


if __name__ == '__main__':
    main()
