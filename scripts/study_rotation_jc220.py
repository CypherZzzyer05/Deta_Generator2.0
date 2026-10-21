"""Small controlled native study; does not modify generator presets or training data."""
import argparse
from dataclasses import replace
import ctypes
import hashlib
import json
import logging
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deta_generator2.config import load_config, Geometry
from deta_generator2.lumerical import FDTDSimulator

C = 299792458.
EPS0 = 8.8541878128e-12


def free_commit_bytes():
    buffer = ctypes.create_string_buffer(64)
    struct.pack_into('I', buffer, 0, 64)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(buffer):
        raise OSError('Cannot read Windows commit availability')
    return struct.unpack_from('Q', buffer, 32)[0]


class StudySimulator(FDTDSimulator):
    def __init__(self, settings, wavelengths, contact_bare=False):
        super().__init__(settings)
        self.field_wavelengths = np.asarray(wavelengths)
        self.contact_bare = contact_bare

    def build(self, geometry, include_silicon=True, substrate_index=None):
        super().build(geometry, include_silicon, substrate_index)
        f = self.fdtd
        if self.contact_bare:
            # Research control: replace the gold layer by the same substrate,
            # leaving the pillars at z=0 and eliminating the intervening air gap.
            f.setnamed('substrate', 'z max', 0.)
            if (int(f.getnamednumber('metal_film')) != 0 or
                    abs(float(f.getnamed('substrate', 'z max'))*1e9) > 1e-6 or
                    f.getnamed('substrate', 'material') != self.settings.substrate_material):
                raise ValueError('Bare contact-control layout mismatch')
        f.cwnorm()
        self.planes = {'mid': geometry.h/2, 'top': geometry.h+10}
        for plane, height in self.planes.items():
            f.addpower(name='field_'+plane, monitor_type='2D Z-normal', x=0., y=0.,
                x_span=geometry.p*1e-9, y_span=geometry.p*1e-9, z=height*1e-9)
            f.set('override global monitor settings', True)
            f.set('sample spacing', 'custom')
            f.set('custom frequency samples', (C/(self.field_wavelengths*1e-9))[::-1])
            f.set('spatial interpolation', 'specified position')
            f.set('down sample x', 1)
            f.set('down sample y', 1)
            for component in ('Ex','Ey','Ez','Hx','Hy','Hz','Px','Py','Pz'):
                f.set('output '+component, component.startswith('E'))
            f.set('output power', False)
        resources = f.runsystemcheck()
        recommended = float(resources.get('Memory_Recommended_Bytes', resources.get('Memory_Recommended')))
        free_commit = free_commit_bytes()
        print(f'Commit available {free_commit/2**30:.2f} GiB; native recommendation {recommended/2**30:.2f} GiB', flush=True)
        if free_commit < recommended+512*2**20:
            raise MemoryError('Insufficient Windows commit for the native solver; no engine launched')

    def read_fields(self):
        result = {}
        for plane, height in self.planes.items():
            dataset = self.fdtd.getresult('field_'+plane, 'E')
            coords = {axis: np.asarray(dataset[axis]).reshape(-1)*1e9 for axis in ('x','y','z')}
            wave = np.asarray(dataset['lambda']).reshape(-1)*1e9
            order = np.argsort(wave)
            if not np.allclose(wave[order], self.field_wavelengths, rtol=0, atol=1e-5):
                raise ValueError('Native field frequencies differ from the study plan')
            shape = tuple(len(coords[axis]) for axis in ('x','y','z'))+(len(wave),3)
            field = np.asarray(dataset['E'])
            if field.shape != shape or not np.isfinite(field).all():
                raise ValueError('Invalid native E dataset')
            intensity = np.asarray(self.fdtd.sourceintensity(np.asarray(dataset['f']).reshape(-1))).reshape(-1)
            incident_squared = 2*intensity/(EPS0*C)
            if not np.isfinite(incident_squared).all() or np.any(incident_squared <= 0):
                raise ValueError('Invalid incident-field normalization')
            enhancement = (np.sum(abs(field)**2, axis=-1)/incident_squared[None,None,None,:])[...,order]
            if len(coords['z']) != 1 or not np.isclose(coords['z'][0], height, atol=1e-5):
                raise ValueError('Field monitor is not at its specified height')
            result.update({f'{plane}_enhancement': enhancement.astype(np.float32),
                           f'{plane}_wavelength_nm': wave[order],
                           f'{plane}_incident_squared': incident_squared[order]})
            result.update({f'{plane}_{axis}_nm': value for axis,value in coords.items()})
        return result


def study_plan():
    with np.load(ROOT/'docs/release_2_2_0/tetramer_sample_000000.npz', allow_pickle=False) as sample:
        values = sample['geometry_values'].copy()
        wave, reflection = sample['wavelength_nm'], sample['reflectance']
        dips = []
        for lo, hi in ((720,900),(950,1300)):
            indices = np.flatnonzero((wave >= lo)&(wave <= hi))
            dips.append(float(wave[indices[np.argmin(reflection[indices])]]))
    probes = np.unique(np.r_[np.arange(700.,1701.,50.), dips])
    odd, even = values[-2:]
    cases = {
        'au_zero_8': (0.,0.,8.,False),
        'au_odd_8': (odd,0.,8.,False),
        'au_even_8': (0.,even,8.,False),
        'au_both_8': (odd,even,8.,False),
        'au_aligned8': (odd,even,float(100/np.ceil(100/8)),False),
        'au_both_8_cv0': (odd,even,8.,False),
        'au_both_4_cv0': (odd,even,4.,False),
        'au_mirror_8': (-even,-odd,8.,False),
        'au_both_4': (odd,even,4.,False),
        'au_both_2': (odd,even,2.,False),
        'bare_contact_8': (odd,even,8.,True),
        'bare_contact_4': (odd,even,4.,True),
    }
    return values, probes, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'research/rotation_jc220')
    parser.add_argument('--case', choices=study_plan()[2], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    values, wavelengths, cases = study_plan()
    odd, even, mesh, bare = cases[args.case]
    settings = replace(config.simulation, metal_mesh_nm=mesh, remove_metal=bare)
    if args.case.endswith('_cv0'):
        settings = replace(settings, mesh_refinement='conformal variant 0')
    geometry = Geometry(*values[:5], theta_odd_deg=odd, theta_even_deg=even)
    folder = args.output.resolve()/args.case
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/'result.npz'
    plan = {'case': args.case, 'geometry_values': geometry.array(include_quadrant=True).tolist(),
            'config': replace(config, simulation=settings).metadata(),
            'field_wavelengths_nm': wavelengths.tolist(),
            'control': 'pillars_touch_glass' if bare else 'pillars_touch_gold',
            'normalization': '|E|^2 / [2*sourceintensity/(eps0*c)] in CWnorm, air incidence',
            'field_planes_nm': {'mid': geometry.h/2, 'top': geometry.h+10},
            'production_settings_changed': False}
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            if str(saved['study_identity']) != identity or int(saved['status']) != 2:
                raise ValueError('Existing result differs from this study plan')
        print(f'Already complete: {target}', flush=True)
        return
    (folder/'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    with StudySimulator(settings, wavelengths, bare) as simulator:
        simulator.fdtd.cd(str(folder))
        material = simulator.material_identity()
        (folder/'materials.json').write_text(json.dumps(material, indent=2)+'\n', encoding='utf-8')
        print(f'Running {args.case}: {plan["geometry_values"]}', flush=True)
        result = simulator.run(geometry, folder/'native.fsp')
        result.update(simulator.read_fields())
        if bare:
            interfaces = json.loads(result['stack_interfaces_nm_json'])
            interfaces['substrate_max'] = 0.
            result['stack_interfaces_nm_json'] = json.dumps(interfaces)
            result['native_glass_contact_verified'] = True
        result.update(study_identity=identity, geometry_values=geometry.array(include_quadrant=True),
                      control=plan['control'])
        np.savez_compressed(target, **result)
        np.savetxt(folder/'spectrum.csv', np.column_stack([result[k] for k in (
            'wavelength_nm','reflectance','transmittance','absorptance')]),
            delimiter=',', header='wavelength_nm,reflectance,transmittance,absorptance', comments='')
        print(f'Saved {args.case}: status={result["status"]}, {result["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
