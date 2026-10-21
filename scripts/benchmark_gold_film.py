"""Independent normal-incidence gold-film FDTD versus analytic thin-film optics.

Research only: production generator settings are not changed. The explicit
positive Drude/Lorentz model is a fit to the CRC dataset, not to Palik.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deta_generator2.lumerical import load_lumapi
from fit_drude_lorentz import permittivity
from probe_metal_fit import metrics


def thin_film(wave, metal, substrate, thickness_nm=100.):
    r01 = (1-metal)/(1+metal)
    r12 = (metal-substrate)/(metal+substrate)
    phase = 2*np.pi*metal*thickness_nm/wave
    denominator = 1+r01*r12*np.exp(2j*phase)
    r = (r01+r12*np.exp(2j*phase))/denominator
    t = 2/(1+metal)*2*metal/(metal+substrate)*np.exp(1j*phase)/denominator
    reflection = abs(r)**2
    transmission = substrate.real*abs(t)**2
    return reflection, transmission, 1-reflection-transmission


def explicit_material(fdtd, parameters):
    name = fdtd.addmaterial('Plasma')
    fdtd.setmaterial(name, 'name', 'Research CRC Drude')
    name = 'Research CRC Drude'
    fdtd.setmaterial(name, 'Permittivity', parameters[0])
    fdtd.setmaterial(name, 'Plasma resonance', parameters[1]*1e15)
    fdtd.setmaterial(name, 'Plasma collision', parameters[2]*1e15)
    native = [{'name': name, 'type': 'Plasma', 'parameters': parameters[:3]}]
    for index, (strength, resonance, damping) in enumerate(np.reshape(parameters[3:], (-1, 3))):
        item = fdtd.addmaterial('Magnetic Electric Lorentz (Version 1.0.0)')
        new_name = f'Research CRC Lorentz {index+1}'
        fdtd.setmaterial(item, 'name', new_name)
        fdtd.setmaterial(new_name, 'base material', name)
        fdtd.setmaterial(new_name, '\u0394\u03b5 electric', np.full((3, 1), strength))
        fdtd.setmaterial(new_name, '\u03c9\u0030 electric', np.full((3, 1), resonance*1e15))
        fdtd.setmaterial(new_name, '\u03b4 electric', np.full((3, 1), damping*.5e15))
        fdtd.setmaterial(new_name, 'exclude magnetic', np.ones((3, 1)))
        native.append({'name': new_name, 'type': 'Magnetic Electric Lorentz (Version 1.0.0)',
            'base': name, 'strength': strength, 'resonance_rad_s': resonance*1e15,
            'lorentz_half_linewidth_rad_s': damping*.5e15,
            'actual_base': str(fdtd.getmaterial(new_name, 'base material'))})
        name = new_name
    return name, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('legacy_palik', 'explicit_crc', 'smooth_crc', 'smooth_jc'), required=True)
    parser.add_argument('--mesh-nm', type=float, default=2.)
    parser.add_argument('--mesh-variant', type=int, choices=(0, 1), default=0)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    output = ROOT/'research/metal_loss_fit/film_benchmark_clear_pml'/f'{args.case}_{args.mesh_nm:g}nm_bg20nm'
    if args.mesh_variant:
        output = output.with_name(output.name+'_cv1')
    output.mkdir(parents=True, exist_ok=True)
    result_path = output/'result.npz'
    if args.run and result_path.exists():
        raise FileExistsError(result_path)
    nm, span = 1e-9, 80e-9
    frequency = 299792458/(np.linspace(700., 1700., 1000)*nm)
    with load_lumapi().FDTD(hide=True) as fdtd:
        fdtd.cd(str(output))
        fdtd.setresource('FDTD', 1, 'processes', '1')
        fdtd.setresource('FDTD', 1, 'threads', '4')
        gold_source = ('Au (Gold) - Palik' if args.case == 'legacy_palik' else
                       'Au (Gold) - Johnson and Christy' if args.case == 'smooth_jc' else 'Au (Gold) - CRC')
        raw = np.asarray(fdtd.getindex(gold_source, frequency)).reshape(-1)
        if args.case != 'legacy_palik':
            dataset, poles, filename = ('JC', 3, 'positive_poles_probe.json') if args.case == 'smooth_jc' else (
                'CRC', 2, 'drude_lorentz_probe.json')
            rows = json.loads((ROOT/'research/metal_loss_fit'/filename).read_text())['results']
            parameters = next(r['parameters'] for r in rows if r['dataset'] == dataset and r['lorentz_count'] == poles)
            rational_index = np.sqrt(permittivity(2*np.pi*frequency/1e15, parameters))
            if args.case == 'explicit_crc':
                material, native_parameters = explicit_material(fdtd, parameters)
                index_model = rational_index
            else:
                material = fdtd.addmaterial('Sampled 3D data')
                fdtd.setmaterial(material, 'sampled 3d data',
                    np.column_stack((frequency[::-1], rational_index[::-1]**2)))
                fit_parameters = {'tolerance': 0., 'max coefficients': 12,
                    'imaginary weight': 10., 'make fit passive': 1, 'improve numerical stability': 1}
                for prop, value in fit_parameters.items():
                    fdtd.setmaterial(material, prop, value)
                native_parameters = {'rational_parameters': parameters,
                    'rational_frequency_unit': '1e15 rad/s', 'native_mcm_settings': fit_parameters}
                index_model = np.asarray(fdtd.getfdtdindex(material, frequency,
                    frequency.min(), frequency.max())).reshape(-1)
        else:
            material, native_parameters = gold_source, []
            index_model = np.asarray(fdtd.getfdtdindex(material, frequency,
                frequency.min(), frequency.max())).reshape(-1)
        glass = np.asarray(fdtd.getfdtdindex('SiO2 (Glass) - Palik', frequency,
            frequency.min(), frequency.max())).reshape(-1)
        raw_glass = np.asarray(fdtd.getindex('SiO2 (Glass) - Palik', frequency)).reshape(-1)
        fdtd.addfdtd(properties=OrderedDict([
            ('dimension', '3D'), ('x', 0.), ('y', 0.), ('x span', span), ('y span', span),
            ('z min', -4000*nm), ('z max', 4000*nm),
            ('x min bc', 'Periodic'), ('x max bc', 'Periodic'),
            ('y min bc', 'Periodic'), ('y max bc', 'Periodic'),
            ('z min bc', 'PML'), ('z max bc', 'PML'),
            ('mesh accuracy', 2), ('simulation time', 1e-12),
            ('mesh refinement', f'conformal variant {args.mesh_variant}'),
            ('auto shutoff min', 1e-7), ('pml profile', 3)]))
        fdtd.addrect(name='substrate', x=0., y=0., x_span=3*span, y_span=3*span,
            z_min=-5000*nm, z_max=-100*nm, material='SiO2 (Glass) - Palik')
        fdtd.addrect(name='gold', x=0., y=0., x_span=3*span, y_span=3*span,
            z_min=-100*nm, z_max=0., material=material)
        # Planar invariance permits a coarse lateral grid, independently of z convergence.
        fdtd.addmesh(name='plane_xy', x=0., y=0., x_span=span, y_span=span,
            z_min=-4000*nm, z_max=4000*nm,
            override_x_mesh=True, override_y_mesh=True, override_z_mesh=True,
            dx=20*nm, dy=20*nm, dz=20*nm)
        fdtd.addmesh(name='metal_z', x=0., y=0., x_span=span, y_span=span,
            z_min=(-100-args.mesh_nm)*nm, z_max=args.mesh_nm*nm,
            override_x_mesh=False, override_y_mesh=False, override_z_mesh=True, dz=args.mesh_nm*nm)
        fdtd.addplane(name='source', injection_axis='z', direction='Backward',
            x=0., y=0., x_span=span, y_span=span, z=800*nm,
            wavelength_start=700*nm, wavelength_stop=1700*nm)
        for name, z in [('R', 1100*nm), ('T', -700*nm)]:
            fdtd.addpower(name=name, monitor_type='2D Z-normal', x=0., y=0.,
                x_span=span, y_span=span, z=z)
            fdtd.set('override global monitor settings', True)
            fdtd.set('sample spacing', 'custom')
            fdtd.set('custom frequency samples', frequency[::-1])
        grid = np.asarray(fdtd.getresult('FDTD', 'z')).reshape(-1)/nm
        metal_steps = np.diff(grid)[(grid[:-1] < 0.) & (grid[1:] > -100.)]
        layers = int(np.asarray(fdtd.getnamed('FDTD', 'pml layers')).item())
        pml_inner_min, pml_inner_max = grid[layers], grid[-layers-1]
        if not pml_inner_min < -700 < 800 < 1100 < pml_inner_max:
            raise ValueError('Source or a power monitor overlaps the PML')
        metadata = {'case': args.case, 'gold_source': gold_source,
            'native_version': str(fdtd.version()), 'native_parameters': native_parameters,
            'material_metrics': metrics(raw, index_model), 'requested_metal_mesh_nm': args.mesh_nm,
            'actual_metal_max_step_nm': float(metal_steps.max()),
            'native_mesh_refinement': str(fdtd.getnamed('FDTD', 'mesh refinement')),
            'pml_layers': layers, 'pml_inner_z_nm': [float(pml_inner_min), float(pml_inner_max)],
            'source_and_monitors_outside_pml': True,
            'planar_reference_background_maximum_z_mesh_nm': 20.,
            'cells': {a: len(np.asarray(fdtd.getresult('FDTD', a)).reshape(-1))-1 for a in ('x', 'y', 'z')},
            'solver_run_performed': False,
            'note': 'For MEL materials getindex reports base material; full response is validated through native planar R/T.'}
        fdtd.save(str(output/'film.fsp'))
        (output/'preflight.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(metadata), flush=True)
        if not args.run:
            return
        start = time.monotonic()
        fdtd.run()
        metadata.update(solver_run_performed=True, elapsed_seconds=time.monotonic()-start,
            status=int(np.asarray(fdtd.getresult('FDTD', 'status')).item()))
        reflection_data, transmission_data = fdtd.getresult('R', 'T'), fdtd.getresult('T', 'T')
        wave = np.asarray(reflection_data['lambda']).reshape(-1)/nm
        order = np.argsort(wave)
        wave = wave[order]
        np.testing.assert_allclose(wave, np.linspace(700., 1700., 1000), atol=1e-6, rtol=0)
        reflection = np.asarray(reflection_data['T']).reshape(-1)[order]
        transmission = -np.asarray(transmission_data['T']).reshape(-1)[order]
        analytic = thin_film(wave, index_model, glass)
        measured_analytic = thin_film(wave, raw, raw_glass)
        for label, reference in [('model', analytic), ('raw_input', measured_analytic)]:
            metadata[label+'_maximum_R_error'] = float(np.max(abs(reflection-reference[0])))
            metadata[label+'_maximum_T_error'] = float(np.max(abs(transmission-reference[1])))
            metadata[label+'_maximum_A_error'] = float(np.max(abs(1-reflection-transmission-reference[2])))
        metadata['energy_range'] = {'R': [float(reflection.min()), float(reflection.max())],
            'T': [float(transmission.min()), float(transmission.max())],
            'A': [float((1-reflection-transmission).min()), float((1-reflection-transmission).max())]}
        np.savez_compressed(result_path, wavelength_nm=wave, reflectance=reflection,
            transmittance=transmission, absorptance=1-reflection-transmission,
            model_index=index_model, raw_index=raw, glass_model_index=glass, glass_raw_index=raw_glass,
            analytic_R=analytic[0], analytic_T=analytic[1], analytic_A=analytic[2],
            measured_analytic_R=measured_analytic[0], measured_analytic_T=measured_analytic[1],
            measured_analytic_A=measured_analytic[2], mesh_z_nm=grid)
        fdtd.save(str(output/'film.fsp'))
        (output/'result.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
