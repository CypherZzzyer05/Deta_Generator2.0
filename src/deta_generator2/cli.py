"""Windows-friendly configuration, native checks, resumable generation and shard merge."""
import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import shutil
import sys
import numpy as np
import psutil
from .config import load_config
from .lumerical import FDTDSimulator, locate_lumapi, mesh_audit
from .pipeline import atomic_json, directory_lock, prepare_manifest, generate_dataset, pack_dataset
from .merge import merge_shards
from . import __version__

ASSETS = Path(__file__).parent/'assets'


def initialize(directory, preset, samples, threads, lumapi, metal_mesh_nm=None):
    directory = Path(directory).resolve()
    config_path = directory/'job.json'
    if config_path.exists() or (directory.exists() and any(directory.iterdir())):
        raise ValueError('Use an empty job directory; existing jobs are never overwritten')
    metal, mesh = preset.split('-')
    source = ASSETS/'configs'/f'as2se3_{"au" if metal == "bare" else metal}_fine8.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    data.update(lumapi_path=lumapi, output_dir='data', sample_count=samples)
    if mesh == 'metal':
        data['simulation'].update(mesh_policy='metal_only', maximum_mesh_nm=None)
        data['simulation']['metal_mesh_nm'] = 8.0 if metal_mesh_nm is None else metal_mesh_nm
    else:
        if metal_mesh_nm is not None:
            raise ValueError('--metal-mesh-nm requires an au-metal, ag-metal or bare-metal preset')
        data['simulation'].update(mesh_nm=float(mesh), maximum_mesh_nm=float(mesh))
    data['simulation'].update(threads=threads, remove_metal=metal == 'bare')
    data['simulation']['resonator_nk_csv'] = 'materials/as2se3_joseph_700nm.csv'
    data['quality_level'] = 'pipeline_validation'
    # Reject invalid metal steps before creating any task files.
    from .config import SimulationSettings
    SimulationSettings(**data['simulation'])
    directory.mkdir(parents=True, exist_ok=True)
    materials = directory/'materials'
    materials.mkdir()
    for name in ('as2se3_joseph_700nm.csv', 'Joseph-700nm.yml', 'README.md'):
        shutil.copyfile(ASSETS/'materials'/name, materials/name)
    atomic_json(config_path, data)
    load_config(config_path)
    return {'config': str(config_path), 'samples': samples,
            'metal_position': data['simulation']['metal_position'],
            'substrate_termination': data['simulation']['substrate_termination'],
            'layer_order_bottom_to_top': ['SiO2', 'metal', 'As2Se3 tetramer'],
            'mesh_policy': data['simulation'].get('mesh_policy', 'legacy'),
            'mesh_nm': None if mesh == 'metal' else float(mesh),
            'metal_mesh_nm': data['simulation']['metal_mesh_nm']}


def configured(args):
    config = load_config(args.config)
    if config.quadrant_rotation_bounds_deg is None or config.simulation.research_grid:
        raise ValueError('This package generates seven-parameter, 1000-point Metaformer datasets')
    if (config.simulation.maximum_mesh_nm is None and config.simulation.mesh_policy != 'metal_only') or not config.simulation.require_auto_shutoff:
        raise ValueError('Use an audited whole-domain or metal-only mesh and automatic shutoff')
    if config.simulation.metal_material is None:
        raise ValueError('Use a bare preset to preserve the finite reference stack')
    settings = config.simulation
    if args.threads is not None:
        settings = replace(settings, threads=args.threads)
    if args.processes is not None:
        settings = replace(settings, processes=args.processes)
    return replace(config, simulation=settings,
                   output_dir=Path(args.output).resolve() if args.output else config.output_dir,
                   lumapi_path=args.lumapi or config.lumapi_path)


def native_check(config, preflight=False):
    api = locate_lumapi(config.lumapi_path)
    executable = api.parents[2]/'bin'/('fdtd-solutions.exe' if sys.platform == 'win32' else 'fdtd-solutions')
    if not executable.is_file():
        raise FileNotFoundError('FDTD executable missing: '+str(executable))
    report = {'python': sys.executable, 'lumapi': str(api), 'fdtd_executable': str(executable),
              'metal_position': config.simulation.metal_position,
              'substrate_termination': config.simulation.substrate_termination,
              'transmission_medium': 'SiO2' if config.simulation.substrate_termination == 'pml' else 'air',
              'metal_thickness_nm': config.simulation.metal_thickness_nm,
              'logical_cpus': psutil.cpu_count(), 'available_ram_gib': psutil.virtual_memory().available/2**30,
              'processes': config.simulation.processes, 'threads': config.simulation.threads,
              'implementation_sha256': config.metadata()['implementation_sha256'],
              'solver_run_performed': False}
    with FDTDSimulator(config.simulation, str(api)) as simulator:
        report.update(fdtd_version=simulator.version, material_identity=simulator.material_identity())
        if preflight:
            geometry = config.sample_geometries()[0]
            simulator.build(geometry)
            report['native_layout_verified'] = True
            grids = {a: np.asarray(simulator.fdtd.getresult('FDTD', a)).reshape(-1) for a in ('x', 'y', 'z')}
            report['mesh_audit'] = mesh_audit(grids, config.simulation)
            report['actual_mesh'] = report['mesh_audit']['actual_grid']
            resources = simulator.fdtd.runsystemcheck()
            recommended = resources.get('Memory_Recommended_Bytes', resources.get('Memory_Recommended'))
            if recommended is None or not np.isfinite(float(recommended)) or float(recommended) <= 0:
                raise ValueError('Native memory recommendation missing')
            report['recommended_ram_gib'] = float(recommended)/2**30
            report['geometry_values'] = config.geometry_values(geometry).tolist()
    # This output belongs to the job, not the package installation directory.
    path = config.output_dir.parent/('preflight.json' if preflight else 'doctor.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, report)
    return {'report': str(path), 'solver_run_performed': False,
            **{key: value for key, value in report.items() if key != 'material_identity'},
            'material_fit_errors': {name: {key: value for key, value in values.items() if key != 'fitted_index_ri'}
                                    for name, values in report['material_identity'].items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('command', choices=('init', 'doctor', 'preflight', 'plan', 'generate', 'pack', 'merge', 'status', 'parallel'))
    parser.add_argument('--config', type=Path, default=Path('job.json'))
    parser.add_argument('--job-dir', type=Path)
    parser.add_argument('--preset', choices=('au-metal', 'ag-metal', 'bare-metal', 'au-4', 'au-8', 'ag-4', 'ag-8', 'bare-4', 'bare-8'), default='au-metal')
    parser.add_argument('--metal-mesh-nm', type=float, help='Metal z step for init with a *-metal preset (default: 8 nm)')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--jobs', type=int, help='Independent samples simulated at the same time, e.g. 4 or 10')
    parser.add_argument('--workers', type=int, help='Fixed shard count; default uses several small shards per active job')
    parser.add_argument('--concurrency', type=int, help='Legacy alias for the active --jobs limit')
    parser.add_argument('--parallel-action', choices=('start', 'prepare', 'run', 'status', 'merge'), default='start')
    parser.add_argument('--threads', type=int, help='Threads per sample; parallel defaults to usable CPU budget divided by jobs')
    parser.add_argument('--processes', type=int)
    parser.add_argument('--lumapi', help='Path to installed lumapi.py or its containing directory')
    parser.add_argument('--output', type=Path, help='Relocate results without changing physical identity')
    parser.add_argument('--limit', type=int, help='Maximum missing/failed samples to attempt this time')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--pack-every', type=int, default=25, help='Refresh dataset.npz after this many attempted samples; always pack on exit')
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shards', nargs='+', type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s %(levelname)s %(message)s')
    try:
        if args.command == 'parallel':
            if args.metal_mesh_nm is not None or args.output is not None or args.limit is not None or args.processes is not None:
                raise ValueError('Parallel uses a fixed Au100/metal8 task; use --job-dir, --workers, --concurrency and --threads')
            from .parallel import run_parallel
            samples = 100 if args.samples is None else args.samples
            return run_parallel(args.job_dir or Path(f'jobs/pml_topmetal_au100_metal8_{samples}'),
                                samples,args.workers,args.concurrency,args.threads,args.lumapi,
                                args.parallel_action,args.pack_every,jobs=args.jobs)
        if args.jobs is not None or args.workers is not None or args.concurrency is not None:
            raise ValueError('--jobs/--workers/--concurrency require the parallel command; generate runs one structure at a time')
        if args.metal_mesh_nm is not None and args.command != 'init':
            raise ValueError('--metal-mesh-nm is an init option; use a new task when changing the mesh')
        if args.command == 'init':
            samples = 12 if args.samples is None else args.samples
            if samples < 1:
                raise ValueError('samples must be positive')
            if args.threads is not None and args.threads < 1:
                raise ValueError('threads must be positive')
            result = initialize(args.job_dir or Path('jobs/as2se3'), args.preset, samples,
                                args.threads if args.threads is not None else 8, args.lumapi, args.metal_mesh_nm)
        else:
            config = configured(args)
            if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
                raise ValueError('Require shard_count>=1 and 0<=shard_index<shard_count')
            if args.command in ('doctor', 'preflight'):
                result = native_check(config, args.command == 'preflight')
            elif args.command in ('plan', 'pack', 'status'):
                with directory_lock(config.output_dir):
                    manifest = prepare_manifest(config)
                    if args.command == 'pack':
                        result = pack_dataset(config, manifest)
                    else:
                        result = {'planned_samples': len(manifest['samples']), 'output_dir': str(config.output_dir),
                                  'config_fingerprint': manifest['config_fingerprint'],
                                  'parameter_names': manifest['parameter_names'],
                                  'counts': {state: sum(row['state'] == state for row in manifest['samples'])
                                             for state in ('pending', 'running', 'success', 'failed')}}
            elif args.command == 'merge':
                result = merge_shards(config, args.shards or [])
            else:
                if args.shard_count == 1:
                    logging.info('SERIAL: one structure at a time; --threads applies to that structure. For simultaneous structures use parallel --jobs 4 or --jobs 10.')
                if args.shard_count > 1 and args.output is None:
                    raise ValueError('Use a separate --output directory for each shard')
                ids = range(args.shard_index, config.sample_count, args.shard_count)
                result = generate_dataset(config, args.limit, args.fail_fast, ids, args.pack_every)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get('failed', 0) else 0
    except KeyboardInterrupt:
        print('Interrupted. Completed samples are preserved; repeat the same command to resume.', file=sys.stderr)
        return 130
    except Exception as error:
        logging.error('%s', error)
        return 1
