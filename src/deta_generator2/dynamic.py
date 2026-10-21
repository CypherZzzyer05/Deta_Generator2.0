"""Default v2 parallel lifecycle using a shared queue of original sample IDs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .cli import initialize
from .config import load_config
from .merge import merge_shards
from .monitor import _read_json
from .parallel import resolve_resources
from .pipeline import directory_lock, prepare_manifest
from .storage import atomic_json, pipeline_write_retries
from .tail import (controller_lock, inspect_remaining, live_workers, run_tail,
                   source_directories)


def assert_task(root, samples, workers):
    config = load_config(root / 'job.json')
    layout = _read_json(root / 'parallel_plan.json')
    if config.sample_count != samples or layout.get('samples') != samples or layout.get('workers') != workers:
        raise ValueError('Sample count and original partition count are fixed; use a new job directory')
    settings = config.simulation
    if (settings.mesh_policy != 'metal_only' or settings.metal_mesh_nm != 8
            or settings.metal_thickness_nm != 100 or settings.remove_metal
            or settings.metal_position != 'above_substrate' or settings.substrate_termination != 'pml'):
        raise ValueError('Parallel requires v2 thick SiO2 / Au100 / tetramer with metal-only 8 nm mesh')
    return config


def prepare(root, samples, workers, threads, lumapi=None):
    root = Path(root).resolve()
    if (root / 'job.json').is_file():
        config = assert_task(root, samples, workers)
    else:
        initialize(root, 'au-metal', samples, threads, lumapi, metal_mesh_nm=8)
        atomic_json(root / 'parallel_plan.json', {'samples': samples, 'workers': workers,
                                                'scheduler': 'per_sample_queue_v1'})
        config = load_config(root / 'job.json')
    with directory_lock(config.output_dir), pipeline_write_retries():
        prepare_manifest(config)
    print(f'Prepared {samples} original sample IDs. Dynamic scheduling enabled. No FDTD run.', flush=True)
    return config


def status(root, jobs):
    path = root / 'tail_resume/status.json'
    if path.exists():
        value = _read_json(path)
        print(f"success={value['success']}/{value['samples']}, running={value['running']}, "
              f"failed={value['failed']}, pending={value['pending']}; "
              f"native_engines={value['native_engines_observed_lower_bound']} (observed lower bound)")
        print('Latest controller snapshot: ' + value['time'])
        for row in value.get('active_samples', []):
            percent = '?' if row['percent_complete'] is None else f"{row['percent_complete']:.2f}%"
            print(f"  sample_{row['sample_id']:06d}: {percent}, "
                  f"Auto Shutoff={row['auto_shutoff']}, worker_alive={row['worker_alive']}")
        return value
    value = inspect_remaining(root)
    print(f"success={value['manifest_success']}/{value['samples']}, "
          f"remaining={value['remaining_count']}; recorded live workers={len(value['live_worker_pids'])}")
    return value


def merge_only(root, config):
    destination = root / 'merged'
    if live_workers(root):
        raise RuntimeError('Workers are still active; merge refused')
    with controller_lock(root):
        sources = [p for p in source_directories(root) if p.resolve() != destination]
        result = merge_shards(replace(config, output_dir=destination), sources)
    if result['success'] != config.sample_count:
        raise RuntimeError(f"Only {result['success']}/{config.sample_count} saved samples; no solver started")
    print('Dataset ready: ' + str(destination / 'dataset.npz'), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'run', 'merge', 'status'), required=True)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--concurrency', type=int)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--lumapi')
    parser.add_argument('--pack-every', type=int, default=25)
    args = parser.parse_args(argv)
    try:
        root = args.job_dir.resolve()
        resources = resolve_resources(root, args.samples, args.workers, args.concurrency, args.threads, args.jobs)
        workers, jobs, threads = (resources[key] for key in ('workers', 'concurrency', 'threads'))
        if args.action == 'prepare':
            prepare(root, args.samples, workers, threads, args.lumapi)
            return 0
        config = assert_task(root, args.samples, workers)
        if args.action == 'status':
            status(root, jobs)
            return 0
        if args.action == 'merge':
            result = merge_only(root, config)
        else:
            print(f'DYNAMIC: up to {jobs} concurrent samples, {threads} threads each. Idle slots take the next sample.', flush=True)
            result = run_tail(root, jobs, threads, args.lumapi,
                              result_dir=root / 'merged', label='PARALLEL')
            print('Dataset ready: ' + str(root / 'merged/dataset.npz'), flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print('Interrupted; saved results remain. Repeat the same parallel command to resume.', file=sys.stderr)
        return 130
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
