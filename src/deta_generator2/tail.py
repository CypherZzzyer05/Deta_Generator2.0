"""Schedule original sample IDs in independent, refillable process slots.

The sampler, electromagnetic solver, physical fingerprint and successful source
files remain unchanged. Each new child receives exactly one original sample ID.
"""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from .config import load_config
from .merge import merge_shards
from .monitor import _read_json, _process_name, _observe_worker, read_progress
from .pipeline import (checksum, directory_lock, load_sample,
                       plan_checksum, utc_now)
from .resources import cpu_inventory
from .storage import atomic_json

write_json = atomic_json


def source_directories(job_dir):
    root = Path(job_dir).resolve()
    layout = _read_json(root / 'parallel_plan.json')
    count = layout.get('workers')
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 64:
        raise ValueError('Invalid original shard count')
    sources = [root / f'shard{i}' for i in range(count)
               if (root / f'shard{i}' / 'manifest.json').is_file()]
    if (root / 'data' / 'manifest.json').is_file():
        sources.append(root / 'data')
    if (root / 'merged' / 'manifest.json').is_file():
        sources.append(root / 'merged')
    sources += sorted(p for p in (root / 'tail_resume').glob('sample_*')
                      if p.is_dir() and (p / 'manifest.json').is_file())
    if not sources:
        raise ValueError('No existing shard manifests; use the original parallel entry first')
    return sources


def worker_records(job_dir):
    root = Path(job_dir).resolve()
    records = []
    for file in (root / 'parallel_workers.json', root / 'tail_resume/workers.json'):
        if file.is_file():
            records.extend(_read_json(file).get('workers', []))
    return records


def live_workers(job_dir, process_api=psutil):
    """Validate both creation time and name before trusting a stored PID."""
    result = {}
    for row in worker_records(job_dir):
        try:
            process = process_api.Process(int(row['pid']))
            if not process.is_running():
                continue
            if _process_name(process.name()) != _process_name(row['process_name']):
                continue
            started = row.get('started_at_unix')
            if started is None:
                raise RuntimeError('Live worker has no recorded creation time; cannot safely take over')
            tolerance = .001 if row.get('creation_time_source') == 'psutil' else 2
            if abs(process.create_time() - float(started)) > tolerance:
                continue  # Windows has reused this PID for another process.
            result[process.pid] = process
        except process_api.NoSuchProcess:
            continue
        except process_api.AccessDenied as error:
            raise RuntimeError('Cannot verify recorded worker identity; run as the same Windows user') from error
    return list(result.values())


def _argument(argv, flag):
    matches = [i for i, value in enumerate(argv[:-1]) if value.lower() == flag.lower()]
    return argv[matches[0] + 1] if len(matches) == 1 else None


def _same_path(left, right):
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))


def verified_controller(process, job_dir):
    """Recognize only the original controller naming this exact task directory."""
    argv = process.cmdline()
    if _process_name(process.name()) not in ('powershell', 'pwsh'):
        module = _argument(argv, '-m')
        directory = _argument(argv, '--job-dir')
        this_script = len(argv) > 1 and argv[1].lower().endswith('.py') and _same_path(argv[1], __file__)
        return bool((module in ('deta_generator2.tail', 'deta_generator2.dynamic') or this_script) and directory
                    and _argument(argv, '--action') == 'run' and _same_path(directory, job_dir))
    script, directory = _argument(argv, '-File'), _argument(argv, '-JobDir')
    return bool(script and directory and Path(script).name.lower() == 'run-parallel.ps1'
                and _same_path(directory, job_dir))


def _stop_processes(processes, process_api=psutil):
    """Stop only supplied live process identities and their captured children."""
    children = {}
    for process in processes:
        for child in process.children(recursive=True):
            children[child.pid] = child
    # Prevent parents from starting another sample before stopping native children.
    roots = {p.pid: p for p in processes}
    ordered = [*roots.values(), *(p for pid, p in children.items() if pid not in roots)]
    for process in ordered:
        try:
            if process.is_running():
                process.terminate()
        except process_api.NoSuchProcess:
            pass
    _, alive = process_api.wait_procs(ordered, timeout=5)
    for process in alive:
        if process.is_running():
            process.kill()
    _, alive = process_api.wait_procs(alive, timeout=5)
    if alive:
        raise RuntimeError('Some owned processes did not stop; takeover refused')


def stop_original(job_dir, process_api=psutil):
    """Explicit stop action. In-flight samples can lose their current progress."""
    workers = live_workers(job_dir, process_api)
    controllers = {}
    for worker in workers:
        parent = worker.parent()
        # Windows venv launchers can insert a Python redirector between the
        # recorded worker and the controller. Follow only Python ancestors.
        for _ in range(5):
            if parent is None:
                break
            if verified_controller(parent, job_dir):
                controllers[parent.pid] = parent
                break
            if not (_process_name(parent.name()).startswith('python') or _process_name(parent.name()) == 'py'):
                break
            parent = parent.parent()
    # Capturing the full tree before stopping the controller retains native PIDs.
    _stop_processes([*controllers.values(), *workers], process_api)
    if live_workers(job_dir, process_api):
        raise RuntimeError('Original workers remain active')
    return {'stopped_workers': len(workers), 'stopped_controllers': len(controllers),
            'successful_files_preserved': True,
            'inflight_progress': 'Unsaved in-flight samples will be recomputed'}


@contextmanager
def controller_lock(job_dir):
    """Use the same controller lock file as the old PowerShell launcher."""
    path = Path(job_dir) / '.parallel-controller.lock'
    try:
        stream = path.open('a+b')
    except OSError as error:
        raise RuntimeError('Original controller is still active; stop it before takeover') from error
    with stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another controller is using this job') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def inspect_remaining(job_dir):
    """Read-only manifest preview; run additionally validates all successful files."""
    root = Path(job_dir).resolve()
    config = load_config(root / 'job.json')
    layout = _read_json(root / 'parallel_plan.json')
    if layout.get('samples') != config.sample_count:
        raise ValueError('Original plan and job disagree on sample count')
    expected = hashlib.sha256(json.dumps(config.metadata(), sort_keys=True).encode()).hexdigest()
    success, running, canonical = {}, set(), None
    sources = source_directories(root)
    target = root / 'tail_resume/merged'
    if (target / 'manifest.json').is_file():
        sources.append(target)
    for directory in sources:
        manifest = _read_json(directory / 'manifest.json')
        if manifest['config_fingerprint'] != expected:
            raise ValueError('Physical implementation/configuration differs: ' + str(directory))
        digest = plan_checksum(manifest)
        if digest != manifest.get('sampling_plan_sha256') or (canonical is not None and digest != canonical):
            raise ValueError('Sampling plans disagree or have changed')
        canonical = digest
        if [r['id'] for r in manifest['samples']] != list(range(config.sample_count)):
            raise ValueError('Incomplete original sampling plan')
        for row in manifest['samples']:
            if row['state'] == 'success':
                previous = success.get(row['id'])
                if previous is not None and previous != row['sha256']:
                    raise ValueError('Conflicting duplicate sample: ' + str(row['id']))
                success[row['id']] = row['sha256']
            elif row['state'] == 'running':
                running.add(row['id'])
    remaining = sorted(set(range(config.sample_count)) - success.keys())
    return {'samples': config.sample_count, 'manifest_success': len(success),
            'remaining_count': len(remaining), 'remaining_ids': remaining,
            'running_ids_without_saved_success': sorted(running - success.keys()),
            'live_worker_pids': [p.pid for p in live_workers(root)],
            'sampling_plan_sha256': canonical,
            'preview_only': 'Successful files are fully validated before run; no FDTD in this preview'}


def _child_command(root, sample_id, output, threads, lumapi):
    config = load_config(root / 'job.json')
    command = [sys.executable, '-m', 'deta_generator2.worker', 'generate',
               '--config', str(root / 'job.json'), '--processes', '1', '--threads', str(threads),
               '--shard-count', str(config.sample_count), '--shard-index', str(sample_id),
               '--output', str(output), '--pack-every', '1', '--fail-fast']
    if lumapi:
        command.extend(['--lumapi', lumapi])
    return command


def _snapshot(root, total, successful, pending, active, failed, jobs, label='TAIL'):
    engines = {}
    workers = []
    samples = []
    for row in active.values():
        observed = _observe_worker(row['identity'], engines, psutil)
        workers.append(observed)
        samples.append({**read_progress(row['output'] / 'projects', row['id'], time.time()),
                        'worker_alive': observed['alive']})
    result = {'time': utc_now(), 'samples': total, 'success': successful,
              'running': len(active), 'failed': len(failed), 'pending': len(pending),
              'capacity': jobs, 'native_engines_observed_lower_bound': len(engines),
              'active_samples': samples, 'workers': workers, 'failed_ids': failed}
    write_json(root / 'tail_resume/status.json', result)
    print(f"{label} success={successful}/{total}, running={len(active)}, failed={len(failed)}, "
          f"pending={len(pending)}; native_engines={len(engines)} (observed lower bound)", flush=True)
    for row in samples:
        percent = '?' if row['percent_complete'] is None else f"{row['percent_complete']:.2f}%"
        print(f"  sample_{row['sample_id']:06d}: {percent}, Auto Shutoff={row['auto_shutoff']}", flush=True)
    return result


def run_tail(job_dir, jobs=10, threads=None, lumapi=None, poll_seconds=.5, result_dir=None, label='TAIL'):
    root = Path(job_dir).resolve()
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 64:
        raise ValueError('jobs must be in [1,64]')
    budget = cpu_inventory()['budget_cpus']
    threads = max(1, budget // jobs) if threads is None else threads
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1 or jobs * threads > budget:
        raise ValueError('jobs times threads exceeds CPU budget; omit threads for automatic allocation')
    config = load_config(root / 'job.json')
    if config.quadrant_rotation_bounds_deg is None or not config.simulation.require_auto_shutoff:
        raise ValueError('Tail takeover requires the existing audited seven-parameter task')
    if live_workers(root):
        raise RuntimeError('Original workers are still alive; run --action stop first')
    output = Path(result_dir).resolve() if result_dir is not None else root / 'tail_resume/merged'
    config = replace(config, output_dir=output, lumapi_path=lumapi or config.lumapi_path)
    with controller_lock(root):
        if live_workers(root):
            raise RuntimeError('Workers started concurrently; takeover refused')
        (root / 'tail_resume').mkdir(parents=True, exist_ok=True)
        inspect_remaining(root)
        sources = [p for p in source_directories(root) if p.resolve() != output]
        # This copies and fully validates all existing successes before new work.
        summary = merge_shards(config, sources)
        manifest = _read_json(output / 'manifest.json')
        successful_ids = {row['id'] for row in manifest['samples'] if row['state'] == 'success'}
        queue = sorted(set(range(config.sample_count)) - successful_ids)
        write_json(root / 'tail_resume/plan.json', {
            'samples': config.sample_count, 'jobs': jobs, 'threads': threads,
            'preserved_success': len(successful_ids), 'remaining_ids': queue,
            'sampling_plan_sha256': manifest['sampling_plan_sha256'],
            'source_directories': [str(p) for p in sources], 'started_at': utc_now()})
        print(f"Preserved and verified {len(successful_ids)} successes. Redistributing {len(queue)} "
              f"original sample IDs, up to {jobs} at once, {threads} threads each.", flush=True)
        active, failures, records = {}, [], []
        last_display = 0.
        # Old files have a single owner again only after the entire takeover ends.
        with ExitStack() as source_locks:
            for directory in sorted(sources):
                if not directory.is_relative_to(root / 'tail_resume'):
                    source_locks.enter_context(directory_lock(directory))
            try:
                while queue or active:
                    for sample_id, row in list(active.items()):
                        code = row['process'].poll()
                        if code is None:
                            continue
                        row['process'].wait()
                        try:
                            if code != 0:
                                raise ValueError('Child exit code: ' + str(code))
                            child = _read_json(row['output'] / 'manifest.json')
                            entry = child['samples'][sample_id]
                            if entry['state'] != 'success':
                                raise ValueError('Child did not save its assigned sample')
                            load_sample(row['output'] / 'samples' / f'sample_{sample_id:06d}.npz',
                                        entry, config, manifest['config_fingerprint'])
                            successful_ids.add(sample_id)
                        except (OSError, ValueError, KeyError) as error:
                            failures.append(sample_id)
                            print(f'Failed sample_{sample_id:06d}: {error}', flush=True)
                        del active[sample_id]
                    # Refill immediately after reaping, before reporting status.
                    while queue and len(active) < jobs:
                        sample_id = queue.pop(0)
                        directory = root / 'tail_resume' / f'sample_{sample_id:06d}'
                        logs = root / 'tail_resume/logs'
                        logs.mkdir(parents=True, exist_ok=True)
                        stdout = (logs / f'sample_{sample_id:06d}.stdout.log').open('ab')
                        stderr = (logs / f'sample_{sample_id:06d}.stderr.log').open('ab')
                        try:
                            process = subprocess.Popen(_child_command(root, sample_id, directory, threads, lumapi),
                                cwd=root, stdout=stdout, stderr=stderr,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                        finally:
                            stdout.close()
                            stderr.close()
                        native = psutil.Process(process.pid)
                        identity = {'index': sample_id, 'pid': process.pid,
                                    'process_name': native.name(), 'started_at_unix': native.create_time(),
                                    'creation_time_source': 'psutil'}
                        active[sample_id] = {'id': sample_id, 'process': process, 'output': directory,
                                             'identity': identity}
                        records.append(identity)
                        write_json(root / 'tail_resume/workers.json', {'workers': records})
                        print(f'Started sample_{sample_id:06d}, PID {process.pid}', flush=True)
                    if time.monotonic() - last_display >= 30 or (not queue and not active):
                        _snapshot(root, config.sample_count, len(successful_ids), queue, active, failures, jobs, label)
                        last_display = time.monotonic()
                    if active:
                        time.sleep(poll_seconds)
            finally:
                if active:
                    _stop_processes([psutil.Process(row['process'].pid) for row in active.values()
                                     if row['process'].poll() is None])
                    for row in active.values():
                        row['process'].wait(timeout=10)
        if len(successful_ids) != config.sample_count:
            raise RuntimeError(f"Only {len(successful_ids)}/{config.sample_count} successful; "
                               'repeat the same parallel command to resume')
        summary = merge_shards(config, [p for p in source_directories(root) if p.resolve() != output])
        write_json(root / 'tail_resume/result.json', summary)
        if summary['success'] != config.sample_count:
            raise RuntimeError(f"Only {summary['success']}/{config.sample_count} successful; rerun the same tail command")
        _snapshot(root, config.sample_count, summary['success'], [], {}, [], jobs, label)
        print('Completed dataset: ' + str(output / 'dataset.npz'), flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--action', choices=('plan', 'stop', 'run', 'status'), default='plan')
    parser.add_argument('--jobs', type=int, default=10)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--lumapi')
    args = parser.parse_args(argv)
    try:
        if args.action == 'plan':
            result = inspect_remaining(args.job_dir)
        elif args.action == 'stop':
            result = stop_original(args.job_dir)
        elif args.action == 'status':
            path = args.job_dir / 'tail_resume/status.json'
            result = _read_json(path) if path.is_file() else inspect_remaining(args.job_dir)
        else:
            result = run_tail(args.job_dir, args.jobs, args.threads, args.lumapi)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Saved original and tail results remain; repeat --action run to resume.', file=sys.stderr)
        return 130
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
