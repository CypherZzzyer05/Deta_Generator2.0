"""Merge independently computed shards while verifying every sample's provenance."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import numpy as np
from .pipeline import directory_lock, prepare_manifest, load_sample, pack_dataset, checksum, plan_checksum
from .storage import atomic_json, resilient_writes, retry_io


@resilient_writes
def merge_shards(config, directories):
    directories = [Path(p).resolve() for p in directories]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError('Supply distinct existing shard directories')
    if config.output_dir.resolve() in directories:
        raise ValueError('Merge output must differ from every shard')
    if any(not (p/'manifest.json').is_file() for p in directories):
        raise ValueError('Each shard must have a manifest')
    with ExitStack() as stack:
        # Lock in a deterministic order; a live worker is rejected rather than
        # taking a partially updated snapshot or moving its files.
        for directory in sorted([*directories, config.output_dir.resolve()]):
            stack.enter_context(directory_lock(directory))
        target = prepare_manifest(config)
        incoming = {}
        identities = []
        versions = {target['fdtd_version']} if 'fdtd_version' in target else set()
        for directory in directories:
            manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
            if manifest['config_fingerprint'] != target['config_fingerprint']:
                raise ValueError('Shard physics/sampling configuration differs: '+str(directory))
            if manifest.get('sampling_plan_sha256') != target['sampling_plan_sha256'] or plan_checksum(manifest) != target['sampling_plan_sha256']:
                raise ValueError('Shard sampling plan differs or was modified')
            if 'fdtd_version' in manifest:
                versions.add(manifest['fdtd_version'])
            if 'material_identity' in manifest:
                identities.append(manifest['material_identity'])
            for entry in manifest['samples']:
                if entry['state'] != 'success':
                    continue
                expected = target['samples'][entry['id']]
                if entry['geometry_values'] != expected['geometry_values'] or entry['split'] != expected['split']:
                    raise ValueError('Shard does not use the canonical sampling plan')
                path = directory/'samples'/f"sample_{entry['id']:06d}.npz"
                load_sample(path, expected | {'sha256': entry['sha256']}, config, target['config_fingerprint'])
                previous = incoming.get(entry['id'])
                if previous and previous[1]['sha256'] != entry['sha256']:
                    raise ValueError('Conflicting duplicate sample ID: '+str(entry['id']))
                if expected['state'] == 'success' and expected['sha256'] != entry['sha256']:
                    raise ValueError('Sample conflicts with merge output: '+str(entry['id']))
                incoming[entry['id']] = (path, entry)
        if len(versions) > 1:
            raise ValueError('Shards use different solver versions')
        if 'material_identity' in target:
            identities.append(target['material_identity'])
        if identities:
            reference = identities[0]
            if any(set(i) != set(reference) or any(not np.allclose(
                    i[name]['fitted_index_ri'], reference[name]['fitted_index_ri'], rtol=1e-7, atol=1e-7)
                    for name in reference) for i in identities[1:]):
                raise ValueError('Shards use different actual material fits')
            target['material_identity'] = reference
        if versions:
            target['fdtd_version'] = versions.pop()
        samples = config.output_dir/'samples'
        samples.mkdir(exist_ok=True)
        for i, (path, entry) in sorted(incoming.items()):
            destination = samples/path.name
            temporary = destination.with_suffix('.npz.tmp')
            retry_io(shutil.copyfile)(path, temporary)
            if checksum(temporary) != entry['sha256']:
                raise ValueError('Copied sample failed checksum')
            retry_io(temporary.replace)(destination)
            csv = path.with_suffix('.csv')
            if csv.is_file():
                retry_io(shutil.copyfile)(csv, destination.with_suffix('.csv'))
            target['samples'][i] = entry
            atomic_json(config.output_dir/'manifest.json', target)
        target['merged_shards'] = [str(p) for p in directories]
        atomic_json(config.output_dir/'manifest.json', target)
        return pack_dataset(config, target)
