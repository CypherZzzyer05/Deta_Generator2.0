"""Persistent sampling plan, atomic sample export, resume and dataset packing."""
# 数据生成调度：先固定几何与划分计划，再逐个调用 FDTD，最后汇总为训练用 NPZ。
# manifest.json 记录每个样本的 ID、几何、划分、状态和校验值，支持中断后补齐。

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import traceback

import numpy as np

from .config import Geometry, PARAMETER_NAMES
from .lumerical import FDTDSimulator, validate_spectrum, mesh_audit

SCHEMA_VERSION = 1
LOG = logging.getLogger(__name__)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 临时文件写完后再替换目标文件，使磁盘上的进度记录尽量保持完整。
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 使用文件句柄，避免 NumPy 自动追加 .npz 后缀，导致临时文件名与替换目标不一致。
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def generation_log(directory):
    handler = logging.FileHandler(directory / "generation.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    try:
        yield
    finally:
        LOG.removeHandler(handler)
        handler.close()


@contextmanager
def directory_lock(directory):
    """OS-held lock is released even if the process is killed in PyCharm."""
    # 使用操作系统文件锁，防止两个进程同时更新同一目录；进程退出会释放锁。
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".generation.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another generator is using {directory}") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def split_assignments(count, seed):
    """Assign before simulation so later successes never move between splits."""
    # 在仿真前划分，避免按仿真成功顺序重新分组；100 个样本时得到 80/10/10。
    order = np.random.default_rng(seed + 1).permutation(count)
    n_val = max(1, round(count * .1)) if count >= 3 else 0
    n_test = max(1, round(count * .1)) if count >= 3 else 0
    n_train = count - n_val - n_test
    result = ["train"] * count
    for i in order[n_train:n_train+n_val]:
        result[int(i)] = "val"
    for i in order[n_train+n_val:]:
        result[int(i)] = "test"
    return result


def plan_checksum(manifest):
    rows = [{key: row[key] for key in ('id', 'geometry_nm', 'geometry_values', 'split')}
            for row in manifest['samples']]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def prepare_manifest(config):
    data = config.metadata()
    fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    path = config.output_dir / "manifest.json"
    schema_version = 3 if config.quadrant_rotation_bounds_deg is not None else (2 if config.rotation_bounds_deg is not None else SCHEMA_VERSION)
    if path.exists():
        # 续跑复用原始计划，并核对整套配置，防止混合不同网格、几何范围或样本计划。
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != schema_version or manifest.get("config_fingerprint") != fingerprint:
            raise ValueError("Output directory contains a different configuration; choose a new output_dir")
        if (manifest.get('sampling_plan_sha256') != plan_checksum(manifest) or
                [row['id'] for row in manifest['samples']] != list(range(config.sample_count))):
            raise ValueError('Sampling plan changed or is incomplete')
        return manifest
    if any(p.name != ".generation.lock" for p in config.output_dir.iterdir()):
        raise ValueError("Output directory has files but no manifest; choose an empty output_dir")
    geometries = config.sample_geometries()
    # 保存完整计划后再启动求解，即使中途失败也不改变样本的几何和数据划分。
    manifest = {
        "schema_version": schema_version, "created_at": utc_now(),
        "config_fingerprint": fingerprint, "config": data,
        "parameter_names": list(config.parameter_names), "geometry_units": "nm",
        "spectral_grid": "uniform ascending wavelength_nm; 1000 points",
        "backend": "Lumerical FDTD lumapi (real electromagnetic simulation)",
        "samples": [
            {"id": i, "geometry_nm": g.array().tolist(), "split": split, "state": "pending"}
            for i, (g, split) in enumerate(zip(geometries, split_assignments(len(geometries), config.seed)))
        ],
    }
    if config.rotation_bounds_deg is not None:
        manifest["geometry_units"] = "mixed; see parameter_units"
        manifest["parameter_units"] = list(config.parameter_units)
        manifest["rotation_mode"] = "each_square_own_center_common_z_angle"
        for entry, geometry in zip(manifest["samples"], geometries):
            entry["rotation_deg"] = float(geometry.rotation_deg)
            entry["geometry_values"] = config.geometry_values(geometry).tolist()
    if config.quadrant_rotation_bounds_deg is not None:
        manifest.update(geometry_units="mixed; see parameter_units",
                        parameter_units=list(config.parameter_units),
                        rotation_mode="self_center_z_I_III_odd_II_IV_even")
        for entry, geometry in zip(manifest["samples"], geometries):
            entry["geometry_values"] = config.geometry_values(geometry).tolist()
    manifest['sampling_plan_sha256'] = plan_checksum(manifest)
    atomic_json(path, manifest)
    return manifest


def load_sample(path, entry, config, fingerprint):
    # 重用样本前核对文件完整性、配置来源、几何标签及光谱质量。
    if entry.get("sha256") != checksum(path):
        raise ValueError(f"Sample checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    if not bool(result.get('native_layout_verified', False)):
        raise ValueError('Sample is missing native layout verification')
    if str(result["config_fingerprint"].item()) != fingerprint:
        raise ValueError("Sample provenance does not match this run")
    if not np.array_equal(result["geometry_nm"], entry["geometry_nm"]):
        raise ValueError("Sample geometry does not match the saved sampling plan")
    values = result["geometry_nm"]
    if config.rotation_bounds_deg is not None:
        values = result["geometry_values"]
        if (not np.array_equal(values, entry["geometry_values"])
                or float(result["rotation_deg"]) != entry["rotation_deg"]
                or not np.array_equal(values[:5], result["geometry_nm"])
                or values[5] != float(result["rotation_deg"])):
            raise ValueError("Sample rotation or mixed-unit geometry does not match the plan")
    if config.quadrant_rotation_bounds_deg is not None:
        values = result["geometry_values"]
        if not np.array_equal(values, entry["geometry_values"]) or not np.array_equal(values[:5], result["geometry_nm"]):
            raise ValueError("Sample quadrant geometry does not match the saved plan")
    if not np.allclose(result["geometry_01"], config.normalize(values), atol=1e-7):
        raise ValueError("Sample labels do not match normalization bounds")
    validate_spectrum(result["wavelength_nm"], result["reflectance"], result["transmittance"],
                      config.simulation, int(result["status"].item()))
    if config.simulation.maximum_mesh_nm is not None or config.simulation.mesh_policy == 'metal_only':
        keys = [f"mesh_{axis}_nm" for axis in ("x", "y", "z")]
        if any(key not in result for key in keys):
            raise ValueError("Mesh-audited samples require saved actual mesh coordinates")
        mesh_audit({axis: result[f"mesh_{axis}_nm"] * 1e-9 for axis in ("x", "y", "z")}, config.simulation)
    return result


def pack_dataset(config, manifest):
    samples, entries = [], []
    # 只汇总成功并通过复查的样本，失败项仍留在 manifest 中等待重试。
    for entry in manifest["samples"]:
        if entry["state"] == "success":
            path = config.output_dir / "samples" / f"sample_{entry['id']:06d}.npz"
            samples.append(load_sample(path, entry, config, manifest["config_fingerprint"]))
            entries.append(entry)
    n = len(entries)
    versions = {str(s['fdtd_version'].item()) for s in samples}
    if len(versions) > 1:
        raise ValueError('Cannot pack samples from different solver versions')
    if versions and versions != {manifest.get('fdtd_version')}:
        raise ValueError('Sample solver version differs from the manifest')
    grid = config.simulation.wavelength_grid()
    # 网络输入使用反射率 [N,1000]；透射率和吸收率另存，供物理检查和展示。
    # 保留源功率归一化的反射率，不对每条光谱再做 min-max 缩放。
    spectra = np.stack([s["reflectance"] for s in samples]).astype(np.float32) if n else np.empty((0, 1000), np.float32)
    geometry = np.stack([s["geometry_nm"] for s in samples]) if n else np.empty((0, 5), np.float64)
    values = geometry
    if config.rotation_bounds_deg is not None:
        values = np.stack([s["geometry_values"] for s in samples]) if n else np.empty((0, 6), np.float64)
    if config.quadrant_rotation_bounds_deg is not None:
        values = np.stack([s["geometry_values"] for s in samples]) if n else np.empty((0, 7), np.float64)
    ids = np.array([e["id"] for e in entries], dtype=np.int64)
    splits = np.array([e["split"] for e in entries], dtype="U5")
    metadata = {
        "schema_version": manifest["schema_version"], "backend": manifest["backend"],
        "config": manifest["config"], "config_fingerprint": manifest["config_fingerprint"],
        "parameter_names": list(config.parameter_names), "successful_samples": n,
        "planned_samples": config.sample_count, "exported_at": utc_now(),
        "time_limit_policy": "Initial configured limit; double on status=1, at most four attempts",
    }
    metadata['material_identity'] = manifest.get('material_identity', {})
    metadata['fdtd_versions'] = sorted(versions)
    arrays = {
        "spectra": spectra, "geometry_nm": geometry,
        "geometry_01": config.normalize(values).astype(np.float32),
        "wavelength_nm": grid, "parameter_names": np.array(config.parameter_names),
        "lower_nm": config.lower[:5], "upper_nm": config.upper[:5],
        "sample_ids": ids, "split_names": splits,
        "status": np.array([int(s["status"]) for s in samples], dtype=np.int64),
        "simulation_time_ps": np.array([float(s.get("simulation_time_ps", config.simulation.simulation_time_ps))
                                         for s in samples], dtype=np.float64),
        "run_attempts": np.array([int(s.get("run_attempts", 1)) for s in samples], dtype=np.int64),
        "metadata_json": np.array(json.dumps(metadata, ensure_ascii=False)),
    }
    if config.rotation_bounds_deg is not None:
        metadata.update(parameter_units=list(config.parameter_units),
                        rotation_mode=manifest["rotation_mode"],
                        angle_equivalence="square orientation period 90deg; +/-theta are mirror-equivalent for this x-polarized reflection task")
        arrays.update(geometry_values=values, rotation_deg=values[:, 5],
                      parameter_units=np.array(config.parameter_units),
                      lower_values=config.lower, upper_values=config.upper,
                      metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
    if config.quadrant_rotation_bounds_deg is not None:
        metadata.update(parameter_units=list(config.parameter_units), rotation_mode=manifest["rotation_mode"],
                        angle_equivalence="each square has 90-degree orientation period; quadrant signs are coupled")
        arrays.update(geometry_values=values, parameter_units=np.array(config.parameter_units),
                      lower_values=config.lower, upper_values=config.upper,
                      metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
    # 同时保存 nm 标签、归一化标签、上下界和波长网格，保证训练/推理的单位一致。
    for name in ("transmittance", "absorptance"):
        arrays[name] = np.stack([s[name] for s in samples]).astype(np.float32) if n else np.empty((0, 1000), np.float32)
    for split in ("train", "val", "test"):
        # 索引指向汇总数组的行；sample_ids 则保留原计划 ID，两者含义不同。
        arrays[split + "_indices"] = np.flatnonzero(splits == split).astype(np.int64)
    path = config.output_dir / "dataset.npz"
    atomic_npz(path, **arrays)
    summary = {
        "planned": config.sample_count, "success": n,
        "failed": sum(e["state"] == "failed" for e in manifest["samples"]),
        "pending": sum(e["state"] not in ("failed", "success") for e in manifest["samples"]),
        "splits": {s: int(np.sum(splits == s)) for s in ("train", "val", "test")},
        "dataset": str(path), "spectra_shape": list(spectra.shape),
        "geometry_shape": [n, len(config.parameter_names)], "quality_level": config.quality_level,
    }
    atomic_json(config.output_dir / "summary.json", summary)
    return summary


def generate_dataset(config, limit=None, fail_fast=False, sample_ids=None, pack_every=25):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not isinstance(pack_every, int) or isinstance(pack_every, bool) or pack_every < 1:
        raise ValueError('pack_every must be a positive integer')
    # limit 控制本次补算多少个样本，不改变已经固定的总采样计划。
    with directory_lock(config.output_dir):
        manifest = prepare_manifest(config)
        with generation_log(config.output_dir):
            return _generate_locked(config, manifest, limit, fail_fast, sample_ids, pack_every)


def _generate_locked(config, manifest, limit, fail_fast, sample_ids=None, pack_every=25):
    samples_dir = config.output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    pending = []
    if sample_ids is not None:
        sample_ids = set(sample_ids)
        if not sample_ids <= set(range(config.sample_count)):
            raise ValueError('Selected sample IDs are outside the fixed plan')
    for entry in manifest["samples"]:
        if sample_ids is not None and entry['id'] not in sample_ids:
            continue
        if entry["state"] == "success":
            path = samples_dir / f"sample_{entry['id']:06d}.npz"
            try:
                # 已完成样本通过校验后跳过；损坏或来源不符时才加入待生成列表。
                load_sample(path, entry, config, manifest["config_fingerprint"])
                continue
            except (OSError, ValueError, KeyError) as error:
                LOG.warning("Sample %s invalid on disk; regenerating: %s", entry["id"], error)
                entry["state"] = "pending"
                entry.pop("sha256", None)
        pending.append(entry)
    atomic_json(config.output_dir / "manifest.json", manifest)
    if not pending:
        LOG.info("All samples verified; packing existing results without launching FDTD")
        return pack_dataset(config, manifest)
    if limit is not None:
        pending = pending[:limit]
    completed = 0
    processed = 0
    try:
        # 多个样本复用同一个 API 会话；每个样本仍会重新构造和运行其 FDTD 模型。
        with FDTDSimulator(config.simulation, config.lumapi_path) as simulator:
            if manifest.get('fdtd_version') not in (None, simulator.version):
                raise ValueError('Solver version changed; use a separate output directory')
            manifest["fdtd_version"] = simulator.version
            manifest["lumapi_file"] = str(simulator.lumapi.__file__)
            identity = simulator.material_identity()
            if 'material_identity' in manifest:
                previous = manifest['material_identity']
                if set(previous) != set(identity) or any(not np.allclose(
                        previous[name]['fitted_index_ri'], identity[name]['fitted_index_ri'],
                        rtol=1e-7, atol=1e-7) for name in identity):
                    raise ValueError('Actual material fits changed; use a separate output directory')
            else:
                manifest['material_identity'] = identity
            manifest.setdefault('runtime_history', []).append({
                'started_at': utc_now(), 'lumapi_file': manifest['lumapi_file'],
                'output_dir': str(config.output_dir), 'processes': config.simulation.processes,
                'threads': config.simulation.threads, 'save_projects': config.simulation.save_projects})
            for entry in pending:
                i = entry["id"]
                g = Geometry(*entry["geometry_nm"], rotation_deg=entry.get("rotation_deg", 0.0))
                if config.quadrant_rotation_bounds_deg is not None:
                    g = config.make_geometry(entry["geometry_values"])
                entry.update(state="running", started_at=utc_now())
                atomic_json(config.output_dir / "manifest.json", manifest)
                LOG.info("Sample %d/%d: %s = %s; units=%s", i+1, config.sample_count,
                         config.parameter_names, np.round(config.geometry_values(g), 2), config.parameter_units)
                try:
                    project = config.output_dir / "projects" / f"sample_{i:06d}.fsp"
                    # 这里实际调用电磁求解器，返回经过方向、能量和收敛检查的光谱。
                    result = simulator.run(g, project)
                    path = samples_dir / f"sample_{i:06d}.npz"
                    # 单样本 NPZ 保存光谱、几何、求解信息和配置指纹，便于独立复查。
                    extra = {} if config.rotation_bounds_deg is None else {
                        "geometry_values": config.geometry_values(g), "rotation_deg": np.float64(g.rotation_deg)}
                    if config.quadrant_rotation_bounds_deg is not None:
                        extra = {"geometry_values": config.geometry_values(g)}
                    atomic_npz(path, **result, **extra, geometry_nm=g.array(),
                               geometry_01=config.normalize(config.geometry_values(g)).astype(np.float32),
                               sample_id=np.int64(i), config_fingerprint=np.array(manifest["config_fingerprint"]))
                    csv = path.with_suffix(".csv")
                    temporary = csv.with_suffix(".csv.tmp")
                    # CSV 按行保存 wavelength_nm/R/T/A，方便直接查看光谱或导入绘图工具。
                    np.savetxt(temporary, np.column_stack([result[k] for k in
                               ("wavelength_nm", "reflectance", "transmittance", "absorptance")]),
                               delimiter=",", header="wavelength_nm,reflectance,transmittance,absorptance", comments="")
                    os.replace(temporary, csv)
                    entry.update(state="success", sha256=checksum(path),
                                 elapsed_seconds=result["elapsed_seconds"], status=result["status"], finished_at=utc_now())
                    entry["simulation_time_ps"] = float(result.get("simulation_time_ps", config.simulation.simulation_time_ps))
                    entry["run_attempts"] = int(result.get("run_attempts", 1))
                    entry.pop("error", None)
                    if not config.simulation.save_projects:
                        # Only remove the exact scratch FSP created by this sample.
                        project.unlink(missing_ok=True)
                    completed += 1
                    LOG.info("Saved sample %d, status=%d, %.1fs", i, result["status"], result["elapsed_seconds"])
                except Exception as error:
                    # 失败样本记录具体异常，排除在训练数据之外；下次运行可继续重试。
                    entry.update(state="failed", error=str(error), finished_at=utc_now())
                    with (config.output_dir / "errors.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"id": i, "time": utc_now(), "error": str(error),
                                                 "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n")
                    LOG.exception("Sample %d failed; excluded from dataset", i)
                    if fail_fast:
                        raise
                finally:
                    # 每处理完一个样本就更新进度并汇总成功数据，减少中断造成的工作损失。
                    atomic_json(config.output_dir / "manifest.json", manifest)
                    processed += 1
                    if processed % pack_every == 0:
                        pack_dataset(config, manifest)
    finally:
        summary = pack_dataset(config, manifest)
    LOG.info("Finished: %d new successes; %s", completed, summary)
    return summary
