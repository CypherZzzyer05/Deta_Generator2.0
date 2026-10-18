"""Validated settings: nm lengths, common or independent quadrant angles in degrees."""
# 仿真配置与几何定义：长度用 nm，在 Lumerical 接口处转换成 m；角度用 deg。
# 五个长度参数顺序固定，旋转任务在末尾追加第六个角度参数。

from dataclasses import asdict, dataclass, fields
import json
import hashlib
from pathlib import Path

import numpy as np

PARAMETER_NAMES = ("p", "l", "h", "alpha_x", "alpha_y")
ROTATION_PARAMETER_NAMES = (*PARAMETER_NAMES, "rotation_deg")
QUADRANT_PARAMETER_NAMES = (*PARAMETER_NAMES, "theta_odd_deg", "theta_even_deg")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Geometry:
    p: float
    l: float
    h: float
    alpha_x: float
    alpha_y: float
    rotation_deg: float = 0.0  # 各方块绕自身中心、共同绕 z 轴旋转；项目正角采用逆时针。
    theta_odd_deg: float | None = None
    theta_even_deg: float | None = None

    def __post_init__(self):
        values = self.array()
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("All geometry dimensions must be finite and positive (nm)")
        # 一个周期单元内沿每个方向放两个正方形；边长与间隙之和不能超出周期。
        if 2 * self.l + max(self.alpha_x, self.alpha_y) >= self.p:
            raise ValueError("Four squares must fit inside a cell: 2*l + gap < p")
        if not np.isfinite(self.rotation_deg):
            raise ValueError("rotation_deg must be finite")
        if (self.theta_odd_deg is None) != (self.theta_even_deg is None):
            raise ValueError("Both quadrant angles must be provided together")
        if self.theta_odd_deg is not None and self.rotation_deg != 0:
            raise ValueError("Common and quadrant rotation cannot be combined")
        if not np.isfinite(self.angles()).all():
            raise ValueError("Quadrant angles must be finite")
        vertices = self.vertices()
        if any(np.max(np.abs(v)) >= self.p / 2 for v in vertices):
            raise ValueError("Rotated squares must stay inside the periodic cell")
        # 各方柱可以独立旋转；用分离轴检查凸多边形相交/接触。
        for i in range(len(vertices)):
            for j in range(i):
                if self.polygons_touch(vertices[i], vertices[j]):
                    raise ValueError("Rotated squares overlap or touch; change gaps or rotation")

    @staticmethod
    def polygons_touch(a, b):
        # Separating-axis theorem works for independently rotated convex polygons.
        for polygon in (a, b):
            edges = np.roll(polygon, -1, axis=0) - polygon
            for edge in edges:
                axis = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
                pa, pb = a @ axis, b @ axis
                if pa.max() < pb.min() - 1e-8 or pb.max() < pa.min() - 1e-8:
                    return False
        return True

    def array(self, include_rotation=False, include_quadrant=False):
        names = QUADRANT_PARAMETER_NAMES if include_quadrant else (ROTATION_PARAMETER_NAMES if include_rotation else PARAMETER_NAMES)
        return np.array([getattr(self, name) for name in names], dtype=np.float64)

    def angles(self):
        # centers() order: III, II, IV, I. Odd quadrants are I and III.
        if self.theta_odd_deg is None:
            return [self.rotation_deg] * 4
        return [self.theta_odd_deg, self.theta_even_deg, self.theta_even_deg, self.theta_odd_deg]

    def rotation_matrix(self, angle_deg=None):
        angle = np.deg2rad(self.rotation_deg if angle_deg is None else angle_deg)
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    def vertices(self):
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * self.l / 2
        return [corners @ self.rotation_matrix(angle).T + center
                for center, angle in zip(self.centers(), self.angles())]

    def centers(self):
        """The alpha parameters denote edge-to-edge gaps, not center distances."""
        # 启用旋转后 alpha 仍描述未旋转排布的边缘间隙；中心保持不变，实际最小间隙会变化。
        # 相邻方块中心距离为 l + alpha；以原点为中心对称放置四个方块。
        x = (self.l + self.alpha_x) / 2
        y = (self.l + self.alpha_y) / 2
        return [(-x, -y), (-x, y), (x, -y), (x, y)]


@dataclass(frozen=True)
class SimulationSettings:
    # 以下为类默认值；实际采样时由 configs/smoke.json 等 JSON 文件覆盖。
    wavelength_start_nm: float = 4000
    wavelength_stop_nm: float = 8000
    spectrum_points: int = 1000
    mesh_accuracy: int = 2
    mesh_nm: float = 80
    simulation_time_ps: float = 3
    auto_shutoff_min: float = 1e-4
    air_padding_nm: float = 4000
    substrate_depth_nm: float = 3000
    background_index: float = 1
    silicon_material: str = "Si (Silicon) - Palik"
    substrate_material: str = "SiO2 (Glass) - Palik"
    processes: int = 1
    threads: int = 4
    hide_fdtd: bool = True
    save_projects: bool = True
    quality_tolerance: float = 0.03
    require_auto_shutoff: bool = True
    resonator_material: str | None = None
    resonator_nk_csv: str | None = None
    material_max_extension_nm: float = 0.0
    metal_material: str | None = None
    remove_metal: bool = False
    metal_thickness_nm: float = 100.0
    metal_position: str = "above_substrate"
    substrate_termination: str = "pml"
    metal_mesh_nm: float = 5.0
    bottom_padding_nm: float = 1000.0
    polarization_deg: float = 0.0
    material_fit_tolerance: float | None = None
    material_imaginary_weight: float = 1.0
    research_grid: bool = False
    maximum_mesh_nm: float | None = None
    mesh_policy: str = "legacy"
    monitor_downsample_xy: int = 1
    power_validation_wavelengths_nm: tuple = ()
    research_wavelengths_nm: tuple = ()

    def __post_init__(self):
        positive = (self.wavelength_start_nm, self.wavelength_stop_nm, self.mesh_nm,
                    self.simulation_time_ps, self.air_padding_nm,
                    self.substrate_depth_nm, self.background_index)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Simulation lengths, time and refractive index must be positive")
        if self.wavelength_stop_nm <= self.wavelength_start_nm:
            raise ValueError("wavelength_stop_nm must exceed wavelength_start_nm")
        if not isinstance(self.research_grid, bool):
            raise ValueError("research_grid must be a boolean")
        if isinstance(self.spectrum_points, bool) or not isinstance(self.spectrum_points, int) or self.spectrum_points < 32:
            raise ValueError("spectrum_points must be an integer of at least 32")
        if self.spectrum_points != 1000 and not self.research_grid:
            raise ValueError("The existing Metaformer requires exactly 1000 spectral points")
        if self.research_wavelengths_nm:
            values = np.asarray(self.research_wavelengths_nm)
            if (not self.research_grid or values.ndim != 1 or len(values) != self.spectrum_points
                    or not np.isfinite(values).all() or np.any(np.diff(values) <= 0)
                    or not np.isclose(values[0], self.wavelength_start_nm, rtol=0, atol=1e-5)
                    or not np.isclose(values[-1], self.wavelength_stop_nm, rtol=0, atol=1e-5)):
                raise ValueError('Focused wavelength samples require an explicit research grid spanning the source band')
        if not 1 <= self.mesh_accuracy <= 8 or int(self.mesh_accuracy) != self.mesh_accuracy:
            raise ValueError("mesh_accuracy must be an integer in [1, 8]")
        if not 0 < self.auto_shutoff_min < 1 or not 0 < self.quality_tolerance < 0.2:
            raise ValueError("Invalid shutoff threshold or quality tolerance")
        for v in (self.processes, self.threads):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError("processes and threads must be positive integers")
        for v in (self.hide_fdtd, self.save_projects, self.require_auto_shutoff, self.remove_metal):
            if not isinstance(v, bool):
                raise ValueError("Boolean settings must use JSON true/false")
        if self.metal_material not in (None, "Au (Gold) - Palik", "Ag (Silver) - Palik (0-2um)"):
            raise ValueError("Use the configured Au or Ag Palik material")
        if self.metal_position not in ("below_substrate", "above_substrate"):
            raise ValueError("metal_position must be below_substrate or above_substrate")
        if self.substrate_termination not in ("pml", "finite"):
            raise ValueError("substrate_termination must be pml or finite")
        if self.substrate_termination == "pml" and self.metal_position == "below_substrate":
            raise ValueError("A bottom-PML substrate requires metal_position=above_substrate")
        if self.remove_metal and self.metal_material is None:
            raise ValueError('Finite bare controls require a declared reference metal stack')
        for value in (self.metal_thickness_nm, self.metal_mesh_nm, self.bottom_padding_nm):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Metal and bottom padding lengths must be positive")
        if self.maximum_mesh_nm is not None:
            if not np.isfinite(self.maximum_mesh_nm) or not 0 < self.maximum_mesh_nm < 10:
                raise ValueError('Formal near-IR maximum mesh must be strictly below 10 nm')
            if self.mesh_nm > self.maximum_mesh_nm or self.metal_mesh_nm > self.maximum_mesh_nm:
                raise ValueError('Local mesh steps must not exceed the maximum mesh')
        if self.mesh_policy not in ('legacy', 'metal_only'):
            raise ValueError('mesh_policy must be legacy or metal_only')
        if self.mesh_policy == 'metal_only':
            if self.maximum_mesh_nm is not None:
                raise ValueError('metal_only cannot enforce a whole-domain mesh limit')
            if self.metal_material is None or not 0 < self.metal_mesh_nm < 10:
                raise ValueError('metal_only requires a reference metal and a metal z mesh below 10 nm')
        if isinstance(self.monitor_downsample_xy, bool) or not isinstance(self.monitor_downsample_xy, int) or self.monitor_downsample_xy < 1:
            raise ValueError('Monitor spatial sampling factor must be a positive integer')
        if self.monitor_downsample_xy > 1 and not self.power_validation_wavelengths_nm:
            raise ValueError('Reduced monitor storage requires independent full-spatial power checks')
        if self.power_validation_wavelengths_nm:
            values = np.asarray(self.power_validation_wavelengths_nm)
            if (values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all()
                    or np.any(np.diff(values) <= 0) or values[0] < self.wavelength_start_nm
                    or values[-1] > self.wavelength_stop_nm):
                raise ValueError('Invalid full-spatial validation wavelengths')
        if not np.isfinite(self.material_max_extension_nm) or self.material_max_extension_nm < 0:
            raise ValueError("Material extension allowance must be nonnegative")
        if not np.isfinite(self.polarization_deg):
            raise ValueError("Polarization must be finite")
        if self.material_fit_tolerance is not None and (not np.isfinite(self.material_fit_tolerance) or self.material_fit_tolerance < 0):
            raise ValueError("Material fit tolerance must be nonnegative")
        if not np.isfinite(self.material_imaginary_weight) or self.material_imaginary_weight <= 0:
            raise ValueError("Material imaginary weight must be positive")
        if self.resonator_nk_csv and not self.resonator_material:
            raise ValueError("Sampled nk data requires a resonator material name")

    def wavelength_grid(self):
        # 按波长均匀排列；训练固定1000点，独立研究可显式采用更密网格。
        if self.research_wavelengths_nm:
            return np.asarray(self.research_wavelengths_nm, dtype=np.float64)
        return np.linspace(self.wavelength_start_nm, self.wavelength_stop_nm, self.spectrum_points)


@dataclass(frozen=True)
class PipelineConfig:
    simulation: SimulationSettings
    bounds_nm: dict
    sample_count: int
    seed: int
    output_dir: Path
    lumapi_path: str | None = None
    parameter_range_source: str = "Engineering example; not verified against Table S3"
    quality_level: str = "pipeline_validation"
    rotation_bounds_deg: tuple[float, float] | None = None
    quadrant_rotation_bounds_deg: dict | None = None

    def __post_init__(self):
        if self.simulation.research_grid:
            raise ValueError("Research spectral grids cannot be packed into Metaformer training datasets")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 1:
            raise ValueError("sample_count must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if set(self.bounds_nm) != set(PARAMETER_NAMES):
            raise ValueError(f"bounds_nm must contain exactly {PARAMETER_NAMES}")
        for name, pair in self.bounds_nm.items():
            if len(pair) != 2 or not np.isfinite(pair).all() or not 0 < pair[0] < pair[1]:
                raise ValueError(f"Invalid lower/upper bounds for {name}")
        if self.rotation_bounds_deg is not None:
            pair = self.rotation_bounds_deg
            if len(pair) != 2 or not np.isfinite(pair).all() or list(pair) != [-45, 45]:
                raise ValueError("Common square rotation currently uses [-45,45] degrees")
        if self.quadrant_rotation_bounds_deg is not None:
            if self.rotation_bounds_deg is not None:
                raise ValueError("Choose common or quadrant rotations")
            if set(self.quadrant_rotation_bounds_deg) != set(QUADRANT_PARAMETER_NAMES[5:]):
                raise ValueError("Quadrant bounds need theta_odd_deg and theta_even_deg")
            if any(list(v) != [-45, 45] for v in self.quadrant_rotation_bounds_deg.values()):
                raise ValueError("Square quadrant angles use [-45,45] degrees")

    @property
    def parameter_names(self):
        if self.quadrant_rotation_bounds_deg is not None:
            return QUADRANT_PARAMETER_NAMES
        return ROTATION_PARAMETER_NAMES if self.rotation_bounds_deg is not None else PARAMETER_NAMES

    @property
    def parameter_units(self):
        return ("nm",) * 5 + ("deg",) * (len(self.parameter_names) - 5)

    def geometry_values(self, geometry):
        return geometry.array(include_rotation=self.rotation_bounds_deg is not None,
                              include_quadrant=self.quadrant_rotation_bounds_deg is not None)

    def make_geometry(self, values):
        if self.quadrant_rotation_bounds_deg is not None:
            return Geometry(*values[:5], theta_odd_deg=values[5], theta_even_deg=values[6])
        return Geometry(*values)

    @property
    def lower(self):
        # 按 PARAMETER_NAMES 排序，不能依赖 JSON 字典键的偶然排列顺序。
        values = [self.bounds_nm[k][0] for k in PARAMETER_NAMES]
        if self.rotation_bounds_deg is not None:
            values.append(self.rotation_bounds_deg[0])
        if self.quadrant_rotation_bounds_deg is not None:
            values.extend(self.quadrant_rotation_bounds_deg[k][0] for k in QUADRANT_PARAMETER_NAMES[5:])
        return np.array(values, dtype=np.float64)

    @property
    def upper(self):
        values = [self.bounds_nm[k][1] for k in PARAMETER_NAMES]
        if self.rotation_bounds_deg is not None:
            values.append(self.rotation_bounds_deg[1])
        if self.quadrant_rotation_bounds_deg is not None:
            values.extend(self.quadrant_rotation_bounds_deg[k][1] for k in QUADRANT_PARAMETER_NAMES[5:])
        return np.array(values, dtype=np.float64)

    def normalize(self, geometry_nm):
        # 长度用 nm，启用旋转时第六列用 deg；各列使用自己的范围归一化。
        return (np.asarray(geometry_nm) - self.lower) / (self.upper - self.lower)

    def sample_geometries(self):
        """Rejection sampling with a fixed seed; one plan is persisted before running."""
        # 在参数范围内均匀采样，丢弃放不进周期单元的组合；固定种子便于复查计划。
        rng = np.random.default_rng(self.seed)
        result = []
        for _ in range(max(10000, self.sample_count * 1000)):
            try:
                geometry = self.make_geometry(rng.uniform(self.lower, self.upper))
            except ValueError:
                continue
            result.append(geometry)
            if len(result) == self.sample_count:
                return result
        raise ValueError("Could not sample valid cells; check gaps, cell size and rotated-square overlap")

    def metadata(self):
        data = asdict(self)
        # Physical identity travels with the job. Machine paths and CPU resource
        # choices are recorded separately and may change when moving a dataset.
        data.pop('output_dir')
        data.pop('lumapi_path')
        for key in ('processes', 'threads', 'hide_fdtd', 'save_projects'):
            data['simulation'].pop(key)
        # 未启用旋转时保持旧配置序列化形式，已有五参数 manifest 指纹仍可复用。
        if self.rotation_bounds_deg is None:
            data.pop("rotation_bounds_deg")
        if self.quadrant_rotation_bounds_deg is None:
            data.pop("quadrant_rotation_bounds_deg")
        # Preserve fingerprints for untouched legacy configurations.
        for key in ("resonator_material", "resonator_nk_csv", "material_max_extension_nm",
                    "metal_material", "metal_mesh_nm", "bottom_padding_nm", "polarization_deg",
                    "material_fit_tolerance", "material_imaginary_weight", "research_grid", "maximum_mesh_nm", "mesh_policy",
                    "monitor_downsample_xy", "power_validation_wavelengths_nm", "research_wavelengths_nm"):
            if getattr(self.simulation, key) == SimulationSettings.__dataclass_fields__[key].default:
                data["simulation"].pop(key)
        if self.simulation.resonator_nk_csv:
            data["resonator_nk_sha256"] = hashlib.sha256(Path(self.simulation.resonator_nk_csv).read_bytes()).hexdigest()
            data['simulation']['resonator_nk_csv'] = 'content-addressed:resonator_nk_sha256'
        data['generator'] = 'deta-generator2/2.1.0'
        data['implementation_sha256'] = {name: hashlib.sha256(
            (Path(__file__).parent/name).read_text(encoding='utf-8').replace('\r\n', '\n').encode()).hexdigest()
            for name in ('config.py', 'lumerical.py', 'pipeline.py')}
        return data


def load_config(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    known = {f.name for f in fields(PipelineConfig)}
    if set(data) - known:
        raise ValueError(f"Unknown config keys: {sorted(set(data) - known)}")
    # 将仿真子配置转换为经过校验的对象，输出路径相对于 JSON 所在目录解析。
    simulation = data.get("simulation", {})
    if simulation.get("metal_position") != "above_substrate":
        raise ValueError("v2 requires explicit metal_position=above_substrate; use a new v2 job, never an old dataset directory")
    if simulation.get("substrate_termination") not in ("pml", "finite"):
        raise ValueError("v2.1 requires explicit substrate_termination=pml or finite; create a new job instead of reinterpreting old data")
    if simulation.get("resonator_nk_csv"):
        simulation["resonator_nk_csv"] = str((path.parent / simulation["resonator_nk_csv"]).resolve())
    data["simulation"] = SimulationSettings(**simulation)
    data["output_dir"] = (path.parent / data.get("output_dir", "../data/fdtd")).resolve()
    return PipelineConfig(**data)
