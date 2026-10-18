"""Load the local 2024 R1 lumapi and control real 3D FDTD sessions."""
# Python 控制真实求解器：连接会话 → 建立周期单元 → 求解 → 读取 R/T → 检查光谱。
# 本模块负责电磁仿真，不包含神经网络训练或人为生成的替代光谱。

from collections import OrderedDict
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil

from .config import Geometry, SimulationSettings

LOG = logging.getLogger(__name__)


def stack_interfaces_nm(settings):
    """Actual extents, computational lower boundary and T plane, all in nm.

    stack_min is only a computational reference in pml mode, not a glass
    backside. Extend glass below the outer domain edge so PML contains glass.
    """
    total_bottom = -settings.substrate_depth_nm-settings.metal_thickness_nm
    if settings.metal_position == 'above_substrate':
        stack = {'substrate_min': total_bottom, 'substrate_max': -settings.metal_thickness_nm,
                'metal_min': -settings.metal_thickness_nm, 'metal_max': 0.,
                'stack_min': total_bottom}
    else:
        stack = {'substrate_min': -settings.substrate_depth_nm, 'substrate_max': 0.,
                 'metal_min': total_bottom, 'metal_max': -settings.substrate_depth_nm,
                 'stack_min': total_bottom}
    stack['domain_min'] = total_bottom-settings.bottom_padding_nm
    stack['transmission_z'] = total_bottom-0.5*settings.bottom_padding_nm
    if settings.substrate_termination == 'pml':
        stack['substrate_min'] = stack['domain_min']-settings.bottom_padding_nm
    return stack


def mesh_grid_summary(grids, maximum_nm=None):
    """Record every returned grid cell; enforce a global limit only when requested."""
    summary = {}
    for axis in ('x', 'y', 'z'):
        positions = np.asarray(grids[axis], dtype=float).reshape(-1)
        steps_nm = np.diff(positions) * 1e9
        if len(steps_nm) < 1 or not np.isfinite(positions).all() or np.any(steps_nm <= 0):
            raise ValueError(f'Invalid solver-generated {axis} mesh')
        largest = float(steps_nm.max())
        if maximum_nm is not None and (largest > maximum_nm + 1e-6 or largest >= 10):
            raise ValueError(f'Actual {axis} mesh exceeds the strict limit: {largest:.9f} nm')
        summary[axis] = {'points': len(positions), 'maximum_step_nm': largest,
                         'minimum_step_nm': float(steps_nm.min())}
    return summary


def mesh_audit(grids, settings):
    """Validate actual metal z cells without limiting the automatic mesh elsewhere."""
    summary = mesh_grid_summary(grids, settings.maximum_mesh_nm)
    audit = {'mesh_policy': settings.mesh_policy,
             'maximum_mesh_nm': settings.maximum_mesh_nm,
             'whole_domain_limit_enforced': settings.maximum_mesh_nm is not None,
             'actual_grid': summary,
             'all_returned_cells_below_10_nm': all(v['maximum_step_nm'] < 10 for v in summary.values())}
    if settings.mesh_policy == 'metal_only':
        z = np.asarray(grids['z'], dtype=float).reshape(-1)*1e9
        stack = stack_interfaces_nm(settings)
        bottom, top = stack['metal_min'], stack['metal_max']
        if z[0] > bottom+1e-6 or z[-1] < top-1e-6:
            raise ValueError('Actual mesh does not cover the reference metal layer')
        mask = (z[:-1] < top-1e-6) & (z[1:] > bottom+1e-6)
        steps = np.diff(z)[mask]
        if not len(steps) or steps.max() > settings.metal_mesh_nm+1e-6 or steps.max() >= 10:
            raise ValueError('Actual metal z mesh exceeds the requested limit')
        audit['metal_region'] = {'axis': 'z', 'z_min_nm': bottom, 'z_max_nm': top,
                                 'cells': len(steps), 'requested_step_nm': settings.metal_mesh_nm,
                                 'maximum_step_nm': float(steps.max()), 'minimum_step_nm': float(steps.min()),
                                 'reference_mesh_without_metal': settings.remove_metal,
                                 'reference_xy_pinned': settings.remove_metal}
    return audit


def validate_power_integration(wavelength, reflection, transmission, check_wavelength, check_r, check_t):
    """Reject reduced spatial storage when full-spatial integrated powers disagree."""
    check_wavelength = np.asarray(check_wavelength).reshape(-1)
    check_r, check_t = np.asarray(check_r).reshape(-1), np.asarray(check_t).reshape(-1)
    if (len(check_wavelength) < 3 or check_r.shape != check_wavelength.shape or check_t.shape != check_wavelength.shape
            or not np.isfinite(np.column_stack([check_wavelength, check_r, check_t])).all()):
        raise ValueError('Invalid full-spatial power-check data')
    order = np.argsort(check_wavelength)
    check_wavelength, check_r, check_t = check_wavelength[order], check_r[order], check_t[order]
    if np.any(np.diff(check_wavelength) <= 0) or check_wavelength[0] < wavelength[0]-1e-5 or check_wavelength[-1] > wavelength[-1]+1e-5:
        raise ValueError('Power-check wavelengths are outside the main spectrum or duplicated')
    errors = {'maximum_R_error': float(np.max(np.abs(np.interp(check_wavelength, wavelength, reflection)-check_r))),
              'maximum_T_error': float(np.max(np.abs(np.interp(check_wavelength, wavelength, transmission)-check_t))),
              'check_points': len(check_wavelength), 'maximum_allowed_R_error': .002, 'maximum_allowed_T_error': 1e-5}
    if errors['maximum_R_error'] > .002 or errors['maximum_T_error'] > 1e-5:
        raise ValueError(f'Reduced monitor storage failed full-spatial power validation: {errors}')
    return errors, check_wavelength, check_r, check_t


def locate_lumapi(explicit=None):
    """Prefer the installed v241 API; never silently use a newer PyLumerical package."""
    # 优先使用配置/环境变量明确指定的本机 API，保持与已安装 2024 R1 的版本一致。
    requested = explicit or os.environ.get("LUMERICAL_LUMAPI_PATH")
    if requested:
        path = Path(requested).expanduser()
        if path.is_dir():
            path = path / "lumapi.py"
        if not path.is_file():
            raise FileNotFoundError(f"Configured lumapi.py does not exist: {path}")
        return path.resolve()
    roots = [Path(r'D:\Program Files\Lumerical'), Path(r'C:\Program Files\Lumerical'),
             Path(r'D:\Program Files\ANSYS Inc'), Path(r'C:\Program Files\ANSYS Inc'),
             Path('/opt/lumerical'), Path('/opt/ansys_inc')]
    matches = {p.resolve() for root in roots if root.is_dir() for pattern in
               ('v*/api/python/lumapi.py', 'v*/Lumerical/api/python/lumapi.py') for p in root.glob(pattern)}
    if len(matches) == 1:
        return matches.pop()
    if len(matches) > 1:
        raise ValueError('Multiple Lumerical APIs found; select one with --lumapi or LUMERICAL_LUMAPI_PATH')
    raise FileNotFoundError('Lumerical API not found. Set --lumapi or LUMERICAL_LUMAPI_PATH')


def load_lumapi(explicit=None):
    path = locate_lumapi(explicit)
    existing = sys.modules.get("lumapi")
    if existing is not None:
        # 同一个 Python 进程已加载另一套 API 时拒绝混用，避免版本与 DLL 不对应。
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError("A different lumapi is already loaded; restart Python with the configured API")
        return existing
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("lumapi", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lumapi"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("lumapi", None)
        raise
    return module


def validate_spectrum(wavelength_nm, reflectance, transmittance, settings, status):
    """Reject bad physics; do not hide sign errors or energy errors by clipping."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64).reshape(-1)
    reflectance = np.asarray(reflectance, dtype=np.float64).reshape(-1)
    transmittance = np.asarray(transmittance, dtype=np.float64).reshape(-1)
    grid = settings.wavelength_grid()
    if any(x.shape != grid.shape for x in (wavelength_nm, reflectance, transmittance)):
        raise ValueError("Monitor must return the configured number of wavelength/R/T points")
    if not all(np.isfinite(x).all() for x in (wavelength_nm, reflectance, transmittance)):
        raise ValueError("Nonfinite monitor values")
    # 按波长统一升序，同步排列 R 和 T，使网络第 i 个输入始终对应同一波长。
    order = np.argsort(wavelength_nm)
    wavelength_nm, reflectance, transmittance = [x[order] for x in (wavelength_nm, reflectance, transmittance)]
    if not np.allclose(wavelength_nm, grid, atol=1e-5, rtol=1e-8):
        raise ValueError("Monitor wavelength grid does not match the configured grid")
    if status not in (1, 2):
        raise ValueError(f"FDTD failed or diverged (status={status})")
    # status=1 表示到达时间上限；status=2 表示自动关断，严格采样要求后者。
    if settings.require_auto_shutoff and status != 2:
        raise ValueError("FDTD reached its time limit before auto-shutoff; increase simulation_time_ps")
    tol = settings.quality_tolerance
    # 保留真实数值并检查功率/能量范围；数值容差允许少量离散误差。
    # 这里不靠裁剪或逐条光谱 min-max 缩放来掩盖符号、归一化或收敛错误。
    if min(reflectance.min(), transmittance.min()) < -tol:
        raise ValueError("Negative R/T beyond tolerance; check propagation/monitor directions")
    if max(reflectance.max(), transmittance.max()) > 1 + tol:
        raise ValueError("R/T exceeds unity beyond tolerance; check convergence/source normalization")
    if np.max(reflectance + transmittance) > 1 + tol:
        raise ValueError("R+T exceeds unity beyond tolerance; check convergence and PML")
    # R、T 由源功率归一化；1-R-T 包含传播到 T 监视器之前的材料吸收损耗。
    return {"wavelength_nm": grid, "reflectance": reflectance,
            "transmittance": transmittance, "absorptance": 1 - reflectance - transmittance}


class FDTDSimulator:
    def __init__(self, settings: SimulationSettings, lumapi_path=None):
        self.settings = settings
        self.lumapi = load_lumapi(lumapi_path)
        self.fdtd = None

    def __enter__(self):
        # with 语句进入时创建 FDTD 会话，设置 CPU 资源并检查所需材料。
        self.fdtd = self.lumapi.FDTD(hide=self.settings.hide_fdtd)
        try:
            self.version = str(self.fdtd.version())
            self.fdtd.setresource("FDTD", 1, "processes", str(self.settings.processes))
            self.fdtd.setresource("FDTD", 1, "threads", str(self.settings.threads))
            s = self.settings
            if s.resonator_nk_csv:
                data = np.loadtxt(s.resonator_nk_csv, delimiter=",", skiprows=1)
                if (data.ndim != 2 or data.shape[1] != 3 or len(data) < 3
                        or not np.isfinite(data).all() or np.any(data[:, :2] <= 0)
                        or np.any(data[:, 2] < 0) or np.any(np.diff(data[:, 0]) <= 0)):
                    raise ValueError("nk CSV must contain increasing wavelength_nm,n,k; n>0,k>=0")
                extension = max(data[0, 0] - s.wavelength_start_nm,
                                s.wavelength_stop_nm - data[-1, 0], 0)
                if extension > s.material_max_extension_nm + 1e-8:
                    raise ValueError("Measured nk data does not cover the wavelength band")
                if extension:
                    LOG.warning("EXPLORATORY MATERIAL EXTENSION: holding endpoint nk over %.3f nm", extension)
                    if s.wavelength_stop_nm > data[-1, 0]:
                        data = np.vstack([data, [s.wavelength_stop_nm, *data[-1, 1:]]])
                    if s.wavelength_start_nm < data[0, 0]:
                        data = np.vstack([[s.wavelength_start_nm, *data[0, 1:]], data])
                # Use only the fitted source band plus neighboring measured knots.
                left = max(0, np.searchsorted(data[:, 0], s.wavelength_start_nm, side='left')-1)
                right = min(len(data), np.searchsorted(data[:, 0], s.wavelength_stop_nm, side='right')+1)
                data = data[left:right]
                name = s.resonator_material
                if not self.fdtd.materialexists(name):
                    material = self.fdtd.addmaterial("Sampled data")
                    self.fdtd.setmaterial(material, "name", name)
                sampled = np.column_stack([299792458 / (data[:, 0]*1e-9), (data[:, 1]+1j*data[:, 2])**2])
                self.fdtd.setmaterial(name, "max coefficients", 12)
                if s.material_fit_tolerance is not None:
                    self.fdtd.setmaterial(name, "tolerance", s.material_fit_tolerance)
                    self.fdtd.setmaterial(name, "imaginary weight", s.material_imaginary_weight)
                self.fdtd.setmaterial(name, "sampled data", sampled[::-1].copy())
                self.material_data = data
            for name in filter(None, (s.resonator_material or s.silicon_material, s.substrate_material, s.metal_material)):
                if not self.fdtd.materialexists(name):
                    raise ValueError(f"Material missing from installed FDTD library: {name}")
        except BaseException:
            self.fdtd.close()
            self.fdtd = None
            raise
        return self

    def __exit__(self, *args):
        # 正常结束或抛出异常都关闭会话，释放该 Python 进程持有的求解器资源。
        if self.fdtd is not None:
            self.fdtd.close()
            self.fdtd = None

    def material_identity(self):
        """Record actual fitted dispersion; it must agree when moving/resuming."""
        frequency = 299792458/(self.settings.wavelength_grid()*1e-9)
        names = filter(None, (self.settings.resonator_material or self.settings.silicon_material,
                              self.settings.substrate_material,
                              None if self.settings.remove_metal else self.settings.metal_material))
        result = {}
        for name in names:
            measured = np.asarray(self.fdtd.getindex(name, frequency)).reshape(-1)
            fitted = np.asarray(self.fdtd.getfdtdindex(name, frequency, frequency.min(), frequency.max())).reshape(-1)
            if measured.shape != frequency.shape or fitted.shape != frequency.shape or not (
                    np.isfinite(measured).all() and np.isfinite(fitted).all()) or np.any(fitted.imag < -1e-8):
                raise ValueError('Invalid or active material fit: '+name)
            result[name] = {'fitted_index_ri': np.c_[fitted.real, fitted.imag].tolist(),
                            'max_n_error': float(abs(measured.real-fitted.real).max()),
                            'max_k_error': float(abs(measured.imag-fitted.imag).max()),
                            'max_imaginary_permittivity_error': float(abs((measured**2-fitted**2).imag).max()),
                            'accuracy_certified': False}
        return result

    def build(self, geometry: Geometry, include_silicon=True, substrate_index=None):
        """Create the tetramer/metal/glass stack, with explicit substrate termination."""
        f, s = self.fdtd, self.settings
        # 项目和标签采用 nm；Lumerical 的长度属性使用 SI 单位 m。
        nm = 1e-9
        p, h = geometry.p * nm, geometry.h * nm
        top, bottom = h + s.air_padding_nm * nm, -s.substrate_depth_nm * nm
        metal_enabled = s.metal_material is not None and substrate_index is None
        stack = stack_interfaces_nm(s)
        substrate_bottom = stack['substrate_min'] * nm
        substrate_top = stack['substrate_max'] * nm if metal_enabled else 0.
        metal_bottom, metal_top = stack['metal_min'] * nm, stack['metal_max'] * nm
        stack_bottom = stack['stack_min'] * nm
        if metal_enabled:
            bottom = stack_bottom - s.bottom_padding_nm * nm
        # 每个样本重新构造几何，防止上一组结构或求解结果遗留到当前样本。
        f.switchtolayout()
        f.selectall()
        f.delete()
        # x/y 周期边界模拟无限阵列，z 方向 PML 吸收上下传播的出射波。
        f.addfdtd(properties=OrderedDict([
            ("dimension", "3D"), ("x", 0.), ("y", 0.),
            ("x span", p), ("y span", p), ("z min", bottom), ("z max", top),
            ("x min bc", "Periodic"), ("x max bc", "Periodic"),
            ("y min bc", "Periodic"), ("y max bc", "Periodic"),
            ("z min bc", "PML"), ("z max bc", "PML"),
            ("index", s.background_index), ("mesh accuracy", s.mesh_accuracy),
            ("simulation time", s.simulation_time_ps * 1e-12),
            ("auto shutoff min", s.auto_shutoff_min), ("pml profile", 3)
        ]))
        if s.maximum_mesh_nm is not None:
            # Cover the complete simulation volume, including substrate, metal,
            # air and PML boundaries. A local pillar override alone is insufficient.
            f.addmesh(name='global_mesh_limit', x=0., y=0., x_span=p, y_span=p,
                      z_min=bottom, z_max=top, override_x_mesh=True,
                      override_y_mesh=True, override_z_mesh=True,
                      dx=s.maximum_mesh_nm*nm, dy=s.maximum_mesh_nm*nm, dz=s.maximum_mesh_nm*nm)
        # pml 模式中玻璃跨越整个下方 PML，绝不创建基板背面的空气界面。
        material = s.substrate_material if substrate_index is None else "<Object defined dielectric>"
        f.addrect(name="substrate", x=0., y=0., x_span=3*p, y_span=3*p,
                  z_min=substrate_bottom if metal_enabled else bottom - p, z_max=substrate_top, material=material)
        if substrate_index is not None:
            f.set("index", float(substrate_index))
        if metal_enabled:
            f.addrect(name="metal_film", x=0., y=0., x_span=3*p, y_span=3*p,
                      z_min=metal_bottom, z_max=metal_top, material=s.metal_material)
            f.addmesh(name="metal_mesh", x=0., y=0., x_span=p, y_span=p,
                      z_min=metal_bottom-s.metal_mesh_nm*nm,
                      z_max=metal_top+s.metal_mesh_nm*nm,
                      override_x_mesh=False, override_y_mesh=False, override_z_mesh=True,
                      dz=s.metal_mesh_nm*nm)
        if include_silicon:
            # 四个材料方柱共用边长 l 和高度 h；中心位置由名义间隙确定。
            for i, ((x, y), angle) in enumerate(zip(geometry.centers(), geometry.angles())):
                f.addrect(name=f"si_{i}", x=x*nm, y=y*nm,
                          x_span=geometry.l*nm, y_span=geometry.l*nm,
                          z_min=0., z_max=h, material=s.resonator_material or s.silicon_material)
                if angle != 0:
                    # Lumerical 的 rotation 1 采用顺时针度数；项目约定正角逆时针。
                    # 只旋转刚建立的方块，自身中心不移动，光源偏振和周期单元不旋转。
                    f.set("first axis", "z")
                    f.set("rotation 1", -float(angle))
        # 在结构及其上下邻近区域设置局部网格，控制几何和共振的离散精度。
        if s.mesh_policy != 'metal_only':
            f.addmesh(name="structure_mesh", x=0., y=0., x_span=p, y_span=p,
                      z_min=-s.mesh_nm*nm, z_max=h + s.mesh_nm*nm,
                      dx=s.mesh_nm*nm, dy=s.mesh_nm*nm, dz=s.mesh_nm*nm)
        # 平面波由空气侧沿 -z 垂直入射，偏振角 0 对应本模型的 x 偏振。
        f.addplane(properties=OrderedDict([
            ("name", "source"), ("injection axis", "z"), ("direction", "Backward"),
            ("x", 0.), ("y", 0.), ("x span", p), ("y span", p),
            ("z", h + 0.5*s.air_padding_nm*nm), ("angle theta", 0.),
            ("polarization angle", s.polarization_deg), ("override global source settings", True),
            ("wavelength start", s.wavelength_start_nm*nm),
            ("wavelength stop", s.wavelength_stop_nm*nm)
        ]))
        # 用波长等间隔采样 1000 点，与训练所用 wavelength_grid() 对齐。
        f.setglobalmonitor("use source limits", True)
        f.setglobalmonitor("use wavelength spacing", True)
        f.setglobalmonitor("frequency points", s.spectrum_points)
        # R is above the source. In pml mode T measures flux inside uniform glass.
        for name, z in (("R", h + 0.75*s.air_padding_nm*nm),
                        ("T", stack_bottom-0.5*s.bottom_padding_nm*nm if metal_enabled else -0.5*s.substrate_depth_nm*nm)):
            f.addpower(name=name, monitor_type="2D Z-normal", x=0., y=0.,
                       x_span=p, y_span=p, z=z)
            if s.research_wavelengths_nm:
                f.set('override global monitor settings', True)
                f.set('sample spacing', 'custom')
                f.set('custom frequency samples', (299792458/(s.wavelength_grid()*nm))[::-1])
            if s.monitor_downsample_xy > 1:
                f.set('down sample x', s.monitor_downsample_xy)
                f.set('down sample y', s.monitor_downsample_xy)
            # Only accumulated power is needed. Full 1000-frequency E/H fields are large.
            for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Px", "Py", "Pz"):
                f.set("output " + field, False)
            f.set("output power", True)
            if s.power_validation_wavelengths_nm:
                f.addpower(name=name+'_powercheck', monitor_type='2D Z-normal', x=0., y=0., x_span=p, y_span=p, z=z)
                f.set('down sample x', 1)
                f.set('down sample y', 1)
                f.set('override global monitor settings', True)
                f.set('sample spacing', 'custom')
                f.set('custom frequency samples', (299792458/(np.asarray(s.power_validation_wavelengths_nm)*nm))[::-1])
                for field in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Px', 'Py', 'Pz'):
                    f.set('output '+field, False)
                f.set('output power', True)

        if s.remove_metal and metal_enabled:
            reference = {a: np.asarray(f.getresult('FDTD', a)).reshape(-1) for a in ('x', 'y', 'z')}
            if s.mesh_policy == 'metal_only':
                # Removal coarsens automatic x/y meshes. Pin those same steps
                # using the existing metal-region override, keeping one object.
                for axis in ('x', 'y'):
                    steps = np.diff(reference[axis])
                    if not np.allclose(steps, steps[0], rtol=0, atol=1e-15):
                        raise ValueError('Bare metal-only control requires uniform reference x/y grids')
                    f.setnamed('metal_mesh', f'override {axis} mesh', True)
                    f.setnamed('metal_mesh', 'd'+axis, float(steps.max()))
            f.select('metal_film')
            f.delete()
            actual = {a: np.asarray(f.getresult('FDTD', a)).reshape(-1) for a in reference}
            if any(reference[a].shape != actual[a].shape or not np.allclose(
                    reference[a], actual[a], rtol=0, atol=1e-15) for a in reference):
                raise ValueError('Removing the metal changed the grid; finite bare control rejected')
        self.validate_native_layout(geometry, include_silicon, substrate_index)

    def validate_native_layout(self, geometry, include_silicon=True, substrate_index=None):
        """Read the constructed native objects, including the two angle groups."""
        f, s = self.fdtd, self.settings
        stack = stack_interfaces_nm(s)
        def near(name, key, expected, scale=1e9):
            actual = float(np.asarray(f.getnamed(name, key)).item())*scale
            if not np.isclose(actual, expected, rtol=0, atol=1e-6):
                raise ValueError(f'Native layout mismatch: {name}.{key}')
        for name in ('FDTD', 'source', 'R', 'T'):
            for key in ('x span', 'y span'):
                near(name, key, geometry.p)
        for key in ('x min bc', 'x max bc', 'y min bc', 'y max bc'):
            if f.getnamed('FDTD', key) != 'Periodic':
                raise ValueError('Native periodic boundary mismatch')
        for key in ('z min bc', 'z max bc'):
            if f.getnamed('FDTD', key) != 'PML':
                raise ValueError('Native PML boundary mismatch')
        near('source', 'polarization angle', s.polarization_deg, 1)
        near('source', 'angle theta', 0, 1)
        near('source', 'z', geometry.h+0.5*s.air_padding_nm)
        near('R', 'z', geometry.h+0.75*s.air_padding_nm)
        near('FDTD', 'z max', geometry.h+s.air_padding_nm)
        near('FDTD', 'index', s.background_index, 1)
        if s.mesh_policy == 'metal_only':
            if any(int(f.getnamednumber(name)) for name in ('global_mesh_limit', 'structure_mesh')):
                raise ValueError('metal_only layout contains a global or pillar mesh override')
            if int(f.getnamednumber('metal_mesh')) != 1:
                raise ValueError('metal_only layout requires exactly one metal mesh override')
            for key, expected in (('z min', stack['metal_min']-s.metal_mesh_nm),
                                  ('z max', stack['metal_max']+s.metal_mesh_nm), ('dz', s.metal_mesh_nm)):
                near('metal_mesh', key, expected)
            for axis in ('x', 'y'):
                if bool(f.getnamed('metal_mesh', f'override {axis} mesh')) != s.remove_metal:
                    raise ValueError('Metal x/y override disagrees with the reference-control policy')
            if not bool(f.getnamed('metal_mesh', 'override z mesh')):
                raise ValueError('Metal z override is disabled')
        near('source', 'wavelength start', s.wavelength_start_nm)
        near('source', 'wavelength stop', s.wavelength_stop_nm)
        if f.getnamed('source', 'direction') != 'Backward' or f.getnamed('source', 'injection axis') not in ('z', 'z-axis'):
            raise ValueError('Native illumination direction mismatch')
        if include_silicon:
            for i, ((x, y), angle) in enumerate(zip(geometry.centers(), geometry.angles())):
                name = f'si_{i}'
                for key, expected in (('x', x), ('y', y), ('x span', geometry.l),
                                      ('y span', geometry.l), ('z min', 0), ('z max', geometry.h)):
                    near(name, key, expected)
                near(name, 'rotation 1', -angle, 1)
                if angle != 0 and f.getnamed(name, 'first axis') != 'z':
                    raise ValueError('Native pillar rotation axis mismatch')
                if f.getnamed(name, 'material') != (s.resonator_material or s.silicon_material):
                    raise ValueError('Native pillar material mismatch')
        if s.metal_material is not None and substrate_index is None:
            near('FDTD', 'z min', stack['stack_min']-s.bottom_padding_nm)
            near('T', 'z', stack['stack_min']-0.5*s.bottom_padding_nm)
            near('substrate', 'z min', stack['substrate_min'])
            near('substrate', 'z max', stack['substrate_max'])
            if s.substrate_termination == 'pml':
                if not stack['substrate_min'] < stack['domain_min'] < stack['transmission_z'] < stack['substrate_max']:
                    raise ValueError('PML substrate must cover the lower boundary and transmission monitor')
            if f.getnamed('substrate', 'material') != s.substrate_material:
                raise ValueError('Native substrate material mismatch')
            count = int(f.getnamednumber('metal_film'))
            if count != int(not s.remove_metal):
                raise ValueError('Native metal-film presence mismatch')
            if not s.remove_metal:
                near('metal_film', 'z min', stack['metal_min'])
                near('metal_film', 'z max', stack['metal_max'])
                if f.getnamed('metal_film', 'material') != s.metal_material:
                    raise ValueError('Native metal-film material mismatch')

    def run(self, geometry, project_path, include_silicon=True, substrate_index=None):
        if geometry.theta_odd_deg is not None and self.settings.mesh_policy != 'metal_only' and (self.settings.maximum_mesh_nm is None
                or not 0 < self.settings.maximum_mesh_nm < 10):
            raise ValueError('Real seven-parameter simulations require a whole-domain maximum mesh strictly below 10 nm; use the fine8 configuration or a finer research mesh')
        self.build(geometry, include_silicon, substrate_index)
        project_path = Path(project_path).resolve()
        project_path.parent.mkdir(parents=True, exist_ok=True)
        resources = self.fdtd.runsystemcheck()
        def plain(value):
            if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
            if isinstance(value, np.ndarray): return value.tolist()
            if isinstance(value, np.generic): return value.item()
            return value
        recommended = resources.get('Memory_Recommended_Bytes', resources.get('Memory_Recommended'))
        if recommended is None or not np.isfinite(float(recommended)) or float(recommended) <= 0:
            raise ValueError('Native resource check did not return a valid memory recommendation')
        available = psutil.virtual_memory().available
        project_path.with_suffix('.resources.json').write_text(json.dumps({
            'native': plain(resources), 'available_memory_bytes': available,
            'recommended_memory_bytes': float(recommended)}, indent=2), encoding='utf-8')
        if available < float(recommended):
            raise MemoryError(f'Available RAM {available/2**30:.2f} GiB < native recommendation '
                              f'{float(recommended)/2**30:.2f} GiB; reduce simultaneous jobs or use more RAM')
        mesh_grids = {}
        mesh_summary = None
        if self.settings.maximum_mesh_nm is not None or self.settings.mesh_policy == 'metal_only':
            mesh_grids = {axis: np.asarray(self.fdtd.getresult('FDTD', axis)).reshape(-1)
                          for axis in ('x', 'y', 'z')}
            audit = mesh_audit(mesh_grids, self.settings)
            mesh_summary = audit['actual_grid']
            if self.settings.maximum_mesh_nm is not None and self.settings.monitor_downsample_xy > 1:
                bound = max(mesh_summary[axis]['maximum_step_nm'] for axis in ('x', 'y'))*self.settings.monitor_downsample_xy
                if bound >= 10:
                    raise ValueError('Monitor recording spacing must also remain strictly below 10 nm')
            project_path.with_suffix('.mesh.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        attempts = []
        elapsed = 0.0
        # 未自动关断时仅延长时间上限，最多四次；几何、网格与波长设置保持相同。
        for attempt in range(4):
            time_limit_ps = self.settings.simulation_time_ps * (2 ** attempt)
            if attempt:
                LOG.warning("Auto-shutoff not reached; retrying %s with %.1f ps time limit",
                            project_path.name, time_limit_ps)
                self.fdtd.switchtolayout()
                self.fdtd.setnamed("FDTD", "simulation time", time_limit_ps * 1e-12)
            self.fdtd.save(str(project_path))
            start = time.monotonic()
            self.fdtd.run()
            wall_seconds = time.monotonic() - start
            elapsed += wall_seconds
            status = int(np.asarray(self.fdtd.getresult("FDTD", "status")).item())
            attempts.append({"time_limit_ps": time_limit_ps, "status": status,
                             "elapsed_seconds": wall_seconds})
            if self.settings.save_projects:
                self.fdtd.save(str(project_path))
            if status != 1 or not self.settings.require_auto_shutoff:
                break
        # 两个监视器必须使用同一波长网格；读取的 "T" 字段代表归一化法向功率。
        r = self.fdtd.getresult("R", "T")
        t = self.fdtd.getresult("T", "T")
        if not np.allclose(np.asarray(r["lambda"]), np.asarray(t["lambda"]), rtol=1e-10, atol=1e-15):
            raise ValueError("R and T monitors returned different wavelength grids")
        # Downward incidence: reflected power flows +z; transmitted power flows -z.
        # 法向参考为 +z：反射功率为正，向 -z 传播的透射原始值为负，故只对 T 取负。
        result = validate_spectrum(np.asarray(r["lambda"])*1e9, r["T"], -np.asarray(t["T"]), self.settings, status)
        if self.settings.power_validation_wavelengths_nm:
            full_r, full_t = self.fdtd.getresult('R_powercheck', 'T'), self.fdtd.getresult('T_powercheck', 'T')
            if not np.allclose(full_r['lambda'], full_t['lambda'], rtol=1e-10, atol=1e-15):
                raise ValueError('Full-spatial R/T validation grids differ')
            errors, check_w, check_r, check_t = validate_power_integration(result['wavelength_nm'], result['reflectance'],
                result['transmittance'], np.asarray(full_r['lambda'])*1e9, full_r['T'], -np.asarray(full_t['T']))
            result.update(power_integration_validation_json=json.dumps(errors), power_check_wavelength_nm=check_w,
                          power_check_reflectance=check_r, power_check_transmittance=check_t)
        # 将最终时间上限、尝试次数和求解器版本一起保存，保留每条光谱的来源记录。
        result.update(status=status, elapsed_seconds=elapsed, fdtd_version=self.version,
                      simulation_time_ps=time_limit_ps, run_attempts=len(attempts),
                      attempts_json=json.dumps(attempts), remove_metal=self.settings.remove_metal,
                      native_layout_verified=True, metal_position=self.settings.metal_position,
                      substrate_termination=self.settings.substrate_termination,
                      transmission_medium='SiO2' if self.settings.substrate_termination == 'pml' else 'air',
                      stack_interfaces_nm_json=json.dumps(stack_interfaces_nm(self.settings)))
        if mesh_summary is not None:
            result.update({f'mesh_{axis}_nm': positions * 1e9 for axis, positions in mesh_grids.items()})
            result['mesh_summary_json'] = json.dumps(mesh_summary)
            result['mesh_audit_json'] = json.dumps(audit)
        return result
