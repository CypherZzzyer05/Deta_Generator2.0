"""Plot genuine saved FDTD results; plotting dependencies are optional."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from scipy.signal import find_peaks, peak_widths

from deta_generator2.config import Geometry


def features(wavelength, reflectance):
    indices, properties = find_peaks(-reflectance, prominence=.03)
    widths = peak_widths(-reflectance, indices, rel_height=.5)[0]*np.diff(wavelength).mean()
    return [dict(wavelength_nm=float(wavelength[i]), reflectance=float(reflectance[i]),
                 prominence=float(prominence), half_prominence_width_nm=float(width))
            for i, prominence, width in zip(indices, properties['prominences'], widths)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/local_stack_test')
    parser.add_argument('--structure-only', action='store_true')
    parser.add_argument('--result-file', default='result.npz')
    parser.add_argument('--allow-time-limit-diagnostic', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.structure_only:
        preflight = json.loads((args.test_dir/'preflight.json').read_text(encoding='utf-8-sig'))
        values = np.asarray(preflight['geometry_values'])
        settings = json.loads((args.test_dir/'settings.json').read_text(encoding='utf-8-sig'))
    else:
        with np.load(args.test_dir/args.result_file, allow_pickle=False) as archive:
            result = {k: archive[k].copy() for k in archive.files}
        with np.load(args.test_dir/'historical_server_sample0.npz', allow_pickle=False) as archive:
            old = {k: archive[k].copy() for k in archive.files}
        diagnostic = int(result['status']) == 1 and args.allow_time_limit_diagnostic
        if (int(result['status']) != 2 and not diagnostic) or str(result['metal_position']) != 'above_substrate':
            raise ValueError('Time-limited results require explicit diagnostic plotting; never treat them as certified samples')
        wavelength, reflectance = result['wavelength_nm'], result['reflectance']
        if not np.allclose(wavelength, old['wavelength_nm'], atol=1e-5, rtol=0):
            raise ValueError('Historical and new spectra have different wavelength grids')
        values = result['geometry_values']
        settings = json.loads(str(result['settings_json']))
    geometry = Geometry(*values[:5], theta_odd_deg=values[5], theta_even_deg=values[6])
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    blue, orange = '#1768AC', '#D16B28'
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]
    p = geometry.p
    ax.add_patch(Rectangle((-p/2, -p/2), p, p, facecolor='#F5F8FB',
                           edgecolor='#657E91', linestyle='--'))
    for i, (vertices, center, angle) in enumerate(zip(geometry.vertices(), geometry.centers(), geometry.angles())):
        ax.add_patch(Polygon(vertices, facecolor=blue if i in (0, 3) else orange, edgecolor='#23354A'))
        ax.text(*center, f"{['III','II','IV','I'][i]}\n{angle:+.2f}°", ha='center', va='center', color='white')
    ax.set(xlim=(-p/2-15, p/2+15), ylim=(-p/2-15, p/2+15), aspect='equal',
           xlabel='x (nm)', ylabel='y (nm)', title='俯视：几何尺寸和两组角度保持一致')
    ax = axes[1]
    thick = settings.get('substrate_termination', 'finite') == 'pml'
    for x, new in (() if thick else ((0, False), (1.7, True))):
        ax.add_patch(Rectangle((x, 0), 1, geometry.h, facecolor=blue))
        gold_bottom, substrate_bottom = (-100, -450) if new else (-450, -350)
        ax.add_patch(Rectangle((x, gold_bottom), 1, 100, facecolor='#D5AC4C'))
        ax.add_patch(Rectangle((x, substrate_bottom), 1, 350, facecolor='#D9E6EE'))
        ax.text(x+.5, geometry.h/2, 'As2Se3\n四聚体', ha='center', va='center', color='white')
        ax.text(x+.5, gold_bottom+50, 'Au 100 nm', ha='center', va='center', fontsize=10)
        ax.text(x+.5, substrate_bottom+175, 'SiO2\n350 nm', ha='center', va='center')
        ax.text(x+.5, -515, '新版：直接接触' if new else '旧版：基板隔开', ha='center', va='center')
    if thick:
        from deta_generator2.config import SimulationSettings
        from deta_generator2.lumerical import stack_interfaces_nm
        stack = stack_interfaces_nm(SimulationSettings(**settings))
        lower = stack['domain_min']-200
        ax.add_patch(Rectangle((0, lower), 1, -100-lower, facecolor='#D9E6EE'))
        ax.add_patch(Rectangle((0, -100), 1, 100, facecolor='#D5AC4C'))
        ax.add_patch(Rectangle((0, 0), 1, geometry.h, facecolor=blue))
        ax.text(.5, geometry.h/2, 'As2Se3', color='white', ha='center', va='center')
        ax.text(1.1, -50, 'Au 100 nm：两侧直接接触', va='center', fontsize=10)
        ax.text(.5, -650, '厚 SiO2\n无背面空气界面', ha='center', va='center')
        for z, text, color in [(stack['transmission_z'], 'T：玻璃内功率通量', orange),
                               (stack['domain_min'], '下方 PML 边界', '#61515B')]:
            ax.hlines(z, -.1, 1.1, color=color, ls='--')
            ax.text(1.15, z, text, va='center', fontsize=9, color=color)
        ax.annotate('', xy=(.5, lower+5), xytext=(.5, lower+120),
                    arrowprops=dict(arrowstyle='->', color=blue))
        ax.set(xlim=(-.15, 3), ylim=(lower, geometry.h+180),
               title='目标层叠：SiO2 → Au → 四聚体')
    else:
        ax.set(xlim=(-.4, 3.2), ylim=(-550, geometry.h+50),
               title='有限厚基板诊断对照（350 nm）')
    ax.axhline(0, color='#777777', lw=.7, linestyle='--')
    ax.set(ylabel='z (nm)', xticks=[])
    fig.suptitle('同一样本几何与实际层叠设置', fontsize=16)
    fig.text(.06, .03, f'P={values[0]:.2f}, L={values[1]:.2f}, H={values[2]:.2f}, dX={values[3]:.2f}, dY={values[4]:.2f} nm；俯视颜色仅区分角度组。', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(args.output/'structure_comparison.png', dpi=180)
    plt.close(fig)
    if args.structure_only:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(wavelength, old['spectra'], color=blue, lw=1.4, label='旧层叠：四聚体 / SiO2 / Au（服务器历史结果）')
    label = '本机真实求解，未自动关断' if diagnostic else '本机真实求解，已自动关断'
    medium = '厚 SiO2，T 在玻璃内' if thick else '有限 SiO2，T 在空气内'
    axes[0].plot(wavelength, reflectance, color=orange, lw=1.5, label=f'新层叠：{medium}（{label}）')
    axes[0].set(ylabel='反射率 R', ylim=(-.02, 1.04))
    axes[0].legend(loc='lower right', fontsize=10)
    axes[1].plot(wavelength, old['transmittance']*100, color=blue, lw=1.3)
    axes[1].plot(wavelength, result['transmittance']*100, color=orange, lw=1.4)
    axes[1].set(xlabel='波长 (nm)', ylabel='透射率 T (%)', xlim=(700, 1700))
    for ax in axes:
        ax.grid(alpha=.22)
    prefix = '诊断预览（未自动关断） · ' if diagnostic else ''
    fig.suptitle(prefix+'真实 FDTD 反射与透射光谱', fontsize=15)
    caption = (f"新谱到 {float(result['simulation_time_ps']):g} ps 上限仍未自动关断，不能作为正式训练样本。"
               if diagnostic else '已达到当前自动关断阈值；未完成时间/网格/材料精度认证。')
    fig.text(.09, .03, caption+' 旧谱 v251，本机 v241；尚未完成同版本网格收敛验证。', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(args.output/'spectra_comparison.png', dpi=180)
    plt.close(fig)

    report = dict(geometry_values=values.tolist(), old_major_features=features(wavelength, old['spectra']),
                  new_major_features=features(wavelength, reflectance),
                  new_status=int(result['status']), elapsed_seconds=float(result['elapsed_seconds']),
                  convergence_confirmed=not diagnostic, not_for_training=diagnostic,
                  fdtd_version=str(result['fdtd_version']),
                  substrate_termination='pml' if thick else 'finite',
                  transmission_medium='SiO2' if thick else 'air',
                  reflectance_range=[float(reflectance.min()), float(reflectance.max())],
                  transmittance_range=[float(result['transmittance'].min()), float(result['transmittance'].max())],
                  maximum_absolute_R_change=float(abs(reflectance-old['spectra']).max()),
                  mean_absolute_R_change=float(abs(reflectance-old['spectra']).mean()),
                  settings=settings)
    (args.output/'analysis.json').write_bytes((json.dumps(report, indent=2, ensure_ascii=False)+'\n').encode('utf-8'))
    name = 'sample0_topmetal_diagnostic.npz' if diagnostic else 'sample0_topmetal.npz'
    np.savez_compressed(args.output/name, **result)
    np.savez_compressed(args.output/'sample0_historical.npz', **old)
    for name in ('preflight.json', 'material_identity.json'):
        (args.output/name).write_bytes((args.test_dir/name).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k != 'settings'},indent=2))


if __name__ == '__main__':
    main()
