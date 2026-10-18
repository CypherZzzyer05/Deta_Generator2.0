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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.structure_only:
        preflight = json.loads((args.test_dir/'preflight.json').read_text(encoding='utf-8-sig'))
        values = np.asarray(preflight['geometry_values'])
        settings = json.loads((args.test_dir/'settings.json').read_text(encoding='utf-8-sig'))
    else:
        with np.load(args.test_dir/'result.npz', allow_pickle=False) as archive:
            result = {k: archive[k].copy() for k in archive.files}
        with np.load(args.test_dir/'historical_server_sample0.npz', allow_pickle=False) as archive:
            old = {k: archive[k].copy() for k in archive.files}
        if int(result['status']) != 2 or str(result['metal_position']) != 'above_substrate':
            raise ValueError('Only an automatically shut-off direct-contact native result may be plotted')
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
    for x, new in ((0, False), (1.7, True)):
        ax.add_patch(Rectangle((x, 0), 1, geometry.h, facecolor=blue))
        gold_bottom, substrate_bottom = (-100, -450) if new else (-450, -350)
        ax.add_patch(Rectangle((x, gold_bottom), 1, 100, facecolor='#D5AC4C'))
        ax.add_patch(Rectangle((x, substrate_bottom), 1, 350, facecolor='#D9E6EE'))
        ax.text(x+.5, geometry.h/2, 'As2Se3\n四聚体', ha='center', va='center', color='white')
        ax.text(x+.5, gold_bottom+50, 'Au 100 nm', ha='center', va='center', fontsize=10)
        ax.text(x+.5, substrate_bottom+175, 'SiO2\n350 nm', ha='center', va='center')
        ax.text(x+.5, -515, '新版：直接接触' if new else '旧版：基板隔开', ha='center', va='center')
    ax.axhline(0, color='#777777', lw=.7, linestyle='--')
    ax.set(xlim=(-.4, 3.2), ylim=(-550, geometry.h+50), ylabel='z (nm)', xticks=[],
           title='层叠对照（柱体高度和层厚按比例）')
    fig.suptitle('同一结构样本：把金膜放到基板上表面', fontsize=16)
    fig.text(.06, .03, f'P={values[0]:.2f}, L={values[1]:.2f}, H={values[2]:.2f}, dX={values[3]:.2f}, dY={values[4]:.2f} nm；俯视颜色仅区分角度组。', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(args.output/'structure_comparison.png', dpi=180)
    plt.close(fig)
    if args.structure_only:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(wavelength, old['spectra'], color=blue, lw=1.4, label='旧层叠：四聚体 / SiO2 / Au（服务器历史结果）')
    axes[0].plot(wavelength, reflectance, color=orange, lw=1.5, label='新层叠：四聚体 / Au / SiO2（本机真实求解）')
    axes[0].set(ylabel='反射率 R', ylim=(-.02, 1.04))
    axes[0].legend(loc='lower right', fontsize=10)
    axes[1].plot(wavelength, old['transmittance']*100, color=blue, lw=1.3)
    axes[1].plot(wavelength, result['transmittance']*100, color=orange, lw=1.4)
    axes[1].set(xlabel='波长 (nm)', ylabel='透射率 T (%)', xlim=(700, 1700))
    for ax in axes:
        ax.grid(alpha=.22)
    fig.suptitle('金膜位置变化后的 R / T 光谱对照 · 同一几何、Au 100 nm、metal8', fontsize=15)
    fig.text(.09, .03, '旧谱来自 v251，当前本机为 v241；这是结构变更的预览对照，尚未完成同版本网格收敛验证。', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(args.output/'spectra_comparison.png', dpi=180)
    plt.close(fig)

    report = dict(geometry_values=values.tolist(), old_major_features=features(wavelength, old['spectra']),
                  new_major_features=features(wavelength, reflectance),
                  new_status=int(result['status']), elapsed_seconds=float(result['elapsed_seconds']),
                  fdtd_version=str(result['fdtd_version']),
                  reflectance_range=[float(reflectance.min()), float(reflectance.max())],
                  transmittance_range=[float(result['transmittance'].min()), float(result['transmittance'].max())],
                  maximum_absolute_R_change=float(abs(reflectance-old['spectra']).max()),
                  mean_absolute_R_change=float(abs(reflectance-old['spectra']).mean()),
                  settings=settings)
    (args.output/'analysis.json').write_bytes((json.dumps(report, indent=2, ensure_ascii=False)+'\n').encode('utf-8'))
    np.savez_compressed(args.output/'sample0_topmetal.npz', **result)
    np.savez_compressed(args.output/'sample0_historical.npz', **old)
    for name in ('preflight.json', 'material_identity.json'):
        (args.output/name).write_bytes((args.test_dir/name).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k != 'settings'},indent=2))


if __name__ == '__main__':
    main()
