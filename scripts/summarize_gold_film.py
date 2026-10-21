"""Summarize native film controls, rejected candidates and convergence evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_gold_film import thin_film

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'research/metal_loss_fit'


def load_case(name):
    folder = OUTPUT/'film_benchmark_clear_pml'/name
    metadata = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    assert metadata['status'] == 2 and metadata['source_and_monitors_outside_pml']
    with np.load(folder/'result.npz', allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    assert np.isfinite(np.column_stack([arrays[k] for k in (
        'wavelength_nm', 'reflectance', 'transmittance', 'absorptance')])).all()
    np.testing.assert_allclose(arrays['reflectance']+arrays['transmittance']+arrays['absorptance'], 1., atol=1e-14)
    metadata['result_npz_sha256'] = hashlib.sha256((folder/'result.npz').read_bytes()).hexdigest()
    return metadata, arrays


def pair_metrics(coarse, fine):
    np.testing.assert_array_equal(coarse['wavelength_nm'], fine['wavelength_nm'])
    return {'maximum_R_absolute_difference': float(max(abs(coarse['reflectance']-fine['reflectance']))),
        'maximum_T_absolute_difference': float(max(abs(coarse['transmittance']-fine['transmittance']))),
        'maximum_T_relative_difference_vs_fine': float(max(abs(coarse['transmittance']/fine['transmittance']-1))),
        'maximum_A_absolute_difference': float(max(abs(coarse['absorptance']-fine['absorptance'])))}


def verify_analytic_limits():
    wave = np.linspace(700., 1700., 1000)
    glass = np.full_like(wave, 1.46, dtype=complex)
    metal = np.full_like(wave, .2+5j, dtype=complex)
    r, t, a = thin_film(wave, metal, glass, 0.)
    fresnel = abs((1-glass)/(1+glass))**2
    np.testing.assert_allclose(r, fresnel, atol=1e-14)
    np.testing.assert_allclose(t, 1-fresnel, atol=1e-14)
    np.testing.assert_allclose(a, 0., atol=1e-14)
    r, t, a = thin_film(wave, np.full_like(wave, 1.7, dtype=complex), glass)
    np.testing.assert_allclose(r+t, 1., atol=1e-14)
    np.testing.assert_allclose(a, 0., atol=1e-14)


def main():
    verify_analytic_limits()
    cases = {}
    for name in ['legacy_palik_2nm_bg20nm', 'explicit_crc_2nm_bg20nm']+[
            f'smooth_jc_{mesh}nm_bg20nm{suffix}' for suffix, meshes in [('', (1, 2, 4, 8)), ('_cv1', (2, 4, 8))]
            for mesh in meshes]:
        cases[name] = load_case(name)
    candidate = cases['smooth_jc_8nm_bg20nm_cv1'][0]
    values = candidate['material_metrics']
    strict_material_gate = bool(values['passive_in_band'] and values['max_n_error'] <= .02
        and values['max_k_error'] <= .05 and values['max_relative_real_epsilon_error'] <= .02
        and values['max_relative_imag_epsilon_error'] <= .05)
    pairs = {variant: pair_metrics(cases['smooth_jc_8nm_bg20nm'+suffix][1],
                                  cases['smooth_jc_4nm_bg20nm'+suffix][1])
             for variant, suffix in [('CV0', ''), ('CV1', '_cv1')]}
    summary = {'scope': 'Planar Air/Au100nm/semi-infinite SiO2 only; no tetramer calculation',
        'native_version': candidate['native_version'],
        'candidate_source': candidate['gold_source'],
        'candidate_material_metrics': values, 'strict_material_gate_passed': strict_material_gate,
        'gate_limits': {'max_n_error': .02, 'max_k_error': .05,
            'max_relative_real_epsilon_error': .02, 'max_relative_imag_epsilon_error': .05},
        'mesh_8_vs_4': pairs, 'cases': {name: data[0] for name, data in cases.items()},
        'analytic_zero_thickness_and_lossless_limits_verified': True,
        'production_generator_modified': False, 'production_release_created': False,
        'original_1000_sample_dataset_modified': False,
        'decision': 'Research candidate only: strict full-band k-error gate is not met; tetramer validation remains outstanding',
        'rejected': {'explicit_crc_nested_MEL': 'Native R/T does not reproduce complete analytic model',
            'legacy_Palik_fit': 'Measured Ohmic loss is almost absent',
            'initial_compact_domain_pilot': 'Source/monitor overlapped upper PML',
            'unrestricted_fit': 'Does not meet accuracy/passivity criteria'}}
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    plot_results(cases)
    write_report(summary, cases)
    print(json.dumps({'candidate_material_gate_passed': strict_material_gate,
        'pairs': pairs, 'cases': len(cases)}, indent=2))


def plot_results(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10})
    old = cases['legacy_palik_2nm_bg20nm'][1]
    new = cases['smooth_jc_8nm_bg20nm_cv1'][1]
    wave = old['wavelength_nm']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    curves = [('Palik 原始输入', old['raw_index'], '#999999', '--'),
              ('旧 Palik 原生拟合', old['model_index'], '#b34036', '-'),
              ('J&C 原始输入（另一数据来源）', new['raw_index'], '#2660a5', '--'),
              ('J&C 被动候选＋原生 MCM', new['model_index'], '#008575', '-')]
    for label, index, color, style in curves:
        for ax, values in zip(axes.flat, [index.real, index.imag, (index**2).imag, (index**2).real]):
            ax.plot(wave, values, style, label=label, color=color)
    for ax, title, ylabel in zip(axes.flat, ['折射率实部', '消光系数', '介电常数虚部（对数）', '介电常数实部'],
                               ['n', 'k', 'Im ε', 'Re ε']):
        ax.set(title=title, xlabel='波长（nm）', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[1, 0].set_yscale('log')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Au 损耗核对：不同输入来源分别标示，候选未达到全部材料误差门槛', fontsize=14)
    fig.savefig(OUTPUT/'material_comparison.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    styles = [(8, '', '#ba5942', '--'), (4, '', '#b4954f', '--'),
              (8, '_cv1', '#008575', '-'), (4, '_cv1', '#2660a5', '-')]
    for mesh, suffix, color, style in styles:
        case = cases[f'smooth_jc_{mesh}nm_bg20nm{suffix}'][1]
        label = f'{mesh} nm / '+('CV1' if suffix else 'CV0')
        axes[0, 0].plot(wave, case['reflectance'], style, color=color, label=label)
        axes[0, 1].plot(wave, case['transmittance'], style, color=color, label=label)
        axes[1, 0].plot(wave, 100*(case['transmittance']/case['analytic_T']-1), style, color=color, label=label)
        axes[1, 1].plot(wave, case['absorptance'], style, color=color, label=label)
    for ax, key in [(axes[0, 0], 'analytic_R'), (axes[0, 1], 'analytic_T'), (axes[1, 1], 'analytic_A')]:
        ax.plot(wave, new[key], color='#333333', lw=1, label='同一候选材料的解析结果')
    titles = [('反射率', 'R'), ('微弱透射（对数）', 'T'),
              ('相对同一材料解析透射的误差', 'T 相对误差（%）'), ('吸收率', 'A')]
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set(title=title, xlabel='波长（nm）', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0, 1].set_yscale('log')
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].axhline(0., color='#333333', lw=1)
    fig.suptitle('仅平面金膜：厚 SiO2 / Au100；网格结论不能外推到四聚体热点或双谷', fontsize=14)
    fig.savefig(OUTPUT/'film_mesh_comparison.png', dpi=160)
    plt.close(fig)


def write_report(summary, cases):
    p0, p1 = summary['mesh_8_vs_4']['CV0'], summary['mesh_8_vs_4']['CV1']
    old = cases['legacy_palik_2nm_bg20nm'][0]
    candidate = cases['smooth_jc_8nm_bg20nm_cv1'][0]
    rows = []
    for variant, suffix, meshes in [('CV0', '', (1, 2, 4, 8)), ('CV1', '_cv1', (2, 4, 8))]:
        for mesh in meshes:
            meta = cases[f'smooth_jc_{mesh}nm_bg20nm{suffix}'][0]
            rows.append(f"| {variant} | {mesh} | {100*meta['model_maximum_R_error']:.5f} | "
                        f"{meta['model_maximum_T_error']:.8f} | {meta['status']} |")
    text = f'''# Au 损耗修正候选与平面金膜网格核对

2026-10-21；Lumerical {summary['native_version']}；独立工作树 `codex/v2-metal-loss-fix`。

已建立恢复金损耗的研究候选，并完成平面 Au100 nm / 厚 SiO₂ 的原生 FDTD 与解析薄膜对照。**候选尚未达到全部材料误差门槛，生产生成器和服务器默认配置未修改，也未发布 2.2.0。** 原始 1000 样本与模型权重保留。

## 已确认的问题

旧 Palik 输入的 Im ε 为 1.277–14.173，原生默认拟合却只有 1.69e-7–2.60e-6。平面金膜验证中，旧模型的 R 与同一 Palik 原始输入的解析结果最大相差 **{100*old['raw_input_maximum_R_error']:.3f} 个百分点**，金属吸收几乎消失。仅仅自动关断到状态 2 不能证明材料正确。

将容差改为零、提高系数或调整虚部权重并没有得到合格 Palik 拟合；关闭被动性/稳定性限制的诊断也未解决问题。相关失败结果均保留。原生 MEL 插件嵌套虽然能保存参数，实际计算未复现完整模型，也已排除，没有用于候选结果。

直接读取数据库原始采样表还发现：Palik 输入在 1240 nm 的 k=8.77，而在 1265 nm 的 k=8.09。该急变来自输入表，不是图表重采样造成；原始查询保存于 `input_samples.json`。这提示应单独审核数据来源，不能只增加拟合阶数；这里不据此判定哪一组数据就是实际沉积薄膜的真值。

## 候选来源和保留的误差

研究候选使用数据库 **Au (Gold) - Johnson and Christy**，与旧 Palik 是不同输入来源，不能把两者差异全部归因于拟合修正。Johnson/Christy 的原始工作通过薄膜反射和透射测量获得金等贵金属光学常数，覆盖本次波段。[原始论文](https://doi.org/10.1103/PhysRevB.6.4370)

先拟合正强度、正阻尼的 Drude＋三个 Lorentz 极点，再导入 Sampled 3D data，由原生 MCM 表示；未用恒定 n/k、PEC 或增益模型替代金。原生 MCM 相对这一解析候选的最大 n/k 差约为 2.1e-7 / 3.1e-7。拟合参数、数据来源和实际原生结果均归档。[原生材料定义与限制](https://optics.ansys.com/hc/en-us/articles/360034394634-Standard-optical-permittivity-material-models-in-FDTD-and-MODE) · [getfdtdindex 的含义](https://optics.ansys.com/hc/en-us/articles/360034409694-getfdtdindex-Script-command)

| 相对 J&C 原始输入的最大误差 | 候选 | 预先采用的研究门槛 |
| --- | ---: | ---: |
| n 绝对误差 | 0.01683 | 0.02，通过 |
| k 绝对误差 | 0.09196 | 0.05，**未通过** |
| Re ε 相对误差 | 1.637% | 2%，通过 |
| Im ε 相对误差 | 4.428% | 5%，通过 |

700–1700 nm 内候选 Im ε 全为正。**上述材料门槛不是用户此前的谷位/深度门槛；未放宽门槛来宣称合格。** 图中 Palik 和 J&C 分别标示。

![不同材料来源及原生拟合](material_comparison.png)

## 平面薄膜验证

层序：从下到上为延伸穿过下 PML 的 SiO₂、100 nm Au、入射空气；没有玻璃背面的空气界面。这里没有四聚体。横向均匀场使用周期边界，横向网格 20 nm；空气与玻璃 z 向参考网格上限 20 nm，仅金膜 z 向改变为 1/2/4/8 nm。该平面参考网格不等于生产四聚体的自动网格。

功率监视器和光源均在 PML 之外，实际坐标检查已通过。解析计算同时保存“原始输入 n/k”与“原生实际拟合 n/k”两组，区分材料近似和数值误差。解析函数通过零厚度 Fresnel 极限与无损膜能量守恒检查；所有下列原生计算都达到状态 2。未裁剪 R/T/A。

| 金属界面处理 | 金属请求网格 nm | R 相对同一候选解析结果的最大误差，百分点 | T 最大绝对误差 | 状态 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

CV0 是当前默认方式，金属界面不使用共形处理；CV1 将共形处理用于金属界面。官方说明 CV1 是否更准确取决于具体网格及材料，不能据平膜结果直接切换所有器件。[官方网格说明](https://optics.ansys.com/hc/en-us/articles/360034382614-Selecting-the-best-mesh-refinement-option-in-the-FDTD-simulation-object)

| 同一候选 8 nm 与 4 nm 的差异 | CV0 | CV1 |
| --- | ---: | ---: |
| 最大 R 绝对差，百分点 | {100*p0['maximum_R_absolute_difference']:.5f} | {100*p1['maximum_R_absolute_difference']:.5f} |
| 最大 T 绝对差 | {p0['maximum_T_absolute_difference']:.8f} | {p1['maximum_T_absolute_difference']:.8f} |
| 最大 T 相对差，以 4 nm 为分母 | {100*p0['maximum_T_relative_difference_vs_fine']:.2f}% | {100*p1['maximum_T_relative_difference_vs_fine']:.2f}% |

这里透射很弱，因此“R 看起来一样”不能替代 T 收敛检查。CV1 明显改善本平膜的透射数值误差，但尚未核对四聚体的双谷位置、线宽、热点或场增强。

候选 CV1/8 nm 相对 J&C 原始输入解析薄膜的最大 R 差为 **{100*candidate['raw_input_maximum_R_error']:.5f} 个百分点**，最大 T 绝对差为 **{candidate['raw_input_maximum_T_error']:.8f}**。这些小误差只支持本平膜案例，不等于完整器件已验证。

![平面金膜的 R/T/A 与网格对照](film_mesh_comparison.png)

## 下一步和复现

继续处理全波段 k 拟合残差，并在真实厚 SiO₂ / Au100 / As₂Se₃ 四聚体上进行材料与网格对照后，再决定 2.0 的生产更新。原有 1000 样本仍仅用于已披露的数值流程训练，不能与新的材料或层序数据混合。

本目录 `summary.json` 保存定量结果及原始结果校验值；`film_benchmark_clear_pml/` 保存实际 NPZ、原生参数 JSON；较早紧凑空间的 pilot 与不合格材料结果保留为诊断，不列为通过证据。原生 FSP 和进度日志保留在 D 盘，未加入 Git。

在安装了 numpy、psutil、scipy、matplotlib 和 Lumerical API 的研究环境下，从仓库根目录执行：

```powershell
python scripts/benchmark_gold_film.py --case smooth_jc --mesh-nm 8 --mesh-variant 1 --run
python scripts/summarize_gold_film.py
```

已有结果目录禁止重复求解覆盖；脚本只用于独立薄膜研究，不启动服务器数据任务。
'''
    (OUTPUT/'README.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
