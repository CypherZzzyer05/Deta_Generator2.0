"""Compare the native angle controls, local-metal meshes, and contact-bare control."""
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon, Rectangle

from study_rotation_jc220 import ROOT, study_plan, Geometry


def axis_weights(coordinates, lower, upper):
    bounds = np.clip(np.r_[lower,(coordinates[1:]+coordinates[:-1])/2,upper], lower, upper)
    return np.diff(bounds)


def area_statistics(values, weights):
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    order = np.argsort(values)
    percentile = np.searchsorted(np.cumsum(weights[order]), .95*weights.sum())
    return {'mean': float(np.sum(values*weights)/weights.sum()),
            'p95': float(values[order[min(percentile,len(values)-1)]])}


def valleys(wave, reflection):
    peaks, properties = find_peaks(-reflection, prominence=.02, distance=5)
    widths, _, left, right = peak_widths(-reflection, peaks, rel_height=.5)
    result = []
    for j, index in enumerate(peaks):
        result.append({'wavelength_nm': float(wave[index]), 'reflectance': float(reflection[index]),
            'prominence': float(properties['prominences'][j]),
            'half_prominence_width_nm': float(np.interp(right[j],np.arange(len(wave)),wave)-
                                             np.interp(left[j],np.arange(len(wave)),wave))})
    return result


def relative_change(reference, candidate):
    return float((candidate-reference)/reference)


def main():
    output = ROOT/'research/rotation_jc220'
    _, probes, cases = study_plan()
    data, summaries = {}, {}
    for name in cases:
        with np.load(output/name/'result.npz', allow_pickle=False) as source:
            data[name] = {key: source[key].copy() for key in source.files}
        case = data[name]
        if int(case['status']) != 2:
            raise ValueError('Incomplete native solve: '+name)
        summary = {'geometry_values': case['geometry_values'].tolist(),
                   'elapsed_seconds': float(case['elapsed_seconds']),
                   'status': int(case['status']), 'control': str(case['control']),
                   'valleys': valleys(case['wavelength_nm'],case['reflectance']),
                   'mesh_audit': json.loads(str(case['mesh_audit_json'])),
                   'metal_interface_node_distance_nm':{str(z):float(np.min(abs(case['mesh_z_nm']-z))) for z in (0.,-100.)},
                   'pml_monitor_audit': json.loads(str(case['pml_monitor_audit_json'])),
                   'fields': {}}
        geometry = Geometry(*case['geometry_values'][:5], theta_odd_deg=case['geometry_values'][5],
                            theta_even_deg=case['geometry_values'][6])
        for plane in ('mid','top'):
            x,y = case[plane+'_x_nm'],case[plane+'_y_nm']
            weights = np.outer(axis_weights(x,-geometry.p/2,geometry.p/2),
                               axis_weights(y,-geometry.p/2,geometry.p/2))
            summary['fields'][plane] = [{'wavelength_nm': float(wave),
                **area_statistics(case[plane+'_enhancement'][:,:,0,i],weights)}
                for i,wave in enumerate(case[plane+'_wavelength_nm'])]
        summaries[name] = summary
    base = data['au_both_8']
    wave = base['wavelength_nm']
    comparisons = {}
    for name, case in data.items():
        differences = {key+'_max_absolute_change': float(np.max(abs(case[key]-base[key])))
                       for key in ('reflectance','transmittance','absorptance')}
        matched = []
        for valley in summaries['au_both_8']['valleys']:
            nearby = [v for v in summaries[name]['valleys'] if abs(v['wavelength_nm']-valley['wavelength_nm']) <= 100]
            candidate = min(nearby,key=lambda v:abs(v['wavelength_nm']-valley['wavelength_nm'])) if nearby else None
            matched.append({'reference_nm': valley['wavelength_nm'], 'candidate': candidate,
                            'shift_nm': candidate['wavelength_nm']-valley['wavelength_nm'] if candidate else None,
                            'width_relative_change': relative_change(valley['half_prominence_width_nm'],
                                candidate['half_prominence_width_nm']) if candidate else None})
        field_changes = {}
        for plane in ('mid','top'):
            original = summaries['au_both_8']['fields'][plane]
            candidate = summaries[name]['fields'][plane]
            field_changes[plane] = [{'wavelength_nm': row['wavelength_nm'],
                'mean_relative_change': relative_change(ref['mean'], row['mean']),
                'p95_relative_change': relative_change(ref['p95'], row['p95'])}
                for ref,row in zip(original,candidate)]
        comparisons[name] = {**differences, 'matched_valleys': matched, 'field_changes': field_changes}
    interaction = (data['au_both_8']['reflectance']-data['au_odd_8']['reflectance']-
                   data['au_even_8']['reflectance']+data['au_zero_8']['reflectance'])
    mesh_comparisons = {}
    for coarse,fine in (('au_both_8','au_both_4'),('au_both_4','au_both_2'),
                        ('au_aligned8','au_both_2'),('au_both_8_cv0','au_both_4_cv0'),
                        ('bare_contact_8','bare_contact_4')):
        reference = summaries[fine]
        candidate = summaries[coarse]
        features = []
        for ref in reference['valleys']:
            nearby = [v for v in candidate['valleys'] if abs(v['wavelength_nm']-ref['wavelength_nm']) <= 100]
            match = min(nearby,key=lambda v:abs(v['wavelength_nm']-ref['wavelength_nm'])) if nearby else None
            features.append({'reference_nm':ref['wavelength_nm'], 'coarse_valley':match,
                'shift_nm': match['wavelength_nm']-ref['wavelength_nm'] if match else None,
                'width_relative_change':relative_change(ref['half_prominence_width_nm'],match['half_prominence_width_nm']) if match else None})
        fields = {}
        for plane in ('mid','top'):
            fields[plane] = {key+'_max_absolute_relative_change':max(
                abs(relative_change(ref[key],row[key])) for ref,row in
                zip(reference['fields'][plane],candidate['fields'][plane])) for key in ('mean','p95')}
        max_R = float(np.max(abs(data[coarse]['reflectance']-data[fine]['reflectance'])))
        gates = {'R':max_R <= .03,
            'valley_position':bool(features) and all(v['shift_nm'] is not None and abs(v['shift_nm']) <= 2 for v in features),
            'valley_width':bool(features) and all(v['width_relative_change'] is not None and abs(v['width_relative_change']) <= .1 for v in features),
            'field_mean':all(v['mean_max_absolute_relative_change'] <= .1 for v in fields.values()),
            'field_p95':all(v['p95_max_absolute_relative_change'] <= .1 for v in fields.values())}
        if not reference['valleys'] and not candidate['valleys']:
            gates['valley_position'] = None
            gates['valley_width'] = None
        mesh_comparisons[coarse+'_vs_'+fine] = {'fine_reference':fine,'coarse_candidate':coarse,
            'reflectance_max_absolute_change':max_R,'features':features,'fields':fields,
            'gates':gates,'all_gates_passed':all(value for value in gates.values() if value is not None)}
    original = Geometry(*base['geometry_values'][:5],theta_odd_deg=base['geometry_values'][5],
                        theta_even_deg=base['geometry_values'][6])
    mirrored = data['au_mirror_8']['geometry_values']
    mirror_geometry = Geometry(*mirrored[:5],theta_odd_deg=mirrored[5],theta_even_deg=mirrored[6])
    corner_error = max(float(np.min(np.linalg.norm((vertex*[-1,1])[:,None,:]-
        mirror_geometry.vertices()[index][None,:,:],axis=-1),axis=1).max())
        for vertex,index in zip(original.vertices(),(2,3,0,1)))
    final = {'native_version': str(base['fdtd_version']), 'cases': summaries,
             'comparisons_to_au_both_8': comparisons,
             'mesh_comparisons_fine_reference':mesh_comparisons,
             'criteria':{'valley_position_nm':2.,'maximum_absolute_R_change':.03,
                         'relative_width_change':.1,'relative_field_mean_change':.1,'relative_field_p95_change':.1},
             'factorial_interaction_max_abs_R': float(np.max(abs(interaction))),
             'factorial_interaction_rms_R': float(np.sqrt(np.mean(interaction**2))),
             'mirror_test': {'mapping': '(theta_odd,theta_even) -> (-theta_even,-theta_odd)',
                'maximum_geometry_corner_error_nm': corner_error,
                'maximum_absolute_R_difference': comparisons['au_mirror_8']['reflectance_max_absolute_change'],
                'scope': 'Normal incidence with x-linear polarization and isotropic materials',
                'possible_training_convention': 'Choose theta_odd+theta_even >= 0 by applying the verified mirror mapping'},
             'width_definition': 'Half of local prominence in reflectance; not a modal Q linewidth',
             'field_definition': 'Area-weighted |E|^2/|Einc|^2 on two XY planes at explicitly sampled frequencies',
             'scope': 'One fixed five-length geometry; no universal angle law or BIC classification is inferred'}
    final['planar_cv0_checks'] = {}
    for mesh in (8,4):
        film = json.loads((ROOT/f'research/metal_loss_fit/film_benchmark_clear_pml/release_jc_{mesh}nm_bg20nm/result.json').read_text())
        final['planar_cv0_checks'][str(mesh)] = {key:film[key] for key in (
            'status','native_mesh_refinement','model_maximum_R_error','model_maximum_T_error',
            'raw_input_maximum_R_error','raw_input_maximum_T_error')}
    (output/'summary.json').write_text(json.dumps(final,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

    labels = {'au_zero_8':'No rotation','au_odd_8':'Odd only','au_even_8':'Even only','au_both_8':'Both groups'}
    fig,axes = plt.subplots(3,1,figsize=(9,9),constrained_layout=True)
    for name,label in labels.items():
        for ax in axes:
            ax.plot(wave,data[name]['reflectance'],label=label,lw=1.6)
    for ax,limits in zip(axes,((700,1700),(720,900),(1000,1250))):
        ax.set_xlim(*limits)
        ax.set_ylabel('Reflectance')
        ax.grid(alpha=.2)
    axes[0].legend(ncol=2)
    axes[2].set_ylim(.58,.96)
    axes[0].text(.98,.02,f'Mirror test: max |dR| = {final["mirror_test"]["maximum_absolute_R_difference"]:.2g}',
                 ha='right',va='bottom',transform=axes[0].transAxes,fontsize=9)
    axes[0].set_title('Independent quadrant-angle controls: corrected Au model')
    axes[-1].set_xlabel('Wavelength (nm)')
    fig.savefig(output/'angle_spectra.png',dpi=160)
    plt.close(fig)

    fig,axes = plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,name in zip(axes[:2],('au_zero_8','au_both_8')):
        values = data[name]['geometry_values']
        geometry = Geometry(*values[:5],theta_odd_deg=values[5],theta_even_deg=values[6])
        for index,vertex in enumerate(geometry.vertices()):
            ax.add_patch(Polygon(vertex,facecolor='#4a91bb' if index in (0,3) else '#d79a45',alpha=.9))
        ax.set_xlim(-geometry.p/2,geometry.p/2)
        ax.set_ylim(-geometry.p/2,geometry.p/2)
        ax.set_aspect('equal')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_title(f'Odd {values[5]:.2f} deg / Even {values[6]:.2f} deg')
    axes[2].add_patch(Rectangle((-1,-350),2,250,facecolor='#b8ccd6'))
    axes[2].add_patch(Rectangle((-1,-100),2,100,facecolor='#d8ae38'))
    axes[2].add_patch(Rectangle((-.75,0),.5,geometry.h,facecolor='#4a91bb'))
    axes[2].add_patch(Rectangle((.25,0),.5,geometry.h,facecolor='#d79a45'))
    for z,label in ((-225,'Thick SiO2 to lower PML'),(-50,'Au 100 nm'),(geometry.h/2,'As2Se3')):
        axes[2].text(0,z,label,ha='center',va='center',fontsize=9)
    axes[2].set_xlim(-1,1)
    axes[2].set_ylim(-350,geometry.h+50)
    axes[2].set_xticks([])
    axes[2].set_ylabel('z (nm)')
    axes[2].set_title('Stack schematic')
    fig.savefig(output/'structure.png',dpi=160)
    plt.close(fig)

    fig,axes = plt.subplots(3,1,figsize=(9,8),sharex=True,constrained_layout=True)
    for name,label in (('au_both_8','Au / 8 nm'),('au_both_4','Au / 4 nm'),('au_both_2','Au / 2 nm'),
                       ('bare_contact_8','Glass contact / 8 nm reference grid'),
                       ('bare_contact_4','Glass contact / 4 nm reference grid')):
        for ax,key in zip(axes,('reflectance','transmittance','absorptance')):
            ax.plot(wave,data[name][key],label=label,lw=1.5)
    for ax,label in zip(axes,('R','T','A')):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.set_xlim(700,1700)
    axes[0].legend()
    axes[0].set_title('Metal and local metal-mesh controls')
    axes[-1].set_xlabel('Wavelength (nm)')
    fig.savefig(output/'metal_mesh_spectra.png',dpi=160)
    plt.close(fig)

    fig,axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for row,group in enumerate((('au_both_8','au_both_4','au_both_2','au_aligned8'),
                                ('au_both_8_cv0','au_both_4_cv0'))):
        for name in group:
            for col in range(2):
                axes[row,col].plot(wave,data[name]['reflectance'],lw=1.5,label=name)
        for col,limits in enumerate(((700,900),(1000,1250))):
            axes[row,col].set_xlim(*limits)
            axes[row,col].set_xlabel('Wavelength (nm)')
            axes[row,col].set_ylabel('Reflectance')
            axes[row,col].grid(alpha=.2)
        axes[row,0].legend(fontsize=8)
        axes[row,1].set_ylim(.55,.96)
    axes[0,0].set_title('Conformal variant 1: short-wave feature')
    axes[0,1].set_title('Conformal variant 1: long-wave feature')
    axes[1,0].set_title('Conformal variant 0: short-wave feature')
    axes[1,1].set_title('Conformal variant 0: long-wave feature')
    fig.savefig(output/'variant_controls.png',dpi=160)
    plt.close(fig)

    reference_valleys = sorted(summaries['au_both_8']['valleys'],key=lambda v:v['prominence'],reverse=True)[:2]
    centers = sorted(v['wavelength_nm'] for v in reference_valleys)
    fig,axes = plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
    names = ('au_both_4','au_both_2','bare_contact_4')
    maximum = max(float(data[name]['mid_enhancement'].max()) for name in names)
    for i,center in enumerate(centers):
        for j,name in enumerate(names):
            case = data[name]
            k = int(np.argmin(abs(case['mid_wavelength_nm']-center)))
            image = axes[i,j].pcolormesh(case['mid_x_nm'],case['mid_y_nm'],
                case['mid_enhancement'][:,:,0,k].T,norm=LogNorm(vmin=.1,vmax=maximum),
                shading='auto',cmap='magma')
            geometry = Geometry(*case['geometry_values'][:5],theta_odd_deg=case['geometry_values'][5],
                                theta_even_deg=case['geometry_values'][6])
            for vertex in geometry.vertices():
                axes[i,j].add_patch(Polygon(vertex,fill=False,edgecolor='cyan',lw=.7))
            axes[i,j].set_aspect('equal')
            axes[i,j].set_title(f'{name}\n{case["mid_wavelength_nm"][k]:.1f} nm')
            axes[i,j].set_xlabel('x (nm)')
            axes[i,j].set_ylabel('y (nm)')
    fig.colorbar(image,ax=axes,label='Intensity enhancement |E|^2 / |Einc|^2',shrink=.75)
    fig.suptitle('Field at mid-height: fixed illumination wavelengths, shared color scale')
    fig.savefig(output/'fields_mid.png',dpi=150)
    plt.close(fig)
    print(json.dumps({'summary':str(output/'summary.json'),
        'comparisons':{name:{k:v for k,v in row.items() if k!='field_changes'}
                       for name,row in comparisons.items()}},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
