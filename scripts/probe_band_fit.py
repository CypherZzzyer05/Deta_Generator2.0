"""Test band-limited sampled Palik input with an explicit near-IR fit range."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from deta_generator2.lumerical import load_lumapi
from probe_metal_fit import metrics


def main():
    output=ROOT/'research/metal_loss_fit'
    output.mkdir(parents=True,exist_ok=True)
    wave=np.linspace(700.,1700.,1000)
    frequency=299792458/(wave*1e-9)
    rows=[]
    arrays={'wavelength_nm':wave}
    with load_lumapi().FDTD(hide=True) as fdtd:
        version=str(fdtd.version())
        fdtd.setglobalsource('wavelength start',700e-9)
        fdtd.setglobalsource('wavelength stop',1700e-9)
        for metal,base in (('Au','Au (Gold) - Palik'),('Ag','Ag (Silver) - Palik (0-2um)')):
            measured=np.asarray(fdtd.getindex(base,frequency)).reshape(-1)
            arrays[metal+'_measured']=measured
            print('BASE RANGE '+metal+': '+json.dumps({key:float(fdtd.getmaterial(base,key)) for key in
                ('specify fit range','wavelength min','wavelength max')}),flush=True)
            for source in ('copied_database','sampled_band'):
                for coefficients,weight in ((6,10),(8,10),(12,10)):
                    if source=='copied_database':
                        name=fdtd.copymaterial(base)
                    else:
                        material=fdtd.addmaterial('Sampled data')
                        name=f'{metal} NIR probe {coefficients}'
                        fdtd.setmaterial(material,'name',name)
                        data=np.column_stack([frequency,measured**2])[::-1].copy()
                        fdtd.setmaterial(name,'sampled data',data)
                    options={'tolerance':0.,'max coefficients':coefficients,'imaginary weight':weight,
                             'make fit passive':1,'improve numerical stability':1,'specify fit range':1,
                             'wavelength min':700e-9,'wavelength max':1700e-9}
                    for key,value in options.items():fdtd.setmaterial(name,key,value)
                    fitted=np.asarray(fdtd.getfdtdindex(name,frequency,frequency.min(),frequency.max())).reshape(-1)
                    values=metrics(measured,fitted)
                    accepted=bool(values['passive_in_band'] and values['max_n_error']<=.02 and
                        values['max_k_error']<=.05 and values['max_relative_real_epsilon_error']<=.02 and
                        values['max_relative_imag_epsilon_error']<=.05)
                    row={'metal':metal,'source':source,'fit_parameters':options,'metrics':values,
                         'passes_provisional_material_checks':accepted}
                    rows.append(row);arrays[f'{metal}_{source}_{coefficients}']=fitted
                    print(json.dumps(row),flush=True)
                    if accepted:break
                if rows[-1]['passes_provisional_material_checks']:break
    (output/'band_probe.json').write_bytes((json.dumps({'native_version':version,'solver_run_performed':False,
        'results':rows},indent=2)+'\n').encode('utf-8'))
    np.savez_compressed(output/'band_probe.npz',**arrays)


if __name__=='__main__':main()
