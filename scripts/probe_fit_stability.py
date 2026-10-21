"""Test documented stability-fit tradeoff while keeping passivity enforced."""
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
    wave=np.linspace(700.,1700.,1000)
    freq=299792458/(wave*1e-9)
    rows=[];arrays={'wavelength_nm':wave}
    with load_lumapi().FDTD(hide=True) as fdtd:
        version=str(fdtd.version())
        for metal,base in (('Au','Au (Gold) - Palik'),('Ag','Ag (Silver) - Palik (0-2um)')):
            original=np.asarray(fdtd.getindex(base,freq)).reshape(-1)
            arrays[metal+'_measured']=original
            for count,weight in ((2,1),(4,2),(6,2),(6,10),(12,10)):
                name=fdtd.copymaterial(base)
                options={'tolerance':0.,'max coefficients':count,'imaginary weight':weight,
                         'make fit passive':1,'improve numerical stability':0,
                         'specify fit range':1,'wavelength min':700e-9,'wavelength max':1700e-9}
                for key,value in options.items():fdtd.setmaterial(name,key,value)
                fitted=np.asarray(fdtd.getfdtdindex(name,freq,freq.min(),freq.max())).reshape(-1)
                values=metrics(original,fitted)
                accepted=bool(values['passive_in_band'] and values['max_n_error']<=.02 and
                    values['max_k_error']<=.05 and values['max_relative_real_epsilon_error']<=.02 and
                    values['max_relative_imag_epsilon_error']<=.05)
                row={'metal':metal,'fit_parameters':options,'metrics':values,
                     'passes_provisional_material_checks':accepted}
                print(json.dumps(row),flush=True);rows.append(row)
                arrays[f'{metal}_coeff{count}_weight{weight}']=fitted
                if accepted:break
    (output/'stability_probe.json').write_bytes((json.dumps({'native_version':version,
        'solver_run_performed':False,'results':rows},indent=2)+'\n').encode('utf-8'))
    np.savez_compressed(output/'stability_probe.npz',**arrays)


if __name__=='__main__':main()
