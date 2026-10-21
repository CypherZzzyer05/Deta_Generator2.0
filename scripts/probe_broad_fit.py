"""Include the visible interband region when fitting the near-IR metal response."""
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
    wave=np.linspace(700.,1700.,1000);freq=299792458/(wave*1e-9)
    rows=[];arrays={'wavelength_nm':wave}
    with load_lumapi().FDTD(hide=True) as f:
        version=str(f.version())
        for metal,base in (('Au','Au (Gold) - Palik'),('Ag','Ag (Silver) - Palik (0-2um)')):
            raw=np.asarray(f.getindex(base,freq)).reshape(-1);arrays[metal+'_measured']=raw
            for lower,upper,count,weight in ((400,2000,6,1),(400,2000,12,2),(500,2000,12,10),(400,2000,24,2)):
                name=f.copymaterial(base)
                options={'tolerance':0.,'max coefficients':count,'imaginary weight':weight,
                         'make fit passive':1,'improve numerical stability':1,
                         'specify fit range':1,'wavelength min':lower*1e-9,'wavelength max':upper*1e-9}
                for key,value in options.items():f.setmaterial(name,key,value)
                fitted=np.asarray(f.getfdtdindex(name,freq,299792458/(upper*1e-9),299792458/(lower*1e-9))).reshape(-1)
                values=metrics(raw,fitted)
                accepted=bool(values['passive_in_band'] and values['max_n_error']<=.02 and
                    values['max_k_error']<=.05 and values['max_relative_real_epsilon_error']<=.02 and
                    values['max_relative_imag_epsilon_error']<=.05)
                row={'metal':metal,'fit_parameters':options,'metrics':values,
                     'passes_provisional_material_checks':accepted}
                print(json.dumps(row),flush=True);rows.append(row)
                arrays[f'{metal}_{lower}_{upper}_{count}_{weight}']=fitted
                if accepted:break
    (output/'broad_probe.json').write_bytes((json.dumps({'native_version':version,'solver_run_performed':False,
        'results':rows},indent=2)+'\n').encode('utf-8'))
    np.savez_compressed(output/'broad_probe.npz',**arrays)


if __name__=='__main__':main()
