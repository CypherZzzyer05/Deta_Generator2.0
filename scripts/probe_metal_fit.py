"""Measure loss fidelity of native metal fits before changing generator defaults."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def metrics(original,fitted):
    raw_eps,fit_eps=original**2,fitted**2
    return {'max_n_error':float(np.max(abs(original.real-fitted.real))),
        'max_k_error':float(np.max(abs(original.imag-fitted.imag))),
        'max_relative_real_epsilon_error':float(np.max(abs((fit_eps.real-raw_eps.real)/raw_eps.real))),
        'max_relative_imag_epsilon_error':float(np.max(abs((fit_eps.imag-raw_eps.imag)/raw_eps.imag))),
        'raw_imag_epsilon_range':[float(raw_eps.imag.min()),float(raw_eps.imag.max())],
        'fitted_imag_epsilon_range':[float(fit_eps.imag.min()),float(fit_eps.imag.max())],
        'passive_in_band':bool(np.all(fit_eps.imag>=0))}


def main():
    from deta_generator2.lumerical import load_lumapi
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lumapi')
    args=parser.parse_args()
    output=ROOT/'research/metal_loss_fit'
    output.mkdir(parents=True,exist_ok=True)
    wave=np.linspace(700.,1700.,1000)
    frequency=299792458/(wave*1e-9)
    api=load_lumapi(args.lumapi)
    results=[]
    spectra={}
    with api.FDTD(hide=True) as fdtd:
        version=str(fdtd.version())
        for metal,base in (('Au','Au (Gold) - Palik'),('Ag','Ag (Silver) - Palik (0-2um)')):
            measured=np.asarray(fdtd.getindex(base,frequency)).reshape(-1)
            default=np.asarray(fdtd.getfdtdindex(base,frequency,frequency.min(),frequency.max())).reshape(-1)
            row={'metal':metal,'label':'legacy_default','metrics':metrics(measured,default),
                 'fit_parameters':{key:float(fdtd.getmaterial(base,key)) for key in
                     ('tolerance','max coefficients','imaginary weight','make fit passive','improve numerical stability')}}
            results.append(row)
            spectra[metal+'_measured']=measured
            spectra[metal+'_legacy']=default
            print(json.dumps(row),flush=True)
            # Follow the documented tolerance=0 recommendation before adding coefficients.
            # Keep passivity and stability restrictions enabled.
            for coefficients,weight in ((6,1),(8,2),(12,2)):
                name=fdtd.copymaterial(base)
                fdtd.setmaterial(name,'tolerance',0.)
                fdtd.setmaterial(name,'max coefficients',coefficients)
                fdtd.setmaterial(name,'imaginary weight',weight)
                fdtd.setmaterial(name,'make fit passive',1)
                fdtd.setmaterial(name,'improve numerical stability',1)
                fitted=np.asarray(fdtd.getfdtdindex(name,frequency,frequency.min(),frequency.max())).reshape(-1)
                values=metrics(measured,fitted)
                accepted=bool(values['passive_in_band'] and values['max_n_error']<=.02 and
                    values['max_k_error']<=.05 and values['max_relative_real_epsilon_error']<=.02 and
                    values['max_relative_imag_epsilon_error']<=.05)
                row={'metal':metal,'label':f'coeff{coefficients}_weight{weight}',
                     'fit_parameters':{'tolerance':0.,'max coefficients':coefficients,'imaginary weight':weight,
                                       'make fit passive':1,'improve numerical stability':1},
                     'metrics':values,'passes_provisional_material_checks':accepted}
                results.append(row)
                spectra[f'{metal}_coeff{coefficients}_weight{weight}']=fitted
                print(json.dumps(row),flush=True)
                if accepted:
                    break
    (output/'fit_probe.json').write_bytes((json.dumps({'native_version':version,'solver_run_performed':False,
        'band_nm':[700,1700],'results':results,
        'note':'Material-only checks are provisional; plane-film FDTD versus analytic R/T still required'},
        indent=2)+'\n').encode('utf-8'))
    np.savez_compressed(output/'fit_probe.npz',wavelength_nm=wave,**spectra)


if __name__=='__main__':
    main()
