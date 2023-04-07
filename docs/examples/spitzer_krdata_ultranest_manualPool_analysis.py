import joblib
import numpy as np
import pandas as pd

from functools import partial
from multiprocessing import Pool, cpu_count

# from exotso.exotso_ultranest import ExoplanetUltranestTSO
from exotso import ExoplanetUltranestTSO
from exotso.utils import (
    grab_data_from_csv,
    visualise_ultranest_traces_corner,
    visualise_ultranest_samples,
    visualise_mle_solution
)


def run_one(
    aper_key,
    n_sig=5,
    aor_dir='r64922368',
    channel='ch2',  # CHANNEL SETTING
    planet_name='hatp26b',
    mast_name='HAT-P-26b',
    inj_fpfs=0 / 1e6,  # no injected signal
    init_fpfs=265 / 1e6,  # no injected signal
    num_live_points=400,
    # aper_key = 'rad_2p5_0p0',
    centering_key='gaussian_fit',
    # centering_key = 'fluxweighted',
    trim_size=1/24,  # one hour in day units
    timebinsize=0/60/24  # 0 minutes in day units
):
    print(f'Running Ultranest on {aper_key}')
    hatp26b_krdata = ExoplanetUltranestTSO(
        df=df_hatp26b,
        planet_name=planet_name,
        mast_name=mast_name,
        trim_size=trim_size,
        timebinsize=timebinsize,
        centering_key=centering_key,
        aper_key=aper_key,
        log_dir='ultranest_savedir',
        init_fpfs=init_fpfs,
        n_sig=n_sig,
        process_ultranest=False,
        run_full_pipeline=False,
        visualise_mle=False,
        visualise_traces_corner=False,
        visualise_samples=False,
        savenow=False,
        standardise_fluxes=True,
        standardise_times=False,
        standardise_centers=False,
        verbose=False
    )

    hatp26b_krdata.initialise_data_and_params()
    '''
        hatp26b_krdata.run_mle_pipeline()

        fpfs_ml, delta_ecenter_ml, log_f_ml = hatp26b_krdata.soln.x

        print(
            f'fpfs_ml={fpfs_ml*1e6}ppm\n'
            f'delta_ecenter_ml={delta_ecenter_ml}\n'
            f'ecenter_ml={hatp26b_krdata.ecenter0 + delta_ecenter_ml}\n'
            f'log_f_ml={log_f_ml}\n'
        )
        '''

    hatp26b_krdata.run_ultranest_pipeline(
        num_live_points=num_live_points,
        log_dir_extra=aper_key
    )

    hatp26b_krdata.save_mle_ultranest(
        num_live_points=num_live_points
    )

    # hatp26b_krdata_apers[aper_key] = hatp26b_krdata

    # sampler = hatp26b_krdata.sampler
    # if np.median(sampler.run_sequence['samples'][:, 0]) < 1:
    #     fpfs_ = sampler.run_sequence['samples'][:, 0]*1e6
    #     sampler.run_sequence['samples'][:, 0] = fpfs_
    #     hatp26b_krdata.sampler = sampler

    # results = hatp26b_krdata.ultranest_results
    # if np.median(results['weighted_samples']['points'][:, 0]) < 1:
    #     fpfs_ = results['weighted_samples']['points'][:, 0]*1e6
    #     results['weighted_samples']['points'][:, 0] = fpfs_
    #     hatp26b_krdata.ultranest_results = results

    # visualise_ultranest_traces_corner(hatp26b_krdata, suptitle=None)

    return hatp26b_krdata


if __name__ == '__main__':
    ppm = 1e6
    n_sig = 5
    aor_dir = 'r64922368'
    channel = 'ch2'  # CHANNEL SETTING
    planet_name = 'hatp26b'
    mast_name = 'HAT-P-26b'
    inj_fpfs = 0 / ppm  # no injected signal
    init_fpfs = 265 / ppm  # no injected signal
    num_live_points = 400
    # aper_key = 'rad_2p5_0p0'
    centering_key = 'gaussian_fit'
    # centering_key = 'fluxweighted'
    trim_size = 1/24  # one hour in day units
    timebinsize = 0/60/24  # 0 minutes in day units

    aornums = [
        # # 'r42621184',  # Passed Eclipse
        # # 'r42624768',  # Passed Eclipse
        # # 'r47023872',  # Transit
        'r50676480',
        'r50676736',
        'r50676992',
        'r50677248',
        'r64922368',
        'r64923136',
        'r64923904',
        'r64924672'
    ]

    df_hatp26b = grab_data_from_csv(filename=None, aornums=aornums)

    aper_keys = [
        col_.replace('flux_', '')
        for col_ in df_hatp26b.columns if 'flux' in col_ and 'rad' in col_
    ]
    run_partial_one = partial(
        run_one,
        n_sig=n_sig,
        aor_dir=aor_dir,
        channel=channel,  # CHANNEL SETTING
        planet_name=planet_name,
        mast_name=mast_name,
        inj_fpfs=inj_fpfs,  # no injected signal
        init_fpfs=init_fpfs,  # no injected signal
        num_live_points=num_live_points,
        centering_key=centering_key,
        trim_size=trim_size,  # one hour in day units
        timebinsize=timebinsize  # 0 minutes in day units
    )
    # hatp26b_krdata_apers = {}
    with Pool(cpu_count()-1) as pool:
        pool.starmap(run_partial_one, zip(aper_keys))
