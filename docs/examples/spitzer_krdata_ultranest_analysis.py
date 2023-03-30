import numpy as np
import pandas as pd

from exotso.exotso_ultranest import ExoplanetUltranestTSO
from spitzer_ultranest_utils import (
    grab_data_from_csv,
    visualise_ultranest_traces_corner,
    visualise_ultranest_samples,
    visualise_mle_solution
)


if __name__ == '__main__':
    ppm = 1e6
    n_sig = 5
    aor_dir = 'r64922368'
    channel = 'ch2'  # CHANNEL SETTING
    planet_name = 'hatp26b'
    mast_name = 'HAT-P-26b'
    inj_fpfs = 0 / ppm  # no injected signal
    init_fpfs = 265 / ppm  # no injected signal
    n_samples = 10000
    nwalkers = 32
    min_num_live_points = 1000
    aper_key = 'rad_2p5_0p0'
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

    # hatp26b_krdata_batches = {}
    for kaor_, aors_ in enumerate([aornums]):
        df_hatp26b = grab_data_from_csv(filename=None, aornums=aors_)

        hatp26b_krdata = ExoplanetUltranestTSO(
            df=df_hatp26b,
            mast_name=mast_name,
            trim_size=trim_size,
            timebinsize=timebinsize,
            centering_key=centering_key,
            aper_key=aper_key,
            inj_fpfs=0,
            init_fpfs=init_fpfs,
            nwalkers=nwalkers,
            n_samples=n_samples,
            n_sig=n_sig,
            process_mcmc=False,
            run_full_pipeline=False,
            visualise_mle=False,
            visualise_nests=False,
            visualise_mcmc_results=False,
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
            min_num_live_points=min_num_live_points
        )
        # hatp26b_krdata.save_mle_ultranest()
        hatp26b_krdata_all_live1000 = hatp26b_krdata

        sampler = hatp26b_krdata.sampler
        if np.median(sampler.run_sequence['samples'][:, 0]) < 1:
            fpfs_ = sampler.run_sequence['samples'][:, 0]*1e6
            sampler.run_sequence['samples'][:, 0] = fpfs_
            hatp26b_krdata.sampler = sampler

        results = hatp26b_krdata.ultranest_results
        if np.median(results['weighted_samples']['points'][:, 0]) < 1:
            fpfs_ = results['weighted_samples']['points'][:, 0]*1e6
            results['weighted_samples']['points'][:, 0] = fpfs_
            hatp26b_krdata.ultranest_results = results

        visualise_ultranest_traces_corner(hatp26b_krdata, suptitle=None)

    if hatp26b_krdata.process_mcmc:
        hatp26b_krdata.run_ultranest_pipeline()

    if hatp26b_krdata.visualise_mle:
        visualise_mle_solution(hatp26b_krdata)

    if hatp26b_krdata.visualise_nests:
        visualise_ultranest_traces_corner(
            hatp26b_krdata,
            discard=100,
            thin=15,
            burnin=0.2,
            verbose=False
        )

    if hatp26b_krdata.visualise_mcmc_results:
        visualise_ultranest_samples(
            hatp26b_krdata,
            discard=0,
            thin=1,
            burnin=0.2,
            verbose=False
        )

    if hatp26b_krdata.savenow:
        hatp26b_krdata.save_mle_ultranest()

    # phase_bins, phase_binned_flux, phase_binned_ferr = phase_bin_data(
    #     df,
    #     planet_name,
    #     n_phases=1000,
    #     min_phase=0.4596,
    #     max_phase=0.5942
    # )

    # plt.errorbar(phase_bins, phase_binned_flux, phase_binned_ferr, fmt='o')
