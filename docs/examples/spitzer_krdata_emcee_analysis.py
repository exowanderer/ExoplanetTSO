import pandas as pd

from datetime import datetime, timezone
from matplotlib import pyplot as plt

from exotso.exotso_emcee import ExoplanetEmceeTSO
from exotso.utils import (
    grab_data_from_csv,
    print_mle_results,
    visualise_emcee_traces_corner,
    visualise_emcee_samples,
    visualise_mle_solution
)


if __name__ == '__main__':
    plt.ion()

    ppm = 1e6
    n_sig = 5
    aor_dir = 'r64922368'
    channel = 'ch2'  # CHANNEL SETTING
    planet_name = 'hatp26b'
    mast_name = 'HAT-P-26b'
    inj_fpfs = 0 / ppm  # no injected signal
    init_fpfs = 265 / ppm  # initial guess from Wallack et al 2019
    n_samples = 10000
    nwalkers = 32
    aper_key = 'rad_2p5_0p0'
    centering_key = 'gaussian_fit'
    # centering_key = 'fluxweighted'
    trim_size = 1/24  # one hour in day units
    timebinsize = 0/60/24  # 0 minutes in day units
    # timebinsize = 0.5/60/24  # 0.5 minutes in day units

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
    # for aors_ in [aornums]:
    #     print(aors_)
    df_hatp26b = grab_data_from_csv(filename=None, aornums=aornums)

    hatp26b_krdata = ExoplanetEmceeTSO(
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
        estimate_pinknoise=True,
        process_mcmc=False,
        run_full_pipeline=False,
        visualise_mle=False,
        visualise_chains=False,
        visualise_mcmc_results=False,
        savenow=False,
        standardise_fluxes=True,
        standardise_times=False,
        standardise_centers=False,
        verbose=False
    )

    hatp26b_krdata.initialise_data_and_params()

    hatp26b_krdata.run_mle_pipeline()

    print_mle_results(hatp26b_krdata.soln.x, hatp26b_krdata.ecenter0)

    hatp26b_krdata.run_emcee_pipeline()

    hatp26b_krdata.postprocess_pipeline(discard=100, thin=15, burnin=0.2)

    """
    isotime = datetime.now(timezone.utc).isoformat()
    filename = (
        f'emcee_spitzer_krdata_1m_binned_all8aors_results_{isotime}'
        '.joblib.save'
    )
    hatp26b_krdata.save_mle_emcee(filename=filename)
    """

    visualise_emcee_traces_corner(
        hatp26b_krdata,
        discard=100,
        thin=15,
        burnin=0.2,
        verbose=False
    )

    if hatp26b_krdata.process_mcmc:
        hatp26b_krdata.run_emcee_pipeline()

    if hatp26b_krdata.visualise_mle:
        visualise_mle_solution(hatp26b_krdata)

    if hatp26b_krdata.visualise_chains:
        visualise_emcee_traces_corner(
            hatp26b_krdata,
            discard=100,
            thin=15,
            burnin=0.2,
            verbose=False
        )

    if hatp26b_krdata.visualise_mcmc_results:
        visualise_emcee_samples(
            hatp26b_krdata,
            discard=0,
            thin=1,
            burnin=0.2,
            verbose=False
        )

    if hatp26b_krdata.savenow:
        isotime = datetime.now(timezone.utc).isoformat()
        filename = f'emcee_spitzer_krdata_results_{isotime}.joblib.save'

        hatp26b_krdata.save_mle_emcee()

    # phase_bins, phase_binned_flux, phase_binned_ferr = phase_bin_data(
    #     df,
    #     planet_name,
    #     n_phases=1000,
    #     min_phase=0.4596,
    #     max_phase=0.5942
    # )

    # plt.errorbar(phase_bins, phase_binned_flux, phase_binned_ferr, fmt='o')
