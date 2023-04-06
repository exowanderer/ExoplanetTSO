import batman
import corner
import joblib
import numpy as np
import pandas as pd
import os

try:
    from ultranest.plot import cornerplot, PredictionBand, traceplot
    HAS_ULTRANEST = True
except ImportError:
    HAS_ULTRANEST = False

from dataclasses import dataclass
from exomast_api import exoMAST_API
from matplotlib import pyplot as plt
from scipy.special import gammaln, ndtri
from tqdm import tqdm

try:
    from wanderer import load_wanderer_instance_from_file
    HAS_WANDERER = True
except ImportError:
    print("Please install `wanderer` before importing")
    HAS_WANDERER = False

from .models import ExoplanetTSOData

"""
@dataclass
class KRDataInputs:
    ind_kdtree: np.ndarray = None
    gw_kdtree: np.ndarray = None
    fluxes: np.ndarray = None
"""
"""
@dataclass
class MCMCArgs:
    times: np.ndarray = None
    fluxes: np.ndarray = None
    flux_errs: np.ndarray = None
    aornums: np.ndarray = None
    krdata_inputs: KRDataInputs = None
    period: float = 0
    tcenter: float = 0
    ecenter: float = 0
    ecenter0: float = 0  # made static to fit delta_ecenter
    inc: float = 0
    aprs: float = 0
    rprs: float = 0
    fpfs: float = 0
    ecc: float = 0
    omega: float = 0
    u1: float = 0
    u2: float = 0
    offset: float = 0
    slope: float = 0
    curvature: float = 0
    ldtype: str = 'uniform'
    transit_type: str = 'secondary'

"""

"""
@dataclass
class ExoplanetTSOData:
    times: np.ndarray = None
    fluxes: np.ndarray = None
    flux_errs: np.ndarray = None
    aornums: np.ndarray = None
    npix: np.ndarray = None
    pld_intensities: np.ndarray = None
    xcenters: np.ndarray = None
    ycenters: np.ndarray = None
    xwidths: np.ndarray = None
    ywidths: np.ndarray = None
"""


def grab_data_from_csv(filename=None, aornums=None):
    if filename is None:
        filename = 'ExtractedData/ch2/hatp26b_ch2_complete_median_full_df.csv'

    df0 = pd.read_csv(filename)

    if aornums is None:
        aornums = df0.aornum.unique()

    hours2days = 24
    n_trim = 1.0 / hours2days
    df_sub_list = []
    for aor_ in aornums:
        df_aor_ = df0.query(f'aornum == "{aor_}"')

        trim_start = df_aor_.bmjd_adjstd.iloc[0] + n_trim
        df_aor_ = df_aor_.query(f'bmjd_adjstd > {trim_start}')

        df_sub_list.append(df_aor_)

    df = pd.concat(df_sub_list)

    # Sort by combined BMJD-ajstd without dropping BMJD-ajstd
    df = df.set_index(keys='bmjd_adjstd', drop=False)
    df.sort_index(inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def bin_df_time(df, timebinsize=0):

    if timebinsize <= 0:
        return

    med_binned = {}
    std_binned = {}

    for aor_ in tqdm(df.aornum.unique(), desc='AORs'):
        df_aor_ = df.query(f'aornum == "{aor_}"')

        time_range = df_aor_.bmjd_adjstd.max() - df_aor_.bmjd_adjstd.min()
        nbins_aor_ = np.ceil(time_range / timebinsize).astype(int)

        start = df_aor_.bmjd_adjstd.min()
        end = start + timebinsize
        med_binned[aor_] = []
        std_binned[aor_] = []
        for _ in tqdm(range(nbins_aor_), desc='Bins'):
            end = start + timebinsize
            df_in_bin = df_aor_.query(f'{start} < bmjd_adjstd < {end}')
            med_binned[aor_].append(df_in_bin.median(numeric_only=True))

            unc_ = df_in_bin.std(numeric_only=True) / np.sqrt(len(df_in_bin))
            std_binned[aor_].append(unc_)

            # print(k, start, end, df.bmjd_adjstd.max(), df_in_bin)
            start = end

        med_binned[aor_] = pd.concat(med_binned[aor_], axis=1).T
        std_binned[aor_] = pd.concat(std_binned[aor_], axis=1).T

        med_binned[aor_]['aornum'] = aor_
        std_binned[aor_]['aornum'] = aor_

    med_binned = pd.concat(med_binned.values())
    std_binned = pd.concat(std_binned.values())

    med_binned = med_binned.set_index(keys='bmjd_adjstd', drop=False)
    med_binned.sort_index(inplace=True)
    med_binned.reset_index(inplace=True, drop=True)

    std_binned = std_binned.set_index(keys='bmjd_adjstd', drop=False)
    std_binned.sort_index(inplace=True)
    std_binned.reset_index(inplace=True, drop=True)

    return med_binned, std_binned


def bin_array_time(times, vector, nbins):

    assert (nbins > 0), '`nbins` must be a positive integer'
    assert (isinstance(nbins, int)), '`nbins` must be a positive integer'

    time_bins = np.linspace(times.min(), times.max(), nbins)

    n_pts_per_bin = times.size / nbins
    delta_time = np.median(np.diff(time_bins))

    med_binned = np.zeros_like(time_bins)
    std_binned = np.zeros_like(time_bins)
    for k, start in enumerate(time_bins):
        end = start + delta_time
        in_bin = (start <= times)*(times < end)
        med_binned[k] = np.nanmedian(vector[in_bin])
        std_binned[k] = np.nanstd(vector[in_bin]) / np.sqrt(n_pts_per_bin)

    return med_binned, std_binned


def trim_initial_timeseries(df, trim_size=0, aornums=None):

    if trim_size <= 0:
        return df

    if aornums is None:
        aornums = df.aornum.unique()

    trimmed = []
    for aor_ in aornums:
        df_aor_ = df.query(f'aornum == "{aor_}"')
        bmjd_min = df_aor_.bmjd_adjstd.min()
        trimmed_ = df_aor_.query(f'bmjd_adjstd >= {bmjd_min + trim_size}')
        trimmed.append(trimmed_)

    df_trimmed = pd.concat(trimmed, axis=0)

    # Sort by combined BMJD-ajstd without dropping BMJD-ajstd
    df_trimmed = df_trimmed.set_index(keys='bmjd_adjstd', drop=False)
    df_trimmed.sort_index(inplace=True)
    df_trimmed.reset_index(inplace=True, drop=True)

    return df_trimmed


def batman_wrapper(
        times, period, tcenter, inc, aprs, rprs, ecc, omega, u1, u2,
        offset, slope, curvature, ecenter=None, fpfs=None,
        ldtype='uniform', transit_type='secondary', verbose=False):
    '''
        Written By Jonathan Fraine
        https://github.com/exowanderer/Fitting-Exoplanet-Transits
    '''
    # print(fpfs, ecenter)
    if verbose:
        print('times', times.mean())
        print('period', period)
        print('tcenter', tcenter)
        print('inc', inc)
        print('aprs', aprs)
        print('rprs', rprs)
        print('ecc', ecc)
        print('omega', omega)
        print('u1', u1)
        print('u2', u2)
        print('offset', offset)
        print('slope', slope)
        print('curvature', curvature)
        print('ldtype', ldtype)
        print('transit_type', transit_type)
        print('t_secondary', ecenter)
        print('fp', fpfs)

    if ecenter is None:
        ecenter = tcenter + 0.5 * period

    bm_params = batman.TransitParams()  # object to store transit parameters
    bm_params.per = period   # orbital period
    bm_params.t0 = tcenter  # time of inferior conjunction
    bm_params.inc = inc      # inclunaition in degrees
    bm_params.a = aprs     # semi-major axis (in units of stellar radii)
    bm_params.rp = rprs     # planet radius (in units of stellar radii)
    bm_params.ecc = ecc      # eccentricity
    bm_params.w = omega    # longitude of periastron (in degrees)
    bm_params.limb_dark = ldtype   # limb darkening model

    if ldtype == 'uniform':
        bm_params.u = []  # limb darkening coefficients

    elif ldtype == 'linear':
        bm_params.u = [u1]  # limb darkening coefficients

    elif ldtype == 'quadratic':
        bm_params.u = [u1, u2]  # limb darkening coefficients
    else:
        raise ValueError(
            "`ldtype` can only be ['uniform', 'linear', 'quadratic']")

    bm_params.t_secondary = ecenter
    bm_params.fp = fpfs

    m_eclipse = batman.TransitModel(
        bm_params,
        times,
        transittype=transit_type
    )

    return m_eclipse.light_curve(bm_params)


def inject_eclipse(inj_fpfs, times, fluxes, planet_params, verbose=False):
    ppm = 1e6

    print(f'Injecting Model with FpFs: {inj_fpfs * ppm}ppm')

    # Inject a signal if `inj_fpfs` is provided
    inj_model = batman_wrapper(
        times,
        period=planet_params['period'],
        tcenter=planet_params['tcenter'],
        inc=planet_params['inc'],
        aprs=planet_params['aprs'],
        rprs=planet_params['rprs'],
        ecc=planet_params['ecc'],
        omega=planet_params['omega'],
        u1=planet_params['u1'],
        u2=planet_params['u2'],
        offset=planet_params['offset'],
        slope=planet_params['slope'],
        curvature=planet_params['curvature'],
        ecenter=planet_params['ecenter'],
        fpfs=inj_fpfs,
        ldtype=planet_params['ldtype'],  # ='uniform',
        transit_type=planet_params['transit_type'],  # ='secondary',
        verbose=verbose
    )

    return fluxes * inj_model


def batman_plotting_wrapper(spitzer_analysis, fpfs=0, delta_ecenter=0):
    return batman_wrapper(
        times=spitzer_analysis.tso_data.times,
        period=spitzer_analysis.period,
        tcenter=spitzer_analysis.tcenter,
        ecenter=spitzer_analysis.ecenter0+delta_ecenter,
        inc=spitzer_analysis.inc,
        aprs=spitzer_analysis.aprs,
        rprs=spitzer_analysis.rprs,
        fpfs=fpfs,
        ecc=spitzer_analysis.ecc,
        omega=spitzer_analysis.omega,
        u1=spitzer_analysis.u1,
        u2=spitzer_analysis.u2,
        offset=spitzer_analysis.offset,
        slope=spitzer_analysis.slope,
        curvature=spitzer_analysis.curvature,
        ldtype=spitzer_analysis.ldtype,
        transit_type=spitzer_analysis.transit_type
    )


def linear_model(times, offset, slope):
    times = times.copy() - times.mean()
    return offset + times * slope


def load_from_df(df, aper_key=None, centering_key=None):
    if aper_key is None:
        aper_key = 'rad_2p5_0p0'

    if centering_key is None:
        centering_key = 'fluxweighted'

    times = df.bmjd_adjstd.values
    fluxes = df[f'flux_{aper_key}'].values
    flux_errs = df[f'noise_{aper_key}'].values
    aornums = df['aornum'].values
    ycenters = df[f'{centering_key}_ycenters'].values
    xcenters = df[f'{centering_key}_xcenters'].values
    npix = df.effective_widths

    # Confirm npix is a DataFrame, Series, or NDArray
    npix = npix.values if hasattr(df.effective_widths, 'values') else npix

    return ExoplanetTSOData(
        times=times,
        fluxes=fluxes,
        flux_errs=flux_errs,
        aornums=aornums,
        ycenters=ycenters,
        xcenters=xcenters,
        npix=npix
    )


def load_from_wanderer(
        wanderer_inst=None, planet_name=None, channel=None, aor_dir=None,
        aper_key=None, centering_key=None):

    if not HAS_WANDERER:
        raise ImportError(
            'please install wanderer via '
            'https://github.com/exowanderer/wanderer'
        )

    if wanderer_inst is None:
        assert (None not in [planet_name, channel, aor_dor]), (
            'please provide either an instance of wanderer or the inputs '
            'required to create one: (`planet_name`, `channel`, `aor_dor`)'
        )

        wanderer_inst, _ = load_wanderer_instance_from_file(
            planet_name=planet_name,
            channel=channel,
            aor_dir=aor_dir,
            check_defaults=False,
            shell=False
        )

    if aper_key is not None:
        aper_key = 'gaussian_fit_annular_mask_rad_2.5_0.0'

    times = wanderer_inst.time_cube
    fluxes = wanderer_inst.flux_tso_df[aper_key].values
    flux_errs = wanderer_inst.noise_tso_df[aper_key].values
    aornums = np.array([aor_dir] * times.size)
    ycenters = wanderer_inst.centering_df[f'{centering_key}_ycenters'].values
    xcenters = wanderer_inst.centering_df[f'{centering_key}_xcenters'].values
    npix = wanderer_inst.effective_widths

    return ExoplanetTSOData(
        times=times,
        fluxes=fluxes,
        flux_errs=flux_errs,
        aornums=aornums,
        ycenters=ycenters,
        xcenters=xcenters,
        npix=npix
    )


def plot_ultranest_trace(sampler, suptitle=None):
    """Make trace plot. From Ultranest
    Write parameter trace diagnostic plots to plots/ directory
    if log directory specified, otherwise show interactively.
    This does essentially::
        from ultranest.plot import traceplot
        traceplot(results=results, labels=paramnames + derivedparamnames)
    """
    if not HAS_ULTRANEST:
        raise ImportError(
            'please install `ultranest` via '
            'https://github.com/JohannesBuchner/UltraNest'
        )

    if sampler.log:
        sampler.logger.debug('Making trace plot ... ')

    paramnames = sampler.paramnames + sampler.derivedparamnames
    # get dynesty-compatible sequences
    fig, axes = traceplot(
        results=sampler.run_sequence,
        labels=paramnames,
        show_titles=True
    )
    if suptitle is not None:
        fig.suptitle(suptitle)

    if sampler.log_to_disk:
        plt.savefig(os.path.join(
            sampler.logs['plots'], 'trace.pdf'), bbox_inches='tight')
        plt.close()
        sampler.logger.debug('Making trace plot ... done')

# sourcery skip: extract-duplicate-method
# sourcery skip: extract-duplicate-method
# sourcery skip: extract-duplicate-method


def visualise_ultranest_traces_corner(spitzer_analysis, suptitle=None):
    if not HAS_ULTRANEST:
        raise ImportError(
            'please install `ultranest` via '
            'https://github.com/JohannesBuchner/UltraNest'
        )

    # Compute Estimators
    spitzer_analysis.sampler.print_results()

    # Plot Traces and Distributions
    # spitzer_analysis.sampler.plot_trace()
    plot_ultranest_trace(spitzer_analysis.sampler, suptitle=suptitle)

    cornerplot(spitzer_analysis.ultranest_results)

    plt.show()


def visualise_ultranest_samples(spitzer_analysis):
    if not HAS_ULTRANEST:
        raise ImportError(
            'please install `ultranest` via '
            'https://github.com/JohannesBuchner/UltraNest'
        )

    # Save White Space Below
    times = spitzer_analysis.tso_data.times
    fluxes = spitzer_analysis.tso_data.fluxes
    yerr = spitzer_analysis.tso_data.flux_errs
    sampler = spitzer_analysis.sampler

    # labels = ["fpfs", "delta-ecenter", "log(f)"]
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.errorbar(
        x=times,
        y=fluxes,
        yerr=yerr,
        marker='.',
        ls=' ',
        color='orange',
        alpha=0.5
    )

    # t_grid = np.linspace(times.min(), times.max(), 400)

    band = PredictionBand(times)

    # go through the solutions
    for fpfs_, delta_ec_, logf_ in sampler.results['samples']:
        ecenter_ = spitzer_analysis.ecenter0 + delta_ec_

        # compute for each time the y value
        band.add(
            spitzer_analysis.batman_wrapper(
                update_fpfs=fpfs_,
                update_ecenter=ecenter_
            )
        )

    band.line(color='k')

    # add 1 sigma quantile
    band.shade(color='k', alpha=0.3)

    # add wider quantile (0.01 .. 0.99)
    band.shade(q=0.49, color='gray', alpha=0.2)


def visualise_mle_solution(spitzer_analysis):
    soln = spitzer_analysis.soln
    fpfs_ml, delta_ecenter_ml, log_f_ml, *offs_slopes = soln.x  #
    # pw_line = piecewise_linear_model(soln.x, spitzer_analysis)
    # pw_line = piecewise_offset_model(soln.x, spitzer_analysis)

    # n_epochs = 695.0
    n_epochs = (spitzer_analysis.tso_data.times.min() -
                spitzer_analysis.tcenter) / spitzer_analysis.period
    n_epochs = np.ceil(n_epochs)

    n_shifts = n_epochs*spitzer_analysis.period

    times = spitzer_analysis.tso_data.times
    fluxes = spitzer_analysis.tso_data.fluxes
    flux_errs = spitzer_analysis.tso_data.flux_errs
    zorder = 3*times.size+1

    ecenter_ml = spitzer_analysis.ecenter0 + n_shifts + delta_ecenter_ml
    plt.errorbar(times, fluxes, yerr=flux_errs, fmt=".k", capsize=0)
    plt.plot(
        times,  # x0
        batman_plotting_wrapper(
            spitzer_analysis,
            fpfs_ml,
            delta_ecenter_ml,
        ),
        "-",
        color="orange",
        label="ML",
        lw=3,
        zorder=zorder
    )
    """
    plt.plot(
        times,
        pw_line,
        "-",
        color="violet",
        label="Piecewiese Linear",
        lw=3,
        zorder=zorder
    )
    """

    plt.axvline(
        ecenter_ml,
        color='green',
        linewidth=1,
        alpha=1,
        zorder=3*times.size+2,
        label='Eclipse Center'
    )

    plt.legend(fontsize=14)
    # plt.xlim(0, 10)
    plt.xlabel("times")
    plt.ylabel("fluxes")
    plt.show()


def plot_phase_data_by_aor(df, planet_name, init_fpfs=None):

    planet_info = exoMAST_API(planet_name=planet_name)

    init_period = planet_info.orbital_period
    init_tcenter = planet_info.transit_time
    init_aprs = planet_info.a_Rs
    init_inc = planet_info.inclination

    init_tdepth = planet_info.transit_depth
    init_rprs = np.sqrt(init_tdepth)
    init_ecc = planet_info.eccentricity
    init_omega = planet_info.omega

    if init_fpfs is None:
        ppm = 1e6
        init_fpfs = 265 / ppm

    init_ecenter = init_tcenter + init_period * 0.5

    init_u1 = 0
    init_u2 = 0
    init_offset = 1.0
    init_slope = 1e-10
    init_curvature = 1e-10

    ldtype = 'uniform'
    transit_type = 'secondary'
    verbose = False

    init_model = batman_wrapper(
        df.bmjd_adjstd.values,
        init_period,
        init_tcenter,
        init_inc,
        init_aprs,
        init_rprs,
        init_ecc,
        init_omega,
        init_u1,
        init_u2,
        init_offset,
        init_slope,
        init_curvature,
        ecenter=init_ecenter,
        fpfs=init_fpfs,
        ldtype=ldtype,
        transit_type=transit_type,
        verbose=verbose
    )

    phased = (df.bmjd_adjstd - init_tcenter) % init_period / init_period

    for k, aor_ in enumerate(df.aornum.unique()):
        df_aor_ = df.query(f'aornum == "{aor_}"')
        times_ = df_aor_.bmjd_adjstd
        phase_aor_ = (times_ - init_tcenter) % init_period / init_period
        plt.plot(phase_aor_, df_aor_.flux_rad_2p0_0p0+k*0.03, '.', label=aor_)
        plt.annotate(aor_, [phase_aor_.max()+0.005, 1+k*0.03])

    plt.tight_layout()

    plt.plot(phased, init_model, 'k.')

    plt.show()


def phase_bin_data(
        df, planet_name, n_phases=1000, min_phase=0.4596, max_phase=0.5942,
        keep_end=False):

    planet_info = exoMAST_API(planet_name=planet_name)
    init_period = planet_info.orbital_period
    init_tcenter = planet_info.transit_time

    phased = (df.bmjd_adjstd - init_tcenter) % init_period / init_period

    phase_bins = np.linspace(min_phase, max_phase, n_phases)

    phase_binned_flux = np.zeros_like(phase_bins)
    phase_binned_ferr = np.zeros_like(phase_bins)
    for k, (phstart, phend) in enumerate(zip(phase_bins[:-1], phase_bins[1:])):
        in_phase = (phstart >= phased.values)*(phased.values <= phend)
        flux_in_phase = df.flux_rad_2p5_0p0.values[in_phase]
        phase_binned_flux[k] = np.median(flux_in_phase)
        phase_binned_ferr[k] = np.std(flux_in_phase)

    if keep_end:
        in_phase = phased.values >= phend
        flux_in_phase = df.flux_rad_2p5_0p0.values[in_phase]
        phase_binned_flux[-1] = np.median(flux_in_phase)
        phase_binned_ferr[-1] = np.std(flux_in_phase)
    else:
        phase_bins = phase_bins[:-1]
        phase_binned_flux = phase_binned_flux[:-1]
        phase_binned_ferr = phase_binned_ferr[:-1]

    return phase_bins, phase_binned_flux, phase_binned_ferr


def trapezoid_transit(time, f, df, p, tt, tf=None, off=0, square=False):
    """
    Flux, from a uniform star source with single orbiting planet, as a function of time
    :param time: 1D array, input times
    :param f: unobscured flux, max flux level
    :param df: ratio of obscured to unobscured flux
    :param p: period of planet's orbit
    :param tt: total time of transit
    :param tf: time during transit in which flux doesn't change
    :param off: time offset. A value of 0 means the transit begins immediately
    :param square: If True, the shape of the transit will be square (tt == tf)
    :return: 1D array, flux from the star
    """
    if tf is None:
        tf = tt
    if tt <= tf:
        # Default to square shaped transit
        square = True

    y = []
    if not square:
        # define slope of sides of trapezium
        h = f*df*tt/(tt-tf)
        grad = 2*h/tt

    for i in time:
        j = (i + off) % p
        if j < tt:
            # transit
            # square shaped transit
            if square:
                y.append(f*(1 - df))

            # trapezium shaped transit
            elif j/tt < 0.5:
                # first half of transit
                val = f - grad*j
                if val < f*(1 - df):
                    y.append(f*(1 - df))
                else:
                    y.append(val)
            else:
                # last half of transit
                val = (grad*j) - 2*h + f
                if val < f*(1 - df):
                    y.append(f*(1 - df))
                else:
                    y.append(val)
        else:
            # no transit
            y.append(f)
    return y


def loglike_trapezoid(theta, fluxes, times):
    """
    Function to return the log likelihood of the trapezium shpaed transit light curve model
    :param theta: tuple or list containing each parameter
    :param fluxes: list or array containing the observed flux of each data point
    :param times: list or array containing the times at which each data point is recorded
    """
    # unpack parameters
    f_like, df_like, p_like, tt_like, tf_like, off_like = theta
    # expected value
    lmbda = np.array(
        trapezoid_transit(
            times, f_like, df_like, p_like, tt_like, tf_like, off=off_like
        )
    )

    # n = len(fluxes)
    a = np.sum(gammaln(np.array(fluxes)+1))
    b = np.sum(np.array(fluxes) * np.log(lmbda))

    return -np.sum(lmbda) - a + b


def prior_transform_trapezoid(theta, priors):
    """
    Transforms parameters from a unit hypercube space to their true space
    for a trapezium transit model
    """
    params = [0 for _ in range(len(theta))]
    for i in range(len(theta)):
        if i == 0:
            # uniform transform for f
            params[i] = (priors[i][1]-priors[i][0])*theta[i] + priors[i][0]
        else:
            # normal transform for remaining parameters
            params[i] = priors[i][0] + priors[i][1]*ndtri(theta[i])

    return np.array(params)


def get_prior_trapezoid():
    # uniform prior on flux
    f_min = 4.9
    f_max = 5.8

    # normal prior on flux drop
    df_mu = 0.19
    df_sig = 0.005

    # normal prior on period
    p_mu = 0.8372
    p_sig = 0.008

    # normal prior on total transit time
    tt_mu = 0.145
    tt_sig = 0.01

    # normal prior on flat transit time
    tf_mu = 0.143
    tf_sig = 0.01

    # normal prior on offset
    off_mu = 0.1502
    off_sig = 0.0008

    return [
        (f_min, f_max),
        (df_mu, df_sig),
        (p_mu, p_sig),
        (tt_mu, tt_sig),
        (tf_mu, tf_sig),
        (off_mu, off_sig)
    ]

    # remove tf for square transit parameters
    # priors_square = priors[:4] + priors[5:]

    # return priors_trapezo


# From Emcee
def add_piecewise_linear(init_params, n_lines=0, add_slope=False):
    # Add a offset and slope for each AOR
    offset_default = 1.0
    slope_default = 0.0
    # for _ in np.sort(tso_data.aornums.unique()):
    for k in range(n_lines):
        offset_ = np.random.normal(offset_default, 1e-4)
        if add_slope:
            slope_ = np.random.normal(slope_default, 1e-5)
            init_params[f'offset{k}'] = offset_
            init_params[f'slope{k}'] = slope_
        else:
            init_params[f'offset{k}'] = offset_

    return init_params


def print_mle_results(soln_x, ecenter0=0):

    fpfs_ml, delta_ecenter_ml, log_f_ml = soln_x[:3]
    if len(soln_x) >= 5:
        sigma_w, sigma_r = soln_x[3:5]
        if len(soln_x) == 6:
            gamma = soln_x[6]

    print(
        f'fpfs_ml={fpfs_ml*1e6:.3f} ppm\n'
        f'delta_ecenter_ml={delta_ecenter_ml:.5f}'
    )
    if ecenter0 > 0:
        print(
            f'ecenter_ml={ecenter0 + delta_ecenter_ml:.3f}'
        )

    print(
        f'log_f_ml={log_f_ml:.3f}'
    )

    if len(soln_x) >= 5:
        print(
            f'sigma_w={sigma_w:.3f}\n'
            f'sigma_r={sigma_r:.3f}'
        )
        if len(soln_x) == 6:
            print(
                f'gamma={gamma:.3f}\n'
            )


def get_labels(ndim=None):

    if ndim is None:
        return ['fpfs', 'delta_center', 'log_f', 'sigma_w', 'sigma_r', 'gamma']

    labels = ["fpfs", "delta_ecenter", "log(f)"]  #
    if ndim >= 5:
        labels.extend(['sigma_w', 'sigma_r'])

    if ndim == 6:
        labels.append('gamma')

    return labels


def trace_plot(sampler):
    # Compute Estimators
    samples = sampler.get_chain()
    n_chain_samples, nwalkers, ndim = samples.shape
    if n_chain_samples < 1000:
        burnin = 0

    samples = samples.copy()
    labels = get_labels(ndim)

    print(labels)
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)

    # for i in range(ndim):
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")


def flatten_chains(sampler, discard=100, thin=15, burnin=0.2):
    n_chain_samples, _, _ = sampler.get_chain().shape
    if n_chain_samples < 1000:
        burnin = 0
        discard = 0
        thin = 1

    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    n_flat_samples, _ = flat_samples.shape

    n_flat_burnin = int(burnin * n_flat_samples)
    return flat_samples.copy()[n_flat_burnin:]


def print_emcee_results(sampler, discard=100, thin=15, burnin=0.2):
    n_chain_samples, _, ndim = sampler.get_chain().shape
    if n_chain_samples < 1000:
        burnin = 0
        discard = 0
        thin = 1

    flat_samples = flatten_chains(
        sampler,
        discard=discard,
        thin=thin,
        burnin=burnin
    )

    labels = get_labels(ndim)

    for i in range(len(labels)):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        if labels[i] == 'fpfs':
            mcmc[1] = mcmc[1]*1e6
            q = q * 1e6

        txt = f"{labels[i]} = {mcmc[1]:.4f}_-{q[0]:.4f}^{q[1]:.4f}"
        print(txt)


def get_truth_emcee_values(sampler, discard=100, thin=15, burnin=0.2):
    n_chain_samples, _, ndim = sampler.get_chain().shape
    if n_chain_samples < 1000:
        burnin = 0
        discard = 0
        thin = 1

    flat_samples = flatten_chains(
        sampler,
        discard=discard,
        thin=thin,
        burnin=burnin
    )

    # TODO: Add upper and lower quantils into Corner plot labels
    perctiles = [16, 50, 84]
    _, fpfs_mcmc, _ = np.percentile(flat_samples[:, 0], perctiles)
    _, delta_ecenter_mcmc, _ = np.percentile(flat_samples[:, 1], perctiles)
    _, log_f_mcmc, _ = np.percentile(flat_samples[:, 2], perctiles)

    so_called_truth = [fpfs_mcmc*1e6, delta_ecenter_mcmc, log_f_mcmc]  #

    if ndim >= 5:
        _, sigma_w_mcmc, _ = np.percentile(flat_samples[:, 3], perctiles)
        _, sigma_r_mcmc, _ = np.percentile(flat_samples[:, 4], perctiles)

        so_called_truth.extend([sigma_w_mcmc, sigma_r_mcmc])

    if ndim == 6:
        _, gamma_mcmc, _ = np.percentile(flat_samples[:, 5], perctiles)
        so_called_truth.append(gamma_mcmc)

    return dict(zip(get_labels(), so_called_truth))


def print_ultranest_results(sampler, discard=100, thin=15, burnin=0.2):
    raise NotImplementedError('`print_ultranest_results` is not implemented')
    n_chain_samples, _, ndim = sampler.get_chain().shape
    if n_chain_samples < 1000:
        burnin = 0
        discard = 0
        thin = 1

    flat_samples = flatten_chains(
        sampler,
        discard=discard,
        thin=thin,
        burnin=burnin
    )

    labels = get_labels(ndim)

    for i in range(len(labels)):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        if labels[i] == 'fpfs':
            mcmc[1] = mcmc[1]*1e6
            q = q * 1e6

        txt = f"{labels[i]} = {mcmc[1]:.4f}_-{q[0]:.4f}^{q[1]:.4f}"
        print(txt)


def get_truth_ultranest_values(sampler, discard=100, thin=15, burnin=0.2):
    raise NotImplementedError(
        '`get_truth_ultranest_values` is not implemented'
    )
    n_chain_samples, _, ndim = sampler.get_chain().shape
    if n_chain_samples < 1000:
        burnin = 0
        discard = 0
        thin = 1

    flat_samples = flatten_chains(
        sampler,
        discard=discard,
        thin=thin,
        burnin=burnin
    )

    # TODO: Add upper and lower quantils into Corner plot labels
    perctiles = [16, 50, 84]
    _, fpfs_mcmc, _ = np.percentile(flat_samples[:, 0], perctiles)
    _, delta_ecenter_mcmc, _ = np.percentile(flat_samples[:, 1], perctiles)
    _, log_f_mcmc, _ = np.percentile(flat_samples[:, 2], perctiles)

    so_called_truth = [fpfs_mcmc*1e6, delta_ecenter_mcmc, log_f_mcmc]  #

    if ndim >= 5:
        _, sigma_w_mcmc, _ = np.percentile(flat_samples[:, 3], perctiles)
        _, sigma_r_mcmc, _ = np.percentile(flat_samples[:, 4], perctiles)

        so_called_truth.extend([sigma_w_mcmc, sigma_r_mcmc])

    if ndim == 6:
        _, gamma_mcmc, _ = np.percentile(flat_samples[:, 5], perctiles)
        so_called_truth.append(gamma_mcmc)

    return dict(zip(get_labels(), so_called_truth))

# sourcery skip: extract-duplicate-method
# sourcery skip: extract-duplicate-method
# sourcery skip: extract-duplicate-method


def visualise_emcee_traces_corner(
        spitzer_analysis, discard=100, thin=15, burnin=0.2, verbose=False):

    # Save White Space Below
    sampler = spitzer_analysis.sampler

    n_chain_samples, _, ndim = sampler.get_chain().shape
    if n_chain_samples < 1000:
        burnin = 0
        discard = 0
        thin = 1

    try:
        tau = sampler.get_autocorr_time()
        print(f'tau: {tau}')
    except Exception as err:
        print(err)

    print_emcee_results(
        sampler,
        discard=discard,
        thin=thin,
        burnin=burnin
    )

    trace_plot(sampler)
    # Display Corner Plots
    so_called_truth = get_truth_emcee_values(
        sampler,
        discard=discard,
        thin=thin,
        burnin=burnin
    )

    ppm = 1e6

    if verbose:
        for key, val in so_called_truth.items():
            # val = val * ppm if key == 'fpfs' else val
            print(f'{key}: {val}')

    labels = get_labels(ndim)

    flat_samples = flatten_chains(
        sampler,
        discard=discard,
        thin=thin,
        burnin=burnin
    )

    truths = list(so_called_truth.values())
    flat_samples[:, 0] = flat_samples[:, 0] * ppm
    fig = corner.corner(
        flat_samples[:, :len(labels)],
        labels=labels,
        truths=truths,
        show_titles=True
    )

    plt.show()


def visualise_emcee_samples(
        spitzer_analysis, discard=100, thin=15, burnin=0.2, verbose=False):

    # Save White Space Below
    times = spitzer_analysis.tso_data.times
    fluxes = spitzer_analysis.tso_data.fluxes
    yerr = spitzer_analysis.tso_data.flux_errs
    sampler = spitzer_analysis.sampler

    n_chain_samples, _, ndim = sampler.get_chain().shape
    if n_chain_samples < 1000:
        print('Too few samples to thin')
        burnin = 0
        discard = 0
        thin = 1

    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    n_flat_samples, _ = flat_samples.shape

    n_flat_burnin = int(burnin * n_flat_samples)
    n_chain_burning = int(burnin * n_chain_samples)

    # Convert from decimal to ppm
    flat_samples = flat_samples.copy()[n_flat_burnin:]

    # TODO: Add upper and lower quantils into Corner plot labels
    perctiles = [16, 50, 84]
    _, fpfs_mcmc, _ = np.percentile(flat_samples[:, 0], perctiles)
    _, delta_ecenter_mcmc, _ = np.percentile(flat_samples[:, 1], perctiles)
    _, log_f_mcmc, _ = np.percentile(flat_samples[:, 2], perctiles)

    if ndim >= 5:
        _, sigma_w_mcmc, _ = np.percentile(flat_samples[:, 3], perctiles)
        _, sigma_r_mcmc, _ = np.percentile(flat_samples[:, 4], perctiles)

    if ndim == 6:
        _, gamma_mcmc, _ = np.percentile(flat_samples[:, 5], perctiles)

    """
    zorder = 3*times.size+1
    pw_line = piecewise_linear_model(soln.x, spitzer_analysis)
    plt.plot(
        times,
        pw_line,
        "-",
        color="violet",
        label="Piecewiese Linear",
        lw=3,
        zorder=zorder
    )
    """

    tcenter = spitzer_analysis.tcenter
    period = spitzer_analysis.period

    n_epochs = (times.min() - tcenter) / period
    n_epochs = np.ceil(n_epochs)
    n_shifts = n_epochs*spitzer_analysis.period

    labels = ["fpfs", "delta_ecenter", "log(f)"]  # "ecenter",
    """
    # Display Distribution of Results Plots
    inds = np.random.randint(len(flat_samples[:, :len(labels)]), size=100)

    for ind in inds:
        fpfs_, _ = flat_samples[ind, :len(labels)]  # delta_ecenter_,
        plt.plot(
            times,  # x0
            batman_plotting_wrapper(
                spitzer_analysis,
                fpfs_,
                delta_ecenter_,
            ),
            "C1",
            alpha=0.1,
            # label="MCMC Estimator",
            lw=3,
            zorder=3*times.size+1
        )

    # n_epochs = 695.0
    ecenter_mcmc = spitzer_analysis.ecenter0 + n_shifts  # + delta_ecenter_mcmc
    plt.errorbar(times, fluxes, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(
        times,  # x0
        batman_plotting_wrapper(
            spitzer_analysis,
            fpfs_mcmc,
            delta_ecenter_,
        ),
        "-",
        color="orange",
        label="MCMC Estimator",
        lw=3,
        zorder=3*times.size+1
    )
    plt.axvline(
        ecenter_mcmc,
        color='violet',
        linewidth=1,
        alpha=1,
        zorder=3*times.size+2
    )

    plt.legend(fontsize=14)
    # plt.xlim(0, 10)
    plt.xlabel("times")
    plt.ylabel("fluxes")
    plt.show()
    """
    # Display Distribution of Results Plots
    inds = np.random.randint(len(flat_samples[:, :len(labels)]), size=100)
    for ind in inds:
        fpfs_, delta_ecenter_ = flat_samples[ind, :len(labels)]
        ecenter_ = spitzer_analysis.ecenter0 + n_shifts + delta_ecenter_
        plt.plot(
            times,  # x0
            batman_plotting_wrapper(
                spitzer_analysis,
                fpfs_,
                delta_ecenter_,
            ),
            "C1",
            alpha=0.1,
            # label="MCMC Estimator",
            lw=3,
            zorder=3*times.size+1
        )
        plt.axvline(
            ecenter_,
            color='violet',
            linewidth=1,
            alpha=0.1,
            zorder=3*times.size+2
        )

    ecenter_mcmc = spitzer_analysis.ecenter0 + n_shifts + delta_ecenter_mcmc
    plt.errorbar(times, fluxes, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(
        times,  # x0
        batman_plotting_wrapper(
            spitzer_analysis,
            fpfs_mcmc,
            delta_ecenter_mcmc,
        ),
        "-",
        color="orange",
        label="MCMC Estimator",
        lw=3,
        zorder=3*times.size+1
    )

    plt.axvline(
        ecenter_mcmc,
        color='violet',
        linewidth=1,
        alpha=1,
        zorder=3*times.size+2
    )

    plt.legend(fontsize=14)
    # plt.xlim(0, 10)
    plt.xlabel("times")
    plt.ylabel("fluxes")
    plt.show()
