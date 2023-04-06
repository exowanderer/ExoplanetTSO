import batman
import emcee
import joblib
import numpy as np
import os

import ultranest
import ultranest.stepsampler as stepsampler

try:
    import pinknoise
    HAS_PINKNOISE = True
except ImportError:
    HAS_PINKNOISE = False

from datetime import datetime, timezone
from exomast_api import exoMAST_API
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from statsmodels.robust import scale
from time import time
from tqdm import tqdm

from skywalker import krdata
from skywalker.utils import configure_krdata_noise_models

from .models import (
    ExoplanetTSOData,
    KRDataInputs,
    exoMastParams
)

from .utils import (
    bin_df_time,
    bin_array_time,
    get_truth_emcee_values,
    get_truth_ultranest_values,
    linear_model,
    load_from_df,
    load_from_wanderer,
    print_emcee_results,
    print_ultranest_results,
    trim_initial_timeseries,
    visualise_emcee_samples,
    visualise_emcee_traces_corner,
    visualise_ultranest_traces_corner,
    visualise_ultranest_samples,
    visualise_mle_solution,
)


class ExoplanetTSO:
    __all__ = [
        'df',
        'aor_dir',
        'channel',
        'planet_name',
        'mast_name',
        'centering_key',
        'aper_key',
        'estimate_pinknoise',
        'n_piecewise_params',
        'init_fpfs',
        'savenow',
        'visualise_mle',
        'trim_size',
        'timebinsize',
        'centering_key',
        'aper_key',
        'n_sig',
        'standardise_fluxes',
        'standardise_times',
        'standardise_centers',
        'verbose'
    ]

    def __init__(
            self, df=None, aor_dir=None, channel=None, planet_name=None,
            trim_size=0, timebinsize=0, mast_name=None,
            estimate_pinknoise=False, centering_key=None,
            aper_key=None, init_fpfs=None, n_piecewise_params=0, n_sig=5,
            savenow=False, visualise_mle=False, standardise_fluxes=False,
            standardise_times=False, standardise_centers=False, verbose=False):
        """Base Exoplanet Time Series Observations object for analysing time
            series observations with transiting exoplanet modeling.

        Args:
            df (pd.DataFrame, optional): base dateframe with times, flux,
                errors, and other vectors (e.g., xcenters, ycenters) useful for
                noise modeling. Defaults to None.
            aor_dir (str, optional): location of processed data to load. Only
                used if loading data from the output of the `wanderer` package.
                Defaults to None.
            channel (str, optional): Spitzer channel directory of stored data.
                Only used if loading data from the output of the `wanderer`
                package. Defaults to None.
            planet_name (str, optional): name of planet for load and save
                functions. Defaults to None.
            trim_size (float, optional): days to trim from start of each AOR.
                Defaults to 0.
            timebinsize (float, optional): width in days (or `times` similar
                value).  Defaults to 0.
            mast_name (str, optional): name of planet in MAST registry.
                Defaults to None.
            estimate_pinknoise (bool, optional): toggle for whether to include
                Carter&Winn2010 wavelet likelihood (`pinknoise`) likelihood.
                Defaults to False.
            centering_key (str, optional): Base for the column name of center
                values: `fluxweighted` or `gaussian_fit`. Defaults to None.
            aper_key (str, optional): Base for column name for flux values.
                Defaults to None.
            init_fpfs (float, optional): Initial guess for the eclipse depth
                (fpfs). Defaults to None.
            n_piecewise_params (int, optional): whether to include an offset
                (=1) or slope (=2) per AOR with multi AOR observations.
                Defaults to 0.
            n_sig (int, optional): Number of sigma for Gaussian thresholding.
                Defaults to 5.
            savenow (bool, optional): Toggle whether to activate the save to
                joblib subroutines. Defaults to False.. Defaults to False.
            visualise_mle (bool, optional): Toggle whether to activate the MLE
                plotting subroutine. Defaults to False.
            standardise_fluxes (bool, optional): Toggle whether to
                gaussian-filter the flux values to remove n_sig outliers.
                Defaults to False.
            standardise_times (bool, optional): Toggle whether to
                median-center the time values to reduce complexity with
                optimsations.  Defaults to False.
            standardise_centers (bool, optional): Toggle whether to
                median-center the centering values to reduce complexity with
                optimsations.  Defaults to False.
            verbose (bool, optional): Toggle whether to activate excessing
                print to stdout. Defaults to False.
        """
        self.df = df
        self.aor_dir = aor_dir
        self.channel = channel
        self.planet_name = planet_name
        self.mast_name = mast_name
        self.centering_key = centering_key
        self.aper_key = aper_key
        self.estimate_pinknoise = estimate_pinknoise and HAS_PINKNOISE
        self.n_piecewise_params = n_piecewise_params
        self.init_fpfs = init_fpfs
        self.savenow = savenow
        self.visualise_mle = visualise_mle
        self.trim_size = trim_size
        self.timebinsize = timebinsize
        self.centering_key = centering_key
        self.aper_key = aper_key
        self.n_sig = n_sig
        self.standardise_fluxes = standardise_fluxes
        self.standardise_times = standardise_times
        self.standardise_centers = standardise_centers
        self.verbose = verbose

        if self.mast_name is None:
            self.mast_name = self.planet_name

    def initialise_data_and_params(self):
        """ Run 6 subroutines to initialize the parameters and data structures

            Routines:
                1. self.preprocess_pipeline
                    Create TSO data structures:
                        (flux, time, err, xcenter, ycenter)
                2. self.configure_krdata: Spitzer Noise model
                3. self.configure_planet_info: exoMAST planet information
                4. self.initialise_fit_params: MLE initial fit parameters
                5. self.initialise_bm_params: Batman parmams for transit model
                6. self.configure_pinknoise_model: Carter & Winn 2010 wavelet
                    likelihood model configuration: padding, initial params
        """
        # create self.tso_data
        self.preprocess_pipeline()

        start_krdata = time()
        self.configure_krdata(ymod=0.7, n_nbr=100)
        end_krdata = time()

        if self.verbose:
            print(f'KRData Creation took {end_krdata - start_krdata} seconds')

        self.configure_planet_info()
        self.initialise_fit_params()
        self.initialise_bm_params()

        if self.estimate_pinknoise:
            self.configure_pinknoise_model()

    def initialise_bm_params(self):
        """Batman parmams for transit model

            Take planet parameters (e.g., from exoMAST) and insert them into
                the `batman.TransitParams` object
        """

        if not hasattr(self, 'period'):
            print(
                '[WARNIGN] Planetary parameters do not exist. Loading from '
                'exoMAST via `https://github.com/exowanderer/exoMAST_API`'
            )
            self.configure_planet_info()

        # object to store transit parameters
        self.bm_params = batman.TransitParams()
        self.bm_params.per = self.planet_params.period  # orbital period
        self.bm_params.t0 = self.planet_params.tcenter  # time of inferior conjunction
        self.bm_params.inc = self.planet_params.inc  # inclunaition in degrees

        # semi-major axis (in units of stellar radii)
        self.bm_params.a = self.planet_params.aprs

        # planet radius (in units of stellar radii)
        self.bm_params.rp = self.planet_params.rprs
        self.bm_params.ecc = self.planet_params.ecc  # eccentricity
        # longitude of periastron (in degrees)
        self.bm_params.w = self.planet_params.omega
        self.bm_params.limb_dark = self.planet_params.ldtype  # limb darkening model

        if self.planet_params.ecenter is None:
            self.planet_params.ecenter = self.planet_params.tcenter + \
                0.5 * self.planet_params.period

        if self.planet_params.ldtype == 'uniform':
            self.bm_params.u = []  # limb darkening coefficients

        elif self.planet_params.ldtype == 'linear':
            # limb darkening coefficients
            self.bm_params.u = [self.planet_params.u1]

        elif self.planet_params.ldtype == 'quadratic':
            # limb darkening coefficients
            self.bm_params.u = [self.planet_params.u1, self.planet_params.u2]
        else:
            raise ValueError(
                "`ldtype` can only be ['uniform', 'linear', 'quadratic']")

        self.bm_params.t_secondary = self.planet_params.ecenter
        self.bm_params.fp = self.planet_params.fpfs

    def get_wavelet_log_likelihood(self, residuals, theta):
        """Take in fit parameters and self parameters to return wavelet log
            likelihood from `pinknoise` package using Carter & Winn 2010
            wavelet likelihood model. 'https://github.com/nespinoza/pinknoise'

        Args:
            residuals (np.ndarray): flux - model residuals from fiting process
            theta (list): list of parameters sigma_w, sigma_r, gamma_w, gamma_r

        Returns:
            float: Carter & Winn 2010 Wavelet log likelihood
        """
        if len(theta[3:]) == 2:
            sigma_w, sigma_r = theta[3:]
            gamma = 1.0
        if len(theta[3:]) == 3:
            sigma_w, sigma_r, gamma = theta[3:]

        if residuals.size != self.ndata_wavelet:
            # Pad up to self.ndata_wavelet
            padding = np.zeros(self.ndata_wavelet - residuals.size)
            residuals = np.r_[residuals, padding]

        return self.wavelet_model.get_likelihood(
            residuals,
            sigma_w,
            sigma_r,
            gamma=gamma
        )

    def log_likelihood(self, params):
        """Compute the normal (and wavelet) log likelihoods

        Args:
            params (list): List of fitting parameters to be used

        Returns:
            float: log likelihood given the fitting parameters
        """
        fpfs, delta_ecenter, log_f = params[:3]  #
        # sigma_w, sigma_r = params[3:5] if self.estimate_pinknoise else (1, 1)
        # gamma = params[5] if len(params) == 6 else 1.0

        # self.planet_params.fpfs = fpfs
        # ecenter0 = self.planet_params.ecenter0
        # self.planet_params.ecenter = ecenter0 + delta_ecenter

        self.bm_params.fp = fpfs
        self.bm_params.t_secondary = self.planet_params.ecenter0 + delta_ecenter

        flux_errs = self.tso_data.flux_errs
        fluxes = self.tso_data.fluxes
        # pw_line = self.piecewise_linear_model(params)
        # pw_line = self.piecewise_offset_model(params)

        model = self.batman_krdata_wrapper()  # * pw_line
        sigma2 = flux_errs**2 + model**2 * np.exp(2 * log_f)

        residuals = fluxes - model

        wavelet_log_likelihood = 0
        if self.estimate_pinknoise and len(params) >= 5:
            # TODO: Confirm if `wavelet_log_likelihood` should be added or
            #   subtracted from `normal_log_likelihood`
            wavelet_log_likelihood = self.get_wavelet_log_likelihood(
                residuals=fluxes - model,
                params=params
            )

        normal_log_likelihood = residuals ** 2 / sigma2 + np.log(sigma2)
        normal_log_likelihood = -0.5 * np.sum(normal_log_likelihood)

        if wavelet_log_likelihood != 0 and self.verbose:
            print('normal_log_likelihood', normal_log_likelihood)
            print('wavelet_log_likelihood', wavelet_log_likelihood)

        return normal_log_likelihood + wavelet_log_likelihood

    def initialise_fit_params(
            self, init_params=None, init_logf=-5.0, spread=1e-4, seed=None):
        """Create initial transit model fitting parameters for

        Args:
            init_params (dict, optional): initial guess from external sourcing.
                Defaults to None.
            init_logf (float, optional): initial guess from external source
                range: [-inf, 1]. Defaults to -5.0.
            spread (float, optional): Width of distribution around intial 
                guesses. Defaults to 1e-4.
            seed (int, optional): random seed for np.random.seed.
                Defaults to None.
        """
        if seed is not None:
            np.random.seed(seed)

        self.batman_fittable_param = [
            'period', 'tcenter', 'ecenter', 'delta_center',
            'inc', 'aprs',
            'rprs', 'fpfs',
            'ecc', 'omega',
            'u1', 'u2',
            'offset', 'slope', 'curvature',
            'log_f', 'sigma_w', 'sigma_r', 'gamma',
        ]

        if init_params is not None and isinstance(init_params, dict):
            self.init_params = init_params
        else:  # approach with standard eclipse fitting parameters
            init_fpfs = self.planet_params.fpfs

            # Standard eclipse fitting parameters
            self.init_params = {
                'fpfs': np.random.normal(init_fpfs, 0.1*spread),
                'delta_center': np.random.normal(0.0, 10*spread),
                'log_f': np.random.normal(init_logf, 100*spread)
            }

        if self.estimate_pinknoise:
            # Activate Carter & Winn 2010 Wavelet LIkelihood Analyis
            #   Implementation provided by Nestor Espinoza
            #   https://github.com/nespinoza/pinknoise

            self.init_params['sigma_w'] = np.random.normal(0.5, spread)
            self.init_params['sigma_r'] = np.random.normal(0.5, spread)

            self.fit_gamma = False
            if self.fit_gamma:
                self.init_params['gamma'] = np.random.normal(1.1, spread)

        if 0 < self.n_piecewise_params <= 2:
            # Add a offset and slope for each AOR
            self.add_piecewise_linear(
                n_lines=len(self.aornum_list),
                add_slope=self.n_piecewise_params == 2  # 2 params
            )

        # Check that the user did not add parameters that the algorithm
        #   does not know how to operate
        for pname_ in self.init_params.keys():
            if 'offset' in pname_:
                # In case of piecewise step function
                pname_ = 'offset'
            if 'slope' in pname_:
                # In case of piecewise linear function
                pname_ = 'slope'
            if 'curvature' in pname_:
                # In case of piecewise parabolic function
                pname_ = 'curvature'

            assert (pname_ in self.batman_fittable_param), (
                f'{pname_} from `init_params` not in '
                '`self.batman_fittable_param`: \n' +
                ', '.join(self.batman_fittable_param)
            )

    def run_mle_pipeline(self, init_fpfs=None):
        """Subroutine for running Maximum Likelihood Estimation pipeline.

        Args:
            init_fpfs (float, optional): Initial guess for eclipse depth
                or FpFs. Defaults to None.
        """
        if init_fpfs is not None:
            self.init_params['fpfs'] = np.random.normal(init_fpfs, 1e-5)

        # nll = lambda *args: -self.log_ultranest_likelihood(*args)
        nlp = lambda *args: -self.log_mle_posterior(*args)

        if self.verbose:
            print('Initial Params:')
            for key, val in self.init_params.items():
                print(f'{key}: {val}')

        self.soln = minimize(nlp, list(self.init_params.values()))

        # Convert MLE soln.x list to mle_estimate dict
        self.mle_estimate = dict(zip(self.init_params.keys(), self.soln.x))

        if self.verbose:
            ppm = 1e6
            print('MLE Params:')
            for key, val in self.mle_estimate.items():
                val = val * ppm if key == 'fpfs' else val
                print(f'{key}: {val}')

        # Plot MLE Results
        # x0 = np.linspace(0, 10, 500)

    def compute_bestfit_residuals(self, fpfs, delta_ecenter):
        """ Helper function to compute the best fit and residuals
                for a given fpfs and delta_ecenter

        Args:
            fpfs (float): eclipse depth or fpfs from best fit
            delta_ecenter (float): time distance for eclipse model
                from circular orbit

        Returns:
            tuple: (residuals, transit_model, sensitivity_map)
        """
        if fpfs > 1:
            ppm = 1e6
            fpfs = fpfs / ppm

        ecenter = self.planet_params.ecenter0 + delta_ecenter

        # Compute the ultranest transit model
        transit_model = self.batman_wrapper(
            update_fpfs=fpfs,
            update_ecenter=ecenter
        )

        krdata_map = krdata.sensitivity_map(
            self.tso_data.fluxes / transit_model,
            self.krdata_inputs.ind_kdtree,
            self.krdata_inputs.gw_kdtree
        )

        spitzer_transit_model = transit_model * krdata_map
        residuals = self.tso_data.fluxes - spitzer_transit_model

        return residuals, transit_model, krdata_map

    def plot_bestfit_and_residuals(
            self, fpfs=None, delta_ecenter=None, phase_fold=False,
            nbins=None, trim=None, ylim=None, xlim=None, fig=None,
            title=None, height_ratios=None, fontsize=20, shownow=False,
            figsize=(20, 10)):
        """ Helper function to visualising best fit and residuals
                for a given fpfs and delta_ecenter

        Args:
            fpfs (float, optional): eclipse depth or fpfs from best fit.
                Defaults to None.
            delta_ecenter (float, optional): time distance for eclipse model
                from circular orbit. Defaults to None.
            phase_fold (bool, optional): Toggle if times should be wrapped into
                circular orbital time coordinates Defaults to False.
            nbins (int, optional): Number of bins to bin the data into
                Defaults to None or "no binning".
            trim (float, optional): Days to remove from the beginning of each
                AOR. Defaults to None or "no trimming".
            ylim (tuple, optional): y-limits for plot Defaults to None.
            xlim (tuple, optional): x-limits for plot Defaults to None.
            fig (plt.figure, optional): Start with existing figure.
                Defaults to None.
            title (str, optional): `suptitle` for plt.figure. Defaults to None.
            height_ratios (tuple, optional): Ratio of heights from
                `plt.subplots`. Defaults to None.
            fontsize (int, optional): Font size for all labels and text.
                Defaults to 20
            shownow (bool, optional): Toggle if plot should be shown
                immediately. Defaults to False.
            figsize (tuple, optional): Size of plt.figure from `plt.subplots`.
                Defaults to (20, 10).

        Usage:
            # Usage Case Expanding on Best Fit
            fig, (ax1, ax2) = hatp26b_krdata.plot_bestfit_and_residuals(
                fpfs=268/1e6,
                delta_ecenter=-0.788/24,
                phase_fold=True,
                nbins=100,
                fig=None,
                trim=1/24,
                ylim=[1-0.0014,1+0.0015],
                height_ratios=[0.1, 0.9],
                title=(
                    'HAT-P-26b Spitzer 8 CH2 AORs - AperRad:2.5 - '
                    '100 Bins Phase Folded'
                )
            )

            # Usage Case Expanding on Residuals
            fig, (ax1, ax2) = hatp26b_krdata.plot_bestfit_and_residuals(
                fpfs=268/1e6,
                delta_ecenter=-0.788/24,
                phase_fold=True,
                nbins=100,
                fig=None,
                trim=1/24,
                ylim=[1-0.0014,1+0.0015],
                height_ratios=[0.9, 0.1],
                title=(
                    'HAT-P-26b Spitzer 8 CH2 AORs - AperRad:2.5 - '
                    '100 Bins Phase Folded'
                )
            )
        """

        if height_ratios is None:
            height_ratios = [0.9, 0.1]

        if fpfs is None:
            raise ValueError('Please provide `fpfs`')
            # ppm = 1e6
            # fpfs = self.nested_estimates['fpfs'] / ppm

        if delta_ecenter is None:
            raise ValueError('Please provide `delta_ecenter`')
            # delta_ecenter = self.nested_estimates['delta_center']

        residuals, transit_model, krsensmap = self.compute_bestfit_residuals(
            fpfs,
            delta_ecenter
        )

        xvar = self.tso_data.times
        starflux = self.tso_data.fluxes / krsensmap
        fluxerr = self.tso_data.flux_errs

        # Sort by time
        argsort_xvar = xvar.argsort()
        starflux = starflux[argsort_xvar]
        fluxerr = fluxerr[argsort_xvar]
        krsensmap = krsensmap[argsort_xvar]
        residuals = residuals[argsort_xvar]
        transit_model = transit_model[argsort_xvar]
        xvar = xvar[argsort_xvar]

        if trim is not None and trim > 0:
            in_range = (xvar - xvar.min()) > trim
            starflux = starflux[in_range]
            fluxerr = fluxerr[in_range]
            krsensmap = krsensmap[in_range]
            residuals = residuals[in_range]
            transit_model = transit_model[in_range]
            xvar = xvar[in_range]

        if phase_fold:
            period = self.planet_params.period
            tcenter = self.planet_params.tcenter
            xvar = ((xvar - tcenter) % period) / period

            # Sort by by Time or Phase
            argsort_xvar = xvar.argsort()
            starflux = starflux[argsort_xvar]
            fluxerr = fluxerr[argsort_xvar]
            krsensmap = krsensmap[argsort_xvar]
            residuals = residuals[argsort_xvar]
            transit_model = transit_model[argsort_xvar]
            xvar = xvar[argsort_xvar]

        if nbins is not None:
            binflux, binflux_unc = bin_array_time(xvar, starflux, nbins)
            binresiduals, binres_unc = bin_array_time(xvar, residuals, nbins)
            binxvar, binxvar_unc = bin_array_time(xvar, xvar, nbins)

        if fig is None:
            fig, (ax1, ax2) = plt.subplots(
                nrows=2,
                height_ratios=height_ratios,
                sharex=True,
                figsize=figsize
            )
        else:
            ax1, ax2 = fig.get_axes()
            ax1.clear()
            ax2.clear()

        ax1.plot(xvar, starflux, '.', ms=2, alpha=0.2, label='Flux')

        if nbins is not None:
            ax1.errorbar(
                binxvar, binflux, yerr=binflux_unc, xerr=binxvar_unc,
                fmt='o', ms=5, alpha=1.0, label='Binned Flux'
            )

        ax1.plot(xvar, transit_model, '-', lw=3, label='Transit')

        ax2.plot(xvar, residuals, '.', ms=2, alpha=0.2, label='Residuals')

        if nbins is not None:
            ax2.errorbar(
                binxvar, binresiduals, binres_unc,
                fmt='o', ms=5, alpha=1.0, label='Binned Residuals'
            )

        ax2.plot(xvar, 0*xvar, '--', lw=3, label='Null')

        # ax1.legend(fontsize=fontsize)
        # ax2.legend(fontsize=fontsize)

        plt.subplots_adjust(
            left=None,
            bottom=None,
            right=None,
            top=None,
            wspace=None,
            hspace=0.0,
        )

        if ylim is not None:
            ax1.set_ylim(ylim)
            ax2.set_ylim([y_ - 1 for y_ in ylim])

        ax1.set_xlim(xlim)

        for tick in ax1.yaxis.get_ticklabels():
            tick.set_fontsize(fontsize=fontsize)
        for tick in ax2.xaxis.get_ticklabels():
            tick.set_fontsize(fontsize=fontsize)
        for tick in ax2.yaxis.get_ticklabels():
            tick.set_fontsize(fontsize=fontsize)

        if title is not None:
            fig.suptitle(title, fontsize=fontsize)

        if shownow:
            plt.show()

        plt.tight_layout()

        return fig, (ax1, ax2)

    def preprocess_pipeline(self):
        """ Pre-process the data from the df or wanderer instance

            Runs:
                1. trim_initial_timeseries
                2. bin_df_time
                3. load_from_df or load_from_wanderer
                4. flags NaNs and Outliers
                5. returns instance of ExoplanetTSOData
        """

        if self.trim_size > 0 and self.df is not None:
            self.df = trim_initial_timeseries(
                df=self.df,
                trim_size=self.trim_size,
                aornums=self.df.aornum.unique()
            )

        if self.timebinsize > 0 and self.df is not None:
            med_df, std_df = bin_df_time(self.df, timebinsize=self.timebinsize)

            # Option 1
            self.df = med_df.copy()

            """
            # Option 2
            self.df = med_df.copy()
            for colname in df.columns:
                if 'noise' in colname:
                    std_colname = colname.replace('noise', 'flux')
                    self.df[colname] = std_df[std_colname]
            """

            del med_df, std_df

        if self.df is None:
            tso_data = load_from_wanderer(
                planet_name=self.planet_name,
                channel=self.channel,
                aor_dir=self.aor_dir,
                aper_key='gaussian_fit_annular_mask_rad_2.5_0.0',
                centering_key=self.centering_key
            )
        else:
            tso_data = load_from_df(
                self.df,
                aper_key=self.aper_key,
                centering_key=self.centering_key
            )

        isfinite = np.isfinite(tso_data.times)
        isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.fluxes))
        isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.flux_errs))
        # isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.aornums))
        isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.ycenters))
        isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.xcenters))
        isfinite = np.bitwise_and(isfinite, np.isfinite(tso_data.npix))

        times = tso_data.times[isfinite]
        fluxes = tso_data.fluxes[isfinite]
        flux_errs = tso_data.flux_errs[isfinite]
        aornums = tso_data.aornums[isfinite]
        ycenters = tso_data.ycenters[isfinite]
        xcenters = tso_data.xcenters[isfinite]
        npix = tso_data.npix[isfinite]

        med_flux = np.median(fluxes)
        flux_errs = flux_errs / med_flux
        fluxes = fluxes / med_flux

        arg_times = times.argsort()
        fluxes = fluxes[arg_times]
        flux_errs = flux_errs[arg_times]
        aornums = aornums[arg_times]
        times = times[arg_times]
        ycenters = ycenters[arg_times]
        xcenters = xcenters[arg_times]
        npix = npix[arg_times]

        if self.standardise_centers:
            # Center by assuming eclipse is near center
            ycenter = (ycenter - ycenter.mean()) / ycenter.std()
            xcenter = (xcenter - xcenter.mean()) / xcenter.std()

        if self.standardise_times:
            # Center by assuming eclipse is near center
            times = times - times.mean()

        if self.standardise_fluxes:
            # Center by assuming eclipse is near center
            med_flux = np.median(fluxes)
            std_flux = scale.mad(fluxes)

            idxkeep = np.abs(fluxes - med_flux) < self.n_sig * std_flux

        self.tso_data = ExoplanetTSOData(
            times=times[idxkeep],
            fluxes=fluxes[idxkeep],
            flux_errs=flux_errs[idxkeep],
            aornums=aornums[idxkeep],
            ycenters=ycenters[idxkeep],
            xcenters=xcenters[idxkeep],
            npix=npix[idxkeep]
        )

    def print_bm_params(self):
        """ Print Batman Model Parameters
        """

        print('times', self.tso_data.times.mean())
        print('period', self.planet_params.period)
        print('tcenter', self.planet_params.tcenter)
        print('inc', self.planet_params.inc)
        print('aprs', self.planet_params.aprs)
        print('rprs', self.planet_params.rprs)
        print('ecc', self.planet_params.ecc)
        print('omega', self.planet_params.omega)
        print('u1', self.planet_params.u1)
        print('u2', self.planet_params.u2)
        print('offset', self.planet_params.offset)
        print('slope', self.planet_params.slope)
        print('curvature', self.planet_params.curvature)
        print('ldtype', self.planet_params.ldtype)
        print('transit_type', self.planet_params.transit_type)
        print('t_secondary', self.planet_params.ecenter)
        print('fp', self.planet_params.fpfs)

    def batman_wrapper(self, update_fpfs=None, update_ecenter=None):
        """ Wrapper function to use batman model with class specific data and
                static model parameters

            Args:
                update_fpfs (float, optional): eclipse depth (fpfs) with which
                    to override `bm_params.fp` Defaults to None
                update_ecenter (float, optional): eclipse center (ecenter) with
                    which to override `bm_params.t_secondary` Defaults to None

            Return:
                (np.ndarray): transit or eclipse model from Batman
        """

        if self.verbose >= 2:
            self.print_bm_params()

        if update_fpfs is not None:
            self.bm_params.fp = update_fpfs

        if update_ecenter is not None:
            self.bm_params.t_secondary = update_ecenter

        m_eclipse = batman.TransitModel(
            self.bm_params,
            self.tso_data.times,
            transittype=self.planet_params.transit_type
        )

        return m_eclipse.light_curve(self.bm_params)

    def batman_krdata_wrapper(self, update_fpfs=None, update_ecenter=None):
        """ Wrapper function to use batman model and Spitzyer KRData noise
            model with class specific data and static model parameters

            Args:
                update_fpfs (float, optional): eclipse depth (fpfs) with which
                    to override `bm_params.fp` Defaults to None
                update_ecenter (float, optional): eclipse center (ecenter) with
                    which to override `bm_params.t_secondary` Defaults to None

            Return:
                (np.ndarray): transit or eclipse model from Batman multiplied
                    by KRData Spitzer noise model
        """
        batman_curve = self.batman_wrapper(
            update_fpfs=update_fpfs,
            update_ecenter=update_ecenter
        )

        # Gaussian Kernel Regression Spitzer Sensivity map
        krdata_map = krdata.sensitivity_map(
            self.krdata_inputs.fluxes / batman_curve,  # OoT_curvature
            self.krdata_inputs.ind_kdtree,
            self.krdata_inputs.gw_kdtree
        )

        # TODO: Check is the OoT should should be multiplied or added
        return batman_curve * krdata_map  # + OoT_curvature

    def piecewise_linear_model(self, params):
        if (
                0 >= self.n_piecewise_params > 2 or  # Wrong values
                not isinstance(self.n_piecewise_params, int)  # Not an integer
        ):
            print(
                f"`piecewise_linear_model` called with "
                f"`n_piecewise_params`={self.n_piecewise_params}.\n"
                f"`n_piecewise_params` must be either 1 or 2.\n"
                f"Returning ones array like self.tso_data.times."
            )
            return np.ones_like(self.tso_data.times)

        times = self.tso_data.times.copy()
        aornums = self.tso_data.aornums.copy()
        sorted_aornums = np.sort(np.unique(aornums))

        if self.n_piecewise_params == 1:
            offsets = params[3:]
            slopes = np.zeros_like(offsets)

        if self.n_piecewise_params == 2:
            offsets = params[3::2]
            slopes = params[4::2]

        # print(offsets, slopes)
        piecewise_line = np.zeros_like(times)
        for offset_, slope_, aornum_ in zip(offsets, slopes, sorted_aornums):
            is_aornum = aornums == aornum_
            line_ = linear_model(times[is_aornum], offset_, slope_)
            piecewise_line[is_aornum] = line_
            # print(aornum_, offset_, slope_, is_aornum.sum(), line_.mean())

        return piecewise_line

    @staticmethod
    def log_mle_prior(params):
        """ Compute the log_prior probability for each parameter values

        Args:
            params (tuples): list of parameters for fitting

        Return:
            (float): Either 0.0 or -np.inf
        """

        # return 0  # Open prior
        fpfs, delta_ecenter, log_f = params[:3]  #
        # print('log_prior', fpfs, delta_ecenter, log_f)  #
        ppm = 1e6
        # TODO: Experiment with negative eclipse depths
        #   to avoid non-Gaussian posterior distributions
        ed_min = 0 / ppm  # Explore existance by permitting non-physical distributions
        ed_max = 500 / ppm  # 500 ppm
        dc_max = 0.05  # day
        dc_min = -0.05  # day

        lf_min = -10  # 1e-10 error modifier
        lf_max = 1  # 10x error modifier

        return (
            0.0 if (
                ed_min <= fpfs < ed_max and
                dc_min < delta_ecenter < dc_max and
                lf_min < log_f < lf_max
            )
            else -np.inf
        )

    def log_mle_posterior(self, params):
        """ Compute the log_posterior probability for each parameter values

        Args:
            params (tuples): list of parameters for fitting

        Return:
            (float): the log_prior + log_likelihood
        """
        lp = self.log_mle_prior(params)

        return (
            self.log_likelihood(params)  # + lp
            if np.isfinite(lp) else -np.inf
        )

    def configure_krdata(self, ymod=0.7, n_nbr=100):
        """ Configure the Spitzer KRData Noise Model cKDTree vectors

        Args:
            ymod (float): inverse of relative importance of
                ycenters vs xcenters. Lower values signify more imporance to
                ycenters than to xcenters. Defaults to 0.7
        """
        self.noise_model_config = configure_krdata_noise_models(
            self.tso_data.ycenters.copy(),
            self.tso_data.xcenters.copy(),
            self.tso_data.npix.copy(),
            ymod=ymod,
            n_nbr=n_nbr
        )

        self.krdata_inputs = KRDataInputs(
            ind_kdtree=self.noise_model_config.ind_kdtree,
            gw_kdtree=self.noise_model_config.gw_kdtree,
            fluxes=self.tso_data.fluxes
        )

    def configure_planet_info(self, init_fpfs=None, init_u1=0, init_u2=0):
        """ Load planetary parameters from exoMAST
                via `https://github.com/exowanderer/exoMAST_API`

            Args:
                init_fpfs (float, optional): Initial eclipse depth for
                    planet_params object. Defaults to None.
                init_u1 (float, optional): Initial linear limb darkening
                    coefficient for planet_params object. Defaults to 0.
                init_u2 (float, optional): Initial parabolic limb darkening
                    coefficient for planet_params object. Defaults to 0.

            Returns:
                store exoMastParams instance in self.planet_params
        """
        if init_fpfs is None:
            init_fpfs = self.init_fpfs  # or 1e-10

        if init_fpfs is None:
            ppm = 1e6
            init_fpfs = 265 / ppm

        self.planet_info = exoMAST_API(planet_name=self.mast_name)

        planet_info = self.planet_info  # save white space below

        planet_info.Rp_Rs = planet_info.Rp_Rs or None  # for later

        if not hasattr(planet_info, 'Rp_Rs') or planet_info.Rp_Rs is None:
            print('[WARNING] Rp_Rs does not exist in `planet_info`')
            print('Assuming Rp_Rs == sqrt(transit_depth)')
            planet_info.Rp_Rs = np.sqrt(planet_info.transit_depth)

        tcenter = self.planet_params.tcenter
        init_ecenter = tcenter + 0.5 * self.planet_params.period

        self.planet_params = exoMastParams(
            period=planet_info.orbital_period,
            tcenter=planet_info.transit_time,
            inc=planet_info.inclination,
            aprs=planet_info.a_Rs,
            tdepth=planet_info.transit_depth,
            rprs=planet_info.Rp_Rs,
            ecc=planet_info.eccentricity,
            omega=planet_info.omega,
            u1=init_u1,
            u2=init_u2,
            offset=1,
            slope=0,
            curvature=0,
            ldtype='uniform',
            transit_type='secondary',
            ecenter=init_ecenter,
            ecenter0=init_ecenter,
            fpfs=init_fpfs
        )

    def configure_pinknoise_model(self):
        """ Configure fit parameters for return wavelet log likelihood from
            `pinknoise` package using Carter & Winn 2010 wavelet likelihood
            model. 'https://github.com/nespinoza/pinknoise'
        """
        if not HAS_PINKNOISE:
            raise ImportError(
                'please install pinknoise via '
                'https://github.com/nespinoza/pinknoise'
            )

        ndata = len(self.tso_data.times)
        power_of_two = np.log(ndata) / np.log(2)
        if power_of_two != int(power_of_two):
            power_of_two = np.ceil(power_of_two)

        self.ndata_wavelet = 2**power_of_two

        lrg_pow2 = np.log(self.ndata_wavelet) / np.log(2)

        assert (lrg_pow2 == int(lrg_pow2)), \
            'Wavelet requires power of two data points'

        self.ndata_wavelet = int(self.ndata_wavelet)
        self.wavelet_model = pinknoise.compute(self.ndata_wavelet)


class ExoplanetUltranestTSO(ExoplanetTSO):

    def __init__(
            self, exotso_inst=None, df=None, aor_dir=None, channel=None,
            planet_name=None, trim_size=0, timebinsize=0, mast_name=None,
            n_fits=400, estimate_pinknoise=False, centering_key=None,
            aper_key=None, init_fpfs=None, n_piecewise_params=0, n_sig=5,
            process_ultranest=False, run_full_pipeline=False,
            log_dir='ultranest_savedir', savenow=False,
            visualise_mle=False, visualise_traces_corner=False,
            visualise_samples=False, standardise_fluxes=False,
            standardise_times=False, standardise_centers=False, verbose=False):
        """Ultranest (Nest Sampling) focused object for analysing
            time series observations.

        Args:
            exotso_inst (ExoplanetTSO, optional): Instance of ExoplanetTSO as
                base to ExoplanetUltranestTSO class
            df (pd.DataFrame, optional): base dateframe with times, flux,
                errors, and other vectors (e.g., xcenters, ycenters) useful for
                noise modeling. Defaults to None.
            aor_dir (str, optional): location of processed data to load. Only
                used if loading data from the output of the `wanderer` package.
                Defaults to None.
            channel (str, optional): Spitzer channel directory of stored data.
                Only used if loading data from the output of the `wanderer`
                package. Defaults to None.
            planet_name (str, optional): name of planet for load and save
                functions. Defaults to None.
            trim_size (float, optional): days to trim from start of each AOR.
                Defaults to 0.
            timebinsize (float, optional): width in days (or `times` similar
                value).  Defaults to 0.
            mast_name (str, optional): name of planet in MAST registry.
                Defaults to None.
            n_fits (int, optional): Used with ultranest plotting. How many
                samples to plot. Defaults to 400.
            estimate_pinknoise (bool, optional): toggle for whether to include
                Carter&Winn2010 wavelet likelihood (`pinknoise`) likelihood.
                Defaults to False.
            centering_key (str, optional): Base for the column name of center
                values: `fluxweighted` or `gaussian_fit`. Defaults to None.
            aper_key (str, optional): Base for column name for flux values.
                Defaults to None.
            init_fpfs (float, optional): Initial guess for the eclipse depth
                (fpfs). Defaults to None.
            n_piecewise_params (int, optional): whether to include an offset
                (=1) or slope (=2) per AOR with multi AOR observations.
                Defaults to 0.
            n_sig (int, optional): Number of sigma for Gaussian thresholding.
                Defaults to 5.
            process_ultranest (bool, optional): Toggle whether to activate the
                ultranest subroutines. Defaults to False.
            run_full_pipeline (bool, optional): Toggle whether to activate the
                full analysis pipeline. Defaults to False.
            log_dir (str, optional): Directory with which to save the ultranest
                output and log files. Defaults to 'ultranest_savedir'.
            savenow (bool, optional): Toggle whether to activate the save to
                joblib subroutines. Defaults to False.. Defaults to False.
            visualise_mle (bool, optional): Toggle whether to activate the MLE
                plotting subroutine. Defaults to False.
            visualise_traces_corner (bool, optional): Toggle whether to
                activate the UltraNest trace and corner plotting subroutine.
                Defaults to False.
            visualise_samples (bool, optional): Toggle whether to activate the
                UltraNest samples over time plotting subroutine.
                Defaults to False.
            standardise_fluxes (bool, optional): Toggle whether to
                gaussian-filter the flux values to remove n_sig outliers.
                Defaults to False.
            standardise_times (bool, optional): Toggle whether to
                median-center the time values to reduce complexity with
                optimsations.  Defaults to False.
            standardise_centers (bool, optional): Toggle whether to
                median-center the centering values to reduce complexity with
                optimsations.  Defaults to False.
            verbose (bool, optional): Toggle whether to activate excessing
                print to stdout. Defaults to False.
        """
        if exotso_inst is not None:
            kwargs = {
                key: val for key, val in exotso_inst.items()
                if key in ExoplanetTSO.__all__
            }
            super().__init__(**kwargs)
        else:
            super().__init__(
                df=df,
                aor_dir=aor_dir,
                channel=channel,
                planet_name=planet_name,
                mast_name=mast_name,
                centering_key=centering_key,
                aper_key=aper_key,
                n_fits=n_fits,
                estimate_pinknoise=estimate_pinknoise and HAS_PINKNOISE,
                n_piecewise_params=n_piecewise_params,
                init_fpfs=init_fpfs,
                savenow=savenow,
                visualise_mle=visualise_mle,
                trim_size=trim_size,
                timebinsize=timebinsize,
                n_sig=n_sig,
                standardise_fluxes=standardise_fluxes,
                standardise_times=standardise_times,
                standardise_centers=standardise_centers,
                verbose=verbose
            )

        self.process_ultranest = process_ultranest
        self.log_dir = log_dir
        self.visualise_samples = visualise_samples
        self.visualise_traces_corner = visualise_traces_corner
        self.run_full_pipeline = run_full_pipeline

        if self.run_full_pipeline:
            self.full_ultranest_pipeline()

    def full_ultranest_pipeline(self):
        """ Runs a sequence of subroutines for the ultranest pipeline

            1. self.initialise_data_and_params
            2. self.run_mle_pipeline
            3. self.run_ultranest_pipeline
            4. self.save_mle_ultranest
            5. visualise_mle_solution
            6. visualise_ultranest_traces_corner
            7. visualise_ultranest_samples
        """
        self.initialise_data_and_params()

        self.run_mle_pipeline()

        if self.process_ultranest:
            self.run_ultranest_pipeline(
                num_live_points=400,
                log_dir_extra=None
            )

        if self.savenow:
            self.save_mle_ultranest()

        if self.visualise_mle:
            visualise_mle_solution(self)

        if self.visualise_traces_corner:
            visualise_ultranest_traces_corner(self)

        if self.visualise_samples:
            visualise_ultranest_samples(
                self,
                discard=100,
                thin=15,
                burnin=0.2,
                verbose=False
            )

    def log_ultranest_prior(self, cube):
        """ Compute relative prior probabilities per nest sample

            Input a set of unit cubic samples and transform them into the
            corresponding prior probabilities from (here) uniform boundaries

            Args:
                cube (list, tuple): (n, 2) list of n prior dimensions from
                    ultranest routine nested samples

            The argument, cube, consists of values from ranging from 0 to 1,
            which we here convert to physical scales:

                param[k] = cube[k] * param_range + param_minimum
        """
        params = cube.copy()

        ppm = 1e6
        # fpfs_min = -500 / ppm
        fpfs_min = 0 / ppm
        fpfs_max = 500 / ppm
        fpfs_rng = fpfs_max - fpfs_min

        delta_ec_min = -0.10
        delta_ec_max = 0.10
        delta_ec_rng = delta_ec_max - delta_ec_min

        log_f_min = -10
        log_f_max = 1.0
        log_f_rng = log_f_max - log_f_min

        # let fpfs level go from -500 to +500 ppm
        params[0] = cube[0] * fpfs_rng + fpfs_min

        # let delta_ecenter go from -0.05 to 0.05
        params[1] = cube[1] * delta_ec_rng + delta_ec_min

        # let log_f go from -10 to 1
        params[2] = cube[2] * log_f_rng + log_f_min

        return params

    def run_ultranest_pipeline(self, num_live_points=400, log_dir_extra=None):
        """ Operate a set of subroutines leading to the full ultranest pipelines

            Args:
                num_live_points (int): number of nested samples to start with.
                    Defaults to 400
                log_dir_extra (str, optional): any specific string to add to
                    the subdir for the log_dir UltaNest kwarg
        """
        # ultranest Sampling
        # init_distro = alpha * np.random.randn(self.nwalkers, len(self.soln.x))

        # # Ensure that "we" are all on the same page
        # self.bm_params.fp = self.soln.x[0]

        # pos = self.soln.x + init_distro
        # nwalkers, ndim = pos.shape

        # Avoid complications with MKI
        # os.environ["OMP_NUM_THREADS"] = "1"  # Emcee
        parameters = ['fpfs', 'delta_ecenter', 'log_f']  #
        ndim = len(parameters)
        log_dir = f'{self.log_dir}_RNS-{ndim}'

        if log_dir_extra is not None:
            log_dir = f'{log_dir}_{log_dir_extra}'

        # with Pool(cpu_count()-1) as pool:
        self.sampler = ultranest.ReactiveNestedSampler(
            parameters,
            self.log_likelihood,
            transform=self.log_ultranest_prior,
            log_dir=log_dir,
            vectorized=False  # TODO: Activte vectorized
            # wrapped_params=[False, False, False, True],
        )
        """ From Tutorial
        if args.slice:
            # set up step sampler. Here, we use a differential evolution slice sampler:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=args.slice_steps,
                generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
            )

        # run sampler, with a few custom arguments:
        sampler.run(
            dlogz=0.5 + 0.1 * ndim,
            update_interval_iter_fraction=0.4 if ndim > 20 else 0.2,
            max_num_improvement_loops=3,
            min_num_live_points=args.num_live_points)
        """
        start = time()
        self.ultranest_results = self.sampler.run(
            min_num_live_points=num_live_points,
            # dKL=np.inf,
            # min_ess=100
        )

        print(f"ULtranest took { time() - start:.1f} seconds")

    def postprocess_ultranest_pipeline(self):
        """ From the process UltraNest pipeline, compute the estimate of the
            fitted parameters and the corresponding transit model + residuals
        """
        raise NotImplementedError('This is not implemented')
        # TODO: This is not implemented
        if self.verbose:
            print_ultranest_results(
                self.sampler, discard=100, thin=15, burnin=0.2)

        self.nested_estimates = get_truth_ultranest_values(
            self.sampler,
            discard=discard,
            thin=thin,
            burnin=burnin
        )

        delta_ecenter = self.nested_estimates['delta_center']

        ppm = 1e6
        self.bm_params.fp = self.nested_estimates['fpfs'] / ppm
        self.bm_params.t_secondary = self.planet_params.ecenter0 + delta_ecenter

        # Compute the ultranest transit model
        self.ultranest_transit_model = self.batman_wrapper()

        self.ultranest_krdata_map = krdata.sensitivity_map(
            self.tso_data.fluxes / self.ultranest_transit_model,
            self.krdata_inputs.ind_kdtree,
            self.krdata_inputs.gw_kdtree
        )

        spitzer_transit_model = self.ultranest_transit_model
        spitzer_full_model = spitzer_transit_model * self.ultranest_krdata_map

        self.ultranest_residuals = self.tso_data.fluxes - spitzer_full_model

    def save_mle_ultranest(self, savedir=None, num_live_points=''):
        """ Save function for ultranest and mle results

            Args:
                savedir (str): Path to save the results
                num_live_points (int): Number of live points to identify the
                    scale of the UltraNest results being saved.
        """
        isotime = datetime.now(timezone.utc).isoformat()

        if savedir is None:
            savedir = 'ultranest_savedir'

        if not os.path.exists(savedir):
            os.mkdir(savedir)

        savename = (
            f'{self.planet_name}_ultranest_krdata_{isotime}_'
            f'{num_live_points}_{self.aper_key}.joblib.save'
        )

        save_path = os.path.join(savedir, savename)

        joblib.dump(
            {
                'ultranest': self.ultranest_results,
                'mle': self.soln if hasattr(self, 'soln') else None
            },
            save_path
        )
        print(f'Saving Ultranest run to {save_path}')

    def load_mle_ultranest(self, filename=None, isotime=None):
        """ Load function for ultranest and mle results

            Args:
                savedir (str): Path to save the results
                isotime (str): ISO Formatted time string to identify the
                    UltraNest results to be loaded.
        """
        assert (filename is not None or isotime is None), \
            'Please provide either `filename` or `isotime`'

        if isotime is not None:
            if filename is not None:
                print(
                    Warning(
                        '`isotime` is not None. '
                        'Therfore `filename` will be overwritten'
                    )
                )

            filename = f'ultranest_spitzer_krdata_ppm_results_{isotime}.joblib.save'

        results = joblib.load(filename)

        self.flat_samples = results['ultranest']['flat_samples']
        self.samples = results['ultranest']['samples']
        self.tau = results['ultranest']['tau']


class ExoplanetEmceeTSO:

    def __init__(
            self, exotso_inst=None, df=None, aor_dir=None, channel=None,
            planet_name=None, trim_size=0, timebinsize=0, mast_name=None,
            n_samples=1000, nwalkers=32, centering_key=None, aper_key=None,
            init_fpfs=None, estimate_pinknoise=False, n_piecewise_params=0,
            n_sig=5, process_mcmc=False, run_full_pipeline=False,
            savenow=False, visualise_mle=False, visualise_chains=False,
            visualise_mcmc_results=False, standardise_fluxes=False,
            standardise_times=False, standardise_centers=False, verbose=False):
        """Emcee (MCMC) focused object for analysing time series observations.

        Args:
            exotso_inst (ExoplanetTSO, optional): Instance of ExoplanetTSO as
                base to ExoplanetEmceeTSO class
            df (pd.DataFrame, optional): base dateframe with times, flux,
                errors, and other vectors (e.g., xcenters, ycenters) useful for
                noise modeling. Defaults to None.
            aor_dir (str, optional): location of processed data to load. Only
                used if loading data from the output of the `wanderer` package.
                Defaults to None.
            channel (str, optional): Spitzer channel directory of stored data.
                Only used if loading data from the output of the `wanderer`
                package. Defaults to None.
            planet_name (str, optional): name of planet for load and save
                functions. Defaults to None.
            trim_size (float, optional): days to trim from start of each AOR.
                Defaults to 0.
            timebinsize (float, optional): width in days (or `times` similar
                value).  Defaults to 0.
            mast_name (str, optional): name of planet in MAST registry.
                Defaults to None.
            estimate_pinknoise (bool, optional): toggle for whether to include
                Carter&Winn2010 wavelet likelihood (`pinknoise`) likelihood.
                Defaults to False.
            centering_key (str, optional): Base for the column name of center
                values: `fluxweighted` or `gaussian_fit`. Defaults to None.
            aper_key (str, optional): Base for column name for flux values.
                Defaults to None.
            init_fpfs (float, optional): Initial guess for the eclipse depth
                (fpfs). Defaults to None.
            n_piecewise_params (int, optional): whether to include an offset
                (=1) or slope (=2) per AOR with multi AOR observations.
                Defaults to 0.
            n_sig (int, optional): Number of sigma for Gaussian thresholding.
                Defaults to 5.
            process_mcmc (bool, optional): Toggle whether to activate the
                mcmc subroutines. Defaults to False.
            run_full_pipeline (bool, optional): Toggle whether to activate the
                full analysis pipeline. Defaults to False.
            savenow (bool, optional): Toggle whether to activate the save to
                joblib subroutines. Defaults to False.. Defaults to False.
            visualise_mle (bool, optional): Toggle whether to activate the MLE
                plotting subroutine. Defaults to False.
            visualise_chains (bool, optional): Toggle whether to
                activate the Emcee chains and corner plotting subroutine.
                Defaults to False.
            visualise_mcmc_results (bool, optional): Toggle whether to activate
                the Emcee samples over time plotting subroutine.
                Defaults to False.
            standardise_fluxes (bool, optional): Toggle whether to
                gaussian-filter the flux values to remove n_sig outliers.
                Defaults to False.
            standardise_times (bool, optional): Toggle whether to
                median-center the time values to reduce complexity with
                optimsations.  Defaults to False.
            standardise_centers (bool, optional): Toggle whether to
                median-center the centering values to reduce complexity with
                optimsations.  Defaults to False.
            verbose (bool, optional): Toggle whether to activate excessing
                print to stdout. Defaults to False.
        """
        if exotso_inst is not None:
            kwargs = {
                key: val for key, val in exotso_inst.items()
                if key in ExoplanetTSO.__all__
            }
            super().__init__(**kwargs)
        else:
            super().__init__(
                df=df,
                aor_dir=aor_dir,
                channel=channel,
                planet_name=planet_name,
                mast_name=mast_name,
                centering_key=centering_key,
                aper_key=aper_key,
                estimate_pinknoise=estimate_pinknoise and HAS_PINKNOISE,
                n_piecewise_params=n_piecewise_params,
                init_fpfs=init_fpfs,
                savenow=savenow,
                visualise_mle=visualise_mle,
                trim_size=trim_size,
                timebinsize=timebinsize,
                n_sig=n_sig,
                standardise_fluxes=standardise_fluxes,
                standardise_times=standardise_times,
                standardise_centers=standardise_centers,
                verbose=verbose
            )

        self.n_samples = n_samples
        self.nwalkers = nwalkers
        self.process_mcmc = process_mcmc
        self.visualise_mcmc_results = visualise_mcmc_results
        self.visualise_chains = visualise_chains
        self.run_full_pipeline = run_full_pipeline

        if self.run_full_pipeline:
            self.full_emcee_pipeline()

    def full_emcee_pipeline(self):
        """ Runs a sequence of subroutines for the emcee pipeline

            1. self.initialise_data_and_params
            2. self.run_mle_pipeline
            3. self.run_emcee_pipeline
            4. self.save_mle_emcee
            5. visualise_mle_solution
            6. visualise_emcee_traces_corner
            7. visualise_emcee_samples
        """
        self.initialise_data_and_params()

        self.run_mle_pipeline()

        if self.process_mcmc:
            self.run_emcee_pipeline()

        if self.savenow:
            self.save_mle_emcee()

        if self.visualise_mle:
            visualise_mle_solution(self)

        if self.visualise_chains:
            visualise_emcee_traces_corner(
                self,
                discard=100,
                thin=15,
                burnin=0.2,
                verbose=False
            )

        if self.visualise_mcmc_results:
            visualise_emcee_samples(
                self,
                discard=100,
                thin=15,
                burnin=0.2,
                verbose=False
            )

    def run_emcee_pipeline(self, alpha=1e-4):
        """ Operate a set of subroutines leading to the full emcee pipelines

            Args:
                alpha (float): Width of distribution around intial guesses.
                Defaults to 1e-4.
        """
        # Emcee Sampling
        init_distro = alpha * np.random.randn(self.nwalkers, len(self.soln.x))

        # Ensure that "we" are all on the same page
        self.bm_params.fp = self.soln.x[0]

        pos = self.soln.x + init_distro
        nwalkers, ndim = pos.shape

        # Avoid complications with MKI
        # os.environ["OMP_NUM_THREADS"] = "1"

        # TODO: Understand why `pool=pool` stopped working after working well
        # with Pool(cpu_count()-1) as pool:
        self.sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_emcee_posterior,  # pool=pool
        )

        start = time()
        self.sampler.run_mcmc(pos, self.n_samples, progress=True)
        print(f"Emcee took { time() - start:.1f} seconds")

    def emcee_postprocess_pipeline(self, discard=100, thin=15, burnin=0.2):
        """ From the process Encee pipeline, compute the estimate of the
            fitted parameters and the corresponding transit model + residuals

            Args:
                discard (int): The number of samples to discard from chains
                thin (int): The number of samples to thin chains
                burnin (float): The percent of samples to discard: range: [0, 1]
        """
        if self.verbose:
            print_emcee_results(self.sampler, discard=100, thin=15, burnin=0.2)

        self.mcmc_estimates = get_truth_emcee_values(
            self.sampler,
            discard=discard,
            thin=thin,
            burnin=burnin
        )

        delta_ecenter = self.mcmc_estimates['delta_center']

        ppm = 1e6
        self.bm_params.fp = self.mcmc_estimates['fpfs'] / ppm
        self.bm_params.t_secondary = self.planet_params.ecenter0 + delta_ecenter

        # Compute the mcmc transit model
        self.mcmc_transit_model = self.batman_wrapper()

        self.mcmc_krdata_map = krdata.sensitivity_map(
            self.tso_data.fluxes / self.mcmc_transit_model,
            self.krdata_inputs.ind_kdtree,
            self.krdata_inputs.gw_kdtree
        )

        spitzer_transit_model = self.mcmc_transit_model * self.mcmc_krdata_map
        self.mcmc_residuals = self.tso_data.fluxes - spitzer_transit_model

    def log_emcee_prior(self, theta):
        """ Compute prior probabilities per parameter in chain

            Input a set of sample `theta` compute the corresponding prior 
                probabilities from (here) uniform boundaries

            Args:
                theta (list, tuple): (n,) list of n prior samples from `emcee`
        """
        # return 0  # Open prior
        fpfs, delta_ecenter, log_f = theta[:3]  #
        sigma_w, sigma_r = theta[3:5] if self.estimate_pinknoise else (1, 1)
        gamma = theta[5] if len(theta) == 6 else 1.0

        # print('log_prior', fpfs, delta_ecenter, log_f)  #
        ppm = 1e6
        # TODO: Experiment with negative eclipse depths
        #   to avoid non-Gaussian posterior distributions
        ed_min = 0 / ppm  # Explore existance by permitting non-physical distributions
        ed_max = 500 / ppm  # 500 ppm
        dc_max = 0.05  # day
        dc_min = -0.05  # day

        lf_min = -10  # 1e-10 error modifier
        lf_max = 1  # 10x error modifier

        sw_min = 0
        sr_min = 0
        gm_min = 1.0

        return (
            0.0 if (
                ed_min <= fpfs < ed_max and
                dc_min < delta_ecenter < dc_max and
                lf_min < log_f < lf_max and
                sw_min <= sigma_w and
                sr_min <= sigma_r and
                gm_min <= gamma
            )
            else -np.inf
        )

    def log_emcee_likelihood(self, theta):
        """Compute the normal (and wavelet) log likelihoods

        Args:
            theta (list): List of fitting parameters to be used

        Returns:
            float: log likelihood given the fitting parameters
        """
        fpfs, delta_ecenter, log_f = theta[:3]  #

        self.bm_params.fp = fpfs
        self.bm_params.t_secondary = self.planet_params.ecenter0 + delta_ecenter

        flux_errs = self.tso_data.flux_errs
        fluxes = self.tso_data.fluxes
        # pw_line = self.piecewise_linear_model(theta)
        # pw_line = self.piecewise_offset_model(theta)

        model = self.batman_krdata_wrapper()  # * pw_line
        sigma2 = flux_errs**2 + model**2 * np.exp(2 * log_f)

        residuals = fluxes - model

        wavelet_log_likelihood = 0
        if self.estimate_pinknoise and len(theta) >= 5:
            wavelet_log_likelihood = -0.5*self.get_wavelet_log_likelihood(
                residuals=fluxes - model,
                theta=theta
            )

        normal_log_likelihood = residuals ** 2 / sigma2 + np.log(sigma2)
        normal_log_likelihood = -0.5 * np.sum(normal_log_likelihood)

        if wavelet_log_likelihood != 0 and self.verbose:
            print('normal_log_likelihood', normal_log_likelihood)
            print('wavelet_log_likelihood', wavelet_log_likelihood)

        return normal_log_likelihood + wavelet_log_likelihood

    def log_emcee_posterior(self, theta):
        """ Compute the log_posterior probability for each parameter values

        Args:
            params (tuples): list of parameters for fitting

        Return:
            (float): the log_prior + log_likelihood
        """
        lp = self.log_emcee_prior(theta)

        return (
            lp + self.log_emcee_likelihood(theta)
            if np.isfinite(lp) else -np.inf
        )

    def save_mle_emcee(self, filename=None):
        """ Save function for ultranest and mle results

            Args:
                filename (str): Path + filename to save the results
        """
        discard = 0
        thin = 1
        self.tau = []

        try:
            self.tau = self.sampler.get_autocorr_time()
        except Exception as err:
            print(err)

        emcee_output = {
            'flat_samples': self.sampler.get_chain(
                discard=discard, thin=thin, flat=True
            ),
            'samples': self.sampler.get_chain(),
            'tau': self.tau
        }

        isotime = datetime.now(timezone.utc).isoformat()
        if filename is None:
            filename = f'emcee_spitzer_krdata_ppm_results_{isotime}.joblib.save'

        joblib.dump({'emcee': emcee_output, 'mle': self.soln}, filename)
        print(f'Saved emcee and mle results to {filename}')

    def load_mle_emcee(self, filename=None, isotime=None):
        """ Load function for ultranest and mle results

            Args:
                filename (str): Path + filename from which load the results
                isotime (str): ISO Formatted time string to identify the
                    UltraNest results to be loaded.
        """

        assert (filename is not None or isotime is None), \
            'Please provide either `filename` or `isotime`'

        if isotime is not None:
            if filename is not None:
                print(
                    Warning(
                        '`isotime` is not None. '
                        'Therfore `filename` will be overwritten'
                    )
                )

            filename = f'emcee_spitzer_krdata_ppm_results_{isotime}.joblib.save'

        results = joblib.load(filename)

        self.flat_samples = results['emcee']['flat_samples']
        self.samples = results['emcee']['samples']
        self.tau = results['emcee']['tau']
