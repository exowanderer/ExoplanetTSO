import batman
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
from scipy.optimize import minimize
from statsmodels.robust import scale
from time import time
from tqdm import tqdm

from skywalker import krdata
from skywalker.utils import configure_krdata_noise_models


from .models import (
    ExoplanetTSOData,
    KRDataInputs
)

from .utils import (
    bin_df_time,
    bin_array_time,
    # get_truth_emcee_values,
    # linear_model,
    load_from_df,
    load_from_wanderer,
    # print_emcee_results,
    trim_initial_timeseries,
    # visualise_emcee_samples,
    # visualise_emcee_traces_corner,
    visualise_ultranest_traces_corner,
    visualise_ultranest_samples,
    visualise_mle_solution,
)


class ExoplanetUltranestTSO:

    def __init__(
            self, df=None, aor_dir=None, channel=None, planet_name=None,
            trim_size=0, timebinsize=0, mast_name=None, n_fits=1000,
            inj_fpfs=0, estimate_pinknoise=False, centering_key=None,
            aper_key=None, init_fpfs=None, n_piecewise_params=0, n_sig=5,
            process_mcmc=False, run_full_pipeline=False,
            log_dir='ultranest_savedir', savenow=False,
            visualise_mle=False, visualise_nests=False,
            visualise_mcmc_results=False, standardise_fluxes=False,
            standardise_times=False, standardise_centers=False, verbose=False):

        self.df = df
        self.aor_dir = aor_dir
        self.channel = channel
        self.planet_name = planet_name
        self.mast_name = mast_name
        self.centering_key = centering_key
        self.aper_key = aper_key
        self.n_fits = n_fits
        self.estimate_pinknoise = estimate_pinknoise and HAS_PINKNOISE
        self.n_piecewise_params = n_piecewise_params
        self.inj_fpfs = inj_fpfs
        self.init_fpfs = init_fpfs
        self.process_mcmc = process_mcmc
        self.log_dir = log_dir
        self.savenow = savenow
        self.visualise_mle = visualise_mle
        self.visualise_mcmc_results = visualise_mcmc_results
        self.visualise_nests = visualise_nests
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

        if run_full_pipeline:
            self.full_pipeline()

    def initialise_data_and_params(self):
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

        if self.inj_fpfs > 0:
            self.inject_eclipse()

    def initialise_bm_params(self):
        # object to store transit parameters
        self.bm_params = batman.TransitParams()
        self.bm_params.per = self.period  # orbital period
        self.bm_params.t0 = self.tcenter  # time of inferior conjunction
        self.bm_params.inc = self.inc  # inclunaition in degrees

        # semi-major axis (in units of stellar radii)
        self.bm_params.a = self.aprs

        # planet radius (in units of stellar radii)
        self.bm_params.rp = self.rprs
        self.bm_params.ecc = self.ecc  # eccentricity
        self.bm_params.w = self.omega  # longitude of periastron (in degrees)
        self.bm_params.limb_dark = self.ldtype  # limb darkening model

        if self.ecenter is None:
            self.ecenter = self.tcenter + 0.5 * self.period

        if self.ldtype == 'uniform':
            self.bm_params.u = []  # limb darkening coefficients

        elif self.ldtype == 'linear':
            self.bm_params.u = [self.u1]  # limb darkening coefficients

        elif self.ldtype == 'quadratic':
            # limb darkening coefficients
            self.bm_params.u = [self.u1, self.u2]
        else:
            raise ValueError(
                "`ldtype` can only be ['uniform', 'linear', 'quadratic']")

        self.bm_params.t_secondary = self.ecenter
        self.bm_params.fp = self.fpfs

    def full_pipeline(self):
        self.initialise_data_and_params()

        self.run_mle_pipeline()

        if self.process_mcmc:
            self.run_ultranest_pipeline(
                num_live_points=400,
                log_dir_extra=None
            )

        if self.savenow:
            self.save_mle_ultranest()

        if self.visualise_mle:
            visualise_mle_solution(self)

        if self.visualise_nests:
            visualise_ultranest_traces_corner(self)

        if self.visualise_mcmc_results:
            visualise_ultranest_samples(
                self,
                discard=100,
                thin=15,
                burnin=0.2,
                verbose=False
            )

    def log_ultranest_prior(self, cube):
        # The argument, cube, consists of values from 0 to 1
        # we have to convert them to physical scales
        # param[k] = cube[k] * param_range + param_minimum

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

    """
    def log_likelihood(self, params):
        # unpack the current parameters:

        fpfs, delta_ecenter, log_f = params  # delta_ecenter,
        # delta_ecenter = 0  # assume circular orbit

        # compute for each x point, where it should lie in y

        self.bm_params.fp = fpfs
        self.bm_params.t_secondary = self.ecenter0 + delta_ecenter

        flux_errs = self.tso_data.flux_errs
        fluxes = self.tso_data.fluxes
        # pw_line = self.piecewise_linear_model(params)
        # pw_line = self.piecewise_offset_model(params)

        model = self.batman_krdata_wrapper()  # * pw_line
        sigma2 = flux_errs**2 + model**2 * np.exp(2 * log_f)

        # compute likelihood
        return -0.5 * np.sum((fluxes - model) ** 2 / sigma2 + np.log(sigma2))
    """
    """
    def get_wavelet_log_likelihood(self, residuals, params):
        if len(params[3:]) == 2:
            sigma_w, sigma_r = params[3:]
            gamma = 1.0
        if len(params[3:]) == 3:
            sigma_w, sigma_r, gamma = params[3:]

        if residuals.size != self.ndata_wavelet:
            # Pad up to self.ndata_wavelet
            padding = np.zeros(self.ndata_wavelet - residuals.size)
            residuals = np.r_[residuals, padding]

        # print(residuals.sum(), sigma_w, sigma_r, gamma)
        return self.wavelet_model.get_likelihood(
            residuals,
            sigma_w,
            sigma_r,
            gamma=gamma
        )
    """

    def log_likelihood(self, params):
        fpfs, delta_ecenter, log_f = params[:3]  #
        # sigma_w, sigma_r = params[3:5] if self.estimate_pinknoise else (1, 1)
        # gamma = params[5] if len(params) == 6 else 1.0
        # print(params)
        # print(fpfs, self.fpfs)
        # self.fpfs = fpfs
        # self.ecenter = self.ecenter0  # + delta_ecenter

        self.bm_params.fp = fpfs
        self.bm_params.t_secondary = self.ecenter0 + delta_ecenter

        flux_errs = self.tso_data.flux_errs
        fluxes = self.tso_data.fluxes
        # pw_line = self.piecewise_linear_model(params)
        # pw_line = self.piecewise_offset_model(params)

        model = self.batman_krdata_wrapper()  # * pw_line
        sigma2 = flux_errs**2 + model**2 * np.exp(2 * log_f)

        residuals = fluxes - model

        wavelet_log_likelihood = 0
        if self.estimate_pinknoise and len(params) >= 5:
            wavelet_log_likelihood = -0.5*self.get_wavelet_log_likelihood(
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
            init_fpfs = self.fpfs

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

    """

    def initialise_fit_params(self, init_logf=-5.0):

        # Compute MLE
        np.random.seed(42)

        init_fpfs = self.fpfs  # 265 ppm
        # init_ecenter = self.ecenter0  # t0 + per/2

        self.init_params = [
            np.random.normal(init_fpfs, 1e-5),
            np.random.normal(0.0, 1e-4),
            np.random.normal(init_logf, 0.01)
        ]

        if 0 < self.n_piecewise_params <= 2:
            # Add a offset and slope for each AOR
            self.add_piecewise_linear(
                n_lines=0,
                add_slope=self.n_piecewise_params == 2  # 2 params
            )
    """

    def inject_eclipse(self):
        ppm = 1e6

        print(f'Injecting Model with FpFs: {self.inj_fpfs*ppm}ppm')

        # Inject a signal if `inj_fpfs` is provided
        inj_model = self.batman_wrapper(
            self.tso_data.times,
            period=self.period,
            tcenter=self.tcenter,
            inc=self.inc,
            aprs=self.aprs,
            rprs=self.rprs,
            ecc=self.ecc,
            omega=self.omega,
            u1=self.u1,
            u2=self.u2,
            offset=self.offset,
            slope=self.slope,
            curvature=self.curvature,
            ecenter=self.ecenter,
            fpfs=self.inj_fpfs,
            ldtype=self.ldtype,  # ='uniform',
            transit_type=self.transit_type,  # ='secondary',
            verbose=self.verbose
        )

        # print(fluxes.mean(), inj_model.min(), inj_model.max())
        self.fluxes = self.fluxes * inj_model

        self.tso_data.fluxes = self.fluxes
        self.krdata_inputs.fluxes = self.fluxes

    def run_mle_pipeline(self, init_fpfs=None):

        if init_fpfs is not None:
            self.init_params['fpfs'] = np.random.normal(init_fpfs, 1e-5)

        # nll = lambda *args: -self.log_ultranest_likelihood(*args)
        nlp = lambda *args: -self.log_mle_probability(*args)

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

    def run_ultranest_pipeline(self, num_live_points=400, log_dir_extra=None):
        # ultranest Sampling
        # init_distro = alpha * np.random.randn(self.nwalkers, len(self.soln.x))

        # # Ensure that "we" are all on the same page
        # self.bm_params.fp = self.soln.x[0]

        # pos = self.soln.x + init_distro
        # nwalkers, ndim = pos.shape

        # Avoid complications with MKI
        os.environ["OMP_NUM_THREADS"] = "16"
        parameters = ['fpfs', 'delta_ecenter', 'log_f']  #
        ndim = len(parameters)
        log_dir = f'{self.log_dir}_RNS-{ndim}'

        if log_dir_extra is not None:
            log_dir = f'{log_dir}_{log_dir_extra}'

        # with Pool(cpu_count()-1) as pool:
        self.sampler = ultranest.ReactiveNestedSampler(
            parameters,
            self.log_likelihood,
            self.log_ultranest_prior,
            log_dir=log_dir
            # wrapped_params=[False, False, False, True],
        )
        """
        self.sampler =  = ReactiveNestedSampler(
            parameters,
            self.log_likelihood,
            transform=self.log_ultranest_prior,
            log_dir=self.log_dir + 'RNS-%dd' % ndim,
            resume=True,
            vectorized=False
        )

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

    def postprocess_pipeline(self, discard=100, thin=15, burnin=0.2):
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
        self.bm_params.t_secondary = self.ecenter0 + delta_ecenter

        # Compute the mcmc transit model
        self.mcmc_transit_model = self.batman_wrapper()

        self.mcmc_krdata_map = krdata.sensitivity_map(
            self.tso_data.fluxes / self.mcmc_transit_model,
            self.krdata_inputs.ind_kdtree,
            self.krdata_inputs.gw_kdtree
        )

        spitzer_transit_model = self.mcmc_transit_model * self.mcmc_krdata_map
        self.mcmc_residuals = self.tso_data.fluxes - spitzer_transit_model

    def compute_bestfit_residuals(self, fpfs, delta_ecenter):

        if fpfs > 1:
            ppm = 1e6
            fpfs = fpfs / ppm

        ecenter = self.ecenter0 + delta_ecenter

        # Compute the mcmc transit model
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
            title=None, height_ratios=None, fontsize=20):

        if height_ratios is None:
            height_ratios = [0.9, 0.1]

        has_none = None in [fpfs, delta_ecenter]
        if has_none and not hasattr(self, 'mcmc_estimates'):
            self.postprocess_pipeline(discard=100, thin=15, burnin=0.2)

        if fpfs is None:
            ppm = 1e6
            fpfs = self.mcmc_estimates['fpfs'] / ppm

        if delta_ecenter is None:
            delta_ecenter = self.mcmc_estimates['delta_center']

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
            period = self.period
            tcenter = self.tcenter
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
                sharex=True
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

        plt.show()

        return fig, (ax1, ax2)

    def preprocess_pipeline(self):

        if self.trim_size > 0:
            df = trim_initial_timeseries(
                df=self.df,
                trim_size=self.trim_size,
                aornums=self.df.aornum.unique()
            )

        if self.timebinsize > 0:
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

        # # TODO: Confirm if this is still required
        # self.tso_data.times = self.tso_data.times
        # self.tso_data.fluxes = self.tso_data.fluxes
        # self.tso_data.flux_errs = self.tso_data.flux_errs
        # self.tso_data.aornums = self.tso_data.aornums

    def print_bm_params(self):
        print('times', self.tso_data.times.mean())
        print('period', self.period)
        print('tcenter', self.tcenter)
        print('inc', self.inc)
        print('aprs', self.aprs)
        print('rprs', self.rprs)
        print('ecc', self.ecc)
        print('omega', self.omega)
        print('u1', self.u1)
        print('u2', self.u2)
        print('offset', self.offset)
        print('slope', self.slope)
        print('curvature', self.curvature)
        print('ldtype', self.ldtype)
        print('transit_type', self.transit_type)
        print('t_secondary', self.ecenter)
        print('fp', self.fpfs)

    def batman_wrapper(self, update_fpfs=None, update_ecenter=None):
        '''
            Written By Jonathan Fraine
            https://github.com/exowanderer/Fitting-Exoplanet-Transits
        '''
        # print(fpfs, ecenter)
        if self.verbose >= 2:
            self.print_bm_params()

        if update_fpfs is not None:
            self.bm_params.fp = update_fpfs

        if update_ecenter is not None:
            self.bm_params.t_secondary = update_ecenter

        m_eclipse = batman.TransitModel(
            self.bm_params,
            self.tso_data.times,
            transittype=self.transit_type
        )

        return m_eclipse.light_curve(self.bm_params)

    def batman_krdata_wrapper(self, update_fpfs=None, update_ecenter=None):
        '''
            Written By Jonathan Fraine
            https://github.com/exowanderer/Fitting-Exoplanet-Transits
        '''
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

    """
    def batman_spitzer_wrapper(self):
        return self.batman_krdata_wrapper(
            times=self.tso_data.times,
            krdata_inputs=self.krdata_inputs,
            period=self.period,
            tcenter=self.tcenter,
            ecenter=self.ecenter,
            inc=self.inc,
            aprs=self.aprs,
            rprs=self.rprs,
            fpfs=self.fpfs,
            ecc=self.ecc,
            omega=self.omega,
            u1=self.u1,
            u2=self.u2,
            offset=self.offset,
            slope=self.slope,
            curvature=self.curvature,
            ldtype=self.ldtype,
            transit_type=self.transit_type
        )
    """
    """
    def piecewise_linear_model(self, params):
        times = self.tso_data.times.copy()
        aornums = self.tso_data.aornums.copy()
        sorted_aornums = np.sort(np.unique(aornums))

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

    def piecewise_offset_model(self, params):
        times = self.tso_data.times.copy()
        aornums = self.tso_data.aornums.copy()
        sorted_aornums = np.sort(np.unique(aornums))

        offsets = params[3:]

        # print(offsets, slopes)
        piecewise_offset = np.zeros_like(times)
        for offset_, aornum_ in zip(offsets, sorted_aornums):
            is_aornum = aornums == aornum_
            line_ = linear_model(times[is_aornum], offset_, 0)
            piecewise_offset[is_aornum] = line_
            # print(aornum_, offset_, slope_, is_aornum.sum(), line_.mean())

        return piecewise_offset
    """

    @staticmethod
    def log_mle_prior(params):
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

    """
    def log_ultranest_likelihood(self, params):
        fpfs, delta_ecenter, log_f = params[:3]  #
        # print(fpfs, self.fpfs)
        # self.fpfs = fpfs
        # self.ecenter = self.ecenter0  # + delta_ecenter

        self.bm_params.fp = fpfs
        self.bm_params.t_secondary = self.ecenter0 + delta_ecenter

        flux_errs = self.tso_data.flux_errs
        fluxes = self.tso_data.fluxes
        # pw_line = self.piecewise_linear_model(params)
        # pw_line = self.piecewise_offset_model(params)

        model = self.batman_krdata_wrapper()  # * pw_line
        sigma2 = flux_errs**2 + model**2 * np.exp(2 * log_f)

        return -0.5 * np.sum((fluxes - model) ** 2 / sigma2 + np.log(sigma2))
    """

    # def log_ultranest_probability(self, params):
    #     return self.log_likelihood(params)

    def log_mle_probability(self, params):
        lp = self.log_mle_prior(params)

        return (
            self.log_likelihood(params)  # + lp
            if np.isfinite(lp) else -np.inf
        )

    def configure_krdata(self, ymod=0.7, n_nbr=100):

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

        if init_fpfs is None:
            init_fpfs = self.init_fpfs

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

        self.period = planet_info.orbital_period
        self.tcenter = planet_info.transit_time
        self.inc = planet_info.inclination
        self.aprs = planet_info.a_Rs
        self.tdepth = planet_info.transit_depth
        self.rprs = planet_info.Rp_Rs
        self.ecc = planet_info.eccentricity
        self.omega = planet_info.omega
        self.u1 = init_u1
        self.u2 = init_u2
        self.offset = 1
        self.slope = 0
        self.curvature = 0
        self.ldtype = 'uniform'
        self.transit_type = 'secondary'

        init_ecenter = self.tcenter + self.period*0.5

        self.ecenter = init_ecenter
        self.ecenter0 = init_ecenter
        self.fpfs = init_fpfs

    def configure_pinknoise_model(self):
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

    def save_mle_ultranest(self, savedir=None, num_live_points=''):
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
