# ExoplanetTSO

ExoplanetTSO: Package for optimal exoplanet &amp; brown dwarf time series observation data analysis

# Nominal Spitzer Use Case with KRData Noise Model

```python
import joblib
import numpy as np
import pandas as pd

from exotso.exotso_ultranest import ExoplanetUltranestTSO
from exotso.utils import (
    grab_data_from_csv,
    visualise_ultranest_traces_corner,
    visualise_ultranest_samples,
    visualise_mle_solution
)
```

```python
ppm = 1e6
n_sig = 5
aor_dir = 'r64922368'
channel = 'ch2'  # CHANNEL SETTING
planet_name = 'planet_name'
mast_name = 'PL-anet-name'
inj_fpfs = 0 / ppm  # no injected signal
init_fpfs = 265 / ppm  # no injected signal
num_live_points = 400
aper_key = 'rad_2p5_0p0'
centering_key = 'gaussian_fit'
# centering_key = 'fluxweighted'
trim_size = 1/24  # one hour in day units
timebinsize = 0/60/24  # 0 minutes in day units
```

```python
df_tso_data = grab_data_from_csv(filename=None, aornums=aornums)
```

```python
exotso_instance = ExoplanetUltranestTSO(
    df=df_tso_data,
    planet_name=planet_name,
    mast_name=mast_name,
    trim_size=trim_size,
    timebinsize=timebinsize,
    centering_key=centering_key,
    aper_key=aper_key,
    inj_fpfs=0,
    init_fpfs=init_fpfs,
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
```

```python
exotso_instance.initialise_data_and_params()

exotso_instance.run_mle_pipeline()

exotso_instance.run_ultranest_pipeline(
    num_live_points=num_live_points,
    log_dir_extra=None
)

exotso_instance.save_mle_ultranest(
    num_live_points=num_live_points
)
```

```python

fpfs_ml, delta_ecenter_ml, log_f_ml = exotso_instance.soln.x

print(
    f'fpfs_ml={fpfs_ml*1e6}ppm\n'
    f'delta_ecenter_ml={delta_ecenter_ml}\n'
    f'ecenter_ml={exotso_instance.ecenter0 + delta_ecenter_ml}\n'
    f'log_f_ml={log_f_ml}\n'
)

visualise_mle_solution(exotso_instance)

sampler = exotso_instance.sampler
if np.median(sampler.run_sequence['samples'][:, 0]) < 1:
    fpfs_ = sampler.run_sequence['samples'][:, 0]*1e6
    sampler.run_sequence['samples'][:, 0] = fpfs_
    exotso_instance.sampler = sampler

results = exotso_instance.ultranest_results
if np.median(results['weighted_samples']['points'][:, 0]) < 1:
    fpfs_ = results['weighted_samples']['points'][:, 0]*1e6
    results['weighted_samples']['points'][:, 0] = fpfs_
    exotso_instance.ultranest_results = results

visualise_ultranest_traces_corner(exotso_instance, suptitle=None)


visualise_ultranest_samples(
    exotso_instance,
    discard=0,
    thin=1,
    burnin=0.2,
    verbose=False
)
```
