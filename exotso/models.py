import numpy as np
from dataclasses import dataclass


@ dataclass
class KRDataInputs:
    ind_kdtree: np.ndarray = None
    gw_kdtree: np.ndarray = None
    fluxes: np.ndarray = None


@ dataclass
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


@ dataclass
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
