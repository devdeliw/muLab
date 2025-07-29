import emcee 
import numpy as np 

from typing import Optional
from dataclasses import dataclass 
from scipy.stats import gaussian_kde


@dataclass 
class LinearMCMCConfig: 
    nwalkers: int = 40 
    nsteps: int   = 6000 
    burnin: int   = 1000 
    thin: int     = 1

class LinearMCMC: 
    """
    Linear-fit MCMC for F115W - F323N vs. F115W CMDs. 
    Systematic uncertainty is estimated via a MCMC jitter. 

    """

    def __init__(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        yerr: np.ndarray,
        boundary: Optional[tuple[float, float, float, float]], 
        top_fraction: float,
        config: LinearMCMCConfig = LinearMCMCConfig()
    ):
        """
        Args: 
            * x, y (np.ndarray): x and y coordinates for Red Clump on CMD 
            * yerr (np.ndarray): 1sigma measurement error on y 
            * boundary (xmin, xmax, ymin, ymax): optinal 
                only points inside this rectangle are used in the fit 
                omit or set to None to keep all stars 
            * top_fraction (float): only use the "top_fraction" most dense 
                stars in the MCMC fit
        """

        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float) 
        yerr = np.asarray(yerr, dtype=float) 

        if x.shape != y.shape or y.shape != yerr.shape: 
            raise ValueError("x, y, yerr must have identical shape.")

        if boundary: 
            xmin, xmax, ymin, ymax = boundary
            mask = (
                (x >= xmin) & (x <= xmax) & 
                (y >= ymin) & (y <= ymax) 
            )
            x, y, yerr = x[mask], y[mask], yerr[mask] 

        if top_fraction: 
            x, y, yerr = self._filter_top_density( x, y, yerr, top_fraction=top_fraction)

        if x.size == 0: 
            raise ValueError("No data points are left after masking!") 

        self.x, self.y, self.yerr = x, y, yerr 
        self.linear_config = config 

        self.samples = None 
        self.best    = None 

    def _filter_top_density(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        yerr: np.ndarray, 
        top_fraction: float 
    ): 
        xy = np.vstack([x, y]) 
        kde = gaussian_kde(xy) 
        density = kde(xy) 

        cutoff = np.quantile(density, 1 - top_fraction)
        mask = density >= cutoff 
        return x[mask], y[mask], yerr[mask]

    def _log_prior(self, theta): 
        m, b, log_sig_int = theta 
        if -1e3 < m < 1e3 and -1e6 < b < 1e6 and -10 < log_sig_int < 3: 
            return -0.5 * log_sig_int**2 
        return -np.inf 

    def _log_likelihood(self, theta): 
        m, b, log_sig_int = theta 
        sig2   = self.yerr**2 + np.exp(2.0 * log_sig_int) 
        resid2 = (self.y - (m * self.x + b))**2 
        return -0.5 * np.sum(resid2 / sig2 + np.log(2.0 * np.pi * sig2)) 

    def _log_prob(self, theta): 
        lp = self._log_prior(theta) 
        return lp + self._log_likelihood(theta) if np.isfinite(lp) else -np.inf 

    def _initial_guess(self): 
        w = 1.0 / self.yerr**2 
        
        A = np.vstack([self.x, np.ones_like(self.x)]).T * np.sqrt(w[:, None])
        b = self.y * np.sqrt(w) 
        m0, b0   = np.linalg.lstsq(A, b, rcond=None)[0]
        sig_int0 = max(1e-3, np.std(self.y - (m0 * self.x + b0))) 
        return np.array([m0, b0, np.log(sig_int0)]) 

    def run(self): 
        nwalkers = self.linear_config.nwalkers 
        nsteps   = self.linear_config.nsteps 
        burnin   = self.linear_config.burnin 
        thin     = self.linear_config.thin 

        p0   = self._initial_guess() 
        ndim = len(p0) 

        # initialize walker positions 
        pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim) 
        
        # run mcmc 
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_prob) 
        sampler.run_mcmc(pos, nsteps)

        self.samples = sampler.get_chain(discard=burnin, thin=thin, flat=True) 

        # medians 
        m_med, b_med, log_sig_med = np.median(self.samples, axis=0)
        m_err = np.std(self.samples[:, 0], ddof=1) 

        self.best = dict(
            slope=m_med, 
            intercept=b_med, 
            sig_int=np.exp(log_sig_med), 
            slope_err=m_err
        )
        return m_med, b_med, m_err 




