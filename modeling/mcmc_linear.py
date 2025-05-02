import numpy as np
import emcee

from scipy.stats import gaussian_kde


class LinearBayes:
    """
    x, y, yerr : 1-D ndarray
        Full catalogue.  yerr is 1-σ measurement error on y.
    box : (xmin, xmax, ymin, ymax), optional
        Only points inside this rectangle are used in the fit.
        Omit or set to None to keep all stars.
    """

    def filter_top_density(self, x: np.ndarray, y: np.ndarray, ye: np.ndarray, top_fraction: float = 0.1):
        """
        Return the subset of (x, y) points that lie in the top `top_fraction`
        most-dense regions of the scatter.

        """

        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)                 
        dens = kde(xy)                         

        cutoff = np.quantile(dens, 1 - top_fraction)

        mask = dens >= cutoff
        return x[mask], y[mask], ye[mask]


    def __init__(self, x, y, yerr, box=None, top_fraction=None):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        yerr = np.asarray(yerr, float)
        if x.shape != y.shape or y.shape != yerr.shape:
            raise ValueError("x, y, yerr must have identical shapes")

        # mask: keep only stars in the user-supplied rectangle
        if box:
            xmin, xmax, ymin, ymax = box
            m = (
                (x >= xmin) & (x <= xmax) &
                (y >= ymin) & (y <= ymax)
            )
            x, y, yerr = x[m], y[m], yerr[m]

        # select top fraction of stars in box
        if top_fraction:
            x, y, yerr = self.filter_top_density(x, y, yerr, top_fraction=top_fraction)
            
        if x.size == 0:
            raise ValueError("No data points left after masking!")

        self.x, self.y, self.yerr = x, y, yerr
        self.samples = None
        self.best = None

    def _log_prior(self, theta):
        m, b, log_sig_int = theta
        # broad uniform for m, b;  weak N(0,1) on log sig_int
        if -1e3 < m < 1e3 and -1e6 < b < 1e6 and -10 < log_sig_int < 3:
            return -0.5 * log_sig_int**2
        return -np.inf

    def _log_likelihood(self, theta):
        m, b, log_sig_int = theta
        sig2 = self.yerr**2 + np.exp(2.0 * log_sig_int)
        resid2 = (self.y - (m * self.x + b))**2
        return -0.5 * np.sum(resid2 / sig2 + np.log(2.0 * np.pi * sig2))

    def _log_prob(self, theta):
        lp = self._log_prior(theta)
        return lp + self._log_likelihood(theta) if np.isfinite(lp) else -np.inf

    def _initial_guess(self):
        """Weighted least-squares seed for (m, b).  sig_int starts at scatter."""
        w = 1.0 / self.yerr**2
        A = np.vstack([self.x, np.ones_like(self.x)]).T * np.sqrt(w[:, None])
        b = self.y * np.sqrt(w)
        m0, b0 = np.linalg.lstsq(A, b, rcond=None)[0]
        sig_int0 = max(1e-3, np.std(self.y - (m0 * self.x + b0)))
        return np.array([m0, b0, np.log(sig_int0)])

    def run(self,
            nwalkers=40, nsteps=6000, burnin=1000, thin=1):
        """
        Execute the sampler.  Returns (slope, slope_error_1sig)
        """
        p0 = self._initial_guess()
        ndim = len(p0)
        pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_prob)
        sampler.run_mcmc(pos, nsteps)

        self.samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        m_med, b_med, log_sig_med = np.median(self.samples, axis=0)
        m_err = np.std(self.samples[:, 0], ddof=1)
        self.best = dict(slope=m_med,
                         intercept=b_med,
                         sig_int=np.exp(log_sig_med),
                         slope_err=m_err)
        return m_med, b_med, m_err

