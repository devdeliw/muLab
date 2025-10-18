import warnings
import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee

from math import log

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) 

class MCMC_Autocorr:
    """
    Fit a 1-D histogram with a compound Gaussian + Linear model via emcee and
    provide basic convergence diagnostics (integrated autocorrelation time tau_int).
    """

    @staticmethod
    def _compound_model(theta, x):
        """ 
        f_RC  : fractional weight of the Gaussian (RC) component, 0 < f < 1 
        amp   :peak height of the Gaussian when f_RC = 1
        mu    : Gaussian mean
        sig   : Gaussian std
        m, b  : slope + intercept of the linear background
        """
        u, amp, mu, sig, m, b = theta
        f = sigmoid(u)
        gauss  = amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)
        linear = m * x + b
        return gauss + (1.0 - f) * linear

    def _log_prior(self, theta):
        u, amp, mu, sig, m, b = theta
        mu0, s0 = np.mean(self.data), np.std(self.data) 

        # logit-space Beta(alpha, beta) prior
        alpha, beta = 4.0, 2.0  
        f_rc = sigmoid(u)
        ln_prior  = (alpha-1)*np.log(f_rc) + (beta-1)*np.log(1-f_rc)
        ln_prior += np.log(f_rc) + np.log(1-f_rc)   # jacobian 

        if not np.isfinite(ln_prior):               # for f_RC = 0,1
             return -np.inf
        if not (0 < amp < 1_000):
            return -np.inf
        if not (mu0 - 2.0 < mu < mu0 + 2.0):
            return -np.inf
        if not (max(0.0, s0 - 0.5) < sig < s0 + 0.5):
            return -np.inf
        if m > 30 or b > 50:
            return -np.inf
        return ln_prior

    def _log_likelihood(self, theta):
        model = self._compound_model(theta, self.bin_centers)
        resid = self.bin_heights - model
        ivar = 1.0 / self.bin_errors**2
        return -0.5 * np.sum(resid**2 * ivar + np.log(2 * np.pi / ivar))

    def _log_prob(self, theta):
        lp = self._log_prior(theta)
        return lp + self._log_likelihood(theta) if np.isfinite(lp) else -np.inf

    def _initial_guess(self):
        h = self.bin_heights
        return [
            np.log(0.1/0.9),      # u 
            h.max() / 4,          # amp
            np.mean(self.data),   # mu
            np.std(self.data),    # sig
            0.0,                  # m
            (h.min() if h.min() > 0 else 1.0),  # b
        ]

    # public api 
    def __init__(self, data, bins=50):
        self.data = np.asarray(data)
        self.bins = bins

        self.bin_heights, self.bin_edges = np.histogram(self.data, bins=bins)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_errors = np.sqrt(self.bin_heights + 1)

        self.sampler = None          # populated by run()
        self.samples = None
        self.tau = None              # \tau_int per parameter
        self._acorr_curve = None     # (N, mean_\tau) pairs for plotting 

    def run(self, nwalkers=64, nsteps=15_000, burnin=1_000, thin=1):
        """
        Launch emcee, store chain and best-fit summary (medians).
        """
        p0 = self._initial_guess()
        ndim = len(p0)
        pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_prob)
        self.sampler.run_mcmc(pos, nsteps, progress=False)

        chain = self.sampler.get_chain(discard=burnin, flat=False)  # shape:(T,W,D)
        self.chain = chain                                          # save for later
        self.samples = self.sampler.get_chain(discard=burnin, thin=thin, flat=True)
        self.log_probs = self.sampler.get_log_prob(discard=0, flat=False)

        u, amp, mu, sig, m, b = np.median(self.samples, axis=0)
        f = sigmoid(u)
        self.best_fit = dict(
            frac_RC=f,
            amplitude=amp, 
            mean=mu, 
            stddev=sig, 
            slope=m, 
            intercept=b
        )
        return self.best_fit, self.samples, self.log_probs

    # autocorrelation utilities
    def integrated_autocorr(self, c=5, tol=50, quiet=True):
        """
        Compute τ_int for each parameter on the post-burn-in chain.
        Returns
        -------
        tau : np.ndarray  shape (ndim,)
        """
        if self.chain is None:
            raise RuntimeError("Run MCMC first.")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tau = emcee.autocorr.integrated_time(
                    self.chain, c=c, tol=tol, quiet=quiet
                )
        except emcee.autocorr.AutocorrError:
            self.tau = np.full(self.chain.shape[-1], np.nan)
        return self.tau

    def autocorr_vs_N(self, n=20, c=5, tol=50):
        """
        Track ⟨\tau_int⟩ as the chain length N grows (Sokal-style diagnostic).
        Stores a curve for plotting.
        """
        if self.chain is None:
            raise RuntimeError("Run MCMC first.")

        n_steps = self.chain.shape[0]
        N_vals = np.linspace(10, n_steps, n, dtype=int)
        means = []

        for N in N_vals:
            try:
                tau = emcee.autocorr.integrated_time(
                    self.chain[:N], c=c, tol=tol, quiet=True
                )
                means.append(np.mean(tau))
            except emcee.autocorr.AutocorrError:
                means.append(np.nan)

        self._acorr_curve = (N_vals, np.asarray(means))
        return self._acorr_curve

    def plot_autocorr(self, bin_num, color='k', fig=None, ax=None):
        """
        Plot ⟨\tau_int⟩ against sample size; 
        call autocorr_vs_N() first.
        """
        if self._acorr_curve is None:
            raise RuntimeError("Call autocorr_vs_N() before plotting.")

        N, tau = self._acorr_curve
        if not fig or not ax: 
            fig, ax = plt.subplots(figsize=(10, 8)) 
            ax.plot(N, N / 50.0, "--k", label=r"$\tau=N/50$")

        ax.plot(N, tau, "o-", label=f"bin {bin_num:02d}", color=color)
        return fig, ax

    # plots 
    def corner_plot(self):
        if self.samples is None:
            raise RuntimeError("Run MCMC first.")

        return corner.corner(
            self.samples,
            labels=["u (logit f_RC)", "amp", "μ", "σ", "slope", "intercept"],
            show_titles=True,
        )

    def plot_fit(self):
        if self.best_fit is None:
            raise RuntimeError("Run MCMC first.")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.errorbar(
            self.bin_centers,
            self.bin_heights,
            yerr=self.bin_errors,
            fmt=".k",
            capsize=2,
            label="Histogram",
        )
        x = np.linspace(self.bin_edges[0], self.bin_edges[-1], 500)

        f   = self.best_fit["frac_RC"]
        u   = np.log(f / (1.0 - f))          # convert back to logit
        theta = [
            u,
            self.best_fit["amplitude"],
            self.best_fit["mean"],
            self.best_fit["stddev"],
            self.best_fit["slope"],
            self.best_fit["intercept"],
        ]
        y = self._compound_model(theta, x)

        ax.plot(x, y, "r-", label="Best-fit")
        ax.set_xlabel("Bin center", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.legend()

        fig.savefig(self.plot_dir / "best_fit.png", dpi=300)
        return fig, ax

