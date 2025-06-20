import warnings
import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee

class MCMC_Autocorr:
    """
    Fit a 1-D histogram with a compound      Gaussian + Linear model via emcee and
    provide basic convergence diagnostics (integrated autocorrelation time τ_int).
    """

    @staticmethod
    def _compound_model(theta, x):
        amp, mu, sig, m, b = theta
        return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2) + m * x + b

    def _log_prior(self, theta):
        amp, mu, sig, m, b = theta
        mu0, s0 = np.mean(self.data), np.std(self.data)
        if not (0 < amp < 1_000):
            return -np.inf
        if not (mu0 - 2.0 < mu < mu0 + 2.0):
            return -np.inf
        if not (max(0.0, s0 - 0.5) < sig < s0 + 0.5):
            return -np.inf
        if m > 30 or b > 50:
            return -np.inf
        return 0.0

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
            h.max() / 4,
            np.mean(self.data),
            np.std(self.data),
            0.0,
            (h.min() if h.min() > 0 else 1.0),
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
        self.tau = None              # τ_int per parameter
        self._acorr_curve = None     # (N, τ̄) pairs for plotting 

    def run(self, nwalkers=64, nsteps=8_000, burnin=1_000, thin=10):
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
        self.samples = self.sampler.get_chain(
            discard=burnin, thin=thin, flat=True
        )
        self.log_probs = self.sampler.get_log_prob(
            discard=0, flat=False
        ) 
        amp, mu, sig, m, b = np.median(self.samples, axis=0)
        self.best_fit = dict(
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
        Track ⟨τ_int⟩ as the chain length N grows (Sokal-style diagnostic).
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
        Plot ⟨τ_int⟩ against sample size; call autocorr_vs_N() first.
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
            labels=["amp", "μ", "σ", "slope", "intercept"],
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
        y = self._compound_model(list(self.best_fit.values()), x)
        ax.plot(x, y, "r-", label="Best-fit")
        ax.set_xlabel("Bin center", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.legend()

        fig.savefig(self.plot_dir / "best_fit.png", dpi=300)
        return fig, ax

