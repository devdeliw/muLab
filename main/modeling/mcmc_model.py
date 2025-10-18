import emcee 
import corner
import numpy as np 
import matplotlib.pyplot as plt 
from dataclasses import dataclass

plt.rcParams["font.family"]      = "serif" 
plt.rcParams['mathtext.fontset'] = 'cm'

def _sigmoid(x): 
    # stable logistic function 
    return np.where(
        x >= 0, 
        1.0 / (1.0 + np.exp(-x)), 
        np.exp(x) / (1.0 + np.exp(x)) 
    )

@dataclass 
class MCMCConfig: 
    nwalkers: int = 64 
    nsteps:   int = 15000
    burnin:   int = 1000 
    thin:     int = 1

class MCMC:
    """
    Emcee sampler for a compound Gaussian+Linear1D histogram model. 

    theta = (u, amplitude, mu, sigma, m, b) 
    f_RC = sigmoid(u) in (0, 1) 
    """

    def __init__(
        self, 
        data,
        bins: int = 50, 
        config: MCMCConfig = MCMCConfig()
    ): 
        self.data = np.asarray(data) 
        self.bins = bins 
        self.config = config 

        self.bin_heights, self.bin_edges = np.histogram(self.data, bins=bins) 
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:]) 
        self.bin_errors  = np.sqrt(self.bin_heights + 1) 

        self.samples                = None 
        self.best_fit_parameters    = None 
        self.log_probabilities      = None 

        # autocorrelation 
        self.tau = None 
        self._acorr_curve = None 

    def _initial_guess(self): 
        u0      = np.log(0.1/0.9) 
        amp0    = self.bin_heights.max() / 4 
        mu0     = np.mean(self.data) 
        sig0    = np.std(self.data) 
        m0      = 0.0 
        b0      = self.bin_heights.min() if self.bin_heights.min() > 0 else 1.0 
        return [u0, amp0, mu0, sig0, m0, b0]

    @staticmethod 
    def _compound_model(theta, x): 
        u, amp, mu, sig, m, b = theta 
        f_rc = _sigmoid(u)
        f_rc = np.clip(f_rc, 1e-6, 1 - 1e-6)

        # compound gaussian + linear
        gaussian = amp * np.exp(-0.5 * ((x-mu) / sig) ** 2) 
        linear   = m * x + b 
        return gaussian + (1.0 - f_rc) * linear

    def integrated_autocorr(self, c=5, tol=50, quiet=True): 
        if self.chain is None: 
            raise RuntimeError("Run MCMC first.") 
        try:
            import warnings 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tau = emcee.autocorr.integrated_time(
                    self.chain, c=c, tol=tol, quiet=quiet
                )
        except emcee.autocorr.AutocorrError:
            self.tau = np.full(self.chain.shape[-1], np.nan)
        return self.tau

    def _autocorr_vs_N(self, n=200, c=5, tol=50, quiet=True): 
        # track tau as chain length N grows. 
        # stores curve for plotting. 
        if self.chain is None: 
            raise RuntimeError("Run the MCMC first.") 

        n_steps = self.chain.shape[0] 
        N_vals = np.linspace(10, n_steps, n, dtype=int)
        means = [] 

        for N in N_vals: 
            try: 
                tau = emcee.autocorr.integrated_time(
                    self.chain[:N], c=c, tol=tol, quiet=quiet
                )
                means.append(np.mean(tau)) 
            except emcee.autocorr.AutocorrError: 
                means.append(np.nan) 

        self._acorr_curve = (N_vals, np.asarray(means)) 
        return self._acorr_curve 

    def plot_autocorr(self, color='k', figure=None, axis=None): 
        # plots tau against sample size 
        if self._acorr_curve is None: 
            raise RuntimeError("Call self._autocorr_vs_N first.") 

        N, tau = self._acorr_curve 
        if not figure or not axis: 
            figure, axis = plt.subplots(1, 1, figsize=(6, 3)) 
            # heuristic 
            axis.plot(N, N/50.0, color="black", linestyle=":", label=r"$\tau=N/50$", alpha=0.5) 

        axis.plot(N, tau, markersize=0, linewidth=0.8, linestyle='-', color=color)
        return figure, axis 

    def _log_prior(self, theta): 
        u, amp, mu, sig, m, b = theta 
        mu0, s0 = np.mean(self.data), np.std(self.data) 

        alpha, beta = 3.0, 2.0 
        f_rc = _sigmoid(u) 
        f_rc = np.clip(f_rc, 1e-6, 1 - 1e-6)

        ln_prior    = (alpha-1)*np.log(f_rc) + (beta-1)*np.log(1-f_rc) 
        ln_prior   += np.log(f_rc) + np.log(1-f_rc) # jacobian 

        if not np.isfinite(ln_prior): 
            return -np.inf 
        if not (0 < amp < 1000): 
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
        ivar  = 1.0 / self.bin_errors ** 2 
        return -0.5 * np.sum(resid**2 * ivar + np.log(2*np.pi / ivar))

    def _log_probability(self, theta): 
        lp = self._log_prior(theta) 
        return -np.inf if not np.isfinite(lp) else lp + self._log_likelihood(theta) 

    def run(self, progress=True): 
        nwalkers = self.config.nwalkers 
        nsteps   = self.config.nsteps 
        burnin   = self.config.burnin 
        thin     = self.config.thin 

        p0 = self._initial_guess() 
        ndim = len(p0) 
        # initialize walker positions 
        pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim) 

        # run the mcmc 
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_probability) 
        sampler.run_mcmc(pos, nsteps, progress=progress)

        self.chain     = sampler.get_chain(discard=burnin, flat=False) 
        self.log_probs = sampler.get_log_prob(discard=0, flat=False) 
        self.samples   = sampler.get_chain(discard=burnin, thin=thin, flat=True) 

        # medians
        u_m, amp_m, mu_m, sig_m, m_m, b_m = np.median(self.samples, axis=0) # type: ignore 
        f_m = _sigmoid(u_m) 

        self.best_fit_parameters = {
            "frac_RC"   : f_m, 
            "amplitude" : amp_m, 
            "mean"      : mu_m, 
            "stddev"    : sig_m, 
            "slope"     : m_m, 
            "intercept" : b_m, 
        }
        return self.best_fit_parameters, self.samples, self.log_probs 

    def plot_corner(self): 
        # corner plot 
        if self.samples is None: 
            raise RuntimeError("Run the MCMC first.") 
        return corner.corner(
            self.samples, 
            labels=["u", "amp", "mu", "sigma", "slope", "int"], 
            show_titles=True, 
            title_kwargs={"fontsize": 14}, 
            label_kwargs={"fontsize": 18},
        )

    def plot_best_fit(self):
        if self.best_fit_parameters is None: 
            raise RuntimeError("Run the MCMC first.") 

        figure, axis = plt.subplots(1, 1, figsize=(8, 6)) 
        axis.errorbar(
            self.bin_centers, self.bin_heights, 
            yerr=self.bin_errors, fmt=".k", capsize=2
        )

        bf = self.best_fit_parameters
        u   = np.log(bf["frac_RC"] / (1.0 - bf["frac_RC"]))
        theta = [
            u, 
            bf["amplitude"], 
            bf["mean"], 
            bf["stddev"],
            bf["slope"], 
            bf["intercept"]
        ]

        x = np.linspace(self.bin_edges[0], self.bin_edges[-1], 500)
        axis.plot(x, self._compound_model(theta, x), "r-")
        axis.set_xlabel("Bin center", fontsize=15)
        axis.set_ylabel("Frequency", fontsize=15)
        return figure, axis





















