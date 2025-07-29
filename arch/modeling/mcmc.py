import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt


def _sigmoid(x):
    """ Numerically stable logistic function. """
    return np.where(
        x >= 0, 
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )

class MCMC:
    """
    emcee sampler for a compound Gaussian+Linear1D histogram model.

    theta = = (u, amp, mu, sigma, m, b)
    f_RC = sigmoid(u) in (0, 1)

    """

    @staticmethod
    def _compound_model(theta, x):
        u, amp, mu, sig, m, b = theta
        f_rc = _sigmoid(u)

        if f_rc == 1.0: 
            f_rc -= 0.01

        gauss  = amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)
        linear = m * x + b
        return gauss + (1.0 - f_rc) * linear

    def _log_prior(self, theta):
        u, amp, mu, sig, m, b = theta
        mu0, s0 = np.mean(self.data), np.std(self.data)

        # logit-space Beta(alpha, beta) prior on f_RC w/ Jacobian
        alpha, beta = 3.0, 2.0  
        f_rc = _sigmoid(u)

        if f_rc == 1.0: 
            f_rc -= 0.01

        ln_prior  = (alpha-1)*np.log(f_rc) + (beta-1)*np.log(1-f_rc)
        ln_prior += np.log(f_rc) + np.log(1-f_rc)   # jacobian 

        if not np.isfinite(ln_prior):
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
        ivar  = 1.0 / self.bin_errors**2
        return -0.5 * np.sum(resid**2 * ivar + np.log(2 * np.pi / ivar))

    def _log_probability(self, theta):
        lp = self._log_prior(theta)
        return -np.inf if not np.isfinite(lp) else lp + self._log_likelihood(theta)

    def __init__(self, data, bins=50):
        self.data = np.asarray(data)
        self.bins = bins

        self.bin_heights, self.bin_edges = np.histogram(self.data, bins=bins)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_errors  = np.sqrt(self.bin_heights + 1)

        self.samples = None
        self.best_fit_params = None
        self.log_probs = None

    def _initial_guess(self):
        u0        = np.log(0.1/0.9)  # logit
        amp0      = self.bin_heights.max() / 4
        mu0       = np.mean(self.data)
        sig0      = np.std(self.data)
        m0        = 0.0
        b0        = self.bin_heights.min() if self.bin_heights.min() > 0 else 1.0
        return [u0, amp0, mu0, sig0, m0, b0]

    def run(self, nwalkers=64, nsteps=5_000, burnin=1_000, thin=1):
        p0   = self._initial_guess()
        ndim = len(p0)
        pos  = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_probability)
        sampler.run_mcmc(pos, nsteps, progress=False)

        self.log_probs = sampler.get_log_prob(discard=0, flat=False)
        self.samples   = sampler.get_chain(discard=burnin, thin=thin, flat=True)

        # median parameters
        u_m, amp_m, mu_m, sig_m, m_m, b_m = np.median(self.samples, axis=0)
        f_m = _sigmoid(u_m)

        self.best_fit_params = {
            "frac_RC":   f_m,
            "amplitude": amp_m,
            "mean":      mu_m,
            "stddev":    sig_m,
            "slope":     m_m,
            "intercept": b_m,
        } 

        import pickle
        pkl = "./best_params.pickle"
        with open(pkl, "wb") as f: 
            pickle.dump(self.best_fit_params, f)
        return self.best_fit_params, self.samples, self.log_probs

    def corner_plot(self):
        if self.samples is None:
            raise RuntimeError("Run the MCMC first.")
        return corner.corner(
            self.samples,
            labels=["u (logit f_RC)", "amp", "μ", "σ", "slope", "intercept"],
            show_titles=True,
            title_kwargs={"fontsize": 14},
            label_kwargs={"fontsize": 18},
        )

    def plot_fit(self):
        if self.best_fit_params is None:
            raise RuntimeError("Run the MCMC first.")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(self.bin_centers, self.bin_heights,
                    yerr=self.bin_errors, fmt=".k", capsize=2)

        bf = self.best_fit_params
        u   = np.log(bf["frac_RC"] / (1.0 - bf["frac_RC"]))
        theta = [u, bf["amplitude"], bf["mean"], bf["stddev"],
                 bf["slope"], bf["intercept"]]

        x = np.linspace(self.bin_edges[0], self.bin_edges[-1], 500)
        ax.plot(x, self._compound_model(theta, x), "r-")
        ax.set_xlabel("Bin centre")
        ax.set_ylabel("Count")
        ax.set_title("Compound Gaussian + Linear fit")
        return fig, ax

