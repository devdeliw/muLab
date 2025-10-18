import emcee 
import corner 
import numpy as np 
import matplotlib.pyplot as plt 


class MCMC(): 
    """ 
    Runs a Markov Chain Monte Carlo (MCMC) to fit a compound Gaussian+Linear1D 
    model to a distribution of stars. 

    Implemented on individual bins of an red clump distribution from `Cells`. 

    Args: 
        * data (array-like)   : 1D data samples.
        * bins (int/sequence) : If int, # bins for np.hist, If sequence, the bin edges.

    run() returns the best fit compound model parameters for the given distribution.

    """

    def __init__(self, data, bins=50): 
        self.data = np.asarray(data) 
        self.bins = bins 

        self.bin_heights, self.bin_edges = np.histogram(self.data, bins=self.bins) 
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:]) 
        self.bin_errors  = np.sqrt(self.bin_heights + 1) 

        self.samples = None 
        self.best_fit_params = None 

    @staticmethod 
    def _compound_model_(theta, x): 
        amplitude, mean, sigma, slope, intercept = theta            # model params  
        gaussian = amplitude * np.exp(-0.5 * ((x-mean)/sigma) ** 2) # gauss part 
        linear   = slope * x + intercept                            # linear part 
        return gaussian + linear # compound model 

    def _log_prior_(self, theta): 
        amplitude, mean, sigma, slope, intercept = theta 

        # some log priors 
        # model mean can't be too far from the intrinsic mean 
        # model stddev can't be too far from the intrinsic stddev 
        # model amplitude can't be larger than size of data itself 
        intrinsic_mean = np.mean(self.data) 
        intrinsic_std  = np.std(self.data) 
        allowed_means  = [intrinsic_mean-2., intrinsic_mean+2.] 
        allowed_stds   = [max(0, intrinsic_std-0.5), min(intrinsic_std+0.5, 1e5)]

        if  not (0 < amplitude < 1000) or \
            not (allowed_stds[0] < sigma < allowed_stds[1]) or \
            not (allowed_means[0] < mean < allowed_means[1]) or \
            not (slope < 30) or \
            not (intercept < 50):
            return -np.inf
        return 0.0 

    def _log_likelihood_(self, theta): 
        model_vals = self._compound_model_(theta, self.bin_centers) 
        residual   = self.bin_heights - model_vals 
        inv_var    = 1.0 / (self.bin_errors ** 2) 
        return -0.5 * np.sum(residual**2 * inv_var + np.log(2.0 * np.pi / inv_var)) 

    def _log_probability_(self, theta): 
        lp = self._log_prior_(theta) 
        if not np.isfinite(lp): 
            return -np.inf 
        return lp + self._log_likelihood_(theta) 

    def _initial_guess_(self): 
        amp_guess   = np.max(self.bin_heights)/4  
        mean_guess  = np.mean(self.data) 
        sigma_guess = np.std(self.data) 
        slope_guess = 0.0 
        inter_guess = np.min(self.bin_heights) if np.min(self.bin_heights) > 0 else 1.
        return [amp_guess, mean_guess, sigma_guess, slope_guess, inter_guess]

    def run(self, nwalkers=64, nsteps=1500, burnin=300, thin=10): 
        """
        Runs the emcee MCMC to fit the histogram distribution 
        with a compound Gaussian+Linear1D model . 

        Args: 
            * nwalkers (int): number of MCMC walkers. 
            * nsteps   (int): total steps in chain.
            * burnin   (int): steps to discard as burn-in. 
            * thin     (int): thinning factor to reduce autocorrelation 

        Returns: 
            * best_fit_params (dict): contains median optimized parameter values
            * samples      (ndarray): 2D array (n_samples, 5) with posterior samples 
                                      after burn-in and thinning. 

        """ 

        init_guess = self._initial_guess_() 
        ndim = len(init_guess) 

        # walker starting positions
        pos = init_guess + 1e-4 * np.random.randn(nwalkers, ndim) 

        # emcee sampler 
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_probability_)  
        sampler.run_mcmc(pos, nsteps, progress=False) 
        
        # log  probability for all walkers at every step 
        self.log_probs = sampler.get_log_prob(discard=0, flat=False) 
        self.samples   = sampler.get_chain(discard=burnin, thin=thin, flat=True) 

        amp_m, mean_m, sigma_m, slope_m, intercept_m = np.median(self.samples, axis=0)
        self.best_fit_params = {
            "amplitude": amp_m,
            "mean": mean_m,
            "stddev": sigma_m,
            "slope": slope_m,
            "intercept": intercept_m
        }
        return self.best_fit_params, self.samples, self.log_probs 

    def corner_plot(self): 
        if self.samples is None: 
            raise RuntimeError("[ERROR] No samples found. Run the MCMC first.") 
        
        fig = corner.corner(
            self.samples, 
            labels=["amplitude", "mean", "stddev", "slope", "intercept"], 
            show_titles=True, 
            label_kwargs={"fontsize": 24} 
        ) 

        return fig 

    def plot_fit(self): 
        if self.best_fit_params is None: 
            raise RuntimeError("[ERROR] No best-fit parameters found. Run the MCMC first.")

        _, _ = plt.subplots(1, 1, figsize=(8, 5)) 
        plt.errorbar(self.bin_centers, self.bin_heights, 
                     yerr=self.bin_errors, fmt=".k", capsize=2, label="Histogram Data")

        bf = self.best_fit_params
        x = np.linspace(self.bin_edges[0], self.bin_edges[-1], 500)
        params = [bf["amplitude"], bf["mean"], bf["stddev"], bf["slope"], bf["intercept"]]
        model_vals = self._compound_model_(params, x)

        plt.plot(x, model_vals, "r-") 

        plt.xlabel("Bin Center")
        plt.ylabel("Frequency")
        plt.title("Compound Fit") 
        plt.legend() 
        return plt 









        

        






