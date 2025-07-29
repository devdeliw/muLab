import logging 
import numpy as np 
import matplotlib.pyplot as plt 

from pathlib import Path 
from dataclasses import dataclass
from modeling.mcmc_model import MCMC, MCMCConfig
from modeling.generating_tiles import GenerateTiles, TileConfig

plt.rcParams["font.family"]      = "serif" 
plt.rcParams['mathtext.fontset'] = 'cm'

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s"
)

@dataclass
class RunMCMCConfig:
    n_tiles:        int     = 10
    max_y_err:      float   = 0.2 
    autocorr:       bool    = False 
    autocorr_bin:   int     = 5 
    slope_minus:    float   = 0.5
    output_dir:     Path    = Path(__file__).parent.parent / "media/MCMC" 

    histogram_bins: int = 50 

    nwalkers: int = 64 
    nsteps:   int = 15000 
    burnin:   int = 1000 
    thin:     int = 1 

class RunMCMC(GenerateTiles): 
    """ 
    Estimate the RC ridge slope by fitting Gaussian+Linear1D models. 
    """

    def __init__(
        self, 
        filter1: str, 
        filter2: str, 
        filtery: str, 
        region1: str, 
        region2: str, 
        regiony: str, 
        load_pkl: bool        = True, 
        config: RunMCMCConfig = RunMCMCConfig()
    ):
        """ 
        Args: 
            * filter1, region1 (str): name of filter & region for lower wavelength
            * filter2, region2 (str): name of filter & region for higher wavelength
            * filtery, regiony (str): name of filter & region on y-axis 
            * load_pkl (bool): load the stored star tiles from pickle? 
            * config   (RunMCMCConfig): configuration dataclass 
        """

        self.filter1 = filter1 
        self.filter2 = filter2 
        self.filtery = filtery 
        self.region1 = region1 
        self.region2 = region2 
        self.regiony = regiony
        self.mcmc_config  = config

        n_tiles     = config.n_tiles 
        slope_minus = config.slope_minus
        self.tileconfig = TileConfig(n_tiles=n_tiles, slope_minus=slope_minus)
        self.mcmcconfig = MCMCConfig(
            nwalkers=config.nwalkers, 
            nsteps=config.nsteps, 
            burnin=config.burnin, 
            thin=config.thin, 
        )             

        super().__init__(
            filter1, 
            filter2, 
            filtery, 
            region1, 
            region2, 
            regiony, 
            self.tileconfig
        )
        self.star_tiles  = self._load_star_tiles(load_pkl=load_pkl)

        theta = np.arctan(self.slope)                   # fritz ansatz 
        cos, sin = np.cos(theta), np.sin(theta) 
        self._rot = np.array([[cos, sin], [-sin, cos]]) # rotation matrix 
        self._inv_rot = self._rot.T                     # inverse rotation matrix 

        self.points = [] 
        self.errors = [] 
        self.fracs  = [] 
        self.amps   = [] 
        self.sigmas = [] 

        self.autocorr       = config.autocorr 
        self.autocorr_bin   = config.autocorr_bin
        self.best_parameters = [] 

        self.logger = logging.getLogger(__name__) 


    def _load_star_tiles(self, load_pkl: bool):
        if load_pkl: 
            import pickle 
            try: 
                out_dir = self.tileconfig.out_dir 
                filename = out_dir / f"{self.filter1}-{self.filter2}_{self.filtery}.pickle" 

                with open(filename, "rb") as f: 
                    star_tiles = pickle.load(f) 
                return star_tiles 
            except: 
                pass 
        return self.tiles(export=False)

    def _rotate_stars(self, stars: np.ndarray): 
        rotated_stars = stars @ self._inv_rot 
        x_rot, y_rot = rotated_stars.T 

        return x_rot, y_rot 

    def run_mcmc_on_stars(self, stars: np.ndarray): 
        x_rot, y_rot = self._rotate_stars(stars)

        self.mcmc = MCMC(
            data=y_rot,
            config=self.mcmcconfig,
            bins=self.mcmc_config.histogram_bins
        )
        best_fit, samples, _ = self.mcmc.run() 

        mean_y = best_fit['mean']
        amp    = best_fit['amplitude']
        sigma  = best_fit['stddev']
        f_rc   = best_fit.get('frac_RC', 1.0)
        sigma_y = np.std(samples[:, 2], ddof=1)
        self.best_parameters.append(best_fit) 

        return x_rot, y_rot, mean_y, amp, sigma, f_rc, sigma_y 

    def analyze_bin(self, stars: np.ndarray, color: str): 
        x_rot, _, mean_y, amp, sigma, f_rc, sigma_y = self.run_mcmc_on_stars(stars)

        # median x in bar frame 
        x_center  = np.median(x_rot) 
        repr_rot  = np.array([x_center, mean_y]) 
        repr_orig = repr_rot @ self._rot 

        # error prop 
        sigma_vec = np.array([0.0, sigma_y]) @ self._rot 
        y_error   = abs(sigma_vec[1])  

        if y_error <= self.mcmc_config.max_y_err: 
            if self.autocorr: 
                self.tau = self.mcmc.integrated_autocorr() 
                self.mcmc._autocorr_vs_N() 

                if not hasattr(self, "fig_autocorr"): 
                    self.fig_autocorr = None 
                    self.ax_autocorr  = None 

                self.fig_autocorr, self.ax_autocorr = self.mcmc.plot_autocorr( 
                    figure=self.fig_autocorr, 
                    axis=self.ax_autocorr, 
                    color=color 
                ) 
        return repr_orig, y_error, f_rc, amp, sigma 

    def run(self): 
        # Iterate through all tiles and run mcmc. 

        self.logger.info(f"Running MCMC with {self.config.n_tiles} tiles.")
        self.logger.info(f"{self.region1}: {self.filter1} - {self.filter2} vs. {self.filtery}") 

        colors = plt.cm.cool(np.linspace(0.2, 0.8, len(self.star_tiles))) # type: ignore 

        for idx, stars in enumerate(self.star_tiles): 
            if stars.size == 0: 
                continue 

            point, y_error, f_rc, amp, sigma = self.analyze_bin(stars, color=colors[idx])
            if (y_error <= self.mcmc_config.max_y_err) and (f_rc > 0.01): 
                self.points.append(point)   # (tile_center, mean_gaussian)
                self.errors.append(y_error) # error 
                self.fracs.append(f_rc)     # fraction of RC in tile 
                self.amps.append(amp)       # amplitude of gaussian 
                self.sigmas.append(sigma)   # stddev of gaussian

        pts = np.vstack(self.points) 
        xs, ys = pts[:, 0], pts[:, 1] 

        area = np.sqrt(2*np.pi) * np.asarray(self.sigmas) * np.asarray(self.amps) 
        A_norm = area / area.max() 

        weights = A_norm * (np.asarray(self.fracs) / np.square(self.errors)) 

        (self.slope, self.intercept), cov = np.polyfit(xs, ys, 1, w=weights, cov=True) 
        self.slope_err = np.sqrt(cov[0, 0])

        if self.autocorr: 
            self.ax_autocorr.set_xlabel(r"Samples $N$", fontsize=15) 
            self.ax_autocorr.set_ylabel(r"Mean $\hat{\tau}_{\mathrm{int}}$", fontsize=15)
            self.ax_autocorr.legend(prop={"size": 10}, frameon=False) 
           
            out_dir = self.mcmc_config.output_dir / "autocorr" 
            out_dir.mkdir(exist_ok=True, parents=True)
            filename = out_dir / f"{self.config.n_tiles}_tiles_{self.region1}_{self.filter1}-{self.filter2}_{self.filtery}.png" 
            self.fig_autocorr.savefig(filename, dpi=300)
            self.logger.info(f"Autocorrelation Figure saved to {Path(*filename.parts[-4:])}.")

        self.logger.info(f" [FINAL] slope={self.slope:.4f} +/- {self.slope_err:.4f}") 
        return self.slope, self.intercept 

    def plot_fit(self, export=True): 
        if not self.points: 
            raise RuntimeError("No points to plot; Call self.run() first.") 

        figure, axis = plt.subplots(1, 1, figsize=(8, 6)) 

        colors = plt.cm.cool(np.linspace(0, 1, len(self.star_tiles))) 
        for idx, stars in enumerate(self.star_tiles): 
            if stars.size: 
                axis.scatter(
                    stars[:, 0], stars[:, 1], 
                    marker='d', color=colors[idx], 
                    alpha=0.5, s=20
                )

        # representative tile points 
        pts = np.vstack(self.points) # (tile_center, mean_y) 
        axis.errorbar(
            pts[:, 0], pts[:, 1], yerr=self.errors, color='k', 
            fmt='h', markersize=6, capsize=4, zorder=3 
        )

        # best-fit line 
        x_vals = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 400)
        y_vals = self.slope * x_vals + self.intercept 
        axis.plot(x_vals, y_vals, c='k', lw=1, linestyle=':', zorder=2) 

        axis.set_xlabel(f"{self.region1} {self.filter1} - {self.filter2} (mag)", fontsize=15) 
        axis.set_ylabel(f"{self.region1} {self.filtery}", fontsize=15)

        axis.text(
            0.02, 0.02,
            f"slope = {self.slope:.3f} +/- {self.slope_err:.3f}",
            transform=axis.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='left',
        )

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False) 
        axis.invert_yaxis() 
        plt.tight_layout() 

        if export: 
            out_dir = self.mcmc_config.output_dir / "Slope" 
            out_dir.mkdir(exist_ok=True, parents=True)
            filename = out_dir / f"{self.config.n_tiles}_tiles_{self.region1}_{self.filter1}-{self.filter2}_{self.filtery}.png"
            figure.savefig(filename, dpi=300) 
            self.logger.info(f"Slope Figure saved to {Path(*filename.parts[-4:])}")
        else: 
            plt.show() 



if __name__ == "__main__": 
    filter1, region1 = "F115W", "NRCB1" 
    filter2, region2 = "F212N", "NRCB1" 
    filtery, regiony = filter1, region1 

    instance = RunMCMC(
        filter1=filter1, 
        filter2=filter2, 
        filtery=filtery, 
        region1=region1, 
        region2=region2, 
        regiony=regiony, 
        config=RunMCMCConfig(autocorr=True)
    )

    slope, intercept = instance.run()
    #instance.plot_fit(export=True)
    







        




    
        
        
        
                




        
    

