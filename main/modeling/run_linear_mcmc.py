import tqdm
import pickle 
import logging
import numpy as np 
import matplotlib.pyplot as plt 

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from scipy.stats import gaussian_kde 
from functools import cached_property 
from modeling.mcmc_jitter import LinearMCMC

plt.rcParams["font.family"]      = "serif" 
plt.rcParams['mathtext.fontset'] = 'cm'

@dataclass 
class RunLinearMCMCConfig: 
    ansatz_slope: tuple[tuple[float, float], tuple[float, float], float]
    output_dir: Path = Path(__file__).parent.parent / "media/MCMC_Linear" 
    data_dir: Path   = Path(__file__).parent.parent / "assets"
    boundary: Optional[tuple[float, float, float, float]] = None 
    top_fraction: float = 1.0 
    n_jitter: int       = 10

class RunLinearMCMC: 
    """ 
    Runs the Linear MCMC on a CMD RC ridge and performs a Monte Carlo Jitter 
    to estimate systematic uncertainty. 

    Meant for F115W - F323N vs. F115W CMDs.
    """ 

    def __init__(
        self, 
        filter1: str, 
        filter2: str, 
        filtery: str, 
        region: str, 
        config: RunLinearMCMCConfig
    ): 
        self.filter1 = filter1 
        self.filter2 = filter2 
        self.filtery = filtery 
        self.region  = region 

        self.data_dir = config.data_dir 
        self.out_dir  = config.output_dir 
        self.run_linear_config = config
        self.out_dir.mkdir(parents=True, exist_ok=True) 

        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s::%(name)s:: %(message)s", 
            force=True
        )

        self.logger = logging.getLogger(__name__)

    @cached_property 
    def load_red_clump_data(self): 
        """ 
        Load and cache the raw red clump .pickle from region
        """ 

        pkl = self.data_dir / f"{self.region}.pickle" 
        try: 
            with open(pkl, "rb") as f: 
                data = pickle.load(f) 
        except FileNotFoundError as e: 
            self.logger.error(f"Red-clump data not found at {pkl}")
            raise e 

        m1 = data[f"m{self.filter1}"] 
        m2 = data[f"m{self.filter2}"] 
        my = data[f"m{self.filtery}"] 
        my_err = data[f"me{self.filtery}"]

        x    = np.subtract(m1, m2) 
        y    = np.asarray(my) 
        yerr = np.asarray(my_err)

        return { 
            "m1": m1, 
            "m2": m2, 
            "my": my, 
            "x": x, 
            "y": y, 
            "ye": yerr
        }

    @staticmethod
    def _in_gap(x, y, m, b, gap): 
        # return mask of points within +/- gap 
        return np.abs(y - (m * x + b)) <= gap 


    def stars_in_gap(self): 
        ansatz_slope    = self.run_linear_config.ansatz_slope 
        red_clump_dict  = self.load_red_clump_data 
        self.x_all = red_clump_dict['x'] 
        self.y_all = red_clump_dict['y'] 
        self.ye_all= red_clump_dict['ye'] 

        (x1, y1), (x2, y2), self.gap = ansatz_slope  
        self.m0 = (y2 - y1) / (x2 - x1) 
        self.b0 = y1 - self.m0 * x1

        # midpoint of ansatz points 
        self.x_anchor = 0.5 * (x1 + x2) 
        self.y_anchor = 0.5 * (y1 + y2) 

        gap_mask = self._in_gap(
            self.x_all, 
            self.y_all, 
            self.m0, 
            self.b0, 
            self.gap
        )
        x_in, y_in, ye_in = (
            self.x_all[gap_mask], 
            self.y_all[gap_mask], 
            self.ye_all[gap_mask] 
        )

        if x_in.size < 10: 
            raise RuntimeError("Fewer than 10 stars within gap.")

        return x_in, y_in, ye_in

    def calculate_slope(self): 
        x_in, y_in, ye_in = self.stars_in_gap()

        self.logger.info(f"Running Linear MCMC ")
        self.logger.info(f"{self.region}: {self.filter1} - {self.filter2} vs. {self.filtery}") 

        # run mcmc 
        mcmc_linear = LinearMCMC(
            x_in, y_in, ye_in, 
            boundary=self.run_linear_config.boundary, 
            top_fraction=self.run_linear_config.top_fraction 
        )

        # medians 
        slope_med, intercept_med, slope_err_stat = mcmc_linear.run()

        self.logger.info(f"Calculated slope: {slope_med:.3f} +/- {slope_err_stat:.3f} (stat)\n")
        
        return slope_med, intercept_med, slope_err_stat 

    def jitter(self): 
        n_jitter = self.run_linear_config.n_jitter 

        rng = np.random.default_rng(42)
        jitter_slopes = [] 

        self.logger.info(f"Starting {n_jitter} MCMC jitters ") 
        for _ in tqdm.tqdm(range(n_jitter)): 
            # 10% jitter 
            m_jitter = self.m0 * (1.0 + 0.10 * rng.normal())
            b_jitter = self.y_anchor - m_jitter * self.x_anchor 

            mask_jitter = self._in_gap(
                self.x_all, 
                self.y_all, 
                m_jitter, 
                b_jitter, 
                self.gap 
            )

            if mask_jitter.sum() < 10: 
                continue # skip, too sparse 

            x_j, y_j, ye_j = (
                self.x_all[mask_jitter],
                self.y_all[mask_jitter], 
                self.ye_all[mask_jitter]
            )

            fit_jitter = LinearMCMC(
                x_j, y_j, ye_j, 
                boundary=self.run_linear_config.boundary, 
                top_fraction=self.run_linear_config.top_fraction, 
            )
            m_jitter, _, _ = fit_jitter.run()
            jitter_slopes.append(m_jitter) 

        jitter_slopes = np.array(jitter_slopes) 
        slope_systematic = jitter_slopes.std(ddof=1) if jitter_slopes.size > 1 else 0.0 

        self.logger.info(f"Systematic Uncertainty: {slope_systematic}")

        return slope_systematic

    def run(self): 
        slope_med, intercept_med, slope_err_statistical = self.calculate_slope()
        slope_systematic = self.jitter() 
        total_err = np.hypot(slope_err_statistical, slope_systematic)

        self.logger.info(f"{self.region} {self.filter1}-{self.filter2} vs {self.filtery}")
        self.logger.info(f"slope  = {slope_med:.4f} +/- {slope_err_statistical:.4f} (statistical)")
        self.logger.info(f"syst   = ±{slope_systematic:.4f} (systematic)\n")
        self.logger.info(f"[FINAL]  = {slope_med:.4f} +/- {total_err:.4f}") 

        self.slope = slope_med 
        self.error = total_err 
        self.incpt = intercept_med 
        return self.slope, self.error 

    def plot_best_fit(self, export=True):
        if not hasattr(self, "x_all"): 
            self.run()

        x = self.x_all 
        y = self.y_all

        finite_mask = np.isfinite(x) & np.isfinite(y) 
        x = x[finite_mask] 
        y = y[finite_mask]

        # plot by density 
        xy = np.vstack([x, y])
        color = gaussian_kde(xy)(xy) 

        figure, axis = plt.subplots(1, 1, figsize=(8, 6)) 
        axis.scatter(x, y, c=color, cmap='magma', alpha=0.6, s=40, marker='d') 
        axis.axline(slope=self.slope, xy1=(0, self.incpt), c='cyan')

        axis.text(
            0.02, 0.02,
            f"slope = {self.slope:.3f} +/- {self.error:.3f}",
            transform=axis.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='left',
        )

        axis.set_xlabel(f"{self.region} {self.filter1} - {self.filter2} vs. {self.filtery}", fontsize=15)
        axis.set_ylabel(f"{self.region} {self.filtery}", fontsize=15) 
        axis.set_xlim(x.min()-0.2, x.max()+0.2)
        axis.set_ylim(y.min()-0.2, y.max()+0.2)
        axis.invert_yaxis()
        plt.tight_layout()

        if export: 
            out_dir  = self.run_linear_config.output_dir 
            filename = out_dir / f"{self.region}_{self.filter1}-{self.filter2}_{self.filtery}.png" 
            figure.savefig(filename, dpi=300) 
            self.logger.info(f"Linear MCMC Figure saved to {Path(*filename.parts[-3:])}")
        else: 
            plt.show()

if __name__ == "__main__": 
    filter1 = "F115W" 
    filter2 = "F323N" 
    filtery = "F115W" 
    region  = "NRCB1" 

    config = RunLinearMCMCConfig(
        ansatz_slope=((7.3, 21.4), (8, 22.5), 0.4), 
        top_fraction=1, 
        n_jitter=10, 
        boundary=None
    )

    instance = RunLinearMCMC(
        filter1=filter1, 
        filter2=filter2, 
        filtery=filtery, 
        region=region,
        config=config
    )

    instance.plot_best_fit()
