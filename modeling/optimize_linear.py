#!/usr/bin/env python3 

import logging 
from os import wait
from pathlib import Path 

import pickle
import numpy as np 
import matplotlib.pyplot as plt 
from functools import cached_property
from scipy.stats import gaussian_kde

from modeling.mcmc_linear import LinearBayes

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
 

class LinearEstimator: 

    def __init__(
        self, 
        filt1, 
        filt2, 
        filty, 
        region, 
        out_dir = Path("./outputs/MCMC"), 
        data_dir= Path("./outputs/red_clump_data"), 
        verbose = True, 
        top_fraction = 1.0, 
    ): 
        self.filt1 = filt1 
        self.filt2 = filt2 
        self.filty = filty 
        self.region = region  
        self.data_dir = data_dir

        self.top_fraction = top_fraction
        self.out_dir = out_dir.expanduser() 
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def filter_top_density(self, x: np.ndarray, y: np.ndarray, top_fraction: float = 0.1):
        """
        Return the subset of (x, y) points that lie in the top `top_fraction`
        most-dense regions of the scatter.

        """

        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)                 
        dens = kde(xy)                         

        cutoff = np.quantile(dens, 1 - top_fraction)

        mask = dens >= cutoff
        return x[mask], y[mask]

    @cached_property
    def red_clump_data(self) -> dict[str, np.ndarray]:
        """
        Load and cache the raw red-clump pickle for reg1.

        """
        pkl = self.data_dir / f"{self.region}.pkl"
        try:
            with pkl.open("rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Red-clump data not found at {pkl!r}")
            raise

        # extract mags
        m1 = data[f"m{self.filt1}"]
        m2 = data[f"m{self.filt2}"]
        my = data[f"m{self.filty}"]
    
        mye= data[f"me{self.filty}"]

        # mask out non-finite values
        mask = np.isfinite(m1) & np.isfinite(m2)

        x = np.subtract(m1[mask], m2[mask]) 
        y = np.array(my[mask])
        ye= np.array(mye[mask])
        
        return {
            "m1": m1[mask],
            "m2": m2[mask],
            "my": my[mask],
            "x": x,
            "y": y,
            "ye": ye, 
        }

    def slope(self, box=None, slopes=None, n_jitter=10): 
        if slopes is None: 
            raise ValueError("need slopes=[(x1, y1), (x2, y2), gap]")

        red_clump_dict = self.red_clump_data
        x_all = np.asarray(red_clump_dict["x"])
        y_all = np.asarray(red_clump_dict["y"])
        ye_all = np.asarray(red_clump_dict["ye"])

        if box is not None:
            xmin, xmax, ymin, ymax = box
            m = (
                (x_all >= xmin) & (x_all <= xmax) &
                (y_all >= ymin) & (y_all <= ymax)
            )
            x_all, y_all, ye_all = x_all[m], y_all[m], ye_all[m]

        (x1, y1), (x2, y2), gap = slopes
        m0 = (y2 - y1) / (x2 - x1)
        b0 = y1 - m0 * x1

        # anchor = midpoint of hand‑drawn points
        x_anchor = 0.5 * (x1 + x2)
        y_anchor = 0.5 * (y1 + y2)

        def in_band(x, y, m, b):
            # Return mask of points within ±ga residual
            return np.abs(y - (m * x + b)) <= gap

        base_mask = in_band(x_all, y_all, m0, b0)
        x_in, y_in, ye_in = x_all[base_mask], y_all[base_mask], ye_all[base_mask]

        if x_in.size < 10:
            raise RuntimeError("Fewer than 10 stars in the base band; check inputs.")

        fit_base = LinearBayes(x_in, y_in, ye_in,
                               box=None, top_fraction=self.top_fraction)
        slope_med, intercept_med, slope_err_stat = fit_base.run()

        rng = np.random.default_rng(42)
        jitter_slopes = []
        for _ in range(n_jitter):
            # jitter slope by ±10 %  (Gaussian, sig=0.10)
            m_jit = m0 * (1.0 + 0.10 * rng.normal())
            # new intercept so that line goes through anchor
            b_jit = y_anchor - m_jit * x_anchor

            mask_jit = in_band(x_all, y_all, m_jit, b_jit)
            if mask_jit.sum() < 10:
                continue                                     # skip too‑sparse trials

            x_j, y_j, ye_j = x_all[mask_jit], y_all[mask_jit], ye_all[mask_jit]
            m_jitter, *_ = np.polyfit(x_j, y_j, 1, w=1/ye_j)  # quick weighted OLS
            jitter_slopes.append(m_jitter)

        jitter_slopes = np.array(jitter_slopes)
        slope_sys = jitter_slopes.std(ddof=1) if jitter_slopes.size > 1 else 0.0

        if self.verbose:
            logging.info(
                f"{self.region} {self.filt1}-{self.filt2} vs {self.filty}\n"
                f"slope  = {slope_med:.4f} ± {slope_err_stat:.4f} (stat)\n"
                f"syst   = ±{slope_sys:.4f} (band‑placement)\n"
                f"final  = {slope_med:.4f} ± {np.hypot(slope_err_stat, slope_sys):.4f}"
            )
        return slope_med, intercept_med, slope_err_stat, slope_sys

    def plot(self, slope, intercept, show=True, save=False): 
        red_clump_dict = self.red_clump_data
        x = np.asarray(red_clump_dict["x"])
        y = np.asarray(red_clump_dict["y"])

        xy = np.vstack([x, y]) 
        z = gaussian_kde(xy)(xy) 

        fig, axis = plt.subplots(1, 1, figsize=(10, 8)) 

        axis.scatter(x, y, c=z, alpha=0.6, s=40, marker="+") 
        axis.axline(slope=slope, xy1=(0, intercept), c='r') 

        axis.set_xlabel(f"{self.region} {self.filt1} - {self.filt2} vs. {self.filty}", fontsize=16) 
        axis.set_ylabel(f"{self.filty} (mag)", fontsize=14)
        axis.set_xlabel(f"{self.filt1} - {self.filt2} (mag)", fontsize=14) 

        axis.invert_yaxis() 

        if show: 
            plt.show() 
        if save: 
            fig.savefig(self.out_dir / f"[MCMC]_{self.region}_{self.filt1}-{self.filt2}_{self.filty}.png", dpi=300)

        


if __name__ == "__main__": 

    filt_combinations = [ 
        ["F212N", "F323N", "F323N", "NRCB1"], 
        ["F212N", "F323N", "F323N", "NRCB2"], 
        ["F212N", "F323N", "F323N", "NRCB3"], 
        ["F212N", "F323N", "F323N", "NRCB4"], 
    ]

    slopes = [
        [(0.8, 14.15), (0.92, 14.45), 0.1],
        [(1.17, 14.5), (1.05, 14.25), 0.12], 
        [(1, 14.254), (1.1, 14.5), 0.12],
        [(1.07, 14.1), (1.2, 14.4), 0.12]
    ]

        

    with open("/Users/devaldeliwala/mulab/outputs/MCMC/slopes.pkl", "rb") as f: 
        slopes_ = pickle.load(f) 

    for idx, comb in enumerate(filt_combinations): 
        filt1, filt2, filty, region = comb 
        est = LinearEstimator(
            filt1=filt1, filt2=filt2, filty=filty, 
            region=region, top_fraction=0.4
        )

        slope_med, intercept, slope_err, slope_sys = est.slope(slopes=slopes[idx], n_jitter=500)

        slope = slope_med 
        error = np.hypot(slope_err, slope_sys)

        est.plot(slope, intercept, show=False, save=True)

        slopes_[region][f"{filt1}-{filt2}_{filty}"] = (slope, error)
        
        with open("./outputs/MCMC/slopes.pkl", "wb") as f: 
            pickle.dump(slopes_, f)




