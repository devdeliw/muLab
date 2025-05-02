#!/usr/bin/env python3 

import logging 
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

    def slope(self, box=None, slopes=None, save=False, show=True):
        red_clump_dict = self.red_clump_data
        x_all = red_clump_dict['x'] 
        y_all = red_clump_dict['y']
        yerr_all = red_clump_dict["ye"]

        if slopes:
            (x1, y1), (x2, y2) = slopes[0], slopes[1] 
            gap = slopes[2] 
            x_vals = np.array(x_all) 
            y_vals = np.array(y_all)

            x = np.asarray(x_vals)
            y = np.asarray(y_vals)

            # slope and intercept
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            # compute residuals
            y_line = m * x + b
            resid  = y - y_line

            mask = np.abs(resid) <= gap

            x, y, yerr_all = x_all[mask], y_all[mask], yerr_all[mask]
        else: 
            x, y = x_all, y_all

        fit = LinearBayes(x, y, yerr_all, box=box, top_fraction=self.top_fraction)

        slope, intercept, slope_err = fit.run() 

        if self.verbose: 
            logging.info(f"{self.region} {self.filt1} - {self.filt2} vs. {self.filty}\n" 
                         f"slope = {slope:.3f} ± {slope_err:.4f}")

        if save or show: 
            fig, axis = plt.subplots(1, 1, figsize=(10, 8)) 

            x = x_all 
            y = y_all

            if box: 
                xmin, xmax, ymin, ymax = box 
                m = (
                    (x >= xmin) & (x <= xmax) &
                    (y >= ymin) & (y <= ymax)
                )
                x, y= x[m], y[m]

            if slopes: 
                (x1, y1), (x2, y2) = slopes[0], slopes[1] 
                gap = slopes[2] 
                x_vals = np.array(x_all) 
                y_vals = np.array(y_all)

                x = np.asarray(x_vals)
                y = np.asarray(y_vals)

                # slope and intercept
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1

                # compute residuals
                y_line = m * x + b
                resid  = y - y_line

                mask = np.abs(resid) <= gap

                x, y = x_all[mask], y_all[mask]


            if self.top_fraction: 
                x, y = self.filter_top_density(x, y, top_fraction=self.top_fraction)

            xy = np.vstack([x_all, y_all]) 
            z = gaussian_kde(xy)(xy) 

            axis.scatter(x_all, y_all, c=z, marker="+", s=40, alpha=0.7)
            axis.scatter(x, y, c="red", marker="+", s=40)
            axis.axline(slope=slope, xy1=(0, intercept), c='r')
            axis.set_xlim(x_all.min(), x_all.max())
            axis.set_ylim(y_all.max(), y_all.min()) # invert y axis

            axis.text(
                0.02, 0.02,
                f"slope = {slope:.3f} ± {slope_err:.3f}",
                transform=axis.transAxes,
                fontsize=12,
                verticalalignment='bottom',
                horizontalalignment='left',
            )

            axis.set_xlabel(f"{self.filt1} - {self.filt2} (mag)", fontsize=14)
            axis.set_ylabel(f"{self.filty} (mag)", fontsize=14)
            axis.set_title(f"{self.region} {self.filt1} - {self.filt2} vs. {self.filty}", fontsize=16)
        
            if show: 
                plt.show()

            if save: 
                fname = self.out_dir / f"[MCMC]_{self.region}_{self.filt1}-{self.filt2}_{self.filty}"
                fig.savefig(fname, dpi=300)

        return slope, intercept, slope_err







if __name__ == "__main__": 

    filt_combinations = [ 
        #["F212N", "F323N", "F323N", "NRCB1"], 
        #["F212N", "F323N", "F323N", "NRCB2"], 
        #["F212N", "F323N", "F212N", "NRCB3"], 
        ["F212N", "F323N", "F323N", "NRCB4"], 
    ]

    for comb in filt_combinations: 
        filt1, filt2, filty, region = comb 
        est = LinearEstimator(
            filt1=filt1, filt2=filt2, filty=filty, 
            region=region, top_fraction=0.2
        )

        slopes = [(0.8, 14.25), (0.92, 14.45), 0.12]
        box = (0.7, 0.95, 14.2, 14.45)

        slopes = [(1.17, 14.45), (1.05, 14.25), 0.12] 

        slopes = [(1, 15.33), (1.1, 15.55), 0.12]

        slopes = [(1.07, 14.2), (1.2, 14.4), 0.12]
        est.slope(show=False, box=None, save=True, slopes=slopes)

