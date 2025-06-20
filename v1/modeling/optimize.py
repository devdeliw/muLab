#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rc_bar_estimator.py
Module to estimate the slope of the Red Clump (RC) bar in a color-magnitude diagram (CMD)
using diagonal bins and MCMC fits for robust uncertainty quantification.
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from modeling.mcmc import MCMC 
from modeling.mcmc_autocorr import MCMC_Autocorr
from modeling.cells import Cells

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Estimator:
    """
    Estimate the RC bar slope by fitting Gaussian+Linear1D models 

    """

    def __init__(
        self,
        filt1: str,
        filt2: str,
        filty: str,
        reg1: str,
        reg2: str,
        regy: str,
        n_bins: int,
        out_dir: Path = Path("./outputs/MCMC/"),
        verbose: bool = False,
        autocorr: bool = False, 
        autocorr_bin: int = 5
    ):
        self.filt1 = filt1
        self.filt2 = filt2
        self.filty = filty
        self.region = reg1
        self.n_bins = n_bins
        self.verbose = verbose

        self.out_dir = out_dir.expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        cells = Cells(filt1, filt2, filty, reg1, reg2, regy)
        self.star_bins, initial_slope = cells.cells(n_bins, write=False)

        theta = np.arctan(initial_slope)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        self._rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]]) # rotation matrix 
        self._inv_rot = self._rot.T                             # inv rotation matrix 

        self.points = []  # representative (x,y) in original frame
        self.errors = []  # corresponding y-errors

        # perform autocorrelation analysis 
        self.autocorr = autocorr
        self.autocorr_bin = autocorr_bin
        self.max_y_err = 0.2


    def _analyze_bin(self, stars: np.ndarray, bin_num: int, color: str) -> tuple[np.ndarray, float]:
        """
        Rotate stars, run MCMC on vertical axis, then transform back.
        Returns:
            * point (2-array): representative (x,y) in original frame
            * y_error (float): uncertainty in y direction 

        """
        # rotate to bar frame
        rot_stars = stars @ self._inv_rot
        x_rot, y_rot = rot_stars.T

        # MCMC on rotated y
        if not self.autocorr: 
            mcmc = MCMC(y_rot)
        else: 
            mcmc = MCMC_Autocorr(y_rot)

        best_fit, samples, _ = mcmc.run()
        mean_y = best_fit['mean']
        sigma_y = np.std(samples[:, 1], ddof=1)

        # median x in bar frame
        x_center = np.median(x_rot)

        # back to original frame
        repr_rot = np.array([x_center, mean_y])
        repr_orig = repr_rot @ self._rot

        # error prop
        sigma_vec = np.array([0.0, sigma_y]) @ self._rot
        y_err = abs(sigma_vec[1])

        if y_err <= self.max_y_err: 
            if self.autocorr: 
                self.tau = mcmc.integrated_autocorr()
                mcmc.autocorr_vs_N() 

                if not hasattr(self, "fig_autocorr"):
                    self.fig_autocorr = None 
                    self.ax_autocorr  = None 
                
                self.fig_autocorr, self.ax_autocorr = mcmc.plot_autocorr(
                    bin_num=bin_num, 
                    fig=self.fig_autocorr, 
                    ax=self.ax_autocorr,
                    color=color
                )
        return repr_orig, y_err

    def run(self) -> tuple[float, float]:
        """
        Process all bins, collect points. 
        Compute global slope & intercept.

        """
        print(f"\nrunning mcmc with {self.n_bins} bins")
        print(f"{self.region}: {self.filt1} - {self.filt2} vs. {self.filty}")

        colors = plt.cm.jet(np.linspace(0, 1, len(self.star_bins)))
        for idx, stars in enumerate(self.star_bins):
            if stars.size == 0:
                continue

            point, y_err = self._analyze_bin(stars, bin_num=idx, color=colors[idx])
            if y_err <= self.max_y_err:
                self.points.append(point)
                self.errors.append(y_err)

            if self.verbose:
                if self.autocorr:
                    if hasattr(self, "tau"): 
                        logger.info(f" bin {idx:02d}: (x,y)=({point[0]:.3f},{point[1]:.3f}) ±{y_err:.3f}, ~tau={np.mean(self.tau):.3f}")
                else: 
                    logger.info(f" bin {idx:02d}: (x,y)=({point[0]:.3f},{point[1]:.3f}) ±{y_err:.3f}")

        pts = np.vstack(self.points)
        xs, ys = pts[:, 0], pts[:, 1]
        weights = 1.0 / np.square(self.errors)

        # Weighted linear fit
        (self.slope, self.intercept), cov = np.polyfit(xs, ys, 1, w=weights, cov=True)
        self.slope_err = np.sqrt(cov[0, 0])

        # saving autocorrelation plots

        if self.autocorr: 
            self.ax_autocorr.set_xlabel(r"Samples $N$", fontsize=14)
            self.ax_autocorr.set_ylabel(r"Mean $\hat{\tau}_{\mathrm{int}}$", fontsize=14)
            self.ax_autocorr.legend()
            self.fig_autocorr.tight_layout()

            fname = self.out_dir / f"[MCMC]_{self.n_bins}bins_autocorrelation.png" 
            self.fig_autocorr.savefig(
                fname, 
                dpi=300
            )
            logging.info(f" png saved to {fname}") 

        print(f"\n[FINAL] slope={self.slope:.4f} ± {self.slope_err:.4f}\n")
        return self.slope, self.intercept

    def plot(self, save: bool = False, show: bool = True) -> None:
        """
        Create a clean, aesthetic plot of the CMD, representative points, and best-fit line.
        """
        if not self.points:
            raise RuntimeError("No points to plot. Call run() first.")

        fig, ax = plt.subplots(figsize=(8, 6))

        # scatter entire RC bar
        colors = plt.cm.jet(np.linspace(0, 1, len(self.star_bins)))
        for i, stars in enumerate(self.star_bins):
            if stars.size: 
                ax.scatter(
                    stars[:, 0], stars[:, 1],
                    marker="+", color=colors[i], 
                    alpha=0.8, s=40
                )

        # Representative points with error bars
        pts = np.vstack(self.points)
        ax.errorbar(
            pts[:, 0], pts[:, 1], yerr=self.errors, color='r', 
            fmt='x', markersize=8, capsize=4, zorder=3
        )

        # Best-fit line
        x_vals = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 400)
        y_vals = self.slope * x_vals + self.intercept
        ax.plot(x_vals, y_vals, c='k', lw=2, zorder=2)

        ax.invert_yaxis()
        ax.set_xlabel(f"{self.filt1} – {self.filt2} (mag)", fontsize=14)
        ax.set_ylabel(f"{self.filty} (mag)", fontsize=14)
        ax.set_title(
            f"{self.filt1} – {self.filt2} vs {self.filty}\n",
            fontsize=16
        )
        ax.text(
            0.02, 0.02,
            f"slope = {self.slope:.3f} ± {self.slope_err:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='left',
        )
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        if save:
            fname = (
                self.out_dir /
                f"[MCMC]_{self.n_bins}bins_{self.region}_{self.filt1}-{self.filt2}_{self.filty}.png"
            )
            fig.savefig(fname, dpi=300)
            logger.info(f" png saved to {fname}")

        if show:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':  # pragma: no cover
    # Example usage
    est = Estimator(
        filt1='F115W', filt2='F212N', filty='F115W',
        reg1='NRCB1', reg2='NRCB1', regy='NRCB1',
        n_bins=15, verbose=True, autocorr=False,
    )
    est.run()
    est.plot(save=False, show=True)

