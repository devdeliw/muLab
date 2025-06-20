# run from v1/ 

import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from functools import cached_property
from preprocessing.isochrones import Isochrones 
from scipy.stats import gaussian_kde

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Cells(Isochrones):
    """
    Generate bins orthogonal to the RC-bar (Fritz slope) on a CMD,
    then group and (optionally) write out star positions per bin.

    """

    def __init__(
        self,
        filt1: str,
        filt2: str,
        filty: str,
        reg1:  str,
        reg2:  str,
        regy:  str, 
        slope_minus: float = 0.0, 
        data_dir:     Path = Path("./outputs/red_clump_data/"),
        cutoffs_file: Path = Path("./assets/NRCB1_cutoffs.pkl"),
        top_fraction = 1.0
    ):
        self.filt1  = filt1
        self.filt2  = filt2
        self.filty  = filty
        self.reg1   = reg1
        self.reg2   = reg2
        self.regy   = regy

        self.slope_minus  = slope_minus
        self.data_dir     = Path(data_dir)
        self.cutoffs_file = Path(cutoffs_file)
        self.top_fraction = top_fraction

    # not used 
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
        pkl = self.data_dir / f"{self.reg1}.pkl"
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

        # mask out non-finite values
        mask = np.isfinite(m1) & np.isfinite(m2)

        x = np.subtract(m1[mask], m2[mask]) 
        y = np.array(my[mask])
        
        x, y = self.filter_top_density(x, y, top_fraction=self.top_fraction)

        return {
            "m1": m1[mask],
            "m2": m2[mask],
            "my": my[mask],
            "x": x,
            "y": y,
        }

    @cached_property
    def slope(self, fritz=True) -> float:
        """
        Compute and cache the slope of the RC-bar from cutoffs.

        """
        if not fritz: 
            try:
                with self.cutoffs_file.open("rb") as f:
                    cutoffs = pickle.load(f)
            except FileNotFoundError:
                logger.error(f"Cutoffs file not found at {self.cutoffs_file!r}")
                raise

            cutoff, _ = cutoffs[f"{self.filt1}_{self.filt2}"][self.filty]
            (x1, y1), (x2, y2) = cutoff
            return (y2 - y1) / (x2 - x1)
        else: 
            iso = Isochrones(self.filt1, self.filt2, self.filty)
            return iso.calculate_slope()

    @cached_property
    def rotation_matrix(self) -> np.ndarray:
        """
        Rotation matrix to align RC-bar vertically.

        """
        theta = np.arctan(self.slope - self.slope_minus)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, s], [-s, c]])

    def rotate(self) -> np.ndarray:
        """
        Apply rotation to all (x,y) points.

        """
        data = self.red_clump_data

        x = data["x"] 
        y = data["y"] 

        xy = np.vstack((x, y)).T    # (N,2)
        return xy @ self.rotation_matrix.T    

    def cells(self, n_bins: int, write: bool = False, out_dir: Path = Path("/outputs/cells/run_001/")):
        """
        Bin stars in rotated x into `n_bins` vertical slices.
        Returns: dict of bin → {edges, stars_rot, stars_orig}, and slope.

        """

        rotated = self.rotate()
        x_rot = rotated[:, 0]
        edges = np.linspace(x_rot.min(), x_rot.max(), n_bins + 1)

        # assign each star to a bin index 0..n_bins-1
        idxs = np.digitize(x_rot, edges, right=False) - 1
        idxs = np.clip(idxs, 0, n_bins - 1)

        star_bins = np.empty(n_bins, dtype=object)
        for i in range(n_bins):
            stars_orig = rotated[idxs==i] @ self.rotation_matrix 
            star_bins[i] = stars_orig 

        if write:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fn = out_dir / f"{self.filt1}-{self.filt2}_{self.filty}.pkl"
            
            with fn.open("wb") as f:
                pickle.dump(star_bins, f)

            logger.info(f"Wrote bins to {fn!r}")

        return star_bins, self.slope

    def plot_bins(self, n_bins: int, color: bool = True) -> None:
        """
        Scatter the original-frame stars per bin, colored or KDE-smoothed.

        """
        star_bins, slope = self.cells(n_bins, write=False)
        _, ax = plt.subplots(figsize=(10, 8))

        if color:
            # Plots bins in different colors 
            cmap = plt.cm.jet(np.linspace(0, 1, n_bins))

            for i, stars in enumerate(star_bins):
                x, y = stars.T 
                ax.scatter(
                    x, y, 
                    c=[cmap[i]], 
                    s=40, 
                    marker="+",
                    label=f"{i:<2}  ({len(x)} stars)"
                )
            ax.legend(loc="best")
        else:
            # Plots entire CMD as function of density 
            # Ignores individual bins 
            from scipy.stats import gaussian_kde

            # concatenate all and evalute KDE 
            all_xy = np.column_stack(star_bins)
            z = gaussian_kde(all_xy)(all_xy)

            ax.scatter(
                all_xy[:, 0], 
                all_xy[:, 1], 
                c=z, 
                marker="+", 
                s=40
            )

        ax.set_xlabel(f"{self.filt1} − {self.filt2}", fontsize=14)
        ax.set_ylabel(self.filty, fontsize=14)
        ax.set_title(f"{self.reg1}: {self.filt1}-{self.filt2} vs. {self.filty}\n" 
                     f"(slope={slope:.3f})", fontsize=16)
       
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    filt1, reg1 = "F212N", "NRCB1" 
    filt2, reg2 = "F323N", "NRCB5" 
    filty, regy = filt2, reg2

    inst = Cells(
        filt1, filt2, filty, 
        reg1, reg2, regy, 
        data_dir=Path("./outputs/red_clump_data/"),
    )
    inst.plot_bins(10, color=True)

