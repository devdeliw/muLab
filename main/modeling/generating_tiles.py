import pickle 
import logging
import numpy as np 
import matplotlib.pyplot as plt 

from pathlib import Path 
from typing import Optional 
from  dataclasses import dataclass
from functools import cached_property
from preprocessing.generating_isochrone import IsochroneSlopes 
from preprocessing.generating_red_clump import RC_CUTOFF, GenerateRedClump

plt.rcParams["font.family"]      = "serif" 
plt.rcParams['mathtext.fontset'] = 'cm'

@dataclass 
class RCConfig: 
    """
    This dataclass determines whether the algorithm will extract RC stars from each 
    NRCB footprint individually, or just be given arbitrary photometric data to perform 
    the RC slope analysis on. 

    If NRCB footprints should be analyzed set m1=m2=my=None, and region1=region2=regiony 
    to the region of choice.

    Otherwise set region1=region2=regiony=None and set m1, m2, my to be the photometric 
    magnitudes to calculate the RC slope from in a m1-m2 vs. my CMD. 

    If both regions and magnitudes are specified, magnitudes take precedence. 
    """

    m1      : Optional[np.ndarray] = None  
    m2      : Optional[np.ndarray] = None 
    my      : Optional[np.ndarray] = None 
    # *
    region1 : Optional[str] = None 
    region2 : Optional[str] = None 
    regiony : Optional[str] = None 

@dataclass
class TileConfig: 
    """
    Args: 
        * n_tiles      (int)   : number of tiles to place across horizontal RC
        * out_dir      (Path)  : directory to place output star-tile dictionary
        * pickle_dir   (Path)  : directory of NRCB red clump data
        * cutoffs_file (Path)  : directory where cutoffs are stored 
        * slope_minus  (float) : how much to bias the fritz slope to truly make RC horizontal
    """ 

    n_tiles: int        = 10 
    out_dir: Path       = Path(__file__).parent.parent / "raw/tiles/"
    plt_dir: Path       = Path(__file__).parent.parent / "media/tiles/"
    pickle_dir: Path    = Path(__file__).parent.parent / "assets/"
    cutoffs_file: Path  = RC_CUTOFF 
    slope_minus: float  = 0.0


class GenerateTiles(GenerateRedClump): 
    """
    Generate tiles (roughly) orthogonal to the RC ridge using the 
    Fritz+11 reddening vector. 

    Exports a ndarray of stars that are in each tile for later analysis. 
    """

    def __init__(
        self, 
        filter1: str, 
        filter2: str, 
        filtery: str, 
        rc_cfg : RCConfig, 
        config : TileConfig = TileConfig()
    ): 
        self.filter1 = filter1 
        self.filter2 = filter2 
        self.filtery = filtery 

        self.m1      = np.asarray(rc_cfg.m1, dtype=float)
        self.m2      = np.asarray(rc_cfg.m2, dtype=float) 
        self.my      = np.asarray(rc_cfg.my, dtype=float)
        self.region1 = rc_cfg.region1 
        self.region2 = rc_cfg.region2 
        self.regiony = rc_cfg.regiony

        # verifying config completeness
        regions_given    = all(r is not None for r in (self.region1, self.region2, self.regiony))
        magnitudes_given = all(m is not None for m in (self.m1, self.m2, self.my))

        if magnitudes_given:
            self.region1 = self.region2 = self.regiony = None
        elif regions_given:
            pass
        else:
            raise ValueError(
                "Either all regions must be specified or all magnitudes must be specified."
            )
         
        self.config    = config 
        self.logger    = logging.getLogger(__name__) 

    @cached_property
    def give_rc_data(self): 
        """ 
        Manually provided red clump data -- no regions.  
        """

        return { 
            "m1": self.m1, 
            "m2": self.m2, 
            "my": self.my, 
            "x" : np.subtract(self.m1, self.m2), 
            "y" : self.my
        }

    @cached_property
    def load_rc_data(self): 
        """
        Load and cache the raw red clump data for given filters & region.
        """

        try:
            pkl = self.config.pickle_dir / f"{self.region1}.pickle" 
            with open(pkl, "rb") as f: 
                    red_clump_data = pickle.load(f) 
        except: 
            self.logger.warning("RC pickle extraction failed. Recalculating Manually")
            red_clump_data = self.extract_red_clump_stars(export=True)

        m1 = red_clump_data[f"m{self.filter1}"] 
        m2 = red_clump_data[f"m{self.filter2}"] 
        my = red_clump_data[f"m{self.filtery}"] 

        x = np.subtract(m1, m2) 
        y = np.asarray(my) 

        return {
            "m1": m1, 
            "m2": m2, 
            "my": my, 
            "x": x, 
            "y": y, 
        }

    @cached_property
    def slope(self): 
        """
        Compute and cache the slope of the RC ridge. 
        """ 
        return 0
        return IsochroneSlopes(self.filter1, self.filter2, self.filtery).reddening_slope()

    @cached_property
    def rotation_matrix(self): 
        theta    = np.arctan(self.slope - self.config.slope_minus) 
        cos, sin = np.cos(theta), np.sin(theta) 
        return np.array([ 
            [cos, sin], 
            [-sin, cos]
        ])

    def rotate_RC(self): 
        if self.m1 is not None and self.m2 is not None and self.my is not None: 
            red_clump_data = self.give_rc_data 
        else: 
            red_clump_data = self.load_rc_data

        x = red_clump_data['x'] 
        y = red_clump_data['y'] 

        xy = np.vstack((x, y)).T 
        return xy #@ self.rotation_matrix.T 

    def tiles(self, export=True): 
        n_tiles = self.config.n_tiles 
        rotated_rc = self.rotate_RC() 

        x_rotated  = rotated_rc[:, 0]
        tile_edges = np.linspace(x_rotated.min(), x_rotated.max(), n_tiles+1)

        # assign each star to bin index 0..n_tiles-1 
        idxs = np.digitize(x_rotated, tile_edges, right=False) - 1 
        idxs = np.clip(idxs, 0, n_tiles - 1)

        star_tiles = np.empty(n_tiles, dtype=object) 
        for i in range(n_tiles):
            star_tiles[i] = rotated_rc[idxs == i]

        if export: 
            out_dir = self.config.out_dir 
            out_dir.mkdir(parents=True, exist_ok=True) 

            filename = out_dir / f"{self.filter1}-{self.filter2}_{self.filtery}.pickle" 

            with open(filename, "wb") as f: 
                pickle.dump(star_tiles, f) 
            self.logger.info(f"Placed tiled-starlist into {Path(*filename.parts[-3:])}.")

        return star_tiles 

    def plot_tiles(self): 
        star_tiles = self.tiles(export=False) 
        figure, axis = plt.subplots(1, 1, figsize=(8, 6)) 

        cmap = plt.cm.cool(np.linspace(0, 1, self.config.n_tiles)) # type: ignore
        for idx, stars in enumerate(star_tiles): 
            x, y = stars.T 
            axis.scatter(
                x, y, 
                c=[cmap[idx]], 
                s=10, 
                marker='d', 
                label=f"{idx:<2}; {len(x)} stars"
            )

        label_prefix = self.region1 if self.region1 is not None else ""
        axis.set_xlabel(f"{label_prefix} {self.filter1} - {self.filter2} (mag)", fontsize=15) 
        axis.set_ylabel(f"{label_prefix} {self.filtery} (mag)", fontsize=15) 
        axis.invert_yaxis()

        # to show orthogonality without any geometric skew 
        axis.axis("equal")

        axis.legend(loc="best", ncol=2) 
        plt.tight_layout()

        self.config.plt_dir.mkdir(exist_ok=True, parents=True)
        loc_prefix = self.region1 if self.region1 else "custom"
        filename = self.config.plt_dir / f"{self.config.n_tiles}_tiles_{loc_prefix}_{self.filter1}-{self.filter2}_{self.filtery}.png"
        figure.savefig(filename, dpi=300) 
        self.logger.info(f"{self.config.n_tiles} tile Figure saved to {Path(*filename.parts[-3:])}.")


if __name__ == "__main__": 
    filter1, region1 = "F115W", "NRCB1" 
    filter2, region2 = "F212N", "NRCB1" 
    filtery, regiony = filter1, region1

    rc_cfg = RCConfig(
        region1 = region1, 
        region2 = region2, 
        regiony = regiony 
    )

    instance = GenerateTiles(
        filter1=filter1, 
        filter2=filter2, 
        filtery=filtery, 
        rc_cfg = rc_cfg,
        config=TileConfig() 
    )

    instance.tiles(export=True)
    instance.plot_tiles()
    












