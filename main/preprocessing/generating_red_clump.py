import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os, ast, pickle, logging 

from pathlib import Path 
from astropy.table import Table 
from functools import lru_cache 
from scripts.catalog_helper_functions import get_matches

plt.rcParams["font.family"]      = "serif" 
plt.rcParams['mathtext.fontset'] = 'cm'
logging.basicConfig(level=logging.INFO) 

ASSETS = Path(__file__).parent.parent / "assets"
FITS_FILE = ASSETS / "jwst_init_NRCB.fits" 
RC_CUTOFF = ASSETS / "rc_cutoffs.pickle" 

class GenerateRedClump(): 
    """ 
    Generates pickle files for each NRCB footprint containing: 
        * catalog indexes of RC stars 
        * x, y centroid positions of RC stars 
        * magnitudes & errors in F115W, F212N, F323N, and F405N

    """

    def __init__(
        self, 
        filter1: str, 
        filter2: str, 
        filtery: str, 
        region1: str, 
        region2: str, 
        regiony: str, 
        pkl_dir: Path = ASSETS, 
    ): 
        """ 
        Args: 
            * filter1, region1 (str): name of filter & region for lower wavelength
            * filter2, region2 (str): name of filter & region for higher wavelength
            * filtery, regiony (str): name of filter & region on y-axis 
            * pkl_dir (Path): directory to place output .pickle files 
        """

        self.filter1 = filter1 
        self.filter2 = filter2 
        self.filtery = filtery 
        self.region1 = region1 
        self.region2 = region2 
        self.regiony = regiony 
        self.catalog = Table.read(FITS_FILE) 
        self.logger  = logging.getLogger(__name__)

        with open(RC_CUTOFF, "rb") as f: 
            self.cutoffs = pickle.load(f) 

        if not os.path.exists(pkl_dir): 
            self.logger.warning(
                "Output pickle directory DNE." 
                "Automatically creating directory"
            ) 
            self.logger.warning(pkl_dir) 
        pkl_dir.mkdir(parents=True, exist_ok=True) 
        self.pkl_dir = pkl_dir

        # Getting magnitudes and errors from catalog 
        
        m1, m2, m1e, m2e = get_matches(
            self.catalog, filter1, region1, filter2, region2, 
        ) 

        # removing any non-finite indices 
        finite_mask = ( 
            np.isfinite(m1) & 
            np.isfinite(m2) & 
            np.isfinite(m1e) & 
            np.isfinite(m2e) 
        )

        if len(m1) - len(finite_mask) > 0: 
            self.logger.info(f" Removing {len(m1) - len(finite_mask)} non-finite stars.")
        self.m1  = m1[finite_mask] 
        self.m2  = m2[finite_mask] 
        self.m1e = m1e[finite_mask] 
        self.m2e = m2e[finite_mask]

        if filtery not in (filter1, filter2): 
            raise ValueError(
                f"`filtery`: {filtery} must either equal " 
                f"`filter1`: {filter1} or `filter2`: {filter2}."
            )
        if regiony not in (region1, region2): 
            raise ValueError(
                f"`regiony`: {regiony} must either equal " 
                f"`region1`: {region1} or `region2`: {region2}."
            )

        if filter1 == filtery: 
            self.my = self.m1 
            self.mye = self.m1e 
        else: 
            self.my = self.m2 
            self.mye = self.mye 

        if len(self.m1) <= 100: 
            raise ValueError("Length of matching stars less than 100.") 

    @lru_cache(maxsize=None) 
    def masking_the_rc(self, expand_factor=1): 
        """ 
        Isolates RC stars from a `filter1` - `filter2` vs. `filtery` CMD. 

        Two parallel cutoffs are defined in RC_CUTOFFS constant. 
        These cutoffs are expanded vertically by `expand_factor` 
        and a mask is generated to isolate RC stars within. 
        """

        try:
            row_in_use = self.cutoffs[ 
                (self.cutoffs["region1"]     == self.region1) & 
                (self.cutoffs["region2"]     == self.region2) & 
                (self.cutoffs["regiony"]     == self.regiony) & 
                (self.cutoffs["catalog1"]    == self.filter1) & 
                (self.cutoffs["catalog2"]    == self.filter2) & 
                (self.cutoffs["catalogy"]    == self.filtery)
            ]
        except KeyError as e:
            self.logger.error(f"` RC_CUTOFFS` has a missing key:\n{e}")
            raise 

        cutoff1, cutoff2, color_range = map( 
            lambda col: ast.literal_eval(row_in_use[col].iloc[0]), 
            ["parallel_cutoff1", "parallel_cutoff2", "x_range"], 
        )

        (x1, y1), (x2, y2) = cutoff1 
        (_, _),   (x4, y4) = cutoff2 

        # slope of RC cutoffs 
        slope = (y2 - y1) / (x2 - x1) 
        intercept1 = y2 - slope * x2 
        intercept2 = y4 - slope * x4 

        # RC cutoff expansion 
        height = abs(intercept1 - intercept2) 
        upper_intercept = max(intercept1, intercept2) + expand_factor*height 
        lower_intercept = min(intercept1, intercept2) - expand_factor*height 
         
        # finding RC within cutoffs
        color = np.subtract(self.m1, self.m2) 
        upper_rc_bound = slope * color + upper_intercept 
        lower_rc_bound = slope * color + lower_intercept 

        rc_mask = (
            (self.my <= upper_rc_bound) & 
            (self.my >= lower_rc_bound) & 
            (color >= color_range[0]) & 
            (color <= color_range[1]) 
        )

        self.logger.info(
            " Number of red clump stars extracted: "
            f"{len(rc_mask)}" 
        )
        return rc_mask 

    def extract_red_clump_stars(self, export=True):
        """ 
        Extracts the red clump stars in every filter 
        F115W, F212N, F323N, and F405N from the rc_mask 
        generated in `self.masking_the_rc()`. 

        Outputs a pickle file to `self.pkl_dir` containing 
        centroid positions, magnitudes, and errors across all filters. 
        
        Note* the centroid positions are taken from F115W detector. 
        """

        rc_idxs = np.flatnonzero(self.masking_the_rc())
        rc_mags = self.my[rc_idxs] 

        full_idxs  = np.where(np.isin(self.catalog['m'], rc_mags))[0] 
        rc_catalog = self.catalog[full_idxs] # extracted RC stars 

        rc_magnitudes    = rc_catalog['m']  # mag 
        rc_magnitude_err = rc_catalog['me'] # mag error 
        rc_centroid_x    = rc_catalog['x']  # x pixel pos 
        rc_centroid_y    = rc_catalog['y']  # y pixel pos 

        REGION_KEYS = {
            "NRCB1": 0, 
            "NRCB2": 1, 
            "NRCB3": 2, 
            "NRCB4": 3
        }

        red_clump_data = {} 

        # get the column idx of the F115W magnitudes 
        # just based on how the catalog is organized 
        f115w_idx = REGION_KEYS.get(self.region1, None)

        if f115w_idx != 0 and not f115w_idx: 
            raise ValueError(
                f"`self.region1`: {self.region1} " 
                "must either be NRCB1, NRCB2, NRCB3, or NRCB4"
            )

        try: 
            red_clump_data['idx']    = np.asarray(full_idxs)

            red_clump_data['x']      = np.asarray(rc_centroid_x[:, f115w_idx], dtype=float)
            red_clump_data['y']      = np.asarray(rc_centroid_y[:, f115w_idx], dtype=float)

            red_clump_data['mF115W'] = np.asarray(rc_magnitudes[:, f115w_idx],    dtype=float) 
            red_clump_data['mF212N'] = np.asarray(rc_magnitudes[:, f115w_idx+4],  dtype=float) 
            red_clump_data['mF323N'] = np.asarray(rc_magnitudes[:, 9], dtype=float) 
            red_clump_data['mF405N'] = np.asarray(rc_magnitudes[:, 8], dtype=float)

            red_clump_data['meF115W'] = np.asarray(rc_magnitude_err[:, f115w_idx],   dtype=float) 
            red_clump_data['meF212N'] = np.asarray(rc_magnitude_err[:, f115w_idx+4], dtype=float) 
            red_clump_data['meF323N'] = np.asarray(rc_magnitude_err[:, 9], dtype=float) 
            red_clump_data['meF405N'] = np.asarray(rc_magnitude_err[:, 8], dtype=float)

            red_clump_data = pd.DataFrame(red_clump_data) 

            if export: 
                out_directory = self.pkl_dir / f"{self.region1}.pickle"
                with open(out_directory, "wb") as f: 
                    pickle.dump(red_clump_data, f) 
                self.logger.info(f" {self.region1}.pickle saved to {out_directory}") 

        except (IndexError, TypeError, ValueError) as e: 
            self.logger.error(f" Failed to extract red clump data: {e}") 
            raise

        return red_clump_data


if __name__ == "__main__": 
    instance = GenerateRedClump( 
        filter1 = "F115W", 
        filter2 = "F212N", 
        filtery = "F115W", 
        region1 = "NRCB4", 
        region2 = "NRCB4", 
        regiony = "NRCB4", 
    )

    instance.extract_red_clump_stars(export=True) 
















    













