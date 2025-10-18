# run from mulab/ 

import os 
import ast 
import pickle 
import logging 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from astropy.table import Table
from functools import cached_property
from unsharp_mask import plot_unsharp_hess 
from utils.catalog_helper_functions import get_matches 

logging.basicConfig(level = logging.INFO) 
FITS = "./assets/jwst_init_NRCB.fits"   # photometric catalog 
CUTOFFS = "./assets/red_clump_cuts.pkl" # cutoffs for RC bar 

class GenerateRC(): 
    """ 
    Generates pickle files containing red clump star data: 
        * catalog index of stars that are Red Clump stars.
        * x, y centroid positions of stars that are Red Clump stars.
        * rc magnitudes/errors in F115W, F212N, F323N, and F405N wavelengths.
    """

    def __init__(self, filt1, filt2, filty, reg1, reg2, regy, out_dir="./outputs/"): 
        """
        Args: 
            * filt1 (str): Name of first magnitude filter.
            * filt2 (str): Name of second magnitude filter. 
            * filty (str): Name of magnitude filter plotted on y-axis.
                Note: Should be equal to either filt1 or filt2.
            * reg1  (str): Name of filt1 region.
            * reg2  (str): Name of filt2 region. 
            * regy  (str): Name of filty region.
            * out_dir (str): Directory to place output files. 

        """
        self.filt1 = filt1 
        self.filt2 = filt2 
        self.filty = filty 
        self.reg1  = reg1 
        self.reg2  = reg2 
        self.regy  = regy 
        self.catalog = Table.read(FITS)

        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

        self.m1, self.m2, self.m1e, self.m2e = get_matches(
            self.catalog, filt1, reg1, filt2, reg2, 
        ) 
        self.my = self.m1 if filty == filt1 else self.m2

        if len(self.m1) <= 100: 
            logging.warning("length of match stars <= 100. recheck inputs.")

    def plot_CMD(self, save=True): 
        from scipy.stats import gaussian_kde
        fig, ax = plt.subplots(1, 1, figsize=(10, 8)) 

        x = np.subtract(self.m1, self.m2) 
        y = np.array(self.my) 

        xy = np.vstack([x, y]) 
        z = gaussian_kde(xy)(xy)
    
        ax.scatter(x, y, c=z, s=40, marker="+", alpha=0.6)
        ax.set_xlabel(f"{self.filt1} - {self.filt2} (mag)", fontsize=16)
        ax.set_ylabel(f"{self.filty} (mag)", fontsize=16) 
        ax.set_title(f"{self.reg1} {self.filt1} - {self.filt2} vs. {self.filty}", fontsize=18)
        ax.invert_yaxis()

        if hasattr(self, "cutoff_point1"): 
            ax.axline(xy1=self.cutoff_point1, slope=self.cutoff_slope, linestyle='--', c='r', label="initial cutoff")
            ax.axline(xy1=self.cutoff_point2, slope=self.cutoff_slope, linestyle='--', c='r')

            height = abs(self.cutoff_point1[1] - self.cutoff_point2[1]) 

            upper_cutoff = max(self.cutoff_point1[1], self.cutoff_point2[1]) 
            lower_cutoff = min(self.cutoff_point1[1], self.cutoff_point2[1])
            upper_cutoff += height 
            lower_cutoff -= height

            ax.axline(xy1=(self.cutoff_point2[0], upper_cutoff), slope=self.cutoff_slope, linestyle='--', c='cadetblue', label="expanded cutoff")
            ax.axline(xy1=(self.cutoff_point1[0], lower_cutoff), slope=self.cutoff_slope, linestyle='--', c='cadetblue')
            ax.legend(fontsize=14)

        if save: 
            plt.tight_layout()
            out_dir = f"{self.out_dir}CMD/"
            os.makedirs(out_dir, exist_ok=True)

            fname   = f"{self.reg1}_{self.filt1}-{self.filt2}_{self.filty}.png"
            fig.savefig(f"{out_dir}{fname}", dpi=300)
            logging.info(f" CMD saved to {out_dir}{fname}.")
            
    @cached_property
    def rc_mask(self, expand_factor=1): 
        with open(CUTOFFS, "rb") as f: 
            cutoffs = pickle.load(f) 

        row = cutoffs[ 
            (cutoffs["region1"]  == self.reg1)  &
            (cutoffs["region2"]  == self.reg2)  &
            (cutoffs["regiony"]  == self.regy)  &
            (cutoffs["catalog1"] == self.filt1) &
            (cutoffs["catalog2"] == self.filt2) & 
            (cutoffs["catalogy"] == self.filty)
        ]

        cutoff1, cutoff2, x_range = map( 
            lambda col: ast.literal_eval(row[col].iloc[0]), 
            ["parallel_cutoff1", "parallel_cutoff2", "x_range"], 
        ) 

        (x1, y1), (x2, y2) = cutoff1 
        (_, _),   (x4, y4) = cutoff2 

        # slope of parallel RC cutoff 
        slope = (y2-y1)/(x2-x1) 
        intercept1 = y2 - slope * x2 
        intercept2 = y4 - slope * x4 

        self.cutoff_point1 = (x1, y1) 
        self.cutoff_point2 = (x4, y4)
        self.cutoff_slope  = slope

        # expand cutoffs 
        height = abs(intercept1 - intercept2) 
        intercept1_ = max(intercept1, intercept2) + expand_factor*height 
        intercept2_ = min(intercept1, intercept2) - expand_factor*height 

        # x coordinates for filt1-filt2 vs filty CMD 
        x = np.subtract(self.m1, self.m2) 
        upper_rc_bound = slope*x + intercept1_ 
        lower_rc_bound = slope*x + intercept2_ 

        rc_mask = (
            (self.my <= upper_rc_bound) & 
            (self.my >= lower_rc_bound) & 
            (x >= x_range[0]) & 
            (x <= x_range[1]) 
        )
        return rc_mask

    def red_clump_stars(self, write=True):
        rc_idxs     = np.flatnonzero(self.rc_mask) 
        rc_mags     = self.my[rc_idxs] 
        full_idxs   = np.where(np.isin(self.catalog['m'], rc_mags))[0] 
        rc_catalog  = self.catalog[full_idxs] 

        rc_magnitudes = rc_catalog['m']     # main mags 
        rc_magnitudee = rc_catalog['me']    # mag error 
        rc_centroid_x = rc_catalog['x']     # x (on pixel) 
        rc_centroid_y = rc_catalog['y']     # y (on pixel) 

        region_keys = {
            "NRCB1": 0, 
            "NRCB2": 1, 
            "NRCB3": 2, 
            "NRCB4": 3, 
        }

        red_clump_data = {} 
        f115w_idx = region_keys.get(self.reg1, None) 
        try: 
            red_clump_data['idx']    = np.array(full_idxs)

            red_clump_data['x']      = np.array(rc_centroid_x[:, f115w_idx], dtype=float)
            red_clump_data['y']      = np.array(rc_centroid_y[:, f115w_idx], dtype=float)

            red_clump_data['mF115W'] = np.array(rc_magnitudes[:, f115w_idx],    dtype=float) 
            red_clump_data['mF212N'] = np.array(rc_magnitudes[:, f115w_idx+4],  dtype=float) 
            red_clump_data['mF323N'] = np.array(rc_magnitudes[:, 9], dtype=float) 
            red_clump_data['mF405N'] = np.array(rc_magnitudes[:, 8], dtype=float)

            red_clump_data['meF115W'] = np.array(rc_magnitudee[:, f115w_idx],   dtype=float) 
            red_clump_data['meF212N'] = np.array(rc_magnitudee[:, f115w_idx+4], dtype=float) 
            red_clump_data['meF323N'] = np.array(rc_magnitudee[:, 9], dtype=float) 
            red_clump_data['meF405N'] = np.array(rc_magnitudee[:, 8], dtype=float)

            red_clump_data = pd.DataFrame(red_clump_data) 

            # saving red_clump_data to pkl file. 
            if write: 
                out_dir = f"{self.out_dir}red_clump_data/"
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, f"{self.reg1}.pkl")
                with open(out_file, "wb") as f: 
                    pickle.dump(red_clump_data, f) 
                logging.info(f" {self.reg1}.pkl saved to {out_dir}") 

        except (IndexError, TypeError, ValueError) as e: 
            logging.error(f" Failed to extract red clump data: {e}") 
            return None

    def filtered_mags(self, filt1, filt2, filty, region): 
        mag_keys = { 
            "F115W": "mF115W", 
            "F212N": "mF212N", 
            "F323N": "mF323N", 
            "F405N": "mF405N", 
        }

        mage_keys = { 
            "F115W": "meF115W", 
            "F212N": "meF212N", 
            "F323N": "meF323N", 
            "F405N": "meF405N", 
        }

        try: 
            with open(f"{self.out_dir}red_clump_data/{region}.pkl", "rb") as f: 
                red_clump_data = pickle.load(f) 
        except FileNotFoundError as e: 
            logging.error(f" Red clump data not found: {e}") 
            return None

        m1  = red_clump_data[mag_keys.get(filt1)] 
        m2  = red_clump_data[mag_keys.get(filt2)] 
        my  = red_clump_data[mag_keys.get(filty)]
        m1e = red_clump_data[mage_keys.get(filt1)] 
        m2e = red_clump_data[mage_keys.get(filt2)] 
        mye = red_clump_data[mage_keys.get(filty)]

        # remove infs or NaNs 
        mask = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(m1e) & np.isfinite(m2e)  
        m1, m2, my      = m1[mask], m2[mask], my[mask] 
        m1e, m2e, mye   = m1e[mask], m2e[mask], mye[mask] 
        return m1, m2, my, m1e, m2e, mye 
    
    def plot(self, filt1, filt2, filty, region, save=False, show=True): 
        from scipy.stats import gaussian_kde 

        m1, m2, my, _, _, _ = self.filtered_mags(filt1, filt2, filty, region)
        mx = np.subtract(m1, m2) 

        # for coloring as function of density 
        # visualizes red clump cluster easier
        xy = np.vstack([mx, my])
        z = gaussian_kde(xy)(xy)

        _, _ = plt.subplots(1, 1, figsize=(10, 8)) 

        plt.scatter(mx, my, c=z, s=20, marker="+")
        plt.xlabel(f"{filt1} - {filt2}", fontsize=14) 
        plt.ylabel(f"{filty}", fontsize=14)
        plt.title(f"Count: {len(m1)}", fontsize=16)

        plt.gca().invert_yaxis() 
        plt.tight_layout()
        
        if save:
            save_dir = f"{self.out_dir}/red_clump_plots/" 
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}{region}_{filt1}-{filt2}_{filty}.png", dpi=300)
        if show: 
            plt.show() 
        
        plt.close()
        return

    def plot_hess(self, filt1, filt2, filty, region, gamma=None): 
        m1, m2, my, m1e, m2e, mye = self.filtered_mags(filt1, filt2, filty, region)
        if not gamma: 
            gamma = 1.25
        plot_unsharp_hess(m1, m2, my, m1e, m2e, mye, filt1, filt2, filty, 
                          region=region, plot_fritz=False, gamma=gamma)




class Type1(): 
    def __init__(self): 
        self.filt_combinations = [
            ["F115W", "F212N", "F115W", "NRCB1", "NRCB1", "NRCB1"],
            ["F115W", "F212N", "F115W", "NRCB2", "NRCB2", "NRCB2"],
            ["F115W", "F212N", "F115W", "NRCB3", "NRCB3", "NRCB3"],
            ["F115W", "F212N", "F115W", "NRCB4", "NRCB4", "NRCB4"],
        ]

    def generate(self, write=True): 
        for comb in self.filt_combinations: 
            filt1, filt2, filty, reg1, reg2, regy = comb 

            instance = GenerateRC(filt1, filt2, filty, reg1, reg2, regy)
            instance.red_clump_stars(write=write)

            sub_filts = [ 
                ["F115W", "F212N", "F115W", "NRCB1", "NRCB1", "NRCB1"],
                ["F115W", "F323N", "F115W", "NRCB1", "NRCB5", "NRCB1"],
                ["F115W", "F405N", "F115W", "NRCB1", "NRCB5", "NRCB1"],
                ["F115W", "F212N", "F115W", "NRCB2", "NRCB2", "NRCB2"],
                ["F115W", "F323N", "F115W", "NRCB2", "NRCB5", "NRCB2"],
                ["F115W", "F405N", "F115W", "NRCB2", "NRCB5", "NRCB2"],
                ["F115W", "F212N", "F115W", "NRCB3", "NRCB3", "NRCB3"],
                ["F115W", "F323N", "F115W", "NRCB3", "NRCB5", "NRCB3"],
                ["F115W", "F405N", "F115W", "NRCB3", "NRCB5", "NRCB3"],
                ["F115W", "F212N", "F115W", "NRCB4", "NRCB4", "NRCB4"],
                ["F115W", "F323N", "F115W", "NRCB4", "NRCB5", "NRCB4"],
                ["F115W", "F405N", "F115W", "NRCB4", "NRCB5", "NRCB4"],
            ]

            for filt in sub_filts: 
                filt1, filt2, filty, region = filt[0], filt[1], filt[2], filt[3]
                instance.plot_hess(filt1, filt2, filty, region)

class Type12(): 
    def __init__(self): 
        self.filt_combinations = [
            ["F115W", "F212N", "F115W", "NRCB1", "NRCB1", "NRCB1"],
            #["F212N", "F323N", "F323N", "NRCB1", "NRCB5", "NRCB5"], 
            #["F212N", "F405N", "F212N", "NRCB1", "NRCB5", "NRCB1"], 
            #["F115W", "F212N", "F115W", "NRCB2", "NRCB2", "NRCB2"],
            #["F212N", "F323N", "F323N", "NRCB2", "NRCB5", "NRCB5"], 
            ["F212N", "F405N", "F212N", "NRCB2", "NRCB5", "NRCB2"], 
            ["F115W", "F212N", "F115W", "NRCB3", "NRCB3", "NRCB3"],
            ["F212N", "F323N", "F323N", "NRCB3", "NRCB5", "NRCB5"], 
            ["F212N", "F405N", "F212N", "NRCB3", "NRCB5", "NRCB3"], 
            ["F115W", "F212N", "F115W", "NRCB4", "NRCB4", "NRCB4"],
            ["F212N", "F323N", "F323N", "NRCB4", "NRCB5", "NRCB5"], 
            ["F212N", "F405N", "F212N", "NRCB4", "NRCB5", "NRCB4"], 
        ]

    def generate(self, write=True): 
        for comb in self.filt_combinations: 
            filt1, filt2, filty, reg1, reg2, regy = comb 

            instance = GenerateRC(filt1, filt2, filty, reg1, reg2, regy) 
            instance.plot_CMD()





if __name__ == "__main__": 

    instance = Type1() 
    instance.generate(write=True)
