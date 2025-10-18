import pickle 
import logging 
import numpy as np 
import matplotlib.pyplot as plt 

from pathlib import Path 
from scipy.stats import gaussian_kde 
from generating_red_clump import GenerateRedClump 

plt.rcParams['font.family']      = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
logging.basicConfig(level=logging.INFO) 


class ColorMagnitudeDiagrams(GenerateRedClump): 
    """     
    Plots color-magnitude diagrams filter1 - filter2 vs. filtery 
    for the entire photometric catalog or just for the RC clump. 
    """

    def __init__( 
        self, 
        filter1: str, 
        filter2: str, 
        filtery: str, 
        region1: str, 
        region2: str, 
        regiony: str, 
        output_dir: Path = Path(__file__).parent.parent / "media" 
    ): 

        # Generates self.m1, self.m2, self.my magnitudes from args  
        super().__init__(filter1, filter2, filtery, region1, region2, regiony) 
        self.logger  = logging.getLogger(__name__)

        output_dir.mkdir(parents=True, exist_ok=True) 
        self.out_dir = output_dir 

    def plot_full_CMD(self, export=True): 
        """ 
        Plots the full CMD colored by density:  
        
        (region1 filter1 - region2 filter2) vs. regiony filtery 
        """

        figure, axis = plt.subplots(1, 1, figsize=(8, 6)) 
        
        color = np.subtract(self.m1, self.m2) # x 
        magnitude = np.asarray(self.my)       # y 

        # coloring by density using a gaussian kde 
        c = self._color_by_density(x=color, y=magnitude) 

        axis.scatter(color, magnitude, c=c, s=20, marker="d", alpha=0.6) 
        axis.set_xlabel(f"{self.region1} {self.filter1} - {self.filter2} (mag)", fontsize=15) 
        axis.set_ylabel(f"{self.regiony} {self.filtery} (mag)", fontsize=15) 
        axis.invert_yaxis() 
        plt.tight_layout() 

        if export: 
            out_dir = self.out_dir / "full_CMDs"
            out_dir.mkdir(exist_ok=True, parents=True)
            filename = out_dir / f"{self.region1}_{self.filter1}-{self.filter2}_{self.filtery}.png" 
            figure.savefig(filename, dpi=300)
            self.logger.info(f" Full CMD Figure saved to {Path(*filename.parts[-3:])}")
        else: 
            plt.show() 

    def _color_by_density(self, x: np.ndarray, y: np.ndarray):         
        xy = np.vstack([x, y]) 
        return gaussian_kde(xy)(xy) 


    def _rc_stars_from_pickle(self, region: str): 
        try: 
            with open(f"{self.pkl_dir}/{region}.pickle", "rb") as f: 
                red_clump_data = pickle.load(f)
            return red_clump_data 
        except FileNotFoundError as e: 
            self.logger.error(f" Pickle dir: {self.pkl_dir}/{region}.pickle not found")
            raise e 

    def _get_rc_stars(self, use_pickle: bool): 
        if use_pickle: 
            try: 
                return self._rc_stars_from_pickle(region=self.region1) 
            except: 
                self.logger.warning(
                    " Pickle extraction failed. "
                    "Extracting RC manually."
                )
                pass 
        return self.extract_red_clump_stars(export=False) 

    def plot_red_clump(self, use_pickle=True, export=True): 
        """ 
        Plots the extracted red clump from the CMD. 

        If pickle is True, it tries taking the RC data directly from 
        stored pickle files in /assets/. Otherwise, it manually 
        recalculates the RC data and plots. 
        """

        red_clump_data = self._get_rc_stars(use_pickle=use_pickle)

        MAG_KEYS = { 
            "F115W": "mF115W", 
            "F212N": "mF212N", 
            "F323N": "mF323N", 
            "F405N": "mF405N", 
        }

        m1 = red_clump_data[MAG_KEYS.get(self.filter1)] 
        m2 = red_clump_data[MAG_KEYS.get(self.filter2)] 
        my = red_clump_data[MAG_KEYS.get(self.filtery)]

        color = np.subtract(m1, m2) 
        magnitude = np.asarray(my) 

        # coloring by density using a gaussian kde 
        c = self._color_by_density(x=color, y=magnitude)

        figure, axis = plt.subplots(1, 1, figsize=(8, 6)) 

        axis.scatter(color, magnitude, c=c, s=20, marker="d", alpha=0.8) 
        axis.set_xlabel(f"{self.region1} {self.filter1} - {self.filter2} (mag)", fontsize=15)
        axis.set_ylabel(f"{self.regiony} {self.filtery} (mag)", fontsize=15) 
        axis.invert_yaxis()
        plt.tight_layout() 

        if export: 
            out_dir = self.out_dir / "red_clump_CMDs" 
            out_dir.mkdir(exist_ok=True, parents=True) 
            filename = out_dir / f"{self.region1}_{self.filter1}-{self.filter2}_{self.filtery}.png" 
            figure.savefig(filename, dpi=300) 
            self.logger.info(f" RC CMD Figure saved to {Path(*filename.parts[-3:])}.")
        else: 
             plt.show() 

if __name__ == "__main__": 
    instance = ColorMagnitudeDiagrams( 
        filter1="F115W", 
        filter2="F212N", 
        filtery="F115W", 
        region1="NRCB1", 
        region2="NRCB1", 
        regiony="NRCB1", 
    )

    instance.plot_full_CMD(export=True)
    instance.plot_red_clump(export=True)


        










        


