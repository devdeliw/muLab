import numpy as np

from typing import Tuple
from pathlib import Path
from spisea import reddening, synthetic, evolution, atmospheres 

SPISEA_FILTER_MAP = {
    "F115W": "jwst,F115W", 
    "F212N": "jwst,F212N", 
    "F323N": "jwst,F323N", 
    "F405N": "jwst,F405N", 
}

class IsochroneSlopes(): 
    """ 
    Calculates the predicted RC slopes from the Fritz+11 extinction law
    for a given filter1 - filter2 vs. filtery CMD. 
    """

    def __init__(
        self,
        filter1: str, 
        filter2: str,
        filtery: str, 
        iso_dir: Path = Path(__file__).parent.parent / "raw/isochrones/"
    ): 
        self.filter1 = filter1 
        self.filter2 = filter2 
        self.filtery = filtery 

        self.evo_model = evolution.MISTv1() 
        self.atm_func  = atmospheres.get_merged_atmosphere 
        self.red_law   = reddening.RedLawFritz11(scale_lambda=2.166)
        
        iso_dir.mkdir(exist_ok=True, parents=True) 
        self.iso_dir = iso_dir 

    def generate_isochrone(
        self, 
        AKs: float, 
        logAge: float = np.log10(10**9), 
        distance: float = 8000.0
    ):
        filter_list = [
            SPISEA_FILTER_MAP.get(filter) 
            for filter in (self.filter1, self.filter2)
        ]
        isochrone = synthetic.IsochronePhot(
            logAge=logAge, 
            AKs=AKs, 
            distance=distance, 
            filters=filter_list, 
            red_law=self.red_law, 
            atm_func=self.atm_func, 
            evo_model=self.evo_model, 
            iso_dir=str(self.iso_dir / f"{self.filter1}-{self.filter2}_{self.filtery}"), 
        )

        mass = isochrone.points["mass"] 
        idx  = np.flatnonzero(abs(mass) == min(abs(mass))) 
        return isochrone, idx 

    def reddening_slope(self): 
        """
        Calculates the slope of the extinction vector for a 
        filter1 - filter2 vs. filtery CMD using the Fritz+11 law. 
        """

        # two isochrones of increasing extinction
        iso_ext_1, idx1 = self.generate_isochrone(AKs=0) 
        iso_ext_2, idx2 = self.generate_isochrone(AKs=1) 

        def get_pt0(ext, key_idx, star_idx):
            key = list(ext.points.keys())[key_idx]
            return ext.points[key][star_idx][0]

        # finding the same mass point on each isochrone 
        # and evaluating slope between
        iso_idx = 8 if self.filter1 == self.filtery else 9 

        y2_y1 = (get_pt0(iso_ext_1, iso_idx, idx1) - get_pt0(iso_ext_2, iso_idx, idx2))
        x2_x1 = (
            (get_pt0(iso_ext_1, 8, idx1) - get_pt0(iso_ext_1, 9, idx1)) -
            (get_pt0(iso_ext_2, 8, idx2) - get_pt0(iso_ext_2, 9, idx2))
        )
        return y2_y1 / x2_x1 


if __name__ == "__main__": 
    filter_combinations = [
        ("F115W", "F212N", "F115W"), 
        ("F115W", "F323N", "F115W"), 
        ("F115W", "F405N", "F115W")
    ]

    for (filt1, filt2, filty) in filter_combinations: 
        instance = IsochroneSlopes(filt1, filt2, filty)
        slope = instance.reddening_slope() 
        print(f"{filt1} - {filt2} vs. {filty}: {slope}")

        
        
        

