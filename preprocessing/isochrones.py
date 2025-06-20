import os 
import numpy as np 
from spisea import evolution, atmospheres, reddening, synthetic 

SPISEA_MAP = { 
    "F115W": "jwst,F115W", 
    "F212N": "jwst,F212N", 
    "F323N": "jwst,F323N", 
    "F405N": "jwst,F405N", 
} 

class Isochrones: 

    """ 
    Calculates the slope of a Fritz+11 extinction law vector for a given 
    `filt1 - filt2 vs. filty` color magnitude diagram. This vector traverses 
    the red clump bar of the diagram. 

    Args: 
        * filt1 (str): first filter for diagram 
        * filt2 (str): second filter for diagram
        * filty (str): filter plotted on y axis of diagram 
            * Note should either be equal to filt1 or filt2 
    
    calculate_slope() returns the slope of the fritz vector. 

    """

    def __init__(self, filt1, filt2, filty): 
        self.filt1 = filt1 
        self.filt2 = filt2 
        self.filty = filty

    def generate_isochrone(self, AKs, filt_list): 
        evo_model = evolution.MISTv1() 
        atm_func  = atmospheres.get_merged_atmosphere 
        red_law   = reddening.RedLawFritz11(scale_lambda=2.166) 

        filt1, filt2 = filt_list 
        filt_list = [SPISEA_MAP.get(filt) for filt in filt_list] 

        # directory to place isochrone files 
        iso_dir = f"./outputs/isochrones/{filt1}-{filt2}/"
        os.makedirs(iso_dir, exist_ok=True)

        # generate isochrone with SPISEA 
        isochrone = synthetic.IsochronePhot( 
            logAge=np.log10(10**9),     # rough age of galactic center 
            AKs=AKs,                    # extinction 
            distance=8000,              # distance in pc to galactic center 
            filters=filt_list,          # filters to calculate isochrone 
            red_law=red_law,            # reddening law is Fritz+11  
            atm_func=atm_func, 
            evo_model=evo_model, 
            iso_dir=iso_dir,            
        ) 

        mass = isochrone.points['mass'] 
        idx = np.flatnonzero(abs(mass) == min(abs(mass)))
        return isochrone, idx  

    def calculate_slope(self): 
        filt_list = [self.filt1, self.filt2] 
        
        iso_ext_1, idx1 = self.generate_isochrone(0, filt_list) 
        iso_ext_2, idx2 = self.generate_isochrone(1, filt_list) 

        iso_idx = 8 if self.filt1 == self.filty else 9 
        y2_y1 = iso_ext_1.points[''+iso_ext_1.points.keys()[iso_idx]][idx1][0] - iso_ext_2.points[''+iso_ext_2.points.keys()[iso_idx]][idx2][0]
        x2_x1 = (iso_ext_1.points[''+iso_ext_1.points.keys()[8]][idx1][0] - iso_ext_1.points[''+iso_ext_1.points.keys()[9]][idx1][0]) - (
                 iso_ext_2.points[''+iso_ext_2.points.keys()[8]][idx2][0] - iso_ext_2.points[''+iso_ext_2.points.keys()[9]][idx2][0])

        return y2_y1 / x2_x1 

if __name__ == "__main__": 
    filt_combinations = [
        ["F115W", "F212N", "F115W"],
        ["F115W", "F212N", "F212N"], 
        ["F212N", "F323N", "F323N"], 
        ["F212N", "F405N", "F212N"], 
    ]

    filt_combinations = [ 
        ["F115W", "F212N", "F115W"], 
        ["F115W", "F323N", "F115W"], 
        ["F115W", "F405N", "F115W"], 
    ]

    for [filt1, filt2, filty] in filt_combinations:
        inst = Isochrones(filt1, filt2, filty) 
        slope = inst.calculate_slope()
        print(f"{inst.filt1}-{inst.filt2} vs. {inst.filty}: {slope}")






         
