import pickle 
import numpy as np 
import pandas as pd 
from pathlib import Path 
from typing import Optional 

ALL_RC_PICKLE        = Path(__file__).parent.parent / "assets/ALL_RC.pickle" 
SAGITTARIUS_A_COORDS = (266.41683, -29.00781) # (ra, dec) 


def load_default_catalog() -> pd.DataFrame: 
    with open(ALL_RC_PICKLE, "rb") as f: 
        return pickle.load(f) 

def dist_from_sagA(
        catalog: Optional[pd.DataFrame] = None
):
    """
    Given a catalog of stars from `ALL_RC_PICKLE`, output a np.ndarray containing 
    the star's distances from Sagittarius A*, the supermassive black hole at center 
    of the Milky Way. 
    """
    
    if catalog is None: 
        catalog = load_default_catalog()

    assert isinstance(catalog, pd.DataFrame), "catalog must be pandas DataFrame" 
    assert "ra" in catalog.columns and "dec" in catalog.columns, \
    (f" 'ra' and 'dec' must be column headers in the catalog at {ALL_RC_PICKLE}.")

    right_ascension = np.asarray(catalog["ra"],  dtype=float) 
    declination     = np.asarray(catalog["dec"], dtype=float)

    coordinates = np.stack([right_ascension, declination], axis=1) 

    return catalog, np.hypot(
        coordinates[:, 0] - SAGITTARIUS_A_COORDS[0], 
        coordinates[:, 1] - SAGITTARIUS_A_COORDS[1] 
    )



