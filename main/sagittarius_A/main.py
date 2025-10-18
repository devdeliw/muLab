import logging
import pandas as pd 
import numpy as np 
import pickle 
import matplotlib.pyplot as plt 

from pathlib import Path 
from typing import Optional, List, Tuple, Dict
from discs import OrganizeByDisc
from modeling.running_mcmc import RunMCMC, RunMCMCConfig, RCConfig 

FILTERS = [ 
    ("F115W", "F212N", "F115W"), 
    ("F115W", "F323N", "F115W"), 
    ("F115W", "F405N", "F115W"), 
]

class RenderSagA(OrganizeByDisc):  
    def __init__(
        self, 
        catalog: Optional[pd.DataFrame] = None, 
        n_discs: int = 5, 
        n_tiles: int = 5, 
    ):

        super().__init__(catalog=catalog, n_discs=n_discs)
        self.n_tiles = n_tiles 
        self.disc_catalog = self.build_disc_catalog()

    def render_individual_disc(
        self, 
        disc_idx: int, 
        filters: Optional[List[Tuple[str, str, str]]] = None, 
        plot_CMD: bool = False,
        save_dir = Path(__file__).parent / "output/disc_CMDs/" 
    ):
        if filters is None: 
            filters = FILTERS

        slope_dict = {} 
        for (filter1, filter2, filtery) in filters: 
            m1 = self.disc_catalog[disc_idx][f"m{filter1}"] 
            m2 = self.disc_catalog[disc_idx][f"m{filter2}"] 
            my = self.disc_catalog[disc_idx][f"m{filtery}"] 

            # remove nans 
            mask = np.isfinite(m1) & np.isfinite(m2)
            m1 = m1[mask]
            m2 = m2[mask]
            my = my[mask]

            run = RunMCMC(
                filter1 = filter1, 
                filter2 = filter2, 
                filtery = filtery, 
                rc_cfg  = RCConfig(
                    m1=m1,
                    m2=m2, 
                    my=my, 
                ), 
                config = RunMCMCConfig( 
                    n_tiles        = self.n_tiles,
                    max_y_err      = 0.2, 
                    autocorr       = False, 
                    autocorr_bin   = 5, 
                    slope_minus    = 0.0,
                    output_dir     = Path(__file__).parent.parent / "media/MCMC/sagA/",
                    histogram_bins = 50, 
                    nwalkers       =  64,
                    nsteps         = 5000, 
                ),
                load_pkl=False, 
            )

            # run the MCMC on the RC disc   
            run.run()

            slope       = run.slope 
            slope_error = run.slope_err 
            intercept   = run.intercept
        
            if plot_CMD: 
                _, _ = plt.subplots(1, 1, figsize=(8, 6)) 

                x = np.subtract(m1, m2) 
                y = np.asarray(my) 
                plt.scatter(x, y, c='k', s=5) 
                plt.axline((0, intercept),  (1, slope + intercept), color='red', linestyle='--', linewidth=2)

                plt.xlim(x.min(), x.max()) 
                plt.ylim(y.max(), y.min())

                plt.xlabel(f"{filter1} - {filter2}", fontsize=15)
                plt.ylabel(f"{filter1}", fontsize=15)

                save_dir.mkdir(exist_ok=True, parents=True)
                fname = save_dir / f"{filter1}-{filter2}_{filtery}_{self.n_discs}discs_idx{disc_idx}.png"
                plt.savefig(fname, dpi=300)
                logging.info(f"plot saved to {fname}")

            slope_dict[(filter1, filter2, filtery)] = (slope, slope_error, intercept)

        return slope_dict 

    def render_all_discs(
        self, 
        filters: Optional[List[Tuple[str, str, str]]] = None, 
        save_pkl: bool = False, 
        load_pkl: bool = False, 
        pkl_path: Path = Path(__file__).parent / "pickle/", 
        plot_CMD: bool = False
    ):
        if load_pkl: 
            with open(pkl_path / "slope_discs.pickle", "rb") as f: 
                full_slope_res = pickle.load(f) 
        else: 
            full_slope_res = {} 

        print(f"Rendering {self.n_discs} Discs")

        # iterating through all discs
        for k in range(self.n_discs):
            slope_dict = self.render_individual_disc(
                disc_idx=k, 
                filters=filters, 
                plot_CMD=plot_CMD,
            )
            full_slope_res[k] = slope_dict 

        if save_pkl: 
            pkl_path.mkdir(parents=True, exist_ok=True) 
            with open(pkl_path / "slope_discs.pickle", "wb") as f: 
                pickle.dump(full_slope_res, f)

        # { 
        #   k: { 
        #       (filter1, filter2, filtery) : (slope, slope_error, intercept)
        #       ... 
        #   } 
        #   ... 
        # }
        return full_slope_res

def output_to_df(
    full_slope_res: Dict[int, Dict[Tuple[str, str, str], Tuple[float, float, float]]],
    disc_radii: List[Tuple[float, float]], 
    save_pkl: bool = True, 
    pkl_path: Path = Path(__file__).parent / "pickle/"
):
    rows = []
    for disc, inner in full_slope_res.items():
        for (f1, f2, f3), (slope, slope_err, intercept) in inner.items():
            rows.append({
                "disc": disc,
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "slope": slope,
                "slope_err": slope_err,
                "intercept": intercept,
                "radii": disc_radii[disc]
            })

    df = pd.DataFrame(rows)

    if save_pkl: 
        with open(pkl_path / "slope_df.pickle", "wb") as f: 
            pickle.dump(df, f)

    return df 


if __name__ == "__main__": 
    filters = [ 
        ("F115W", "F212N", "F115W"), 
    ]

    renderer = RenderSagA(n_discs=12, n_tiles=10)
    res = renderer.render_all_discs(filters=filters, save_pkl=True, load_pkl=False, plot_CMD=True)

    output_to_df(res, renderer.radii, save_pkl=True)
    
