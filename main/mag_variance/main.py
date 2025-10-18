import logging 
import numpy as np 
import pickle 
import matplotlib.pyplot as plt 

from pathlib import Path 
from scipy.stats import gaussian_kde
from modeling.running_mcmc import RunMCMC, RunMCMCConfig, RCConfig 

pkl_path = Path(__file__).parent / "pickle/" 
slope_dict = {} 

def run(n: int, min_size: int): 
    with open(pkl_path / f"catalog_n={n}_min_size={min_size}.pickle", "rb") as f: 
        catalog = pickle.load(f) 

    logging.info(f"running for {len(catalog)} bins")
    for idx in catalog.keys(): 
        t = catalog[idx] 

        mF212N = t["mF212N"] 
        mF115W = t["mF115W"] 
    
        mask = np.isfinite(mF212N) & np.isfinite(mF115W) 
        mF115W, mF212N = mF115W[mask], mF212N[mask] 

        run = RunMCMC( 
            filter1 = "F212N", 
            filter2 = "", 
            filtery = "F115W - F212N", 
            rc_cfg  = RCConfig( 
                m1=mF212N, 
                m2=np.zeros_like(mF212N), 
                my=np.subtract(mF115W, mF212N), 
            ), 
            config= RunMCMCConfig( 
                n_tiles     = 8, 
                output_dir  = Path(__file__).parent.parent / "media/MCMC/mag_variance/", 
                nwalkers    = 32, 
                nsteps      = 5000, 
            ), 
            load_pkl=False
        )

        run.run() 

        slope = run.slope 
        slope_error = run.slope_err 
        intercept = run.intercept 

        slope_dict[idx] = (slope, slope_error, intercept) 

        _, _ = plt.subplots(1, 1, figsize=(8, 6)) 

        x = mF212N 
        y = np.subtract(mF115W, mF212N) 

        xy = np.vstack([x, y]) 
        z = gaussian_kde(xy)(xy)

        plt.scatter(x, y, c=z, s=5) 
        plt.axline((0, intercept), (1, slope+intercept), color='red', linestyle='--', linewidth=2)

        plt.xlim(x.min(), x.max()) 
        plt.ylim(y.max(), y.min()) 

        plt.xlabel(r"$m_{F212N}$", fontsize=15) 
        plt.ylabel(r"$m_{F115W} - m_{F212N}$", fontsize=15)

        save_dir = Path(__file__).parent / "media/"
        save_dir.mkdir(exist_ok=True, parents=True) 
        fname = save_dir / f"mF212N_mF115W-mF212N_bin{idx}.png"
        plt.savefig(fname, dpi=300)
        logging.info(f"plot saved to {fname}") 

        with open(f"mag_variance/pickle/slope_dict_{len(catalog)}bins.pickle", "wb") as f: 
            pickle.dump(slope_dict, f) 


if __name__ == "__main__": 
    run(n=6, min_size=400)




