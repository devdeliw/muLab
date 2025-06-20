import os, pickle, logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import PowerNorm
from astropy.convolution import Gaussian2DKernel, convolve


logging.basicConfig(level=logging.INFO)

def plot_unsharp_hess(
    mag1, mag2, magy,                # photometry
    mag1err, mag2err, magyerr,       # errors
    filt1, filt2, filty, 
    magerr_max   = 1.0,              # σ cut
    binsize_mag  = 0.02,             # y-axis bin width  (mag)
    binsize_clr  = 0.02,             # x-axis bin width  (mag)
    gauss_sigma  = 0.6,              # σ of blur kernel (mag)
    gamma        = 3.0,              # PowerNorm scaling 
    amount       = 5.0,              # sharpening strength; 0→off
    extent       = None,             # (xmin,xmax,ymax,ymin)
    figsize      = (10,8),
    cmap         = 'magma',
    savepath     = "./outputs/unsharp_mask/plot/",             # path or None
    picklepath   = "./outputs/unsharp_mask/data/",             # path or None
    verbose      = True
):
    """
    Plot an unsharp–masked Hess diagram.

    Parameters
    ----------
    gauss_sigma : float
        Mask width *in magnitudes* (same definition as De Marchi +2016).
    amount : float
        0  → original Hess  
        1  → classic unsharp mask (original − blurred)  
        >1 → extra contrast
    savepath / picklepath : str or None
        If given, save PNG or (hist,magbins,clrbins) pickle.
    """

    # preprocessing
    good = ( 
        np.isfinite(mag1) & np.isfinite(mag2) & 
        (mag1err <= magerr_max) & 
        (mag2err <= magerr_max) &
        (mag1err > 0) & 
        (mag2err > 0)
    )

    if verbose:
        logging.info(f"\n[INFO] Hess Diagram for {filt1} - {filt2} vs. {filty}.")
        print(f"[INFO] Keeping {good.sum():,} / {mag1.size:,} stars.")

    m1, m2, my  = mag1[good],  mag2[good],  magy[good]
    e1, e2, ey  = mag1err[good], mag2err[good], magyerr[good]

    colour   = m1 - m2
    colour_e = np.hypot(e1, e2)
    mag      = my
    mag_e    = ey

    ## bin edges 
    mag_bins = np.arange(mag.min()-mag_e.max(),
                         mag.max()+mag_e.max(),  binsize_mag)
    clr_bins = np.arange(colour.min()-colour_e.max(),
                         colour.max()+colour_e.max(), binsize_clr)

    n_y, n_x = len(mag_bins)-1, len(clr_bins)-1     # (rows, cols)

    ## error-weighted histogram
    hess = np.zeros((n_y, n_x))

    for m, dm, c, dc in zip(mag, mag_e, colour, colour_e):
        pdf_y = np.diff(norm(m,dm).cdf(mag_bins))
        pdf_x = np.diff(norm(c,dc).cdf(clr_bins))
        hess += np.outer(pdf_y, pdf_x)              # auto-broadcast

    # unsharp mask
    kernel = Gaussian2DKernel(gauss_sigma / binsize_mag)   # σ in pixels (y)
    blurred = convolve(hess, kernel)
    sharpen = (1 + amount) * hess - amount * blurred       # std. formula

    floor = sharpen[sharpen>0].min() * 1e-2
    data_for_plot = np.clip(sharpen, floor, None)
    pwr_norm = PowerNorm(gamma=gamma, vmin=floor, vmax=sharpen.max())

    # plot 
    if extent is None:
        extent = (clr_bins[0], clr_bins[-1], mag_bins[-1], mag_bins[0])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data_for_plot,
        origin='upper',
        extent=extent,
        cmap=cmap,
        norm=pwr_norm,
        aspect='auto'
    )
    ax.set_xlabel(f'{filt1} - {filt2}  (mag)', fontsize=14)
    ax.set_ylabel(f'{filty}  (mag)', fontsize=14)
    plt.colorbar(im, ax=ax, label='stars / bin')
    ax.set_title(f'Hess Diagram {filt1} - {filt2} vs. {filty}', fontsize=16)
    plt.tight_layout()

    if savepath:
        os.makedirs(savepath, exist_ok=True) 
        filename = f"{savepath}HESS_{filt1}-{filt2}_{filty}.png" 
        fig.savefig(filename, dpi=300)
        if verbose:
            print(f"\n[INFO] Hess diagram written to {filename}")

    # save to pkl
    if picklepath:
        os.makedirs(picklepath, exist_ok=True)
        filename = f"{picklepath}_{filt1}_{filt2}_{filty}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(sharpen,  f)
            pickle.dump(mag_bins, f)
            pickle.dump(clr_bins, f)
        if verbose:
            print(f"[INFO] Hess pkl written to {filename}")

    return fig, ax, sharpen, mag_bins, clr_bins



