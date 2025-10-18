import os, pickle, logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from matplotlib.colors import PowerNorm
from astropy.convolution import Gaussian2DKernel, convolve
from sklearn.neighbors import NearestNeighbors
from isochrones import Isochrones
from scipy.stats import gaussian_kde

logging.basicConfig(level=logging.INFO)

def filter_top_density(x: np.ndarray, y: np.ndarray, ye: np.ndarray, top_fraction: float = 0.1):
    """
    Return the subset of (x, y) points that lie in the top `top_fraction`
    most-dense regions of the scatter.

    """

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)                 
    dens = kde(xy)                         
    cutoff = np.quantile(dens, 1 - top_fraction)

    mask = dens >= cutoff
    return x[mask], y[mask], ye[mask]


def plot_unsharp_hess(
    mag1, mag2, magy,                # photometry
    mag1err, mag2err, magyerr,       # errors
    filt1, filt2, filty, 
    region       = "NRCB1",  
    magerr_max   = 1.0,              # sigma cut
    binsize_mag  = 0.02,             # y-axis bin width  (mag)
    binsize_clr  = 0.02,             # x-axis bin width  (mag)
    gauss_sigma  = 0.3,              # sigma of blur kernel (mag)
    gamma        = 3.0,              # PowerNorm scaling
    amount       = 0.0,               # sharpening strength; 0 is off
    extent       = None,             # (xmin,xmax,ymax,ymin)
    figsize      = (10,8),
    cmap         = 'viridis',
    savepath     = "./outputs/unsharp_mask/plot/",             
    picklepath   = "./outputs/unsharp_mask/data/",      
    verbose      = True, 
    plot_fritz   = True, 
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
        logging.info(f" Hess Diagram for {filt1} - {filt2} vs. {filty}.")
        logging.info(f" Keeping {good.sum():,} / {mag1.size:,} stars.")

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
        hess += np.outer(pdf_y, pdf_x)              

    # unsharp mask
    kernel = Gaussian2DKernel(gauss_sigma / binsize_mag)   # sigma in pixels (y)
    blurred = convolve(hess, kernel)
    sharpen = (1 + amount) * hess - amount * blurred       # std

    floor = sharpen[sharpen>0].min() * 1e-2
    data_for_plot = np.clip(sharpen, floor, None)
    pwr_norm = PowerNorm(gamma=gamma, vmin=floor, vmax=sharpen.max())

    fig, ax = plt.subplots(figsize=figsize)
    orig_extent = (clr_bins[0], clr_bins[-1], mag_bins[-1], mag_bins[0])
    im = ax.imshow(
        data_for_plot,
        origin='upper',
        extent=orig_extent,
        cmap=cmap,
        norm=pwr_norm,
        aspect='auto'
    ) 

    import pickle 
    with open("./xy.pickle", "rb") as f: 
        [x_rot, y_rot] = pickle.load(f) 

    colors = plt.cm.cool(np.linspace(0.2, 0.7, len(x_rot))) 
    count = 0
    for x, y in zip(x_rot, y_rot): 
        ax.scatter(
            x-16.5, y+17,
            c=colors[count],             # or some other scalar array
            cmap='cool',          # your choice
            norm=PowerNorm(gamma=1.0),  # optional normalization
            s=4,
            edgecolors='none',
            alpha=0.7,
            zorder=2,
        )
        count+=1

    import matplotlib.cm as cm
    cmap_obj    = cm.get_cmap(cmap)               # map 'viridis' → Colormap
    normed_floor = pwr_norm(pwr_norm.vmin)        # e.g. 0.0
    bg_rgba     = cmap_obj(normed_floor)          # RGBA tuple

    ax.set_facecolor(bg_rgba)


    """
    # ─── NEW: overplot the top‐density points ─────────────────────────
    # compute color & mag arrays for scatter
    full_x = m1 - m2            # x-axis = filt1 - filt2
    full_y = my                 # y-axis = filty
    # pick out, say, the top 10% most dense:
    x_top, y_top, _ = filter_top_density(full_x, full_y, mag_e, top_fraction=0.4)
    # scatter them as small neon points

    xy = np.vstack([x_top, y_top])
    z = gaussian_kde(xy)(xy) 

    ax.scatter(
        x_top, y_top,
        s=4,                   # small points
        marker="D",
        c=z,             # neon‐style color
        cmap='plasma',
        edgecolors='none',
        alpha=0.8,
        label=r'$f_{top}=40$'
    )

    x1, y1 = 7.25, 21.25
    x2, y2 = 8.4, 23.45

    m0 = (y2 - y1)/(x2 - x1)
    xm = 0.5*(x1 + x2)
    ym = m0*xm + (y1 - m0*x1)  # or 0.5*(y1+y2)

    # 2) extreme slopes ±10%, both forced through (xm,ym)
    δ    = 0.30
    m_lo = m0*(1 - δ)
    m_hi = m0*(1 + δ)
    b_lo = ym - m_lo*xm
    b_hi = ym - m_hi*xm

    # 3) full x-range & compute y
    xmin, xmax = extent[0], extent[1]
    x = np.linspace(xmin, xmax, 200)
    y0   = m0  * x + (y1 - m0*x1)  # central
    y_lo = m_lo* x + b_lo          # lower edge
    y_hi = m_hi* x + b_hi          # upper edge

    # 4) plot the wedge
    ax.plot(   x, y0,   color='white', lw=2, label=r'MCMC Linear Fit to $f_{top}$')
    ax.fill_between(x, y_lo, y_hi,
                    color='white', alpha=0.3,
                    label='±30% sweep')

    # 5) tiny horizontal tick at the midpoint as a pivot cue
    y_tick = ym
    dx = (xmax - xmin)*0.005
    ax.plot([xm-dx, xm+dx], [y_tick, y_tick],
            color='white', lw=3)


    """
    ax.set_xlabel(f'{filt1} - {filt2}  (mag)', fontsize=16)
    ax.set_ylabel(f'{filty}  (mag)', fontsize=16)
    plt.colorbar(im, ax=ax, label='stars / bin')
    plt.tight_layout()
    
    def densest_point(x: np.ndarray, y: np.ndarray, k: int = 10) -> tuple[float, float]:
        
        # Given 1D arrays x, y of equal length N, returns the (x, y) coordinate
        # among the samples that has the highest kNN-based density estimate.
        
        pts = np.column_stack((x, y))
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(pts)

        distances, _ = nbrs.kneighbors(pts)
        r_k = distances[:, k]
        densities = k / (np.pi * r_k**2 * len(pts))
        idx = np.argmax(densities)
        return (float(x[idx]), float(y[idx]))

    if plot_fritz: 
        slope = Isochrones(filt1, filt2, filty).calculate_slope() 

        x = np.subtract(mag1, mag2) 
        y = np.array(magy)

        xy1 = densest_point(x, y)
        ax.axline(xy1=xy1, slope=slope, c='r')

    if savepath:
        os.makedirs(savepath, exist_ok=True) 
        filename = f"{savepath}HESS_{region}_{filt1}-{filt2}_{filty}.png" 
        fig.savefig(filename, dpi=300)
        if verbose:
            logging.info(f" Hess written to {filename}")

    # save to pkl
    if picklepath:
        os.makedirs(picklepath, exist_ok=True)
        filename = f"{picklepath}_{region}_{filt1}_{filt2}_{filty}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(sharpen,  f)
            pickle.dump(mag_bins, f)
            pickle.dump(clr_bins, f)
        if verbose:
            logging.info(f" pkl written to {filename}")

    return fig, ax, sharpen, mag_bins, clr_bins



