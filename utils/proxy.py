#!/usr/bin/env python3
"""
m212 vs. colour-ratio proxy plot with density-coloured scatter
-------------------------------------------------------------
* Viridis  → F323N points
* Magma    → F405N points
* Mean ± SEM overlay in black

Inputs
------
mag_dict : {
    "F115W": np.ndarray,
    "F212N": np.ndarray,
    "F323N": np.ndarray,
    "F405N": np.ndarray,
}
nbins : int (equal-count bins for overlay), default 8
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ---------- helper ----------------------------------------------------------
def _density_colours(x, y, cmap, bw=0.15):
    """Return RGBA colours mapped to point density; fallback to solid colour."""
    try:
        xy   = np.vstack([x, y])
        kde  = gaussian_kde(xy, bw_method=bw)
        dens = kde(xy)
        dens = (dens - dens.min()) / (dens.max() - dens.min())
        return plt.cm.get_cmap(cmap)(dens)
    except Exception:                      # NaNs, singular cov, any failure
        solid = plt.cm.get_cmap(cmap)(0.6)
        return np.repeat(solid[None, :], len(x), axis=0)
# ---------------------------------------------------------------------------


def proxy_plot(mag_dict, region, nbins=8):
    m115, m212 = map(np.asarray, (mag_dict["F115W"], mag_dict["F212N"]))
    m323, m405 = map(np.asarray, (mag_dict["F323N"], mag_dict["F405N"]))

    denom = m115 - m212
    base_mask = (denom > 0)

    # keep only fully finite rows
    finite = np.isfinite(m115) & np.isfinite(m212) & \
             np.isfinite(m323) & np.isfinite(m405)
    mask = base_mask & finite

    x = m212[mask]                                # proxy for A_212
    c323 = (m323[mask] - m212[mask]) / denom[mask]
    c405 = (m405[mask] - m212[mask]) / denom[mask]

    col323 = _density_colours(x, c323, "viridis")
    col405 = _density_colours(x, c405, "magma")

    # equal-count bins → mean ± SEM
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    mid   = 0.5 * (edges[:-1] + edges[1:])
    idx   = np.digitize(x, edges, right=False) - 1

    def _stats(arr):
        mean = np.array([arr[idx == k].mean() for k in range(nbins)])
        sem  = np.array([arr[idx == k].std(ddof=1) /
                         max((idx == k).sum(), 1)**0.5 for k in range(nbins)])
        return mean, sem

    m323, s323 = _stats(c323)
    m405, s405 = _stats(c405)

    # -------- plot --------
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(x, c323, s=10, c=col323, marker='o', lw=0, label='F323N')
    ax.scatter(x, c405, s=10, c=col405, marker='^', lw=0, label='F405N')

    ax.errorbar(mid, m323, yerr=s323, fmt='o', mfc='none', mec='k',
                ecolor='k', lw=1.2, capsize=2, label='323 mean')
    ax.errorbar(mid, m405, yerr=s405, fmt='s', mfc='none', mec='k',
                ecolor='k', lw=1.2, capsize=2, label='405 mean')

    ax.set_xlabel(r'$m_{212}$  (proxy for total $A_{212}$)')
    ax.set_ylabel(r'$(m_\lambda - m_{212})/(m_{115} - m_{212})$')
    ax.set_title('Extinction-law colour-ratio proxy')
    ax.grid(alpha=0.25, linestyle='--', linewidth=0.6)
    ax.legend(frameon=False)
    plt.tight_layout()

    fname = "/Users/devaldeliwala/mulab/outputs/proxy/"
    os.makedirs(fname, exist_ok=True)
    plt.savefig(f"{fname}{region}.png", dpi=300)
    plt.close()
    return fig


# --------------------------- driver ----------------------------------------
if __name__ == "__main__":
    import pickle, pathlib
    data_dir = pathlib.Path("/Users/devaldeliwala/mulab/outputs/red_clump_data")
    region   = "NRCB3"

    with open(data_dir / f"{region}.pkl", "rb") as f:
        rc = pickle.load(f)

    mags = {
        "F115W": rc["mF115W"],
        "F212N": rc["mF212N"],
        "F323N": rc["mF323N"],
        "F405N": rc["mF405N"],
    }
    proxy_plot(mags, region=region, nbins=10)

