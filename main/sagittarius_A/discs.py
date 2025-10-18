import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from pathlib import Path 
from dataclasses import dataclass 
from typing import List, Tuple, Optional 

from distances import dist_from_sagA, SAGITTARIUS_A_COORDS

plt.rcParams["font.family"] = "serif" 

@dataclass(frozen=True)
class EqualAreaAnnuli: 
    edges: np.ndarray 

    @property 
    def pairs(self) -> List[Tuple[float, float]]: 
        """ 
        Returns [(r0, r1), (r1, r2), ... (r_{n-1}, r_n)] 

        Each tuple defines an inner and outer radii for a disc of 
        stars surrounding Sag A*. Each disc is of equal area. 
        """

        e = self.edges 
        return list(zip(e[:-1], e[1:])) 

    @classmethod 
    def render(
        cls, 
        distances : np.ndarray, 
        n_discs   : int, 
        rmax      : Optional[float] = None, 
        dropna    : bool = True  
    ) -> "EqualAreaAnnuli": 
        """ 
        Builds n equal-area annuli that span [0, rmax] where r_max is the farthest star 
        from Sagittarius A* (or the provided rmax if specified). 
        """

        d = np.asarray(distances, dtype=float).ravel() 
        if dropna: 
            d = d[~np.isnan(d)] 
        if d.size == 0: 
            raise ValueError("distance array empty after NaNs removed") 
        if (d < 0).any(): 
            raise ValueError("distances should be non-negative") 
        if n_discs < 1: 
            raise ValueError("n must be >= 1.") 

        r_max = float(np.max(d)) if rmax is None else rmax 
        if r_max <= 0: 
            raise ValueError("rmax should be > 0") 

        # equal-area edges: r_k = sqrt(k/n) * r_max 
        edges = np.sqrt(np.linspace(0.0, 1.0, n_discs + 1)) * r_max 
        return cls(edges=edges) 

    def bin_idxs(self, distances: np.ndarray) -> np.ndarray: 
        """ 
        For each distance in distances, reutrn the annuli index [0, n-1] 
        that star is contained in. distances beyond the edge are clipped to the 
        outermost bin. 

        """

        d   = np.asarray(distances, dtype=float).ravel() 
        idx = np.searchsorted(self.edges, d, side="right") - 1 
        
        return np.clip(idx, 0, len(self.edges) - 2) 



class OrganizeByDisc: 
    def __init__(
        self, 
        catalog: Optional[pd.DataFrame] = None, 
        n_discs: int = 6, 
    ):
        self.catalog, self.distances = dist_from_sagA(catalog)

        annuli = EqualAreaAnnuli.render(self.distances, n_discs) 
        self.radii    = annuli.pairs 
        self.bin_idxs = annuli.bin_idxs(self.distances)
        self.n_discs  = n_discs

    def build_disc_idxs(self): 
        # { 0: [<idxs of stars in disc 0>], ..., n:  } 
        disc_idxs = {i: [] for i in range(0, self.n_discs)} 

        for idx, bin_idx in enumerate(self.bin_idxs): 
            disc_idxs[bin_idx].append(idx)

        return disc_idxs 

    def build_disc_catalog(self, min_count: int = 1000):
        disc_idxs    = self.build_disc_idxs()
        assert self.catalog is not None, "catalog is empty"

        # { 0: <pd.DataFrame> of stars in disc 0, ..., n-1: ... }
        disc_catalog = {
            k: self.catalog.iloc[idxs] for k, idxs in disc_idxs.items() 
        }

        # fuse sparse bins; require at least min_count in each bin 
        # otherwise fuse to bin below it 
        frames = [disc_catalog[k] for k in range(self.n_discs)]   
        pairs  = list(self.radii)                                  

        i = 0
        while i < len(frames):
            # enough; skip 
            if len(frames[i]) >= min_count or len(frames) == 1:
                i += 1
                continue

            if i == 0:
                # inner most; fuse to outer 
                frames[1] = pd.concat([frames[1], frames[0]], ignore_index=False)
                pairs[1]  = (pairs[0][0], pairs[1][1])  # expand inward boundary
                del frames[0]; del pairs[0]
            else:
                # general; fuse to inner 
                frames[i-1] = pd.concat([frames[i-1], frames[i]], ignore_index=False)
                pairs[i-1]  = (pairs[i-1][0], pairs[i][1])  # extend outer boundary
                del frames[i]; del pairs[i]
                i -= 1

        # reindex 0..m-1 
        disc_catalog_fused = {k: df.sort_index() for k, df in enumerate(frames)}

        # update metadata
        self.radii   = pairs
        self.n_discs = len(pairs)

        return disc_catalog_fused

    def plot_disc(
        self, 
        out_dir: Path = Path(__file__).parent / "output/disc_plots/" 

    ): 
        disc_catalog = self.build_disc_catalog() 
        n = [k for k in range(self.n_discs)]

        colors = plt.cm.cool(np.linspace(0, 1, len(n))) # type: ignore

        _, _ = plt.subplots(1, 1, figsize=(8, 8)) 

        # idx equal to k, just idiomatic 
        for idx, k in enumerate(n): 
            disc_unique  = disc_catalog[k] 

            plt.scatter(
                disc_unique["ra"], 
                disc_unique["dec"], 
                s=5, 
                color=colors[idx], 
                label=f"disc {k} | {len(disc_unique)} stars", 
                marker="d"
            )

        plt.scatter(
            SAGITTARIUS_A_COORDS[0], 
            SAGITTARIUS_A_COORDS[1], 
            s=40, 
            c='yellow', 
            marker="*", 
            label="Sagittarius A*", 
        )



        plt.xlabel("right ascension (deg)", fontsize=15)
        plt.ylabel("declination (deg)",     fontsize=15) 
        plt.legend() 

        out_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{n[-1] + 1}_discs_plot.png"
        plt.savefig(out_dir / fname, dpi=300)

if __name__ == "__main__": 

    OrganizeByDisc(n_discs=100).plot_disc()


