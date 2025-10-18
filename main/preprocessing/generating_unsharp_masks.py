import logging 
import numpy as np 
import matplotlib.pyplot as plt 

from pathlib import Path
from scipy.stats import norm
from dataclasses import dataclass
from matplotlib.colors import PowerNorm 
from generating_red_clump import GenerateRedClump
from astropy.convolution import Gaussian2DKernel, convolve  

logging.basicConfig(level=logging.INFO)

@dataclass 
class UnsharpMaskConfig: 
    """
    Configuration dataclass for UnsharpMask 

    Args: 
        * magerr_max  (float):  sigma cut 
        * binsize_mag (float):  y-axis bin width (mag) 
        * binsize_clr (float):  x-axis bin width (mag) 
        * gauss_sigma (float):  sigma of blur kernel (mag)
        * gamma       (float):  PowerNorm scaling 
        * sharpen     (float):  sharpening strength; 0 is off 
        * output_dir  (Path) :  output directory for unsharp mask figures
    """ 

    magerr_max:  float = 1.0 
    binsize_mag: float = 0.02
    binsize_clr: float = 0.02
    gauss_sigma: float = 0.3 
    gamma:       float = 3.0
    sharpen:     float = 0.0 
    output_dir:  Path  = Path(__file__).parent.parent / "media/unsharp_masks/" 

class UnsharpMask(GenerateRedClump): 
    """ 
    Builds the unsharp-masked Hess Diagram outlined in DeMarchi (2016) 
    with an extra pixel intensity gamma scaling. 

    """

    def __init__(
        self, 
        filter1: str, 
        filter2: str, 
        filtery: str, 
        region1: str, 
        region2: str, 
        regiony: str,
        config: UnsharpMaskConfig = UnsharpMaskConfig()
    ): 
        """
        Args: 
            * filter1, region1 (str): name of filter & region for lower wavelength
            * filter2, region2 (str): name of filter & region for higher wavelength
            * filtery, regiony (str): name of filter & region on y-axis 
        """


        # Generates self.m1, self.m2, self.my magnitudes and errors from args 
        super().__init__(filter1, filter2, filtery, region1, region2, regiony) 
        self.config = config 
        self.logger = logging.getLogger(__name__)

    def _usable_stars(self): 
        """ 
        Returns usable stars for unsharp masking. 
        * star errors are inside the sigma cut 
        * star errors are positive and finite 
        """
        usable_star_mask = (
            (self.m1e <= self.config.magerr_max) & 
            (self.m2e <= self.config.magerr_max) & 
            (self.m1e > 0) & 
            (self.m2e > 0) 
        )

        self.logger.info(f" Hess Diagram for {self.filter1} - {self.filter2} vs. {self.filtery}")
        self.logger.info(f" Keeping {usable_star_mask.sum():,} / {self.m1.size:,} stars")

        m1, m2, my = self.m1[usable_star_mask], self.m2[usable_star_mask], self.my[usable_star_mask] 
        e1, e2, ey = self.m1e[usable_star_mask], self.m2e[usable_star_mask], self.mye[usable_star_mask] 
         
        color = np.subtract(m1, m2) 
        color_error = np.hypot(e1, e2) 

        magnitude = my 
        magnitude_error = ey 

        return color, color_error, magnitude, magnitude_error 

    def unsharp_mask(self, export=True): 
        color, color_error, magnitude, magnitude_error = self._usable_stars() 

        # bin edges 
        mag_bins = np.arange(
            magnitude.min() - magnitude_error.max(), 
            magnitude.max() + magnitude_error.max(), 
            self.config.binsize_mag
        )
        clr_bins = np.arange( 
            color.min() - color_error.max(), 
            color.max() + color_error.max(), 
            self.config.binsize_clr 
        )

        # number of rows, cols in 2D histogram 
        n_magnitude, n_color = len(mag_bins)-1, len(clr_bins) - 1

        # build the error-weighted 2D histogram 
        hess = np.zeros((n_magnitude, n_color))
        for m, dm, c, dc in zip(
            magnitude, magnitude_error, color, color_error 
        ): 
            pdf_mag = np.diff(norm(m, dm).cdf(mag_bins)) 
            pdf_clr = np.diff(norm(c, dc).cdf(clr_bins)) 
            hess += np.outer(pdf_mag, pdf_clr) 

        # unsharp mask 
        kernel    = Gaussian2DKernel(self.config.gauss_sigma / self.config.binsize_mag)  
        blurred   = convolve(hess, kernel) 
        sharpened = (1 + self.config.sharpen) * hess - self.config.sharpen * blurred 

        # power norm 
        vmin = sharpened[sharpened>0].min() * 1e-2 
        vmax = sharpened.max() 
        pwr_norm = PowerNorm(gamma=self.config.gamma, vmin=vmin, vmax=vmax) 

        figure, axis = plt.subplots(1, 1, figsize=(8, 6))
        plot_extent  = (clr_bins[0], clr_bins[-1], mag_bins[-1], mag_bins[0]) 

        hess_diagram = np.clip(sharpened, vmin, None) 
        im = axis.imshow(
            hess_diagram, 
            origin='upper', 
            extent=plot_extent, 
            cmap='viridis', 
            norm=pwr_norm, 
            aspect='auto' 
        ) 

        axis.set_xlabel(f"{self.region1} {self.filter1} - {self.filter2}", fontsize=15)
        axis.set_ylabel(f"{self.region1} {self.filtery}", fontsize=15)
        plt.colorbar(im, ax=axis, label="stars / bin") 
        plt.tight_layout()
        

        if export: 
            self.config.output_dir.mkdir(parents=True, exist_ok=True) 
            filename = self.config.output_dir / f"{self.region1}_{self.filter1}-{self.filter2}_{self.filtery}.png" 
            figure.savefig(filename, dpi=300) 
            self.logger.info(f" Unsharp Mask Figure saved to {Path(*filename.parts[-3:])}")  
 

if __name__ == "__main__": 
    instance = UnsharpMask(
        filter1 = "F115W", 
        filter2 = "F212N", 
        filtery = "F115W", 
        region1 = "NRCB1", 
        region2 = "NRCB1", 
        regiony = "NRCB1", 
        config  = UnsharpMaskConfig() 
    )
    instance.unsharp_mask(export=True)




        







