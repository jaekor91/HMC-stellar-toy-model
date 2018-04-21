# RHMC prototype sampler. 
# Base class provides the infrastructure common to both single source and multiple source
# inference problem.
# - The truth image based on the user input
# - Constants including experimental condition
# Single source inference class is used to achieve the following.
# - Inference is possible even when the initial guess is orders of mangitude off from the true answer.
# - Show that the energy is conserved throughout its trajectory.
# - Show that the path is reversible as long as the base step size is made small.
# Multiple source class: 
# - Perform inference with many sources with large dynamic range [7, 22] magnitude!

from utils import *

class base_class(object):
    def __init__(self):
        """
		Sets default experimental or observational values, which can be changed later.
        """

        # Placeholder for data
        self.D = None

        # Placeholder for model
        self.M = None

        # Default experimental set-up
        self.num_rows, self.num_cols, self.flux_to_count, self.PSF_FWHM_pix, \
            self.B_count, self.arcsec_to_pix = self.default_exp_setup()

        return

    def gen_mock_data(self, q_true=None, return_data = False):
        """
        Given the properties of mock q_true (Nobjs, 3) with f, x, y for each, 
        generate a mock data image. The following conditions must be met.
        - An object must be within the image.

        Gaussian PSF is assumed with the width specified by PSF_FWHM_pix.
        """
        # Generate an image with background.
        data = np.ones((self.num_rows, self.num_cols), dtype=float) * self.B_count

        # Add one star at a time.
        for i in xrange(q_true.shape[0]):
            f, x, y = q_true[i]
            data += f * gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM = self.PSF_FWHM_pix)

        # Poission realization D of the underlying truth D0
        data = poisson_realization(data)

        if return_data:
            return data
        else:
            self.D = data            


    def compute_factors(self):
        """
        Compute constant factors that are useful RHMC_diag method.
        """
        self.g0, self.g1, self.g2 = factors(self.num_rows, self.num_cols, self.num_rows/2., self.num_cols/2., self.PSF_FWHM_pix)

        return

    def default_exp_setup(self):
        #---- A note on conversion
        # From Stephen: If you want to replicate the SDSS image of M2, you could use:
        # 0.4 arcsec per pixel, seeing of 1.4 arcsec
        # Background of 179 ADU per pixel, gain of 4.62 (so background of 179/4.62 = 38.7 photoelectrons per pixel)
        # 0.00546689 nanomaggies per ADU (ie. 183 ADU = 22.5 magnitude; see mag2flux(22.5) / 0.00546689)
        # Interpretation: Measurement goes like: photo-electron counts per pixel ---> ADU ---> nanomaggies.
        # The first conversion is called gain.
        # The second conversion is ADU to flux.

        # Flux to counts conversion
        # flux_to_count = 1./(ADU_to_flux * gain)

        #---- Global parameters
        arcsec_to_pix = 0.4
        PSF_FWHM_arcsec = 1.4
        PSF_FWHM_pix = PSF_FWHM_arcsec / arcsec_to_pix
        PSF_sigma = PSF_FWHM_arcsec
        gain = 4.62 # photo-electron counts to ADU
        ADU_to_flux = 0.00546689 # nanomaggies per ADU
        B_ADU = 179 # Background in ADU.
        B_count = B_ADU/gain
        flux_to_count = 1./(ADU_to_flux * gain) # Flux to count conversion

        #---- Default mag set up
        self.mB = 23
        B_count = mag2flux(self.mB) * flux_to_count

        # Size of the image
        num_rows = num_cols = 48 # Pixel index goes from 0 to num_rows-1

        return num_rows, num_cols, flux_to_count, PSF_FWHM_pix, B_count, arcsec_to_pix
