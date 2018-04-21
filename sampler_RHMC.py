# RHMC prototype sampler --- Fixed dimensional case
# Base class provides the infrastructure common to both single source and multiple source
# inference problem.
# - The truth image based on the user input
# - Constants including experimental condition
# Single trajectory class is used to achieve the following.
# - Show that the energy is conserved throughout its trajectory.
# - Show that the path is reversible as long as the base step size is made small.
# - Multiple sources can be included.
# - Produce a video that convinces for each of the case above.
# General inference class: 
# - Inference is possible even when the initial guess is orders of mangitude off from the true answer.
# - Perform inference of a single source with varying magnitudes.
# - Perform inference with many sources with large dynamic range [7, 22] magnitude!
# - Produce a video that convinces for each of the case above.

# Conventions:
# - All flux are in units of counts. The user only deals with magnitudes and the magitude to
# flux (in counts) conversion is done automatically and internally.
# - All object inputs are in the form (Nobjs, 3) with each row corresponding to an object 
# and its mag, x, y information.
# - The initial point is saved in 0-index position. If the user asks for N samples, then 
# the returned array cotains N+1 samples.

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

		# Compute factors to be used repeatedly.
		self.compute_factors()

		return

	def gen_mock_data(self, q_true=None, return_data = False):
		"""
		Given the properties of mock q_true (Nobjs, 3) with mag, x, y for each, 
		generate a mock data image. The following conditions must be met.
		- An object must be within the image.

		Gaussian PSF is assumed with the width specified by PSF_FWHM_pix.
		"""
		# Generate an image with background.
		data = np.ones((self.num_rows, self.num_cols), dtype=float) * self.B_count

		# Add one star at a time.
		for i in xrange(q_true.shape[0]):
			mag, x, y = q_true[i]
			data += self.mag2flux_converter(mag) * gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM = self.PSF_FWHM_pix)

		# Poission realization D of the underlying truth D0
		data = poisson_realization(data)

		if return_data:
			return data
		else:
			self.D = data            

	def mag2flux_converter(self, mag):
		"""
		Given user input magnitude convert it to flux counts.
		"""

		return mag2flux(mag) * self.flux_to_count


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


class single_gym(base_class):
	def __init__(self, Nsteps = 100, dt = 1.):
		"""
		Single trajectory simulation.
		- Nsteps: Number of steps to be taken.
		- dt: Global time step size factor.
		"""

		self.Nsteps = Nsteps
		self.dt = dt
		# ---- Place holder for various variables. 
		self.q_chain = None
		self.p_chain = None
		self.E_chain = None
		self.V_chain = None
		self.T_chain = None

		return
		