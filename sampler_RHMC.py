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
	def __init__(self, dt = 1., g_xx = 10, g_ff = 10):
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

		# Global time step / Factors that appear in H computation.
		self.dt = dt
		self.g_xx = g_xx
		self.g_ff = g_ff

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

	def u_sample(self, d):
		"""
		Return a random sample from a unit multi-variate normal of dimension D.
		"""
		return np.random.randn(d)

	def format_q(self, q):
		"""
		Given an input of shape ((Nobjs, 3)) with f, x, y for each row, output
		m, x, y array of shape ((Nobjs * 3))
		"""
		for i in range(q.shape[0]):
			q[i, 0] = self.mag2flux_converter(q[i, 0])

		return q.reshape((q.size, ))

	def H(self, q):
		"""
		Diagonal matrix that corresponds to q in pi(p|q) = Norm(0, H)

		q has the dimension (Nobjs * 3) where each chunk of three variable
		corresponds to f, x, y
		"""
		H_diag = np.zeros(q.size)
		for i in xrange(self.Nobjs):
			f = q[3 * i]
			H_diag[3 * i] = self.H_ff(f)
			H_diag[3 * i + 1] = H_diag[3 * i + 2] = self.H_xx(f)

		return H_diag

	def H_ff(self, f):
		"""
		Given the object flux, returns the approximate H matrix element corresponding to flux. 
		"""
		return self.g_ff * np.min([1./f, self.g0 / self.B_count])

	def H_xx(self, f):
		"""
		Given the object flux, returns the approximate H matrix element corresponding to position. 
		"""
		return self.g_xx * np.min([f * self.g1, f**2 * self.g2 / self.B_count])

	def V(self, q, f_pos=False):
		"""
		Negative Poisson log-likelihood given data and model.

		The model is specified by the list of objs, which are provided
		as a flattened list [Nobjs x 3](e.g., [f1, x1, y1, f2, x2, y2, ...])

		Assume a fixed background.
		"""
		if f_pos: # If f is required to be positive
			# Check whether all f is positive
			all_f_pos = True
			for i in xrange(self.Nobjs):
				if q[3 * i] < 0:
					all_f_pos = False
					break

			if not all_f_pos: # If not all f is positive, then return infinity.
				return np.infty

		Lambda = np.ones_like(self.D) * self.B_count # Model set to background
		for i in range(self.Nobjs): # Add every object.
			f, x, y = q[3*i:3*i+3]
			Lambda += f * gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)
		return np.sum(Lambda - self.D * np.log(Lambda))

	def T(self, p, H_diag):
	    """
	    Gaussian potential energy
	    """
	    # Exponential argument term
	    term1 = np.sum(p**2 / H_diag)

	    # Determinant term. # Note that the flux could be negative so the absolute sign is necessary.
	    term2 = np.log(np.abs(np.prod(H_diag)))

	    return (term1 + term2) / 2.


class single_gym(base_class):
	def __init__(self, Nsteps = 100, dt = 0.1, g_xx = 10, g_ff = 10):
		"""
		Single trajectory simulation.
		- Nsteps: Number of steps to be taken.
		- dt: Global time step size factor.
		- g_xx, g_ff: Factors that scale the momenta.
		"""
		# ---- Call the base class constructor
		base_class.__init__(self, dt = 1., g_xx = 10, g_ff = 10)

		# ---- Global variables
		self.Nsteps = Nsteps

		# ---- Place holder for various variables. 
		self.q_chain = None
		self.p_chain = None
		self.E_chain = None
		self.V_chain = None
		self.T_chain = None

		return

	def run_single(self, q_model_0=None, f_pos=False):
		"""
		Perform Bayesian inference with HMC with the initial model given as q_model_0.
		f_pos: Enforce the condition that total flux counts for individual sources be positive.
		"""

		#---- Number of objects should have been already determined via optimal step search
		self.Nobjs = q_model_0.shape[0]
		self.d = self.Nobjs * 3 # Total dimension of inference
		q_model_0 =  self.format_q(q_model_0) # Converter the magnitude to flux counts and reformat the array.

		#---- Allocate storage for variables being inferred.
		self.q_chain = np.zeros((self.Nsteps+1, self.Nobjs * 3))
		self.p_chain = np.zeros((self.Nsteps+1, self.Nobjs * 3))
		self.E_chain = np.zeros(self.Nsteps+1)
		self.V_chain = np.zeros(self.Nsteps+1)
		self.T_chain = np.zeros(self.Nsteps+1)

		#---- Loop over each step. 
		# Recall the 0-index corresponds to the intial model.
		# Set the initial values.
		q_initial = q_model_0
		H_diag = self.H(q_initial) # 
		p_initial = self.u_sample(self.d) / np.sqrt(H_diag)
		self.q_chain[0] = q_initial
		self.p_chain[0] = p_initial
		self.V_chain[0] = self.V(q_initial, f_pos=f_pos)
		self.T_chain[0] = self.T(p_initial, H_diag)
		self.E_chain[0] = self.V_chain[0] + self.T_chain[0]

		E_previous = self.E_chain[0]
		q_tmp = q_initial
		p_tmp = p_initial
		# p_half = p_tmp - self.dt * self.dVdq(q_tmp) / 2. # First half step

		# #---- Looping over steps
		# for i in xrange(1, self.Nsteps+1, 1):
		# 	#---- Initial point 
		# 	q_initial = q_tmp
		# 	H_diag = self.H(q_initial) # 

		# 	iflip = np.zeros(self.d, dtype=bool) # Flip array.                
		# 	for _ in xrange(steps_sample): 
		# 		flip = False
		# 		q_tmp = q_tmp + self.dt * p_half
		# 		# We only consider constraint in the flux direction.
		# 		# If any of the flux is below the limiting point, then change the momentum direction
		# 		for l in xrange(self.Nobjs):
		# 			if q_tmp[3 * l] < self.f_lim:
		# 				iflip[3 * l] = True
		# 				flip = True
		# 		if flip: # If fix due to constraint.
		# 			p_half_tmp = -p_half[iflip] # Flip the direction.
		# 			p_half = p_half - self.dt * self.dVdq(q_tmp) # Update as usual
		# 			p_half[iflip] = p_half_tmp # Make correction
		# 		else:
		# 			p_half = p_half - self.dt * self.dVdq(q_tmp) # If no correction, then regular update.

		# 	# Final half step correction
		# 	if flip:
		# 		p_half_tmp = p_half[iflip] # Save 
		# 		p_half = p_half + self.dt * self.dVdq(q_tmp) / 2.# Update as usual 
		# 		p_half[iflip] = p_half_tmp # Make correction                    
		# 	else:
		# 		p_tmp = p_half + self.dt * self.dVdq(q_tmp) / 2. # Account for the overshoot in the final run.

		# 	# Compute final energy and save.
		# 	E_final = self.E(q_tmp, p_tmp)
				
		return