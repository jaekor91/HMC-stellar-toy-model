# RHMC prototype sampler --- Fixed dimensional case
# Base class provides the infrastructure common to both single source and multiple source
# inference problem.
# - The truth image based on the user input
# - Constants including experimental condition
# Single trajectory class is used to achieve the following.
# - Show that the energy is conserved throughout its trajectory.
# - Show that the path is reversible as long as the base step size is made small. (Not shown)
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

		# Contrast information
		self.vmin = None
		self.vmax = None

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

	def gen_model(self, q_model):
		"""
		Given model sample (Nobjs, 3) with mag, x, y for each, 
		generate the model image.

		Gaussian PSF is assumed with the width specified by PSF_FWHM_pix.
		"""
		# Generate an image with background.
		model = np.ones((self.num_rows, self.num_cols), dtype=float) * self.B_count

		# Add one star at a time.
		for i in xrange(q_model.shape[0]):
			mag, x, y = q_model[i]
			model += self.mag2flux_converter(mag) * gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM = self.PSF_FWHM_pix)

		return model

	def mag2flux_converter(self, mag):
		"""
		Given user input magnitude convert it to flux counts.
		"""

		return mag2flux(mag) * self.flux_to_count

	def flux2mag_converter(self, flux):
		"""
		Given user input magnitude convert it to flux counts.
		"""

		return flux2mag(flux /self.flux_to_count)

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
		self.f_lim = mag2flux(self.mB+3) * flux_to_count

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

	def reverse_format_q(self, q):
		"""
		Reverse process of format_q
		"""
		q = np.copy(q.reshape((self.Nobjs, 3)))
		for i in range(self.Nobjs):
			q[i, 0] = self.flux2mag_converter(q[i, 0])

		return q

	def H(self, q, grad=False):
		"""
		Diagonal matrix that corresponds to q in pi(p|q) = Norm(0, H)

		q has the dimension (Nobjs * 3) where each chunk of three variable
		corresponds to f, x, y

		If grad=True, then retrun gradient respect to flu as well.		
		"""
		H_diag = np.zeros(q.size)
		if not grad:
			for i in xrange(self.Nobjs):
				f = q[3 * i]
				H_diag[3 * i] = self.H_ff(f)
				H_diag[3 * i + 1] = H_diag[3 * i + 2] = self.H_xx(f)

			return H_diag
		else:
			H_grad_diag = np.zeros(q.size)			
			for i in xrange(self.Nobjs):
				f = q[3 * i]
				val, grad = self.H_ff(f, grad=True)
				H_diag[3 * i] = val
				H_grad_diag[3 * i] = grad

				val, grad = self.H_xx(f, grad=True)
				H_diag[3 * i + 1] = H_diag[3 * i + 2] = val
				H_grad_diag[3 * i + 1] = H_grad_diag[3 * i + 2] = grad

			return H_diag, H_grad_diag

	def H_xx(self, f, grad=False):
		"""
		Given the object flux, returns the approximate H matrix element corresponding to position. 

		If grad=True, then retrun gradient respect to flux.
		"""

		fmin = self.f_lim

		if fmin > f:
			f = fmin
			f_low = True
		else: 
			f_low = False

		if not grad:
			return self.g_xx * f * self.g1
		else:
			if f_low:
				return self.g_xx * f * self.g1, 0.
			else:
				return self.g_xx * f * self.g1, self.g_xx * self.g1

	def H_ff(self, f, grad=False):
		"""
		Given the object flux, returns the approximate H matrix element corresponding to flux. 

		If grad=True, then retrun gradient respect to flux.
		"""
		fmin = self.f_lim

		if fmin > f:
			f = fmin
			f_low = True
		else: 
			f_low = False

		if not grad:
			return self.g_ff / f
		else:
			if not f_low:
				return self.g_ff / f, -self.g_ff / f**2
			else:
				return self.g_ff / f, 0.				
		
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

		for i in xrange(self.Nobjs):
			x, y = q[3*i+1: 3*i+3]
			if  (x < 0) or (x > self.num_rows-1) or (y < 0) or (y > self.num_cols-1):
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

	def dVdq(self, q):
		"""
		Gradient of Poisson pontential above.    
		"""
		# Place holder for the gradient.
		grad = np.zeros(q.size)

		# Compute the model.
		Lambda = np.ones_like(self.D) * self.B_count # Model set to background
		for i in range(self.Nobjs): # Add every object.
			f, x, y = q[3*i:3*i+3]
			Lambda += f * gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)

		# Variable to be recycled
		rho = (self.D/Lambda)-1.# (D_lm/Lambda_lm - 1)
		# Compute f, x, y gradient for each object
		lv = np.arange(0, self.num_rows)
		mv = np.arange(0, self.num_cols)
		mv, lv = np.meshgrid(lv, mv)
		var = (self.PSF_FWHM_pix/2.354)**2 
		for i in range(self.Nobjs):
			f, x, y = q[3*i:3*i+3]
			PSF = gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)
			grad[3*i] = -np.sum(rho * PSF) # flux grad
			grad[3*i+1] = -np.sum(rho * (lv - x + 0.5) * PSF) * f / var
			grad[3*i+2] = -np.sum(rho * (mv - y + 0.5) * PSF) * f / var
		return grad

	def dVdq_RHMC(self, q, p):
		"""
		Second gradient term unique to RHMC.
		"""
		grads = np.zeros_like(q)

		# Compute H matrix and their gradients
		H, H_grad = self.H(q, grad=True)

		# For each object compute the gradient
		for i in xrange(self.Nobjs):
			# Quadratic term
			term1 = (p[3 * i] ** 2) * (-H_grad[3 * i] / H[3 * i]**2)

			# Log Det term
			term2 = (H_grad[3 * i] / H[3 * i]) + (2 * H_grad[3 * i + 1] / H[3 * i + 1])

			grads[3 * i] = (term1 + term2) / 2.

		return grads

	def dphidq(self, q):
		"""
		As in the general metric paper.
		"""
		# Gradient contribution
		grads = self.dVdq(q)

		# Compute H matrix and their gradients
		H, H_grad = self.H(q, grad=True)

		# For each object compute the gradient
		for i in xrange(self.Nobjs):
			# Log Det term
			term2 = (H_grad[3 * i] / H[3 * i]) + (2 * H_grad[3 * i + 1] / H[3 * i + 1])

			grads[3 * i] += term2 / 2.

		return grads

	def dtaudq(self, q, p):
		"""
		As in the general metric paper.
		"""
		grads = np.zeros_like(q)

		# Compute H matrix and their gradients
		H, H_grad = self.H(q, grad=True)

		# For each object compute the gradient
		for i in xrange(self.Nobjs):
			# Quadratic term
			term1 = (p[3 * i] ** 2) * (-H_grad[3 * i] / H[3 * i]**2)

			grads[3 * i] = term1 / 2.

		return grads

	def dtaudp(self, q, p):
		"""
		As in the general metric paper.
		"""
		# Compute H matrix
		H_diag = self.H(q, grad=False)

		return p / H_diag

	def display_image(self, show=True, save=None, figsize=(5, 5), num_ticks = 6, \
			vmin=None, vmax=None):
		"""
		Display the data image
		"""
		#---- Contrast
		# If the user does not provide the contrast
		if vmin is None:
			# then check whether there is contrast stored up. If not.
			if self.vmin is None:
				D_raveled = self.D.ravel()
				self.vmin = np.percentile(D_raveled, 1.)
				self.vmax = np.percentile(D_raveled, 80.)
			vmin = self.vmin
			vmax = self.vmax

		fig, ax = plt.subplots(1, figsize = figsize)
		ax.imshow(self.D,  interpolation="none", cmap="gray", vmin=vmin, vmax = vmax)
		yticks = ticker.MaxNLocator(num_ticks)
		xticks = ticker.MaxNLocator(num_ticks)		
		ax.yaxis.set_major_locator(yticks)
		ax.xaxis.set_major_locator(xticks)		
		if show:
			plt.show()
		if save is not None:
			plt.savefig(save, dpi=100, bbox_inches = "tight")
		plt.close()


class single_gym(base_class):
	def __init__(self, Nsteps = 100, dt = 0.1, g_xx = 10, g_ff = 10):
		"""
		Single trajectory simulation.
		- Nsteps: Number of steps to be taken.
		- dt: Global time step size factor.
		- g_xx, g_ff: Factors that scale the momenta.
		"""
		# ---- Call the base class constructor
		base_class.__init__(self, dt = dt, g_xx = g_xx, g_ff = g_ff)

		# ---- Global variables
		self.Nsteps = Nsteps

		# ---- Place holder for various variables. 
		self.q_chain = None
		self.p_chain = None
		self.E_chain = None
		self.V_chain = None
		self.T_chain = None

		return

	def run_single_HMC(self, q_model_0=None, f_pos=False):
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
		# H_diag = self.H(q_initial, grad=False) # 
		p_initial = self.u_sample(self.d) #  * np.sqrt(H_diag)
		self.q_chain[0] = q_initial
		self.p_chain[0] = p_initial
		self.V_chain[0] = self.V(q_initial, f_pos=f_pos)
		self.T_chain[0] = self.T(p_initial, np.ones_like(p_initial)) # H_diag)
		self.E_chain[0] = self.V_chain[0] + self.T_chain[0]

		E_previous = self.E_chain[0]
		q_tmp = q_initial
		p_tmp = p_initial

		#---- Looping over steps
		# Using incorrect and naive leapfrog method
		for i in xrange(1, self.Nsteps+1, 1):

			# First half step for momentum
			p_half = p_tmp - self.dt * (self.dVdq(q_tmp)) / 2.

			# Leap frog step
			q_tmp = q_tmp + self.dt * p_half # / H_diag
			# H_diag = self.H(q_tmp) # immediately compute the new H_diag.

			# Second half step for momentum
			p_tmp = p_half  - self.dt * (self.dVdq(q_tmp)) / 2.

			# Store the variables and energy
			self.q_chain[i] = q_tmp
			self.p_chain[i] = p_tmp
			self.V_chain[i] = self.V(q_tmp, f_pos=f_pos)
			self.T_chain[i] = self.T(p_tmp, np.ones_like(p_initial))# H_diag)
			self.E_chain[i] = self.V_chain[i] + self.T_chain[i]
				
		return

	def run_single_RHMC(self, q_model_0=None, f_pos=False, solver="naive", delta=1e-6, p_initial=None,\
		counter_max = 100):
		"""
		Perform Bayesian inference with HMC with the initial model given as q_model_0.
		f_pos: Enforce the condition that total flux counts for individual sources be positive.

		If p_initial is not None, then use that as the initial seed.
		"""
		if solver == "leap_frog":
			print "Leap frog solver doesn't work quite well."

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
		H_diag = self.H(q_initial, grad=False) # 
		if p_initial is None:
			p_initial = self.u_sample(self.d) * np.sqrt(H_diag)
		self.q_chain[0] = q_initial
		self.p_chain[0] = p_initial
		V_initial = self.V(q_initial, f_pos=f_pos)
		T_initial = self.T(p_initial, H_diag)
		E_initial = V_initial + T_initial

		q_tmp = q_initial
		p_tmp = p_initial

		#---- Looping over steps
		for i in xrange(1, self.Nsteps+1, 1):
			if solver == "naive":
				# Compute new position
				q_tmp_new = q_tmp + self.dt * p_tmp / H_diag

				# Momentum update
				p_tmp_old = p_tmp
				p_tmp = p_tmp - self.dt * (self.dVdq(q_tmp) + self.dVdq_RHMC(q_tmp, p_tmp)) 			

				if f_pos: # If flux must be kept positive always.
					iflip = np.zeros(q_tmp_new.size, dtype=bool)
					for k in xrange(self.Nobjs):
						f = q_tmp_new[3 * k]
						# If flux is negative, then reverse the direction of the momentum corresponding to the flux
						if f < self.f_lim: 
							p_tmp[3 * k] = p_tmp_old[3 * k] * -1.

				# Update the position
				q_tmp = q_tmp_new
				H_diag = self.H(q_tmp)
			elif solver == "leap_frog":
				# First momentum half-step
				p_half = p_tmp - self.dt * (self.dVdq(q_tmp) + self.dVdq_RHMC(q_tmp, p_tmp)) / 2.

				# Compute new position
				q_tmp = q_tmp + self.dt * p_half / H_diag

				# Momentum update
				p_tmp = p_half - self.dt * (self.dVdq(q_tmp) + self.dVdq_RHMC(q_tmp, p_half)) / 2.

				if f_pos: # If flux must be kept positive always.
					iflip = np.zeros(q_tmp.size, dtype=bool)
					for k in xrange(self.Nobjs):
						f = q_tmp[3 * k]
						# If flux is negative, then reverse the direction of the momentum corresponding to the flux
						if f < self.f_lim: 
							p_tmp[3 * k] = p_half[3 * k] * -1.

				# Update the position
				H_diag = self.H(q_tmp)
			elif solver == "implicit":
				# First update phi-hat
				p_tmp = p_tmp - (self.dt/2.) * self.dphidq(q_tmp)

				# p-tau update
				rho = np.copy(p_tmp)
				dp = np.infty
				counter = 0
				while (dp > delta) and (counter < counter_max):
					p_prime = rho - (self.dt/2.) * self.dtaudq(q_tmp, p_tmp) 
					dp = np.max(np.abs(p_tmp - p_prime))
					p_tmp = np.copy(p_prime)
					counter +=1

				# q-tau update
				sig = np.copy(q_tmp)
				dq = np.infty
				counter = 0				
				while (dq > delta) and (counter < counter_max):
					q_prime = sig + (self.dt/2.) * (self.dtaudp(sig, p_tmp) + self.dtaudp(q_tmp, p_tmp))
					dq = np.max(np.abs(q_tmp - q_prime))
					q_tmp = np.copy(q_prime)					
					counter +=1		

				# p-tau update
				p_tmp = p_tmp - (self.dt/2.) * self.dtaudq(q_tmp, p_tmp)

				# Last update phi-hat
				p_tmp = p_tmp - (self.dt/2.) * self.dphidq(q_tmp)

				# Diagonal H update
				H_diag = self.H(q_tmp, grad=False)

				for k in xrange(self.Nobjs):
					f, x, y= q_tmp[3 * k : 3 * k + 3]					
					# ---- Check for any source with flux < f_lim.
					# If flux is negative, then reverse the direction of the momentum corresponding to the flux
					if f < self.f_lim: 
						p_tmp[3 * k] *= -1.
					# ---- Reflect xy momenta if xy outside boundary						
					if (x < 0) or (x > self.num_rows-1):
						p_tmp[3 * k + 1] *= -1.
					if (y < 0) or (y > self.num_cols-1):
						p_tmp[3 * k + 2] *= -1.
			else: # If the user input the non-existing solver.
				assert False

			# Store the variables and energy
			self.q_chain[i] = q_tmp
			self.p_chain[i] = p_tmp
			self.V_chain[i] = self.V(q_tmp, f_pos=f_pos) - V_initial
			self.T_chain[i] = self.T(p_tmp, H_diag) - T_initial
			self.E_chain[i] = self.V_chain[i] + self.T_chain[i]
				
		return

	def diagnostics_first(self, q_true, show=True, figsize=(18, 12), \
		plot_E = True, plot_V = False, plot_T = False, plot_flux=True, num_ticks=5,\
		ft_size = 20, pt_size1=20, save=None):
		"""
		Scatter plot of the first source inference.
		"""
		X = self.q_chain[:, 1]
		Y = self.q_chain[:, 2]
		if plot_flux:
			F = self.q_chain[:, 0]
			F_true = self.mag2flux_converter(q_true[0, 0])
		else:
			F = self.flux2mag_converter(self.q_chain[:, 0])
			F_true = q_true[0, 0]
		E = self.E_chain
		V = self.V_chain
		T = self.T_chain

		# ---- Min Max
		Xmin, Xmax = np.min(X), np.max(X)
		Ymin, Ymax = np.min(Y), np.max(Y)		
			
		fig, ax_list = plt.subplots(2, 2, figsize=figsize)
		ax_list[0, 0].axis("equal")
		if (Xmin < 0) or (Xmax > self.num_rows - 1) or (Ymin < 0) or (Ymax > self.num_cols - 1):
			ax_list[0, 0].set_xlim([0, self.num_rows-1])
			ax_list[0, 0].set_ylim([0, self.num_cols-1])

		# ---- Joining certain axis
		ax_list[0, 0].get_shared_x_axes().join(ax_list[0, 0], ax_list[1, 0])
		ax_list[0, 0].get_shared_y_axes().join(ax_list[0, 0], ax_list[0, 1])

		# ---- XY plot
		ax_list[0, 0].scatter([q_true[0, 2]], [q_true[0, 1]], c="green", s=200, edgecolor="none")				
		ax_list[0, 0].scatter(Y, X, c="black", s=pt_size1, edgecolor="none")
		ax_list[0, 0].scatter([Y[0]], [X[0]], c="red", s=100, edgecolor="none")
		ax_list[0, 0].set_xlabel("Y", fontsize=ft_size)					
		ax_list[0, 0].set_ylabel("X", fontsize=ft_size)
		yticks00 = ticker.MaxNLocator(num_ticks)
		xticks00 = ticker.MaxNLocator(num_ticks)		
		ax_list[0, 0].yaxis.set_major_locator(yticks00)
		ax_list[0, 0].xaxis.set_major_locator(xticks00)		

		# --- Flux - Y
		ax_list[1, 0].scatter([q_true[0, 2]], [q_true[0, 0]], c="green", s=200, edgecolor="none")						
		ax_list[1, 0].scatter(Y, F, c="black", s=pt_size1, edgecolor="none")
		ax_list[1, 0].axhline(y = F_true, c="red", lw=2., ls="--")
		ax_list[1, 0].scatter([Y[0]], [F[0]], c="red", s=100, edgecolor="none")
		if plot_flux:
			ax_list[1, 0].set_ylabel("Flux", fontsize=ft_size)
		else:
			ax_list[1, 0].set_ylabel("Mag", fontsize=ft_size)
		ax_list[1, 0].set_xlabel("Y", fontsize=ft_size)
		yticks10 = ticker.MaxNLocator(num_ticks)
		xticks10 = ticker.MaxNLocator(num_ticks)		
		ax_list[1, 0].yaxis.set_major_locator(yticks10)
		ax_list[1, 0].xaxis.set_major_locator(xticks10)		

		# --- Flux - X 
		ax_list[0, 1].scatter([q_true[0, 0]], [q_true[0, 1]], c="green", s=200, edgecolor="none")						
		ax_list[0, 1].scatter(F, X, c="black", s=pt_size1, edgecolor="none")
		ax_list[0, 1].axvline(x = F_true, c="red", lw=2., ls="--")
		ax_list[0, 1].scatter([F[0]], [X[0]], c="red", s=100, edgecolor="none")
		if plot_flux:
			ax_list[0, 1].set_xlabel("Flux", fontsize=ft_size)
		else:
			ax_list[0, 1].set_xlabel("Mag", fontsize=ft_size)
		ax_list[0, 1].set_ylabel("X", fontsize=ft_size)
		yticks01 = ticker.MaxNLocator(num_ticks)
		xticks01 = ticker.MaxNLocator(num_ticks)
		ax_list[0, 1].yaxis.set_major_locator(yticks01)
		ax_list[0, 1].xaxis.set_major_locator(xticks01)

		# --- Energy plot
		if plot_V:
			ax_list[1, 1].plot(range(self.Nsteps+1), V, c="green", lw=2, label="V")
		if plot_T:
			ax_list[1, 1].plot(range(self.Nsteps+1), T, c="red", lw=2, label="T")
		if plot_E:
			ax_list[1, 1].plot(range(self.Nsteps+1), E, c="blue", lw=2, label="H")
		ax_list[1, 1].legend(loc="upper right", fontsize=ft_size)
		ax_list[1, 1].set_xlabel("Step", fontsize=ft_size)
		ax_list[1, 1].set_ylabel(r"$\Delta E$", fontsize=ft_size)
		yticks11 = ticker.MaxNLocator(num_ticks)
		xticks11 = ticker.MaxNLocator(num_ticks)				
		ax_list[1, 1].yaxis.set_major_locator(yticks11)
		ax_list[1, 1].xaxis.set_major_locator(xticks11)

		if show:
			plt.show()
		if save is not None:
			plt.savefig(save, dpi=100, bbox_inches = "tight")
		plt.close()

		return 



class multi_gym(base_class):
	def __init__(self, Nsteps = 100, dt = 0.1, g_xx = 1., g_ff = 1.):
		"""
		Single trajectory simulation.
		- Nsteps: Number of steps to be taken.
		- dt: Global time step size factor.
		- g_xx, g_ff: Factors that scale the momenta.
		"""
		# ---- Call the base class constructor
		base_class.__init__(self, dt = dt, g_xx = g_xx, g_ff = g_ff)

		# ---- Global variables
		self.Nsteps = Nsteps

		# ---- Place holder for various variables. 
		self.q_chain = None
		self.p_chain = None
		self.E_chain = None
		self.V_chain = None
		self.T_chain = None
		self.A_chain = None # Accepted or not. The initial point deemed accepted by default.

		return

	def run_RHMC(self, q_model_0, f_pos=True, delta=1e-6, Niter = 100, Nsteps=100,\
				dt = 1e-1, save_traj=False, counter_max = 1000):
		"""
		- Perform Bayesian inference with RHMC with the initial model given as q_model_0. 
		q_model_0 given in (Nobjs, 3) format.
		- f_pos: Enforce the condition that total flux counts for individual sources be positive.
		- If p_initial is not None, then use that as the initial seed.
		- save_traj: If True, save the intermediate 
		- counter_max: Maximum number to try to converge at a solution.

		Convention:
		- q_model_0 is used as the first point.
		- In each iteration
			- There are Nsteps evaluation, leading to Nsteps+1 points collected.
		- The initial energy is saved.
		- Accept/reject index conicides with the Niter index. If a particular trajectory is accepted,
		then accepted point will appear as the first point of the next iteration.

		Note: 
		- Current version only runs a single chain. Multiple chains can be added later
		or obtained by calling this function multiple times.
		- No thinning or burn-in is applied. All of these can be done post-hoc.
		- Nsteps is fixed from iteration to iteration.
		- Total number of samples is Niter + 1, where 1 is for the initial point.
		"""
		#---- Set global variables
		self.dt = dt
		self.Niter = Niter
		self.Nsteps = Nsteps
		self.save_traj = save_traj

		#---- Number of objects should have been already determined via optimal step search
		self.Nobjs = q_model_0.shape[0]
		self.d = self.Nobjs * 3 # Total dimension of inference
		q_model_0 =  self.format_q(q_model_0) # Converter the magnitude to flux counts and reformat the array.

		#---- Allocate storage for variables being inferred.
		# The intial point is saved in the zero index of the first axis.
		# The last point is saved in the last index of the first axis.
		if save_traj:
			# Note that for the (Niter+1)-th sample does not go through steps.
			# The last point in an iteration is the same the first point in the next iteration if the proposal is accepted.
			self.q_chain = np.zeros((self.Niter+1, self.Nsteps+1, self.Nobjs * 3))
			self.p_chain = np.zeros((self.Niter+1, self.Nsteps+1, self.Nobjs * 3))
			self.E_chain = np.zeros((self.Niter+1, self.Nsteps+1))
			self.V_chain = np.zeros((self.Niter+1, self.Nsteps+1))
			self.T_chain = np.zeros((self.Niter+1, self.Nsteps+1))
		else:
			# Save the first point energy.			
			self.q_chain = np.zeros((self.Niter+1, self.Nobjs * 3))
			self.p_chain = np.zeros((self.Niter+1, self.Nobjs * 3))
			self.E_chain = np.zeros(self.Niter+1)
			self.V_chain = np.zeros(self.Niter+1)
			self.T_chain = np.zeros(self.Niter+1)
		self.A_chain = np.zeros(self.Niter+1, dtype=bool)

		#---- Set the very first initial point.
		q_tmp = np.copy(q_model_0)

		#---- Perform the iterations
		for l in xrange(self.Niter+1):
			# The initial q_tmp has already been set at the end of the previous run.			
			# Resample momentum
			H_diag = self.H(q_tmp, grad=False)
			p_tmp = self.u_sample(self.d) * np.sqrt(H_diag)

			# Compute the initial energies
			V_initial = self.V(q_tmp, f_pos=f_pos)
			T_initial = self.T(p_tmp, H_diag)
			E_initial = V_initial + T_initial # Necessary to save to compute dE

			#---- Save the initial point and energies
			if save_traj:
				self.q_chain[l, 0] = q_tmp
				self.p_chain[l, 0] = p_tmp
				self.V_chain[l, 0] = V_initial
				self.E_chain[l, 0] = E_initial
				self.T_chain[l, 0] = T_initial			
			else:
				# Only time energy is saved in the whole iteration.
				self.q_chain[l] = q_tmp
				self.p_chain[l] = p_tmp
				self.V_chain[l] = V_initial
				self.E_chain[l] = E_initial
				self.T_chain[l] = T_initial

			#---- Looping over steps
			for i in xrange(1, self.Nsteps+1, 1):
				# First update phi-hat
				p_tmp = p_tmp - (self.dt/2.) * self.dphidq(q_tmp)

				# p-tau update
				rho = np.copy(p_tmp)
				dp = np.infty
				counter = 0
				while (dp > delta) and (counter < counter_max):
					p_prime = rho - (self.dt/2.) * self.dtaudq(q_tmp, p_tmp) 
					dp = np.max(np.abs(p_tmp - p_prime))
					p_tmp = np.copy(p_prime)
					counter +=1

				# q-tau update
				sig = np.copy(q_tmp)
				dq = np.infty
				counter = 0				
				while (dq > delta) and (counter < counter_max):
					q_prime = sig + (self.dt/2.) * (self.dtaudp(sig, p_tmp) + self.dtaudp(q_tmp, p_tmp))
					dq = np.max(np.abs(q_tmp - q_prime))
					q_tmp = np.copy(q_prime)					
					counter +=1					

				# p-tau update
				p_tmp = p_tmp - (self.dt/2.) * self.dtaudq(q_tmp, p_tmp)

				# Last update phi-hat
				p_tmp = p_tmp - (self.dt/2.) * self.dphidq(q_tmp)

				# Boundary condition checks
				for k in xrange(self.Nobjs):
					f, x, y= q_tmp[3 * k : 3 * k + 3]					
					# ---- Check for any source with flux < f_lim.
					# If flux is negative, then reverse the direction of the momentum corresponding to the flux
					if f < self.f_lim: 
						p_tmp[3 * k] *= -1.
					# ---- Reflect xy momenta if xy outside boundary						
					if (x < 0) or (x > self.num_rows-1):
						p_tmp[3 * k + 1] *= -1.
					if (y < 0) or (y > self.num_cols-1):
						p_tmp[3 * k + 2] *= -1.

				# Intermediate variables save if asked
				if save_traj:
					# Diagonal H update
					H_diag = self.H(q_tmp, grad=False)
					self.q_chain[l, i] = q_tmp
					self.p_chain[l, i] = p_tmp
					self.V_chain[l, i] = self.V(q_tmp, f_pos=f_pos)
					self.T_chain[l, i] = self.T(p_tmp, H_diag)
					self.E_chain[l, i] = self.V_chain[l, i] + self.T_chain[l, i]

			# Compute the energy difference between the initial and the final energy
			if save_traj: # If the energy has been already saved.
				E_final = self.E_chain[l, -1]
			else:
				H_diag = self.H(q_tmp, grad=False)				
				E_final = self.V(q_tmp, f_pos=f_pos) + self.T(p_tmp, H_diag)
			dE = E_final - E_initial

			# Accept or reject and set the next initial point accordingly.
			lnu = np.log(np.random.random(1))
			if (dE < 0) or (lnu < -dE): # If accepted.
				self.A_chain[l] = 1
			else: # Otherwise, proposal rejected.
				# Reseting the position variable to the previous.
				if save_traj:
					q_tmp = self.q_chain[l, 0]
				else:
					q_tmp = self.q_chain[l]

		# ---- Compute the total acceptance rate.
		self.R_accept = np.sum(self.A_chain) / float(self.Niter + 1)
		print "Acceptance rate without warm-up: %.2f %%" % (self.R_accept * 100)

		return		


	def diagnostics_first(self, q_true, show=True, figsize=(18, 12), \
		plot_flux=False, num_ticks=5, ft_size = 20, pt_size1=20, save=None, Nskip=0, \
		title_str = None):
		"""
		Scatter plot of the first source inference over the entire chain.
		
		Nskip: Number of samples to skip
		"""
		#---- Extracting the proper points
		if self.save_traj:
			X = self.q_chain[Nskip:, 0, 1]
			Y = self.q_chain[Nskip:, 0, 2]
			if plot_flux:
				F = self.q_chain[Nskip:, 0, 0]
				F_true = self.mag2flux_converter(q_true[0, 0])
			else:
				F = self.flux2mag_converter(self.q_chain[Nskip:, 0, 0])
				F_true = q_true[0, 0]
			E = self.E_chain[Nskip:, 0]
		else:
			X = self.q_chain[Nskip:, 1]
			Y = self.q_chain[Nskip:, 2]
			if plot_flux:
				F = self.q_chain[Nskip:, 0]
				F_true = self.mag2flux_converter(q_true[0, 0])
			else:
				F = self.flux2mag_converter(self.q_chain[Nskip:, 0])
				F_true = q_true[0, 0]
			E = self.E_chain[Nskip:]
			A = self.A_chain[Nskip:]

		# ---- Min Max
		Xmin, Xmax = np.min(X), np.max(X)
		Ymin, Ymax = np.min(Y), np.max(Y)

		fig, ax_list = plt.subplots(2, 2, figsize=figsize)
		ax_list[0, 0].axis("equal")
		if (Xmin < 0) or (Xmax > self.num_rows - 1) or (Ymin < 0) or (Ymax > self.num_cols - 1):
			ax_list[0, 0].set_xlim([0, self.num_rows-1])
			ax_list[0, 0].set_ylim([0, self.num_cols-1])

		# ---- Joining certain axis
		ax_list[0, 0].get_shared_x_axes().join(ax_list[0, 0], ax_list[1, 0])
		ax_list[0, 0].get_shared_y_axes().join(ax_list[0, 0], ax_list[0, 1])

		# ---- XY plot
		ax_list[0, 0].scatter(Y, X, c="black", s=pt_size1, edgecolor="none")
		ax_list[0, 0].scatter([Y[0]], [X[0]], c="red", s=100, edgecolor="none")
		ax_list[0, 0].scatter([q_true[0, 2]], [q_true[0, 1]], c="green", s=200, edgecolor="none")    
		ax_list[0, 0].set_xlabel("Y", fontsize=ft_size)
		ax_list[0, 0].set_ylabel("X", fontsize=ft_size)
		yticks00 = ticker.MaxNLocator(num_ticks)
		xticks00 = ticker.MaxNLocator(num_ticks)
		ax_list[0, 0].yaxis.set_major_locator(yticks00)
		ax_list[0, 0].xaxis.set_major_locator(xticks00)

		# --- Flux - Y
		ax_list[1, 0].scatter(Y, F, c="black", s=pt_size1, edgecolor="none")
		ax_list[1, 0].axhline(y = F_true, c="red", lw=2., ls="--")
		ax_list[1, 0].scatter([Y[0]], [F[0]], c="red", s=100, edgecolor="none")
		ax_list[1, 0].scatter([q_true[0, 2]], [q_true[0, 0]], c="green", s=200, edgecolor="none")    
		if plot_flux:
			ax_list[1, 0].set_ylabel("Flux", fontsize=ft_size)
		else:
			ax_list[1, 0].set_ylabel("Mag", fontsize=ft_size)
		ax_list[1, 0].set_xlabel("Y", fontsize=ft_size)
		yticks10 = ticker.MaxNLocator(num_ticks)
		xticks10 = ticker.MaxNLocator(num_ticks)
		ax_list[1, 0].yaxis.set_major_locator(yticks10)
		ax_list[1, 0].xaxis.set_major_locator(xticks10)

		# --- Flux - X 
		ax_list[0, 1].scatter(F, X, c="black", s=pt_size1, edgecolor="none")
		ax_list[0, 1].axvline(x = F_true, c="red", lw=2., ls="--")
		ax_list[0, 1].scatter([q_true[0, 0]], [q_true[0, 1]], c="green", s=200, edgecolor="none")    
		ax_list[0, 1].scatter([F[0]], [X[0]], c="red", s=100, edgecolor="none")
		if plot_flux:
			ax_list[0, 1].set_xlabel("Flux", fontsize=ft_size)
		else:
			ax_list[0, 1].set_xlabel("Mag", fontsize=ft_size)
		ax_list[0, 1].set_ylabel("X", fontsize=ft_size)
		yticks01 = ticker.MaxNLocator(num_ticks)
		xticks01 = ticker.MaxNLocator(num_ticks)
		ax_list[0, 1].yaxis.set_major_locator(yticks01)
		ax_list[0, 1].xaxis.set_major_locator(xticks01)

		# --- Energy histograms
		# Marginal energy distribution
		Emean = np.mean(E)
		ax_list[1, 1].hist(E-Emean, bins=50, label="E", histtype="step", lw=2, color="blue")
		# Transition energy distribution
		dE = E[1:] - E[:-1]
		ax_list[1, 1].hist(dE, bins=50, label="dE", histtype="step", lw=2, color="black")
		ax_list[1, 1].legend(loc="upper right", fontsize=ft_size)
		ax_list[1, 1].set_xlabel("E", fontsize=ft_size)

		# --- Title
		R_accept = np.sum(A) / float(self.Niter - Nskip + 1)
		Raccept_str = "R_accept: %.2f%%" % (R_accept * 100)
		if title_str is None:
			title_str = Raccept_str
		else:
			title_str = title_str + "  /  " + Raccept_str
		plt.suptitle(title_str, fontsize = 20)

		if show:
			plt.show()
		if save is not None:
			plt.savefig(save, dpi=100, bbox_inches = "tight")
		plt.close()

		return

	def diagnostics_all(self, q_true, idx_iter = -1, idx_step = None, figsize = (16, 11), \
						color_truth="red", color_model="blue", ft_size = 15, num_ticks = 5, \
						show=False, save=None, title_str = None):
		"""
		- idx_iter: Index of the iteration to plot.
		- idx_step: Iddex of the step to plot. (Only applicable if save_traj = True.)
		"""
		# Contrast
		if vmin is None:
			# then check whether there is contrast stored up. If not.
			if self.vmin is None:
				D_raveled = self.D.ravel()
				self.vmin = np.percentile(D_raveled, 1.)
				self.vmax = np.percentile(D_raveled, 80.)
			vmin = self.vmin
			vmax = self.vmax

		# Obtain the model q
		if self.save_traj:
			q_model = self.reverse_format_q(self.q_chain[idx_iter, idx_step])
		else:
			q_model = self.reverse_format_q(self.q_chain[idx_iter])

		# --- Extract X, Y, Mag variables
		# Truth 
		F0 = q_true[:, 0]
		X0 = q_true[:, 1]
		Y0 = q_true[:, 2]
		S0 = linear_func(F0, m=-30, b = 20, s0=23, y_min=10)
		# Model
		F = q_model[:, 0]
		X = q_model[:, 1]
		Y = q_model[:, 2]
		S = linear_func(F, m=-30, b = 20, s0=23, y_min=10)

		# --- Make the plot
		fig, ax_list = plt.subplots(2, 3, figsize=figsize)
		# ---- Joining certain axis
		ax_list[0, 0].get_shared_x_axes().join(ax_list[0, 0], ax_list[1, 0])
		ax_list[0, 0].get_shared_y_axes().join(ax_list[0, 0], ax_list[0, 1])
		ax_list[0, 0].get_shared_y_axes().join(ax_list[0, 0], ax_list[1, 1])
		ax_list[0, 0].get_shared_x_axes().join(ax_list[0, 0], ax_list[1, 1])
		ax_list[0, 0].get_shared_y_axes().join(ax_list[0, 0], ax_list[1, 2])
		ax_list[0, 0].get_shared_x_axes().join(ax_list[0, 0], ax_list[1, 2])

		# (0, 0): Image
		ax_list[0, 0].imshow(self.D, cmap="gray", interpolation="none", vmin=vmin, vmax=vmax)
		# Truth locs
		ax_list[0, 0].scatter(Y0, X0, c=color_truth, s=S0, edgecolor="none", marker="x")
		# Model locs
		ax_list[0, 0].scatter(Y, X, c=color_model, s=S, edgecolor="none", marker="x")
		# Decorations
		ax_list[0, 0].set_title("Data", fontsize=ft_size)
		ax_list[0, 0].set_xlabel("Y", fontsize=ft_size)
		ax_list[0, 0].set_ylabel("X", fontsize=ft_size)
		yticks00 = ticker.MaxNLocator(num_ticks)
		xticks00 = ticker.MaxNLocator(num_ticks)
		ax_list[0, 0].yaxis.set_major_locator(yticks00)
		ax_list[0, 0].xaxis.set_major_locator(xticks00)

		# (0, 1): Mag - X
		ax_list[0, 1].scatter(F0, X0, c=color_truth, s=S0, edgecolor="none", marker="x")
		ax_list[0, 1].scatter(F, X, c=color_model, s=S, edgecolor="none", marker="x")
		# Decorations
		ax_list[0, 1].axvline(x=self.flux2mag_converter(self.f_lim), c="green", lw=1.5, ls="--")
		ax_list[0, 1].set_ylabel("X", fontsize=ft_size)
		ax_list[0, 1].set_xlabel("Mag", fontsize=ft_size)
		yticks10 = ticker.MaxNLocator(num_ticks)
		xticks10 = ticker.MaxNLocator(num_ticks)
		ax_list[0, 1].yaxis.set_major_locator(yticks10)
		ax_list[0, 1].xaxis.set_major_locator(xticks10)	


		# (1, 0): Y - Mag
		ax_list[1, 0].scatter(Y0, F0, c=color_truth, s=S0, edgecolor="none", marker="x")
		ax_list[1, 0].scatter(Y, F, c=color_model, s=S, edgecolor="none", marker="x")
		# Decorations
		ax_list[1, 0].axhline(y=self.flux2mag_converter(self.f_lim), c="green", lw=1.5, ls="--")
		ax_list[1, 0].set_ylabel("Mag", fontsize=ft_size)
		ax_list[1, 0].set_xlabel("Y", fontsize=ft_size)
		yticks10 = ticker.MaxNLocator(num_ticks)
		xticks10 = ticker.MaxNLocator(num_ticks)
		ax_list[1, 0].yaxis.set_major_locator(yticks10)
		ax_list[1, 0].xaxis.set_major_locator(xticks10)	

		# (1, 1): Model
		model = self.gen_model(q_model)
		ax_list[1, 1].imshow(model, cmap="gray", interpolation="none", vmin=vmin, vmax=vmax)
		ax_list[1, 1].scatter(Y0, X0, c=color_truth, s=S0, edgecolor="none", marker="x")
		ax_list[1, 1].scatter(Y, X, c=color_model, s=S, edgecolor="none", marker="x")
		ax_list[1, 1].set_title("Model", fontsize=ft_size)
		yticks11 = ticker.MaxNLocator(num_ticks)
		xticks11 = ticker.MaxNLocator(num_ticks)
		ax_list[1, 1].yaxis.set_major_locator(yticks11)
		ax_list[1, 1].xaxis.set_major_locator(xticks11)
		ax_list[1, 1].set_xlabel("Y", fontsize=ft_size)
		ax_list[1, 1].set_ylabel("X", fontsize=ft_size)

		# (1, 2): Residual
		sig_fac = 5
		residual = self.D - model
		sig = np.sqrt(self.B_count)
		ax_list[1, 2].imshow(residual, cmap="gray", interpolation="none", vmin=-sig_fac * sig, vmax=sig_fac * sig)
		ax_list[1, 2].scatter(Y0, X0, c=color_truth, s=S0, edgecolor="none", marker="x")
		ax_list[1, 2].scatter(Y, X, c=color_model, s=S, edgecolor="none", marker="x")
		ax_list[1, 2].set_title("Residual", fontsize=ft_size)
		yticks12 = ticker.MaxNLocator(num_ticks)
		xticks12 = ticker.MaxNLocator(num_ticks)
		ax_list[1, 2].yaxis.set_major_locator(yticks12)
		ax_list[1, 2].xaxis.set_major_locator(xticks12)
		ax_list[1, 2].set_xlabel("Y", fontsize=ft_size)
		ax_list[1, 2].set_ylabel("X", fontsize=ft_size)

		# (0, 2): Residual histogram
		sig_fac2 = 7 # Histogram should plot wider range
		binsize = sig_fac * sig / 12.5
		bins = np.arange(-sig_fac2 * sig, sig_fac2 * sig, binsize)
		x = np.arange(-sig_fac2 * sig, sig_fac2 * sig, sig_fac2 * sig / 1000.)
		ax_list[0, 2].plot(x, self.num_rows * self.num_cols * gaussian_1D(x, sig = sig) * binsize, lw=1.5)
		ax_list[0, 2].hist(residual.ravel(), bins=bins, color="black", lw=1.5, histtype="step")
		ax_list[0, 2].set_xlim([-sig_fac2 * sig, sig_fac2 * sig])
		ax_list[0, 2].set_title("Res. hist", fontsize=ft_size)
		yticks02 = ticker.MaxNLocator(num_ticks)
		xticks02 = ticker.MaxNLocator(num_ticks)
		ax_list[0, 2].yaxis.set_major_locator(yticks02)
		ax_list[0, 2].xaxis.set_major_locator(xticks02)

		# Add title
		if title_str is not None:
			plt.suptitle(title_str, fontsize=25)

		if show:
			plt.show()
		if save is not None:
			plt.savefig(save, dpi=100, bbox_inches = "tight")
		plt.close()
