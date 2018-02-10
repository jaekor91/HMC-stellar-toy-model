from utils import *

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
gain = 4.62 # photo-electron counts to ADU
ADU_to_flux = 0.00546689 # nanomaggies per ADU
B_ADU = 179 # Background in ADU.
B_count = B_ADU/gain
flux_to_count = 1./(ADU_to_flux * gain) # Flux to count conversion

#---- Step size coeff
dt_xy_coeff = [1., 5e-1, 1e-1] # Smaller coefficient for deeper image
dt_f_coeff = [1e-2, 1e-2, 1e-2]

#---- Friction coefficients
# alphas = np.array([[1e-3, 1e-1, 1e-1], [1e-3, 1e-2, 1e-2], [1e-3, 1e-2, 1e-2]])
alphas = np.array([[0, 0, 0]]*3)


#---- Background chosen to be integer.
for j, mag_B in enumerate([23., 25., 27.]):
	if j  == -1:
		pass
	else: 
		print "/---- Background magnitude: %d" % mag_B
		B_count = mag2flux(mag_B) * flux_to_count

		# Minimum flux
		fmin = B_count

		# Size of the image
		num_rows = num_cols = 48 # Pixel index goes from 0 to num_rows-1

		# Number of time steps
		Nsample = 1000

		dir_figures = "./figures/"


		#---- Single history to pick the best mode inference parameters including viscosity parameters alpha and 
		# adaptive time step size
		#---- Select magnitude of the star to use
		mag_arr = np.arange(mag_B-4, mag_B, 1)
		for mag_model in mag_arr:
			print "Initial mag: %d" % mag_model
			#---- Generate blank test image
			D0 = np.ones((num_rows, num_cols), dtype=float) * B_count 
			# Poission realization D of the underlying truth D0
			D = poisson_realization(D0)

			#---- Define grads and potential
			# Define potential
			def V(objs_flat):
				"""
				Negative Poisson log-likelihood given data and model.

				The model is specified by the list of objs, which are provided
				as a flattened list [Nobjs x 3](e.g., [f1, x1, y1, f2, x2, y2, ...])

				Assume a fixed background.
				"""
				Nobjs = objs_flat.size // 3 # Number of objects
				Lambda = np.ones_like(D) * B_count # Model set to background
				for i in range(Nobjs): # Add every object.
					f, x, y = objs_flat[3*i:3*i+3]
					Lambda += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
				return -np.sum(D * np.log(Lambda) - Lambda)

			# Define gardients
			def dVdq(objs_flat):
				"""
				Gradient of Poisson pontentia above.    
				"""
				# Place holder for the gradient.
				grad = np.zeros(objs_flat.size)

				# Compute the model.
				Nobjs = objs_flat.size // 3 # Number of objects
				Lambda = np.ones_like(D) * B_count # Model set to background
				for i in range(Nobjs): # Add every object.
					f, x, y = objs_flat[3*i:3*i+3]
					Lambda += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)

				# Variable to be recycled
				rho = (D/Lambda)-1.# (D_lm/Lambda_lm - 1)
				# Compute f, x, y gradient for each object
				lv = np.arange(0, num_rows)
				mv = np.arange(0, num_cols)
				mv, lv = np.meshgrid(lv, mv)
				var = (PSF_FWHM_pix/2.354)**2 
				for i in range(Nobjs):
					f, x, y = objs_flat[3*i:3*i+3]
					PSF = gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
					grad[3*i] = -np.sum(rho * PSF) # flux grad
					grad[3*i+1] = -np.sum(rho * (lv - x + 0.5) * PSF) * f / var
					grad[3*i+2] = -np.sum(rho * (mv - y + 0.5) * PSF) * f / var
				return grad
			
			#---- Allocate memory for the trajectory and initialize
			phi_t = np.zeros((Nsample, 3)) # We are considering only one particle.
			phi_t[0] = np.array([mag2flux(mag_model) * flux_to_count, num_rows/2., num_cols/2.])
			p_t = np.zeros((Nsample, 3)) # Momentum initially set to zero.



			# Memory for time and gradients
			dt_t = np.zeros_like(p_t)
			grads_t = np.zeros_like(p_t)

			# Viscosity term for velocity
			alpha = alphas[j]

			#---- *Time* evolution where step sizes are adjusted by flux.
			# Compute intial time steps
			dt_f =  dt_f_coeff[j] * phi_t[0, 0] # Fraction of previous flux
			dt_xy = dt_xy_coeff[j]/phi_t[0, 0] # Inverse of previous flux
			dt = np.array([dt_f, dt_xy, dt_xy])
			dt_t[0] = dt
			# Compute the first leap frog step t = 1/2
			i = 0 
			grad = dVdq(phi_t[i, :])
			grads_t[i]= grad
			p_t[i, :] = np.array([-mag2flux(25) * flux_to_count, 0, 0]) - dt * grad / 2. # We assume unit covariance momentum matrix.

			for i in range(1, Nsample):
				# Update position
				q_tmp = phi_t[i-1] + dt * p_t[i-1]
				if q_tmp[0] < fmin:
					break
				phi_t[i] = q_tmp

				# Compute new step size
				dt_f = dt_f_coeff[j] * phi_t[i, 0] # Fraction of previous flux
				dt_xy = dt_xy_coeff[j]/phi_t[i, 0] # Inverse of previous flux
				dt = np.array([dt_f, dt_xy, dt_xy])
				dt_t[i] = dt

				# Update momentum to next half step t = i + 0.5
				grad = dVdq(phi_t[i])
				grads_t[i]= grad    
				p_t[i] = (1.-alpha * dt) * p_t[i-1] - dt * grad

			plt.close()
			fig, ax_list = plt.subplots(6, 2, figsize=(25, 20))
			#---- xy
			# xy
			ax_list[0, 0].plot(range(i), phi_t[:i, 1], label="x", c="black")
			ax_list[0, 0].plot(range(i), phi_t[:i, 2], label="y", c="red")
			ax_list[0, 0].set_ylabel("xy", fontsize=20)
			# Time step xy
			ax_list[1, 0].plot(range(i), dt_t[:i, 1], label="x", c="black")
			ax_list[1, 0].set_ylabel("dt_xy", fontsize=20)
			# Gradient xy
			ax_list[2, 0].plot(range(i), grads_t[:i, 1], label="x", c="black")
			ax_list[2, 0].plot(range(i), grads_t[:i, 2], label="y", c="red")
			ax_list[2, 0].set_ylabel("dV/dxy", fontsize=20)
			# Gradient * time step
			ax_list[3, 0].plot(range(i), grads_t[:i, 1] * dt_t[:i, 1], label="x", c="black")
			ax_list[3, 0].plot(range(i), grads_t[:i, 2] * dt_t[:i, 2], label="y", c="red")
			ax_list[3, 0].set_ylabel("dt_xy * dV/dxy", fontsize=20)
			# Momentum xy
			ax_list[4, 0].plot(range(i), p_t[:i, 1], label="x", c="black")
			ax_list[4, 0].plot(range(i), p_t[:i, 2], label="y", c="red")
			ax_list[4, 0].set_ylabel("p_xy", fontsize=20)
			# Momentum xy * time_step
			ax_list[5, 0].plot(range(i), p_t[:i, 1]* dt_t[:i, 1], label="x", c="black")
			ax_list[5, 0].plot(range(i), p_t[:i, 2]* dt_t[:i, 2], label="y", c="red")
			ax_list[5, 0].set_ylabel("p_xy * dt_xy", fontsize=20)


			#---- Mag
			Nsample_sub = i # int(200 + (4 - (mag_B - mag_model)) * 100)
			# Mag
			ax_list[0, 1].plot(range(Nsample_sub), flux2mag(phi_t[:Nsample_sub, 0]/flux_to_count), c="black")
			ax_list[0, 1].axhline(y=flux2mag(B_count/flux_to_count), c="blue", ls="--", lw=1)
			ax_list[0, 1].set_ylabel("Mag", fontsize=20)
			# Time step f
			ax_list[1, 1].plot(range(Nsample_sub), dt_t[:Nsample_sub, 0], c="black")
			ax_list[1, 1].set_ylabel("dt_f", fontsize=20)
			# Gradient f
			ax_list[2, 1].plot(range(Nsample_sub), grads_t[:Nsample_sub, 0], c="black")
			ax_list[2, 1].set_ylabel("dV/df", fontsize=20)
			# Gradient f * time step
			ax_list[3, 1].plot(range(Nsample_sub), grads_t[:Nsample_sub, 0] * dt_t[:Nsample_sub, 0], c="black")
			ax_list[3, 1].set_ylabel("dt_f * dV/df", fontsize=20)
			# Momentum f
			ax_list[4, 1].plot(range(Nsample_sub), p_t[:Nsample_sub, 0], c="black")
			ax_list[4, 1].set_ylabel("p_f", fontsize=20)
			# Momentum f
			ax_list[5, 1].plot(range(Nsample_sub), p_t[:Nsample_sub, 0] * dt_t[:Nsample_sub, 0], c="black")
			ax_list[5, 1].set_ylabel("p_f * dt_f", fontsize=20)

		#     plt.show()
			plt.savefig(dir_figures+"one-star-in-mag%d-background-model-mag%d-single-history.png" % (mag_B, mag_model), dpi=200, bbox_inches="tight")
			plt.close()        
		print "Completed"