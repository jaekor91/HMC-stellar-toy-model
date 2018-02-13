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


# Custom background
flux_to_count = 1./(ADU_to_flux * gain) # Flux to count conversion

#---- Step size coeff
dt_xy_coeff = [1e-1, 5e-1, 5e-2] # Smaller coefficient for deeper image
dt_f_coeff = [1e-1, 2e-2, 1e-2]

# Number of time steps
Nsample = 1000

dir_figures = "./figures/"

# Number of histories
Nhistory = 1000

# Size of the image
num_rows = num_cols = 48 # Pixel index goes from 0 to num_rows-1

#---- Background chosen to be integer.
for j, mag_B in enumerate([23.]):# ., 25., 27.]):
    if j == -1:
        pass
    else:

        print "Background mag: %d" % mag_B
        B_count = mag2flux(mag_B) * flux_to_count # B_ADU/gain

        # Minimum flux
        fmin = mag2flux(mag_B-1.5) * flux_to_count # B_ADU/gain

        #---- Select magnitude of the star to use
        mag_arr = np.arange(mag_B-4, mag_B-1, 1)
        for mag_model in mag_arr:
            print "/---- mag = %d" % mag_model
            plt.close()
            fig, ax_list = plt.subplots(1, 2, figsize=(15, 7)) # Collect histories in
            for _ in xrange(Nhistory):
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

                #---- *Time* evolution where step sizes are adjusted by flux.
                # Compute intial time steps
                dt_f =  dt_f_coeff[j] * phi_t[0, 0] # Fraction of previous flux
                dt_xy = dt_xy_coeff[j]/phi_t[0, 0] # Inverse of previous flux
                dt = np.array([dt_f, dt_xy, dt_xy])

                # Compute the first leap frog step t = 1/2
                i = 0 
                for i in range(1, Nsample):
                    grad = dVdq(phi_t[i-1])
                    q_tmp = phi_t[i-1] - dt * grad 
                    if q_tmp[0] < fmin:
                        break
                    phi_t[i] = q_tmp

                    # Compute new step size
                    dt_f = dt_f_coeff[j] * phi_t[i, 0] # Fraction of previous flux
                    dt_xy = dt_xy_coeff[j]/phi_t[i, 0] # Inverse of previous flux
                    dt = np.array([dt_f, dt_xy, dt_xy])

                #--- plot current trajectory
                # xy 
                ax_list[0].plot(phi_t[:i, 1], phi_t[:i, 2], lw=1, c="black", alpha=0.2)

                # xm and ym
                dr = np.sqrt((phi_t[:i, 1]-num_rows/2.)**2 + (phi_t[:i, 2]-num_cols/2.)**2)
                ax_list[1].plot(flux2mag(phi_t[:i, 0] / flux_to_count), dr, lw=1, c="black", alpha=0.2)

            # xy
            ax_list[0].axis("equal")    
            ax_list[0].set_xlim([num_rows/2.-1., num_rows/2.+1.])
            ax_list[0].set_ylim([num_cols/2.-1., num_cols/2.+1.])        

            # xm and ym
            ax_list[1].set_ylim([0, 1.])

            plt.savefig(dir_figures+"one-star-in-mag%d-background-model-mag%d-many-trajectory.png" % (mag_B, mag_model), dpi=200, bbox_inches="tight")
            # plt.show()
            plt.close()


        print "Completed"