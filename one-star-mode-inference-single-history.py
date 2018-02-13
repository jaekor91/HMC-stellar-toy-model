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
dt_xy_coeff = [1e-1] * 3 # Smaller coefficient for deeper image
dt_f_coeff = [1e-1] * 3

# Size of the image
num_rows = num_cols = 48 # Pixel index goes from 0 to num_rows-1

# Number of time steps
Nsample = 1000

dir_figures = "./figures/one-star-mode-inferenece/"

dxs = np.arange(0, 5, 1)

num_cases = 0

#---- Background chosen to be integer.
for j, mag_B in enumerate([23.]): #, 25., 27.]):
    if j  == -1: # 
        pass
    else: 
        print "/---- Background magnitude: %d" % mag_B
        B_count = mag2flux(mag_B) * flux_to_count

        # Minimum flux
        fmin = mag2flux(mag_B-1.5) * flux_to_count
        
        #---- True magnitudes
        mTs = np.arange(15, mag_B, 1.)
        for mT in mTs:
            print "/-- mT =", mT
            #---- Select magnitude of the star to use
            mag_arr = [mag_B-2] # Initial model magnitude
            for mag_model in mag_arr:
                print "/- Initial mag: %d" % mag_model
                for dx in dxs:
                    num_cases +=1
                    print "Initial dx: %.2f" % dx
                    xT = num_rows/2. + np.random.random() - 0.5
                    yT = num_cols/2. + np.random.random() - 0.5
                    # print "Truth xy: %.2f, %.2f" % (xT, yT)
                
                    #---- Generate blank test image
                    D0 = np.ones((num_rows, num_cols), dtype=float) * B_count  # Background
                    # Truth image by adding a star
                    f = mag2flux(mT) * flux_to_count
                    D0 += f * gauss_PSF(num_rows, num_cols, xT, yT, FWHM=PSF_FWHM_pix)
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
                    phi_t[0] = np.array([mag2flux(mag_model) * flux_to_count, xT+dx, yT+dx])
                    
                    # Memory for time and gradients
                    dt_t = np.zeros_like(phi_t)
                    grads_t = np.zeros_like(phi_t)
                    V_t = np.zeros(Nsample)
                    
                    #---- *Time* evolution where step sizes are adjusted by flux.
                    # Compute intial time steps
                    dt_f =  dt_f_coeff[j] * phi_t[0, 0] # Fraction of previous flux
                    dt_xy = dt_xy_coeff[j]/phi_t[0, 0] # Inverse of previous flux
                    dt = np.array([dt_f, dt_xy, dt_xy])
                    dt_t[0] = dt
                    
                    for i in range(1, Nsample):
                        grad = dVdq(phi_t[i-1])
                        grads_t[i]= grad                                                
                        phi_t[i] = phi_t[i-1] - dt * grad 
                        V_t[i] = V(phi_t[i])

                        if (phi_t[i, 0] - phi_t[i-1, 0])/phi_t[i, 0] < 1e-6:
                            break


                        # Compute new step size
                        dt_f = dt_f_coeff[j] * phi_t[i, 0] # Fraction of previous flux
                        dt_xy = dt_xy_coeff[j]/phi_t[i, 0] # Inverse of previous flux
                        dt = np.array([dt_f, dt_xy, dt_xy])
                        dt_t[i] = dt


                    plt.close()
                    fig, ax_list = plt.subplots(5, 2, figsize=(21, 20))
                    #---- xy
                    # xy
                    ax_list[0, 0].plot(range(i), phi_t[:i, 1]-xT, label="x", c="black")
                    ax_list[0, 0].plot(range(i), phi_t[:i, 2]-yT, label="y", c="red")
                    ax_list[0, 0].axhline(y=0, c="blue", ls="--", lw=1)
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
                    # Potential
                    ax_list[4, 0].plot(range(i), V_t[:i], label="x", c="black")
                    ax_list[4, 0].set_ylabel("V_t", fontsize=20)


                    #---- Mag
                    Nsample_sub = i # int(200 + (4 - (mag_B - mag_model)) * 100)
                    # Mag
                    ax_list[0, 1].plot(range(Nsample_sub), flux2mag(phi_t[:Nsample_sub, 0]/flux_to_count), c="black")
                    ax_list[0, 1].axhline(y=mag_B, c="red", ls="--", lw=1)
                    ax_list[0, 1].axhline(y=mT, c="blue", ls="--", lw=1)

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
                    ax_list[4, 1].axis("off")

                #     plt.show()
                    plt.savefig(dir_figures+"one-star-modal-inference-mB%d-mT%d-mM%d-dx%d-single-history.png" % (mag_B, mT, mag_model, dx), dpi=200, bbox_inches="tight")
                    plt.close()        
                print "Completed\n"                    


print num_cases