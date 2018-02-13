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


# Size of the image
num_rows = num_cols = 48 # Pixel index goes from 0 to num_rows-1
dir_figures = "./figures/one-star-mag-mode-inference/"


#---- Background chosen to be integer.
for j, mag_B in enumerate([23.]):
    print "/---- Background magnitude: %d" % mag_B
    B_count = mag2flux(mag_B) * flux_to_count
    
    #---- True magnitudes
    mT_min = 17
    mTs = np.arange(mT_min, mag_B, 0.5)
    for mT in mTs:
        print "/-- mT =", mT
        #---- Core plot
        plt.close()
        fig, ax = plt.subplots(1, figsize=(7, 7))
        
        #---- Select magnitude of the star to use
        mag_arr = [mag_B - 1.5] # These are model magnitudes
        xT = num_rows/2. # + np.random.random() - 0.5
        yT = num_cols/2. # + np.random.random() - 0.5
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

        #---- Compute the potential and plot
        mags = np.arange(mT-1., mag_B, 0.05)
        Vs = np.zeros_like(mags)
        for l in xrange(mags.size):
            Vs[l] = V(np.array([mag2flux(mags[l]) * flux_to_count, xT, yT]))    
        ax.plot(mags, Vs, label="mT=%.2f" % mT)
        ax.axvline(x=mT, c="black", ls="--", lw=1)
        ax.set_xlim([mT-1., mag_B])
        ax.legend(loc="lower right", fontsize=15)
        plt.savefig(dir_figures + "one-star-mag-mode-inference-mB%d-mT%d.png" % (mag_B, mT*10), dpi=200, bbox_inches="tight")
        # plt.show()
        plt.close()
