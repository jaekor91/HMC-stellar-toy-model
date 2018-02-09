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

# # Prior parameters
# alpha = -1.1# f**alpha
# fmin = mag2flux(20) # Minimum flux of an object.
# fmax = 17. # Maximum flux in the simulation

# Figure directory
dir_figures ="./figures/"








#---- dVdx(dx) at various y's including the correct one; f variation [15, 24]
# Rows correspond to different true f values
# Columns correspond to different y values
# Each panel shows V(dx) at various model f values
dx = np.arange(-10, 10+0.5/2., 0.2)
dy = np.arange(-10, 0.+0.5/2., 2)
mag_arr = np.arange(15, 23, 1)

# Number of rows and columns
Nrows = mag_arr.size
Ncols = dy.size

# Create figure
fig, ax_list = plt.subplots(Nrows, Ncols, figsize=(70, 100))

# Loop through panels 
for i, mag_true in enumerate(mag_arr):
    for j, dy_fixed in enumerate(dy):
        #---- Generate a mock image given a list of objects x, y, f.
        objs = np.array([[mag2flux(mag_true) * flux_to_count, num_rows/2., num_cols/2.]])
        # Generate a blank image with background
        D0 = np.ones((num_rows, num_cols), dtype=float) * B_count 
        # Insert objects: For each object, evaluate the PSF located at x and y position
        f, x, y = objs[0]
        D0 += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
        # Poission realization D of the underlying truth D0
        D = poisson_realization(D0)

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

        # For each flux guess, generate a V(dx) curve corresponding to true f value
        for mag_model in mag_arr:
            if mag_model >= (mag_true-2): # Only plot those that are at most two magnitudes brigther.
                dVdx_dx = np.zeros_like(dx)
                for k, dx_tmp in enumerate(dx):
                    objs1 = np.copy(objs) # Create model objects 
                    objs1[0, 0] = mag2flux(mag_model) * flux_to_count
                    objs1[0, 1] += dx_tmp # Perturb model objects
                    objs1[0, 2] += dy_fixed
                    objs_flat = objs1.flatten()     
                    dVdx_dx[k] = dVdq(objs_flat)[1] # Compute grad based on the model object
                ax_list[i, j].plot(dx, dVdx_dx, lw=3, label="M=%d" % mag_model)

        # Panel decoration
        ft_size = 25
        ax_list[i, j].set_title("mag_true/dy: %.1f/%.2f" % (mag_true, dy_fixed), fontsize=ft_size)
        ax_list[i, j].set_xlabel("dx", fontsize=ft_size)
        ax_list[i, j].set_ylabel("dV/dx(dx)", fontsize=ft_size)            
        ax_list[i, j].legend(loc="lower right", fontsize=ft_size*0.8)
        ax_list[i, j].set_xlim([-10, 10])        
        ax_list[i, j].axvline(x=0, c="black", lw=2, ls="--")
        ax_list[i, j].axhline(y=0, c="black", lw=2, ls="--")
        
        
plt.savefig(dir_figures+"one-star-dVdx_dx.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()







#---- dV/df(m) at various dx; m variation [15, 24]
# Row 1 correspond to m = 15 through 18
# Row 2 to m = 19 through 22
# Columns correspond to different x values
# Each panel shows V(m) at various model f values
dx = np.arange(-10, 0+0.5/2., 2.)
mag_arr = np.arange(15, 23, 1)

# Number of rows and columns
Nrows = 2
Ncols = 4

# Create figure
plt.close()
fig, ax_list = plt.subplots(Nrows, Ncols, figsize=(50, 20))

# Loop through panels 
for i, mag_true in enumerate(mag_arr):
    idx_row = i // Ncols
    idx_col = i % Ncols 
    #---- Generate a mock image given a list of objects x, y, f.
    objs = np.array([[mag2flux(mag_true) * flux_to_count, num_rows/2., num_cols/2.]])
    # Generate a blank image with background
    D0 = np.ones((num_rows, num_cols), dtype=float) * B_count 
    # Insert objects: For each object, evaluate the PSF located at x and y position
    f, x, y = objs[0]
    D0 += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
    # Poission realization D of the underlying truth D0
    D = poisson_realization(D0)

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
    
    # For each flux guess, generate a V(dx) curve corresponding to true f value
    mag_model = np.arange(13, 25, 0.2)    
    for dx_tmp in dx:
        dVdf_m = np.zeros_like(mag_model)
        for k, m in enumerate(mag_model):
            objs1 = np.copy(objs) # Create model objects 
            objs1[0, 0] = mag2flux(m) * flux_to_count
            objs1[0, 1] += dx_tmp # Perturb model objects
            objs_flat = objs1.flatten()     
            dVdf_m[k] = dVdq(objs_flat)[0] # Compute potential based on the model object
        ax_list[idx_row, idx_col].plot(mag_model, dVdf_m, lw=3, label="dx=%.2f" % dx_tmp)
            
    # Panel decoration
    ft_size = 25
    ax_list[idx_row, idx_col].set_title("mag_true: %.2f" % (mag_true), fontsize=ft_size)
    ax_list[idx_row, idx_col].axvline(x=mag_true, c="black", ls="--", lw=2)    
    ax_list[idx_row, idx_col].axhline(y=0, c="black", ls="--", lw=2)        
#     ax_list[idx_row, idx_col].axhline(y=np.median(V_m[-10:]), c="black", ls="--", lw=2)        
    ax_list[idx_row, idx_col].set_xlabel("m", fontsize=ft_size)
    ax_list[idx_row, idx_col].set_ylabel("dV/df(m)", fontsize=ft_size)            
    ax_list[idx_row, idx_col].set_xlim([13, 25])
    ax_list[idx_row, idx_col].legend(loc="lower left", fontsize=ft_size*0.8)
plt.savefig(dir_figures+"one-star-dVdf_m.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()



#---- dVdx(dx, dy)
# Rows correspond to different true mag values
# Columns correspond to different model mag values
# Each panel shows 2D image where magnitude on the x-axis.
dx = np.arange(-10, 10+0.5/2., 0.2) 
dy = np.arange(-10, 10.+0.5/2., 0.2)
mag_arr = np.arange(16, 24, 1) 

# Number of rows and columns
Nrows = Ncols = mag_arr.size

# Create figure
plt.close()
fig, ax_list = plt.subplots(Nrows, Ncols, figsize=(80, 80))

# Loop through panels 
for i, mag_true in enumerate(mag_arr):
    for j, mag_model in enumerate(mag_arr):
        #---- Generate a mock image given a list of objects x, y, f.
        objs = np.array([[mag2flux(mag_true) * flux_to_count, num_rows/2., num_cols/2.]])
        # Generate a blank image with background
        D0 = np.ones((num_rows, num_cols), dtype=float) * B_count 
        # Insert objects: For each object, evaluate the PSF located at x and y position
        f, x, y = objs[0]
        D0 += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
        # Poission realization D of the underlying truth D0
        D = poisson_realization(D0)

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

        # For each flux guess, generate a V(dx) curve corresponding to true f value
        dVdx_dx_dy = np.zeros((dy.size, dx.size))        
        for k, dy_tmp in enumerate(dy):
            for l, dx_tmp in enumerate(dx):
                objs1 = np.copy(objs) # Create model objects 
                objs1[0, 0] = mag2flux(mag_model) * flux_to_count
                objs1[0, 1] += dx_tmp # Perturb model objects
                objs1[0, 2] += dy_tmp
                objs_flat = objs1.flatten()     
                dVdx_dx_dy[k, l] = dVdq(objs_flat)[1] # Compute potential based on the model object
                
        ax_list[i, j].imshow(dVdx_dx_dy, interpolation="None", cmap="RdBu_r",\
                             vmin=np.min(dVdx_dx_dy), vmax=np.max(dVdx_dx_dy),\
                            extent=(-10, 10, -10, 10))
        # True minimum
#         ax_list[i, j].scatter([int(mag_arr_model.size * (mag_true-10.)/float(15.))], [dx.size//2], s=10, c="red")
        ax_list[i, j].scatter([0], [0], s=500, c="red", edgecolor="None")
        ax_list[i, j].axvline(x=0, c="red", ls="--", lw=1)
        ax_list[i, j].axhline(y=0, c="red", ls="--", lw=1)
        
        # Panel decoration
        ft_size = 25
        ax_list[i, j].set_title("mag_true/model: %.1f/%.1f" % (mag_true, mag_model), fontsize=ft_size)
        ax_list[i, j].set_xlabel("dy", fontsize=ft_size)
        ax_list[i, j].set_ylabel("dx", fontsize=ft_size)            
        ax_list[i, j].axis("equal")
fig.suptitle("(Red, Positive) ---- (White, Zero) ---- (Blue, Negative)", fontsize=30)        
plt.savefig(dir_figures+"one-star-dVdx_dx_dy.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()






#---- dVdf(dx, m) and dVdx(dx, m)
# Rows correspond to different true mag values
# Columns correspond to different dy values
# Each panel shows 2D image where magnitude on the x-axis.
dx = np.arange(-10, 10+0.5/2., 0.2) 
dy = np.arange(-8, 0.+0.5/2., 1.)
mag_arr = np.arange(15, 24, 1) 
mag_arr_model = np.arange(10, 25, 0.2)

# Number of rows and columns
Nrows = mag_arr.size
Ncols = dy.size

# Create figure
plt.close()
fig1, ax_list1 = plt.subplots(Nrows, Ncols, figsize=(80, 80))
fig2, ax_list2 = plt.subplots(Nrows, Ncols, figsize=(80, 80))


# Loop through panels 
for i, mag_true in enumerate(mag_arr):
    for j, dy_fixed in enumerate(dy):
        #---- Generate a mock image given a list of objects x, y, f.
        objs = np.array([[mag2flux(mag_true) * flux_to_count, num_rows/2., num_cols/2.]])
        # Generate a blank image with background
        D0 = np.ones((num_rows, num_cols), dtype=float) * B_count 
        # Insert objects: For each object, evaluate the PSF located at x and y position
        f, x, y = objs[0]
        D0 += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
        # Poission realization D of the underlying truth D0
        D = poisson_realization(D0)

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

        
        # For each flux guess, generate a V(dx) curve corresponding to true f value
        dVdf_dx_m = np.zeros((dx.size, mag_arr_model.size))        
        dVdx_dx_m = np.zeros((dx.size, mag_arr_model.size))                
        for k, mag_model in enumerate(mag_arr_model):            
            for l, dx_tmp in enumerate(dx):
                objs1 = np.copy(objs) # Create model objects 
                objs1[0, 0] = mag2flux(mag_model) * flux_to_count
                objs1[0, 1] += dx_tmp # Perturb model objects
                objs1[0, 2] += dy_fixed
                objs_flat = objs1.flatten()     
                grads = dVdq(objs_flat) # Compute potential based on the model object
                dVdf_dx_m[l, k] = grads[0]
                dVdx_dx_m[l, k] = grads[1]                
        #--- dVdf
        orig_cmap = matplotlib.cm.RdBu_r
        shifted_cmap = shiftedColorMap(orig_cmap, start=np.min(dVdf_dx_m), stop=1, midpoint=0., name='shifted')
        ax_list1[i, j].imshow(dVdf_dx_m, interpolation="None", cmap=shifted_cmap,\
                             vmin=np.min(dVdf_dx_m), vmax=1.,\
                            extent=(13, 25, -10, 10))
        # True minimum
        ax_list1[i, j].scatter([mag_true], [0], s=500, c="red", edgecolor="None")
        ax_list1[i, j].axvline(x=mag_true, c="red", ls="--", lw=1)
        ax_list1[i, j].axhline(y=0, c="red", ls="--", lw=1)
        
        # Panel decoration
        ft_size = 25
        ax_list1[i, j].set_title("mag_true/dy: %.1f/%.2f" % (mag_true, dy_fixed), fontsize=ft_size)
        ax_list1[i, j].set_xlabel("mag", fontsize=ft_size)
        ax_list1[i, j].set_ylabel("dx", fontsize=ft_size)            
        
        #--- dVdx
        ax_list2[i, j].imshow(dVdx_dx_m, interpolation="None", cmap="RdBu_r",\
                             vmin=np.min(dVdx_dx_m)*0.5, vmax=np.max(dVdx_dx_m)*0.5,\
                            extent=(13, 25, -10, 10))
        # True minimum
        ax_list2[i, j].scatter([mag_true], [0], s=500, c="red", edgecolor="None")
        ax_list2[i, j].axvline(x=mag_true, c="red", ls="--", lw=1)
        ax_list2[i, j].axhline(y=0, c="red", ls="--", lw=1)
        
        # Panel decoration
        ft_size = 25
        ax_list2[i, j].set_title("mag_true/dy: %.1f/%.2f" % (mag_true, dy_fixed), fontsize=ft_size)
        ax_list2[i, j].set_xlabel("mag", fontsize=ft_size)
        ax_list2[i, j].set_ylabel("dx", fontsize=ft_size)            
# fig1.suptitle("(Red, Positive) ---- (White, Zero) ---- (Blue, Negative)", fontsize=30)        
fig1.savefig(dir_figures+"one-star-dVdf_dx_m.png", dpi=200, bbox_inches="tight")
# fig2.suptitle("(Red, Positive) ---- (White, Zero) ---- (Blue, Negative)", fontsize=30)
fig2.savefig(dir_figures+"one-star-dVdx_dx_m.png", dpi=200, bbox_inches="tight")

# plt.show()
plt.close()



