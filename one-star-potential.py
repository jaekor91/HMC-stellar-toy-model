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






#---- V(dx) at various y's including the correct one; f variation [15, 24]
# Rows correspond to different true f values
# Columns correspond to different y values
# Each panel shows V(dx) at various model f values
dx = np.arange(-10, 10+0.5/2., 0.5)
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
        
        # For each flux guess, generate a V(dx) curve corresponding to true f value
        for mag_model in mag_arr:
            if mag_model >= (mag_true-2): # Only plot those that are at most two magnitudes brigther.
                V_dx = np.zeros_like(dx)
                for k, dx_tmp in enumerate(dx):
                    objs1 = np.copy(objs) # Create model objects 
                    objs1[0, 0] = mag2flux(mag_model) * flux_to_count
                    objs1[0, 1] += dx_tmp # Perturb model objects
                    objs1[0, 2] += dy_fixed
                    objs_flat = objs1.flatten()     
                    V_dx[k] = V(objs_flat) # Compute potential based on the model object
                ax_list[i, j].plot(dx, V_dx, lw=3, label="M=%d" % mag_model)
            
        # Panel decoration
        ft_size = 25
        ax_list[i, j].set_title("mag_true/dy: %.1f/%.2f" % (mag_true, dy_fixed), fontsize=ft_size)
        ax_list[i, j].set_xlabel("dx", fontsize=ft_size)
        ax_list[i, j].set_ylabel("V(dx)", fontsize=ft_size)            
        ax_list[i, j].legend(loc="lower right", fontsize=ft_size*0.8)
plt.savefig(dir_figures+"one-star-V_dx.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()



#---- V(m) at various dx; m variation [15, 24]
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

    # For each flux guess, generate a V(dx) curve corresponding to true f value
    mag_model = np.arange(max(13, mag_true-2), min(24, mag_true+3), 0.2)    
    for dx_tmp in dx:
        V_m = np.zeros_like(mag_model)
        for k, m in enumerate(mag_model):
            objs1 = np.copy(objs) # Create model objects 
            objs1[0, 0] = mag2flux(m) * flux_to_count
            objs1[0, 1] += dx_tmp # Perturb model objects
            objs_flat = objs1.flatten()     
            V_m[k] = V(objs_flat) # Compute potential based on the model object
        ax_list[idx_row, idx_col].plot(mag_model, V_m, lw=3, label="dx=%.2f" % dx_tmp)
            
    # Panel decoration
    ft_size = 25
    ax_list[idx_row, idx_col].set_title("mag_true: %.2f" % (mag_true), fontsize=ft_size)
    ax_list[idx_row, idx_col].axvline(x=mag_true, c="black", ls="--", lw=2)    
    ax_list[idx_row, idx_col].axhline(y=np.median(V_m[-10:]), c="black", ls="--", lw=2)        
    ax_list[idx_row, idx_col].set_xlabel("m", fontsize=ft_size)
    ax_list[idx_row, idx_col].set_ylabel("V(m)", fontsize=ft_size)            
    ax_list[idx_row, idx_col].set_xlim([max(13, mag_true-2), min(24, mag_true+3)])
    ax_list[idx_row, idx_col].legend(loc="upper right", fontsize=ft_size*0.8)
plt.savefig(dir_figures+"one-star-V_m.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()	






#---- V(dx, m)
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
fig, ax_list = plt.subplots(Nrows, Ncols, figsize=(100, 100))

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
        
        # For each flux guess, generate a V(dx) curve corresponding to true f value
        V_dx_m = np.zeros((dx.size, mag_arr_model.size))        
        for k, mag_model in enumerate(mag_arr_model):
            for l, dx_tmp in enumerate(dx):
                objs1 = np.copy(objs) # Create model objects 
                objs1[0, 0] = mag2flux(mag_model) * flux_to_count
                objs1[0, 1] += dx_tmp # Perturb model objects
                objs1[0, 2] += dy_fixed
                objs_flat = objs1.flatten()     
                V_dx_m[l, k] = V(objs_flat) # Compute potential based on the model object
                
        ax_list[i, j].imshow(V_dx_m, interpolation="None", cmap="gray",\
                             vmin=np.min(V_dx_m), vmax=np.percentile(V_dx_m[:, -20:], 90),\
                            extent=(10, 25, -10, 10))
        # True minimum
#         ax_list[i, j].scatter([int(mag_arr_model.size * (mag_true-10.)/float(15.))], [dx.size//2], s=10, c="red")
        ax_list[i, j].scatter([mag_true], [0], s=500, c="red", edgecolor="None")
        ax_list[i, j].axvline(x=mag_true, c="red", ls="--", lw=1)
        ax_list[i, j].axhline(y=0, c="red", ls="--", lw=1)
        
        # Panel decoration
        ft_size = 25
        ax_list[i, j].set_title("mag_true/dy: %.1f/%.2f" % (mag_true, dy_fixed), fontsize=ft_size)
        ax_list[i, j].set_xlabel("mag", fontsize=ft_size)
        ax_list[i, j].set_ylabel("dx", fontsize=ft_size)            
#         ax_list[i, j].set_axis_off()
#         ax_list[i, j].axis("equal")
plt.savefig(dir_figures+"one-star-V_dx_m.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()





#---- V(dx, dy)
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
        
        # For each flux guess, generate a V(dx) curve corresponding to true f value
        V_dx_dy = np.zeros((dy.size, dx.size))        
        for k, dy_tmp in enumerate(dy):
            for l, dx_tmp in enumerate(dx):
                objs1 = np.copy(objs) # Create model objects 
                objs1[0, 0] = mag2flux(mag_model) * flux_to_count
                objs1[0, 1] += dx_tmp # Perturb model objects
                objs1[0, 2] += dy_tmp
                objs_flat = objs1.flatten()     
                V_dx_dy[k, l] = V(objs_flat) # Compute potential based on the model object
                
        ax_list[i, j].imshow(V_dx_dy, interpolation="None", cmap="gray",\
                             vmin=np.min(V_dx_dy), vmax=np.percentile(V_dx_dy, 90),\
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
        
plt.savefig(dir_figures+"one-star-V_dx_dy.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()