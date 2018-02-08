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




mag_arr = np.arange(15, 24, 0.25)

# Number of rows and columns
Nrows = Ncols = 6

# Create figure
plt.close()
fig, ax_list = plt.subplots(Nrows, Ncols, figsize=(50, 50))

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
    ax_list[idx_row, idx_col].imshow(D, vmin=np.percentile(D, 10), vmax=np.max(D), interpolation="none",\
                                    extent=(-32, 32, -32, 32), cmap="gray")    
    ax_list[idx_row, idx_col].set_title("Mag = %.2f" % mag_true, fontsize=25)
    ax_list[idx_row, idx_col].axvline(x=0, c="red", lw=2, ls="--")    
    ax_list[idx_row, idx_col].axhline(y=0, c="red", lw=2, ls="--")    
    
plt.savefig(dir_figures+"one-star-image.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()



