from samplers import *
import time

# Truths
mTs = [15, 16, 17, 18, 19, 20, 21, 21.4, 21.5, 21.6]
dts = []
fs = []

# Min/max number of steps to use
steps_max = 20
steps_min = 5

for mT in mTs: 
    print "/---- mT %.1f" % mT
    gym = lightsource_gym()
    gym.num_rows = 32
    gym.num_cols = 32
    # arcsec_to_pix = 0.2
    # PSF_FWHM_arcsec = 0.7
    # gym.PSF_FWHM_pix = PSF_FWHM_arcsec / arcsec_to_pix

    # Starting point
    # One source
    q0 = np.array([[mag2flux(mT) * gym.flux_to_count, gym.num_rows/2.+np.random.randn(), gym.num_cols/2.+np.random.randn()]])

    
    # Generating mock data
    print "Generate mock data."
    start = time.time()
    gym.gen_mock_data(q_true=q0)
    print "Time taken: %.2f\n" % (time.time() - start)

    # Determine the best step sizes for each variable
    print "Finding best step sizes"
    start = time.time()
    gym.HMC_find_best_dt(q0, default=False, dt_f_coeff=1, dt_xy_coeff=10, Niter_per_trial=1000, A_target_f=0.99, A_target_xy=0.5)
    print "Time taken: %.2f\n" % (time.time() - start)
    
    # Save flux and dt
    fs.append(q0[0, 0])
    dts.append(gym.dt)



fs = np.asarray(fs)
dts = np.vstack(dts)

dts_f = dts[:, 0]
dts_xy = dts[:, 1]
ms = flux2mag(fs/gym.flux_to_count)


arcsec_to_pix = 0.4
PSF_FWHM_arcsec = 1.4
PSF_FWHM_pix = PSF_FWHM_arcsec / arcsec_to_pix

num_rows = 50
x = y = 15.5 

def gauss_PSF_sumsquare(num_rows, num_cols, x, y, PSF_FWHM_pix):
    return np.sum(np.square(gauss_PSF(num_rows, num_rows, x, y, PSF_FWHM_pix)))

def dVdxx_factor(num_rows, num_cols, x, y, PSF_FWHM_pix):
    # Compute f, x, y gradient for each object
    lv = np.arange(0, num_rows)
    mv = np.arange(0, num_cols)
    mv, lv = np.meshgrid(lv, mv)
    var = (PSF_FWHM_pix/2.354)**2 
    PSF = gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
    factor1 = 1./ np.sqrt(np.sum((lv - x + 0.5)**2 * PSF) / var**2)
    factor2 = 1./ np.sqrt(np.sum((lv - x + 0.5)**2 * PSF**2) / var**2)
        
    return factor1, factor2

factor1, factor2 = dVdxx_factor(num_rows, num_rows, x, y, PSF_FWHM_pix)
lam = gauss_PSF_sumsquare(num_rows, num_rows, x, y, PSF_FWHM_pix)
# Faint limit flux step size
dt_f_faint = np.sqrt(gym.B_count/lam)
dts_xy_faint = factor2 * np.sqrt(gym.B_count)/fs
# Bright limit
dts_f_bright = np.sqrt(fs)
dts_xy_bright = factor1/np.sqrt(fs)



# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
dt_f_coeff = 0.5
dt_xy_coeff = 1.

# Flux step size
ax1.scatter(ms, dts_f)
ax1.plot(ms, dts_f_bright * dt_f_coeff, c="red")
ax1.axhline(y=dt_f_faint * dt_f_coeff, c="blue")

# xy
ax2.scatter(ms, dts_xy)
ax2.plot(ms, dts_xy_bright * dt_xy_coeff, c="red")
ax2.plot(ms, dts_xy_faint * dt_xy_coeff, c="blue")
ax2.set_ylim([0., 0.8])

plt.show()
plt.close()