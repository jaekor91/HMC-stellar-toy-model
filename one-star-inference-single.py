from samplers import *
import time

# Truths
mTs = [15, 16, 17, 18, 19, 20, 21, 21.4, 21.5, 21.6]
# mag_lims = [21.5, 22, 25]
# mTs = [20, 21, 21.4, 21.5, 21.6]
mag_lims = [23, 22, 21.5]

hist_height = 1000

# Number of iterations
Niter = 10000

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
    # Two sources
    # sep = 5.
    # q0 = np.array([[mag2flux(mT) * gym.flux_to_count, gym.num_rows/2. + sep/2., gym.num_cols/2.], \
    #               [mag2flux(mT) * gym.flux_to_count, gym.num_rows/2. - sep/2., gym.num_cols/2.]])

    # Generating mock data
    print "Generate mock data."
    start = time.time()
    gym.gen_mock_data(q_true=q0)
    print "Time taken: %.2f\n" % (time.time() - start)

    # # Find peaks via gradient descent
    # print "Commence peak finding."
    # start = time.time()
    # gym.find_peaks(linear_pix_density=1., no_perturb=False)
    # print "# of peaks found: %d" % gym.q_seed.shape[0]
    # print "Time taken: %.2f\n" % (time.time() - start)

    # # Dispaly if interested
    # print "Display the data."
    # start = time.time()
    # gym.display_data(vmax_percentile=100, vmin_percentile=0, plot_seed=False, size_seed = 20)
    # print "Time taken: %.2f\n" % (time.time() - start)

    # Determine the best step sizes for each variable
    print "Finding best step sizes"
    start = time.time()
    gym.HMC_find_best_dt(q0, default=False, dt_f_coeff=1, dt_xy_coeff=10, Niter_per_trial=100, A_target_f=0.99, A_target_xy=0.5)
    print "Time taken: %.2f\n" % (time.time() - start)

    counter = 0
    for mag_lim in mag_lims:
        counter += 1 # For setting axes
        print "/-- mag_lim %.1f" % mag_lim
        # Calculate f_lim
        f_lim = mag2flux(mag_lim) * gym.flux_to_count

        # Perform HMC with the best step sizes determined above.
        print "Performing inference."
        start = time.time()
        gym.HMC_random(q0, Nchain=1, Niter=Niter, steps_max=steps_max, steps_min=steps_min, f_lim_default=False, f_lim=f_lim)
        print "Time taken: %.2f\n" % (time.time() - start)

        #---- Plot the results!
        pt_size = 2.
        lw = 2

        N_warmup = int(Niter * 0)
        m = flux2mag(gym.q_chain[0, N_warmup:, 0] / gym.flux_to_count)
        x = gym.q_chain[0, N_warmup:, 1]
        y = gym.q_chain[0, N_warmup:, 2]

        colors_est = ["green", "orange"]
        # Min max
        if counter == 1:
            x_min, x_max = plot_range(x)
            y_min, y_max = plot_range(y)
            m_min, m_max = plot_range(m)

        # Estimators -- mode
        med_m = np.median(m)
        med_x = np.median(x)
        med_y = np.median(y)

        # Estimators -- median
        mean_m = np.mean(m)
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        # Std
        std_m = np.std(m)
        std_x = np.std(x)
        std_y = np.std(y)

        fig, ax_list = plt.subplots(2, 3, figsize = (20, 15))
        plt.suptitle("G: Mean; O: Med; R: Truth; Blue: Min mag.", fontsize=25)

        # xy scatter plot
        ax_list[0, 0].scatter(y, x, s=pt_size, edgecolors="none", c="black")
        ax_list[0, 0].axvline(x=q0[0, 2], c="red", ls="--", lw=lw)
        ax_list[0, 0].axhline(y=q0[0, 1], c="red", ls="--", lw=lw)
        # Estimators
        ax_list[0, 0].axvline(x=med_y, c=colors_est[1], ls="--", lw=lw)
        ax_list[0, 0].axhline(y=med_x, c=colors_est[1], ls="--", lw=lw)
        ax_list[0, 0].axvline(x=mean_y, c=colors_est[0], ls="--", lw=lw)
        ax_list[0, 0].axhline(y=mean_x, c=colors_est[0], ls="--", lw=lw)
        ax_list[0, 0].axis("equal")
        ax_list[0, 0].set_xlabel("y", fontsize=15)
        ax_list[0, 0].set_ylabel("x", fontsize=15)
        # Range
        ax_list[0, 0].set_xlim([y_min, y_max])
        ax_list[0, 0].set_ylim([x_min, x_max])


        # xm xcatter plot
        ax_list[0, 1].scatter(m, x, s=pt_size, edgecolors="none", c="black")
        ax_list[0, 1].axvline(x=mT, c="red", ls="--", lw=lw)
        ax_list[0, 1].axvline(x=mag_lim, c="blue", ls="--", lw=lw)
        ax_list[0, 1].axhline(y=q0[0, 1], c="red", ls="--", lw=lw)
        # Est
        ax_list[0, 1].axvline(x=med_m, c=colors_est[1], ls="--", lw=lw)
        ax_list[0, 1].axhline(y=med_x, c=colors_est[1], ls="--", lw=lw)
        ax_list[0, 1].axvline(x=mean_m, c=colors_est[0], ls="--", lw=lw)
        ax_list[0, 1].axhline(y=mean_x, c=colors_est[0], ls="--", lw=lw)
        ax_list[0, 1].set_xlabel("mag", fontsize=15)
        ax_list[0, 1].set_ylabel("x", fontsize=15)
        ax_list[0, 1].set_xlim([m_min, m_max])        
        ax_list[0, 1].set_ylim([x_min, x_max])                

        # ym scatter plot
        ax_list[1, 2].scatter(m, y, s=pt_size, edgecolors="none", c="black")
        ax_list[1, 2].axvline(x=mT, c="red", ls="--", lw=lw)
        ax_list[1, 2].axvline(x=mag_lim, c="blue", ls="--", lw=lw)
        ax_list[1, 2].axhline(y=q0[0, 2], c="red", ls="--", lw=lw)
        # Est
        ax_list[1, 2].axvline(x=med_m, c=colors_est[1], ls="--", lw=lw)
        ax_list[1, 2].axhline(y=med_y, c=colors_est[1], ls="--", lw=lw)
        ax_list[1, 2].axvline(x=mean_m, c=colors_est[0], ls="--", lw=lw)
        ax_list[1, 2].axhline(y=mean_y, c=colors_est[0], ls="--", lw=lw)
        ax_list[1, 2].set_xlabel("mag", fontsize=15)
        ax_list[1, 2].set_ylabel("y", fontsize=15)
        ax_list[1, 2].set_xlim([m_min, m_max])                
        ax_list[1, 2].set_ylim([y_min, y_max])                        

        # x hist
        ax_list[0, 2].hist(x, bins=50, range=(x_min, x_max), histtype="step", orientation="horizontal", color="black", lw=lw,\
                           label="std: %.3f" % std_x)
        ax_list[0, 2].legend(loc="upper right", fontsize=15)
        ax_list[0, 2].axhline(y=q0[0, 1], c="red", ls="--", lw=lw)
        ax_list[0, 2].set_xlabel("x", fontsize=15)
        # Est
        ax_list[0, 2].axhline(y=mean_x, c=colors_est[0], ls="--", lw=lw)
        ax_list[0, 2].axhline(y=med_x, c=colors_est[1], ls="--", lw=lw)
        ax_list[0, 2].set_ylim([x_min, x_max])  
        ax_list[0, 2].set_xlim([0, hist_height])                        

        # y hist
        ax_list[1, 0].hist(y, bins=50, range=(y_min, y_max), histtype="step", color="black", lw=lw, label="std: %.3f" % std_y)
        ax_list[1, 0].axvline(x=q0[0, 2], c="red", ls="--", lw=lw)
        ax_list[1, 0].legend(loc="upper right", fontsize=15)
        # Est
        ax_list[1, 0].axvline(x=mean_y, c=colors_est[0], ls="--", lw=lw)
        ax_list[1, 0].axvline(x=med_y, c=colors_est[1], ls="--", lw=lw)
        ax_list[1, 0].set_xlabel("y", fontsize=15)
        # Lim
        ax_list[1, 0].set_xlim([y_min, y_max])
        ax_list[1, 0].set_ylim([0, hist_height])                                

        # m hist
        ax_list[1, 1].hist(m, bins=50, range=(m_min, m_max), histtype="step", color="black", lw=lw, label="std: %.3f" % std_m)
        ax_list[1, 1].legend(loc="upper right", fontsize=15)
        ax_list[1, 1].axvline(x=mT, c="red", ls="--", lw=lw)
        ax_list[1, 1].axvline(x=mag_lim, c="blue", ls="--", lw=lw)
        # Est
        ax_list[1, 1].axvline(x=mean_m, c=colors_est[0], ls="--", lw=lw)
        ax_list[1, 1].axvline(x=med_m, c=colors_est[1], ls="--", lw=lw)
        ax_list[1, 1].set_xlabel("mag", fontsize=15)
        ax_list[1, 1].set_xlim([m_min, m_max])                
        ax_list[1, 1].set_ylim([0, hist_height])                                

        fig_str = "./figures/one-star-inference/single/one-star-inference-single-mB%d-mT%.1f-mlim%.1f.png" % (gym.mB, mT, mag_lim)
        plt.savefig(fig_str, dpi=100, bbox_inches="tight")
        plt.close()

        print "\n\n"