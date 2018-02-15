from utils import *

class lightsource_gym(object):
    """
    This class provides facilities for exploring light source inference problem.
    - Storage for real or simulated data. 
    - Perform inference given seed initial values.
    - Make useful diagonostic plots.

    The current version only implements stellar inference though it was designed with an eye towards galaxy
    inference as well.
    """
    def __init__(self):
        """
        The constructor define placeholders for variables and sets default experimental or observational values, which
        can be changed later.
        """

        # Placeholder for data
        self.D = None

        # Placeholder for model
        self.M = None

        # Default experimental set-up
        self.num_rows, self.num_cols, self.flux_to_count, self.PSF_FWHM_pix, \
            self.B_count, self.arcsec_to_pix = self.default_exp_setup()

        #---- Inferential variables
        self.Nchain = None 
        self.Niter = None
        self.thin_rate = None
        self.Nwarmup = None
        self.q_chain = None
        self.p_chain = None
        self.V_chain = None
        self.E_chain = None
        self.dE_chain = None
        self.A_chain = None # Acceptance rate chain
        self.dt = None # Step size to use for each variable

        return

    def gen_mock_data(self, q_true=None, return_data = False):
        """
        Given the properties of mock q_true (Nobjs, 3) with f, x, y for each, 
        generate a mock data image. The following conditions must be met.
        - An object must be within the image.

        Gaussian PSF is assumed with the width specified by PSF_FWHM_pix.
        """
        # Generate an image with background.
        data = np.ones((self.num_rows, self.num_cols), dtype=float) * self.B_count

        # Add one star at a time.
        for i in xrange(q_true.shape[0]):
            f, x, y = q_true[i]
            data += f * gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM = self.PSF_FWHM_pix)

        # Poission realization D of the underlying truth D0
        data = poisson_realization(data)

        if return_data:
            return data
        else:
            self.D = data


    def dVdq_single(self, q_single, model_data, f_only=True):
        """
        Return the gradient of a single object based on model data where an object with f, x, y is to be added. 
        If f_only is True, then return only f gradient. If False, return xy gradients.
        """
        f, x, y = q_single
        Lambda = model_data + f * gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)

        # Variable to be recycled
        rho = (self.D/Lambda)-1.# (D_lm/Lambda_lm - 1)

        # Compute f, x, y gradient for each object
        PSF = gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)

        if f_only:
            grad_f = -np.sum(rho * PSF) # flux grad       
            return grad_f
        else:
            lv = np.arange(0, self.num_rows)
            mv = np.arange(0, self.num_cols)
            mv, lv = np.meshgrid(lv, mv)
            var = (self.PSF_FWHM_pix/2.354)**2 
            grad_x = -np.sum(rho * (lv - x + 0.5) * PSF) * f / var
            grad_y = -np.sum(rho * (mv - y + 0.5) * PSF) * f / var
            return grad_x, grad_y

    def E_single(self, q_single, p_single, model_data):
        """
        Energy of a single particle given q_singe = [f, x, y] and its corresponding momenta.
        """
        f0, x0, y0 = q_single
        Lambda = model_data + f0 * gauss_PSF(self.num_rows, self.num_cols, x0, y0, FWHM=self.PSF_FWHM_pix)
        return -np.sum(self.D * np.log(Lambda) - Lambda) + np.dot(p_single, p_single) / 2.
    


    def HMC_find_best_dt(self, q_model_0, steps_min=10, steps_max=50, Niter_per_trial = 10, Ntrial = 10, \
        dt_f_coeff=1e-1, dt_xy_coeff=1., default=False):
        """
        Find the optimal dt through the following heuristic method.

        Set the initial step size as follows. For each particle,
        - dt_xy: dt_xy_coeff/f
        - dt_f: dt_f_coeff * f
    
        For each parameter, fixing every other parameter, do the following search
            Coarse finding:
            Perform Niter_per_trial HMC iterations. Compute the acceptance rate.
            Until acceptance rate becomes 1, either increase or decrease the step size by a factor of 2. 

            Fine finding (NOT IMPLEMENTED):
            Once the acceptance has become equal to 1, increase the step size by a factor 2.
            Compute the acceptance rate again, and if it falls below 1, then the next step size to try is
            the mid-point of the current and the previous step size. Continue with the "bi-section search".
            This process terminates when Ntrial number has been exhausted.

        Note that only one particle is being "perturbed" at a time. Also, position step sizes dt_xy are tuned together.

        If default False, then use the intial step size.
        """
        #--- Number of objects
        self.Nobjs = q_model_0.shape[0]
        self.d = self.Nobjs * 3

        #---- Placeholder for step size
        self.dt = np.zeros(self.d)

        #---- If default parameters asked for.
        if default:
            for i in xrange(self.Nobjs):
                #---- Initialize step sizes
                f0, x0, y0 = q_model_0[i]
                dt_xy = dt_xy_coeff / f0
                dt_f = dt_f_coeff * f0
                self.dt[3*i:3*i+3] = np.array([dt_f, dt_xy, dt_xy])
            return 

        #---- Find the optimal step size for each parameter.
        for l in xrange(self.Nobjs):
            #---- Initialize step sizes
            f0, x0, y0 = q_model_0[l]
            dt_xy = dt_xy_coeff / f0
            dt_f = dt_f_coeff * f0

            #---- Set up grad functions for the current object's parameters
            model_data = self.gen_mock_data(q_model_0, return_data=True) # Based on the initial seed.       
            model_data -= f0 * gauss_PSF(self.num_rows, self.num_cols, x0, y0, FWHM=self.PSF_FWHM_pix) # Subtract the object of interest.

            for k in range(2): # k = 0 for flux and 1 for xy.
                if k == 0: # Flux step size adjustment
                    dt = np.array([dt_f, 0, 0])
                else:
                    dt = np.array([0, dt_xy, dt_xy])            

                #---- Coarse finding
                A_rate = 0
                while np.abs(A_rate - 1) > 0.05: # While the acceptance is not equal to one, keep adjusting the step sizes
                    # To be used as a counter until division at the end
                    A_rate = 0 

                    # Set the initial values.
                    q_tmp = np.array([f0, x0, y0])

                    #---- Looping over iterations
                    for i in xrange(1, Niter_per_trial+1, 1):
                        q_initial = q_tmp # Save the initial just in case.

                        #---- Initial
                        # Resample moementum
                        p_tmp = np.random.randn(3)
                        if k == 0:
                            p_tmp[1:] = 0
                        else:
                            p_tmp[0] = 0

                        # Compute initial
                        E_initial = self.E_single(q_tmp, p_tmp, model_data)

                        #---- Looping over a random number of steps
                        steps_sample = np.random.randint(low=steps_min, high=steps_max, size=1)[0]
                        p_half = p_tmp - dt * self.dVdq_single(q_tmp, model_data) / 2. # First half step                        
                        # Leap frogging
                        for _ in xrange(steps_sample):
                            q_tmp = q_tmp + dt * p_half 
                            p_half = p_half - dt * self.dVdq_single(q_tmp, model_data)
                        p_tmp = p_half + dt * self.dVdq_single(q_tmp, model_data) / 2. # Account for the overshoot in the final run.

                        # Compute final energy and save.
                        E_final = self.E_single(q_tmp, p_tmp, model_data)
                            
                        # With correct probability, accept or reject the last proposal.
                        dE = E_final - E_initial
                        E_previous = E_initial # Save the energy so that the energy differential can be computed during the next run.
                        lnu = np.log(np.random.random(1))        
                        if (dE < 0) or (lnu < -dE): # If accepted. 
                            A_rate +=1 
                        else: # Otherwise, proposal rejected.
                            q_tmp = q_initial

                    A_rate /= float(Niter_per_trial)

                    if np.abs(A_rate-1) > 0.05:
                        dt /= 2.
                    else:
                        dt *= 2.

                if k == 0:
                    dt_f = dt[0]
                else:
                    dt_xy = dx[1]
                
            self.dt[3*i:3*i+3] = np.array([dx_f, dx_xy, dt_xy])



    def HMC_random(self, q_model_0, Nchain, Niter, thin_rate=0, Nwarmup=0, steps_min=10, steps_max = 50):
        """
        Perform Bayesian inference with HMC given an initial model q_model_0 (Nobjs, 3). 
        No change in dimension is implemented. 

        Random trajectory length is used with steps ~ [steps_min, steps_max]
        dt_xy_coeff and dt_f_coeff are parameters.
        """
        #---- Set inference variables as provided by the user. 
        assert Nchain == 1 # Currently we do not support any other.
        self.Nchain = Nchain
        self.Niter = Niter
        self.thin_rate = thin_rate
        self.Nwarmup = Nwarmup

        #---- Number of objects should have been already determined via optimal step search
        assert self.d is  not None
        
        #---- Reshape the initial point        
        q_model_0 =  q_model_0.reshape((self.d,))# Flatten 

        #---- Allocate storage for variables being inferred.
        # The zeroth slot is reserved for the initial. The first iteration takes 0 --> 1.
        self.q_chain = np.zeros((self.Nchain, self.Niter+1, self.d))
        # self.p_chain = np.zeros((self.Nchain, self.Niter+1, self.d))
        # self.V_chain = np.zeros((self.Nchain, self.Niter+1, 1))
        self.E_chain = np.zeros((self.Nchain, self.Niter+1, 1))
        self.dE_chain = np.zeros((self.Nchain, self.Niter+1, 1))
        self.A_chain = np.zeros((self.Nchain, self.Niter, 1)) # Acceptance rate chain. There are only Niter transitions.
        # Samples are taken from [1, Niter + 1]

        #---- Looping over chains
        for m in xrange(self.Nchain):
            # Set the initial values.
            q_initial = q_model_0
            p_initial = self.p_sample()            
            self.q_chain[m, 0, :] = q_model_0
            # self.p_chain[m, 0, :] = self.p_sample()[0]
            # self.V_chain[m, 0, 0] = self.V(self.q_chain[m, 0, :])
            self.E_chain[m, 0, 0] = self.E(q_initial, p_initial)
            self.dE_chain[m, 0, 0] = 0 # Arbitrarily set to zero.
            E_previous = self.E_chain[m, 0, 0]
            q_tmp = q_initial
            #---- Looping over iterations
            for i in xrange(1, self.Niter+1, 1):
                #---- Initial
                q_initial = q_tmp

                # Resample moementum
                p_tmp = self.p_sample()

                # Compute E and dE and save
                E_initial = self.E(q_tmp, p_tmp)
                self.E_chain[m, i, 0] = E_initial
                self.dE_chain[m, i, 0] = E_initial - E_previous                    

                #---- Looping over a random number of steps
                steps_sample = np.random.randint(low=steps_min, high=steps_max, size=1)[0]
                p_half = p_tmp - self.dt * self.dVdq(q_tmp) # First half step                                        
                for _ in xrange(steps_sample):
                    q_tmp = q_tmp + self.dt * p_half 
                    p_half = p_half - self.dt * self.dVdq(q_tmp)
                p_tmp = p_half + self.dt * self.dVdq(q_tmp) / 2. # Account for the overshoot in the final run.


                # Compute final energy and save.
                E_final = self.E(q_tmp, p_tmp)
                    
                # With correct probability, accept or reject the last proposal.
                dE = E_final - E_initial
                E_previous = E_initial # Save the energy so that the energy differential can be computed during the next run.
                lnu = np.log(np.random.random(1))        
                if (dE < 0) or (lnu < -dE): # If accepted.
                    self.A_chain[m, i-1, 0] = 1
                    self.q_chain[m, i, :] = q_tmp # save the new point
                else: # Otherwise, proposal rejected.
                    self.q_chain[m, i, :] = q_initial # save the old point
                    q_tmp = q_initial

            print "Chain %d Acceptance rate: %.2f%%" % (m, np.sum(self.A_chain[m, :] * 100)/float(self.Niter))

        return 


    def dVdq(self, objs_flat):
        """
        Gradient of Poisson pontential above.    
        """
        # Place holder for the gradient.
        grad = np.zeros(objs_flat.size)

        # Compute the model.
        Lambda = np.ones_like(self.D) * self.B_count # Model set to background
        for i in range(self.Nobjs): # Add every object.
            f, x, y = objs_flat[3*i:3*i+3]
            Lambda += f * gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)

        # Variable to be recycled
        rho = (self.D/Lambda)-1.# (D_lm/Lambda_lm - 1)
        # Compute f, x, y gradient for each object
        lv = np.arange(0, self.num_rows)
        mv = np.arange(0, self.num_cols)
        mv, lv = np.meshgrid(lv, mv)
        var = (self.PSF_FWHM_pix/2.354)**2 
        for i in range(self.Nobjs):
            f, x, y = objs_flat[3*i:3*i+3]
            PSF = gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)
            grad[3*i] = -np.sum(rho * PSF) # flux grad
            grad[3*i+1] = -np.sum(rho * (lv - x + 0.5) * PSF) * f / var
            grad[3*i+2] = -np.sum(rho * (mv - y + 0.5) * PSF) * f / var
        return grad
    

    def V(self, objs_flat):
        """
        Negative Poisson log-likelihood given data and model.

        The model is specified by the list of objs, which are provided
        as a flattened list [Nobjs x 3](e.g., [f1, x1, y1, f2, x2, y2, ...])

        Assume a fixed background.
        """
        Lambda = np.ones_like(self.D) * self.B_count # Model set to background
        for i in range(self.Nobjs): # Add every object.
            f, x, y = objs_flat[3*i:3*i+3]
            Lambda += f * gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)
        return -np.sum(self.D * np.log(Lambda) - Lambda)

    def K(self, p):
        """
        The user supplies potential energy and its gradient.
        User also sets the form of kinetic distribution.
        Kinetic energy -ln P(p)
        """
        # return np.dot(p, np.dot(self.inv_cov_p, p)) / 2. 
        return np.dot(p, p) / 2. # We assume identity covariance.

    def E(self, q, p):
        """
        Kinetic plus potential energy    
        """
        return self.V(q) + self.K(p)

    def p_sample(self):
        """
        Return a random sample from a unit multi-variate normal of dimension D
        """
        return np.random.randn(self.d) # np.random.multivariate_normal(np.zeros(self.d), self.cov_p, size=1)
        
    def leap_frog(self, p_old, q_old, dt):
        """
        dVdq is the gradient of poential function supplied by the user.
        """
        p_half = p_old - dt * self.dVdq(q_old) / 2.
        q_new = q_old + dt * p_half 
        p_new = p_half - dt * self.dVdq(q_new)/ 2. 

        return p_new, q_new

    def display_data(self, figsize=(5, 5), vmax_percentile = 100):
        """
        A light display of the mock data.
        """
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(self.D, interpolation="none", vmin=np.min(self.D), vmax=np.percentile(self.D, vmax_percentile), cmap="gray")
        plt.show()
        plt.close()

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

        # Size of the image
        num_rows = num_cols = 48 # Pixel index goes from 0 to num_rows-1

        return num_rows, num_cols, flux_to_count, PSF_FWHM_pix, B_count, arcsec_to_pix


    


# class sampler(object):
#     """
#     Parent class to MH_sampler and HMC_sampler. Contains common functions such as the plot function.
#     """
    
#     def __init__(self, D, target_lnL, Nchain=2, Niter=1000, thin_rate=1, warm_up_num=0):
#         """
#         Args:
#         - D: Number of paramters to be inferred on.
#         - target_lnL: Function that takes in a D-dimensional input q and outputs the log-like of the target distribution.
#         - Nchain: Number of chains.
#         - Niter: Number of iterations.
#         - Thin rate
#         - warm_up_num: Sample index at which to start computing statistics. 
        
#         None of these variable can be changed. In other words, if you want to change the variables, 
#         then one must create a new sampler object as only the appropriate number of samples are retained.        
#         """
    
#         self.D = D 
#         self.target_lnL = target_lnL        
#         self.Nchain = Nchain
#         self.Niter = Niter
#         self.thin_rate = thin_rate
#         self.warm_up_num = warm_up_num
                    
#         # Allocate memory for samples
#         self.L_chain = 1+ ((self.Niter-self.warm_up_num)//self.thin_rate) # Length of individual chain
#         # Simple example: Niter = 3, N_warm_up = 0 --> N_total = 4.
#         self.q_chain = np.zeros((self.Nchain, self.L_chain, self.D), dtype=np.float) # Samples
#         self.lnL_chain = np.zeros((self.Nchain, self.L_chain, 1)) # loglikelihood
        
#         # Stats 
#         self.R_q = None # R statistics for each parameter
#         self.R_lnL = None # R stats for the loglikelihood.
#         self.n_eff_q = None # Effective number of samples
#         self.accept_R_warm_up = None # Acceptance rate during warm up
#         self.accept_R = None # Acceptance rate after warm up

#         # Time for measuring total time taken for inference
#         self.dt_total = 0 # Only for chain inference.

#         # Total number of computations. All of the following computations has a unit cost.
#         # None other computations figure in this accounting.
#         # - Gradient computation per variable: 1.
#         # - Likelihood evaluation: 1
#         self.N_total_steps = 0

        
#     def compute_convergence_stats(self):
#         """
#         Compute stats on the chain and return.
        
#         Remember, the chain has already been warmed up and thinned.
#         """

#         # Note that we should not include the first point.
#         self.R_q, self.n_eff_q = convergence_stats(self.q_chain[:, 1:, :], warm_up_num=0, thin_rate=1)
#         # self.R_lnL, _ = convergence_stats(self.lnL_chain, warm_up_num=0, thin_rate=1)

#         return
    
    
#     def plot_samples(self, title_prefix, show=False, savefig=False, xmax = None, dx = None, plot_normal=True, plot_cov=True, q0=None, cov0=None,\
#         var_idx1=0, var_idx2=1):
#         """
#         Plot the samples after warm up and thinning.
#         Args:
#         - show: If False, don't show the plot, whic wouldn't make sense unless savefig is True.
#         - xmax: Unless specified by the user, the plot limit is set automatically,
#          by first computing inner 95% tile and expanding it by 200 percent keeping the center the same. Assumes zero centering.
#         - dx: Unless specified by the user, Bin size is set as 1% of the 95% tile range. This means with 1000 samples there should be on average
#         10 samples in each bin.
#         - savefig: Saves figure with the name fname. 
#         - title_prefix: Title prefix must be provided by the user
#         - plot_normal: If true, the user specified normal marginal for q1 and q2 are plotted with proper normalizatin.
#         - plot_cov: If true, the the user provided normal models are plotted with proper normalizatin.
#         - q0, cov0: Are parameters of the normal models to be overlayed.
#         """
        
#         #---- Extract samples from all chains
#         q_chain_tmp_1 = self.q_chain[:, :, var_idx1].flatten()
#         q_chain_tmp_2 = self.q_chain[:, :, var_idx2].flatten()
#         E_chain_tmp = self.E_chain[:, 1:, :].flatten() # Only HMC samplers, so these are always defined.
#         E_chain_tmp -= np.mean(E_chain_tmp)# Center the energies
#         dE_chain_tmp = self.dE_chain[:, 1: , :].flatten()
        
#         #---- Setting the boundary and binwidth
#         # Boundary
#         if xmax is None: # If the user didn't specify the range
#             # q1 range
#             # Compute 95 percent tile
#             q1_max = np.percentile(q_chain_tmp_1, 97.5)
#             q1_min = np.percentile(q_chain_tmp_1, 2.5)
#             q1_range = q1_max - q1_min
#             q1_center = (q1_max + q1_min)/2.
#             # Adjust the range
#             q1_range *= 2.5
#             q1_max = q1_center + (q1_range/2.)
#             q1_min = q1_center - (q1_range/2.)

#             # q2 range
#             # Compute 95 percent tile
#             q2_max = np.percentile(q_chain_tmp_2, 97.5)
#             q2_min = np.percentile(q_chain_tmp_2, 2.5)
#             q2_range = q2_max - q2_min
#             q2_center = (q2_max + q2_min)/2.
#             # Adjust the range
#             q2_range *= 2.5
#             q2_max = q2_center + (q2_range/2.)
#             q2_min = q2_center - (q2_range/2.)  
#         else:
#             xmin = -xmax
#             q1_min, q1_max = xmin, xmax
#             q2_min, q2_max = xmin, xmax
#             q1_range = q2_range = q1_max - q1_min            

#         # Bin width
#         if dx is None:
#             dq1 = q1_range/100.
#             dq2 = q2_range/100.
#         else:
#             dq1 = dq2 = dx 

#         print q1_min, q1_max 
#         print q2_min, q2_max
        
#         #---- Plot normal models, properly normalized.
#         if plot_normal:
#             assert (q0 is not None) and (cov0 is not None)
#             assert q0.size == self.D
#             xgrid1 = np.arange(q1_min, q1_max, dq1/10.)
#             q1_marginal = multivariate_normal.pdf(xgrid1, mean=q0[0], cov=cov0[0, 0]) * self.L_chain * dq1 * self.Nchain
#             xgrid2 = np.arange(q2_min, q2_max, dq2/10.)            
#             q2_marginal = multivariate_normal.pdf(xgrid2, mean=q0[1], cov=cov0[1, 1]) * self.L_chain * dq2 * self.Nchain

#         #---- Start of the figure generation ----#
#         plt.close() # Clear any open panels.
#         fig, ax_list = plt.subplots(3, 3, figsize = (20, 20))

#         # Font sizes
#         ft_size = 25 # axes labels, 
#         ft_size2 = 20 # Legend
#         ft_size_title = 30

#         #---- Scatter plot
#         ax_list[0, 0].scatter(q_chain_tmp_1, q_chain_tmp_2, s=2, c="black")
#         if plot_cov:
#             plot_cov_ellipse(ax_list[0, 0], [q0], [cov0], 0, 1, MoG_color="Blue", lw=2)
#         ax_list[0, 0].set_xlabel("q1", fontsize=ft_size)
#         ax_list[0, 0].set_ylabel("q2", fontsize=ft_size)
#         # ax_list[0, 0].axis("equal")
#         ax_list[0, 0].set_xlim([q1_min, q1_max])
#         ax_list[0, 0].set_ylim([q2_min, q2_max])
        
#         #---- q2 histogram
#         ax_list[0, 1].hist(q_chain_tmp_2, bins=np.arange(q2_min, q2_max, dq2), histtype="step", \
#             color="black", orientation="horizontal", lw=2, label=(r"R = %.3f" % self.R_q[var_idx2]))
#         if plot_normal:
#             assert (q0 is not None) and (cov0 is not None)
#             ax_list[0, 1].plot(q2_marginal, xgrid2, c="green", lw=3)
#         ax_list[0, 1].set_ylim([q2_min, q2_max])
#         ax_list[0, 1].set_ylabel("q2", fontsize=ft_size)
#         ax_list[0, 1].legend(loc="upper right", fontsize=ft_size2)
        
#         #---- q1 histogram
#         ax_list[1, 0].hist(q_chain_tmp_1, bins=np.arange(q1_min, q1_max, dq1), histtype="step", \
#             color="black", lw=2, label=(r"R = %.3f" % self.R_q[var_idx1]))
#         if plot_normal:        
#             ax_list[1, 0].plot(xgrid1, q1_marginal, c="green", lw=3)
#         ax_list[1, 0].set_xlim([q1_min, q1_max])
#         ax_list[1, 0].set_xlabel("q1", fontsize=ft_size)
#         ax_list[1, 0].legend(loc="upper right", fontsize=ft_size2)

#         #---- E and dE histograms
#         # Compute the proper range
#         E_min = np.percentile(E_chain_tmp, 2.5)
#         E_max = np.percentile(E_chain_tmp, 97.5)
#         E_range = (E_max - E_min) * 2.5
#         E_center = (E_min+E_max)/2.
#         E_min = E_center - (E_range/2.)
#         E_max = E_center + (E_range/2.)
#         bin_E = E_range/100.
#         Egrid = np.arange(E_min, E_max, bin_E)
#         ax_list[0, 2].hist(E_chain_tmp, bins=Egrid, histtype="step", color="black", label="E", lw=2)
#         ax_list[0, 2].hist(dE_chain_tmp, bins=Egrid, histtype="step", color="red", label="dE", lw=2)        
#         ax_list[0, 2].set_xlim([E_min, E_max])
#         ax_list[0, 2].set_xlabel("Energy", fontsize=ft_size)
#         ax_list[0, 2].legend(loc="upper right", fontsize=ft_size2)

#         #---- Rhat distribution
#         # Compute the proper range
#         R_min = np.percentile(self.R_q, 2.5)
#         R_max = np.percentile(self.R_q, 97.5)
#         R_range = (R_max - R_min) * 2.5
#         R_center = (R_min+R_max)/2.
#         R_min = R_center - (R_range/2.)
#         R_max = R_center + (R_range/2.)
#         bin_R = R_range/50.
#         Rgrid = np.arange(R_min, R_max, bin_R)
#         ax_list[1, 2].hist(self.R_q, bins=Rgrid, histtype="step", color="black", lw=2, \
#             label = ("R med/std: %.3f/ %.3f" % (np.median(self.R_q), np.std(self.R_q))))
#         ax_list[1, 2].set_xlim([R_min, R_max])
#         ax_list[1, 2].set_xlabel("Rhat", fontsize=ft_size)
#         ax_list[1, 2].legend(loc="upper right", fontsize=ft_size2)           

#         # #---- Inferred standard deviations
#         # # Extracting true diagonal covariances
#         # cov0_diag = []
#         # cov_diag = [] # Inferred covariances
#         # for i in range(self.D):
#         #     cov0_diag.append(cov0[i, i])
#         #     cov_diag.append(np.std(self.q_chain[:, 1:, i])**2)
#         # # Converting
#         # cov0_diag = np.asarray(cov0_diag)
#         # cov_diag = np.asarray(cov_diag)        

#         # # Setting x-ranges for both plots below
#         # xmax = np.max(cov0_diag) * 1.1
#         # xmin = np.min(cov0_diag) * 0.9

#         # # Plotting true vs. inferred
#         # ymin = 0.5 * np.min(cov_diag)
#         # ymax = 1.5 * np.max(cov_diag)                
#         # ax_list[2, 1].scatter(cov0_diag, cov_diag, s=50, c="black", edgecolor="none")
#         # ax_list[2, 1].plot([xmin, xmax], [xmin, xmax], c="black", lw=2, ls="--")
#         # ax_list[2, 1].set_xlim([xmin, xmax])
#         # ax_list[2, 1].set_ylim([ymin, ymax])
#         # ax_list[2, 1].set_xlabel("True cov", fontsize=ft_size)
#         # ax_list[2, 1].set_ylabel("Estimated cov", fontsize=ft_size)

#         # # Plotting the ratio
#         # cov_ratio = cov_diag/cov0_diag
#         # ymin = 0.5 * np.min(cov_ratio)
#         # ymax = 1.5 * np.max(cov_ratio)        
#         # ax_list[2, 2].scatter(cov0_diag, cov_ratio, s=50, c="black", edgecolor="none")
#         # ax_list[2, 2].axhline(y=1, lw=2, c="black", ls="--")
#         # ax_list[2, 2].set_xlim([xmin, xmax])
#         # ax_list[2, 2].set_ylim([ymin, ymax])
#         # ax_list[2, 2].set_xlabel("True cov", fontsize=ft_size)
#         # ax_list[2, 2].set_ylabel("Ratio cov", fontsize=ft_size)

#         # #---- Inferred means
#         # q_mean = []
#         # for i in range(self.D):
#         #     q_mean.append(np.mean(self.q_chain[:, 1:, i]))
#         # q_mean = np.asarray(q_mean)

#         # # Calculate the bias
#         # bias = q_mean-q0

#         # # Setting x-ranges for both plots below
#         # xmax = np.max(cov0_diag) * 1.1
#         # xmin = np.min(cov0_diag) * 0.9

#         # # Plotting histogram of bias
#         # ymax = np.max(bias)
#         # ymin = np.min(bias)
#         # y_range = ymax-ymin
#         # y_range *= 2.5
#         # y_center = (ymax+ymin)/2.
#         # ymax = y_center + (y_range/2.)
#         # ymin = y_center - (y_range/2.)       
#         # ax_list[2, 0].scatter(cov0_diag, bias, s=50, c="black", edgecolor="none")
#         # ax_list[2, 0].axhline(y=0, c="black", ls="--", lw=2)
#         # ax_list[2, 0].set_xlim([xmin, xmax])
#         # ax_list[2, 0].set_ylim([ymin, ymax])
#         # ax_list[2, 0].set_xlabel("True cov", fontsize=ft_size)
#         # ax_list[2, 0].set_ylabel("bias(mean)", fontsize=ft_size)             

#         #----- Stats box
#         ax_list[1, 1].scatter([0.0, 1.], [0.0, 1.], c="none")
#         if self.warm_up_num > 0:
#             ax_list[1, 1].text(0.1, 0.8, "RA before warm-up: %.3f" % (self.accept_R_warm_up), fontsize=ft_size2)
#         ax_list[1, 1].text(0.1, 0.7, "RA after warm-up: %.3f" % (self.accept_R), fontsize=ft_size2)
#         ax_list[1, 1].text(0.1, 0.6, "Total time: %.1f s" % self.dt_total, fontsize=ft_size2)
#         ax_list[1, 1].text(0.1, 0.5, "Total steps: %.2E" % self.N_total_steps, fontsize=ft_size2)        
#         ax_list[1, 1].text(0.1, 0.4, "Ntot/eff med: %.1E/%.1E" % (self.L_chain * self.Nchain, np.median(self.n_eff_q)), fontsize=ft_size2)                
#         ax_list[1, 1].text(0.1, 0.3, "#steps/ES med: %.2E" % (self.N_total_steps/np.median(self.n_eff_q)), fontsize=ft_size2)                
#         ax_list[1, 1].text(0.1, 0.2, "#steps/ES best: %.2E" % (self.N_total_steps/np.max(self.n_eff_q)), fontsize=ft_size2)                
#         ax_list[1, 1].text(0.1, 0.1, "#steps/ES worst: %.2E" % (self.N_total_steps/np.min(self.n_eff_q)), fontsize=ft_size2)                
#         ax_list[1, 1].set_xlim([0, 1])
#         ax_list[1, 1].set_ylim([0, 1])

#         plt.suptitle("D/Nchain/Niter/Warm-up/Thin = %d\%d\%d\%d\%d" % (self.D, self.Nchain, self.Niter, self.warm_up_num, self.thin_rate), fontsize=ft_size_title)
#         if savefig:
#             fname = title_prefix+"-samples-D%d-Nchain%d-Niter%d-Warm%d-Thin%d.png" % (self.D, self.Nchain, self.Niter, self.warm_up_num, self.thin_rate)
#             plt.savefig(fname, dpi=400, bbox_inches = "tight")
#         if show:
#             plt.show()
#         plt.close()

        



# class HMC_sampler(sampler):
#     """
#     HMC sampler for a general distribution.
    
#     The main user functions are the constructor and the gen_sample.
#     """
    
#     def __init__(self, D, V, dVdq, Nchain=2, Niter=1000, thin_rate=1, warm_up_num=0, \
#                  cov_p=None, sampler_type="Fixed", L=None, global_dt = True, dt=None, \
#                  L_low=None, L_high=None, log2L=None, d_max=10):
#         """
#         Args: See Sampler class constructor for other variables.
#         - D: Dimensin of inference.
#         - V: The potential function which is the negative lnL.
#         - dVdq: The gradient of the potential function to be supplied by the user. Currently, numerical gradient is not supported.
#         - cov_p: Covariance matrix of the momentum distribution assuming Gaussian momentum distribution.
#         - global_dt: If True, then a single uniform time step is used for all variables. Otherwise, one dt for each variable is used.
#         - dt: Time step(s). An integer or numpy array of dimension D.
#         - L: Number of steps to be taken for each sample if sampler type is "Fixed". 
#         - L_low, L_high: If "Random" sampler is chosen, then vary the trajectory length as a random sample from [L_low, L_high]. 
#         - log2L: log base 2 of trajectory length for the static scheme        
#         - sampler_type = "Fixed", "Random", "Static", or "NUTS"; 
#         - d_max: Maximum number of doubling allowed by the user.

#         Note: "Fixed" and "Static" is no longer supported.
#         """
#         # parent constructor
#         sampler.__init__(self, D=D, target_lnL=None, Nchain=Nchain, Niter=Niter, thin_rate=thin_rate, warm_up_num=warm_up_num)
        
#         # Save the potential and its gradient functions
#         self.V = V
#         self.dVdq = dVdq
        
#         # Which sampler to use?
#         assert (sampler_type=="Fixed") or (sampler_type=="Random") or (sampler_type=="NUTS") or (sampler_type=="Static")
#         assert (dt is not None)
#         self.dt = dt
#         self.global_dt = global_dt 
#         self.sampler_type = sampler_type
#         if self.sampler_type == "Fixed":
#             assert (L is not None)
#             self.L = L
#         elif self.sampler_type == "Random":
#             assert (L_low is not None) and (L_high is not None)
#             self.L_low = L_low
#             self.L_high = L_high
#         elif self.sampler_type == "Static":
#             assert (log2L is not None)
#             self.log2L =log2L
#         elif self.sampler_type == "NUTS":
#             assert d_max is not None
#             self.d_max = d_max


#         # Momentum covariance matrix and its inverse
#         if cov_p is None: 
#             self.cov_p = np.diag(np.ones(self.D))
#         else:
#             self.cov_p = cov_p                    
#         self.inv_cov_p = np.linalg.inv(self.cov_p)             
        
#         # Save marginal energy and energy difference.
#         self.E_chain = np.zeros((self.Nchain, self.L_chain, 1), dtype=np.float)
#         self.dE_chain = np.zeros((self.Nchain, self.L_chain, 1), dtype=np.float)
        
        
#     def gen_sample(self, q_start, N_save_chain0=0, verbose=True):
#         """
#         Save Nchain of (Niter-warm_up_num+1)//thin_rate samples.
#         Also, record acceptance/rejection rate before and after warm-up.
        
#         Appropriate samplers "Random" and "NUTS". Others not supported.
        
#         Args:
#         - q_start: The function takes in the starting point as an input.
#         This requirement gives the user a fine degree of choice on how to
#         choose the starting point. Dimensions (self.Nchain, D)
#         - N_save_chain0: (# of samples - 1) to save from chain0 from the beginning in order to produce a video.
#         - verbose: If true, then print how long each chain takes.
#         """
        
#         if (self.sampler_type == "Random"):
#             self.gen_sample_random(q_start, N_save_chain0, verbose)
#         elif (self.sampler_type == "NUTS"):
#             self.gen_sample_NUTS(q_start, N_save_chain0, verbose)
            
#         return



#     def gen_sample_random(self, q_start, N_save_chain0, verbose):
#         """
#         Random trajectory length sampler.

#         Same arguments as gen_sample.
#         """
    
#         #---- Param checking/variable construction before run
#         # Check if the correct number of starting points have been provided by the user.
#         assert q_start.shape[0] == self.Nchain
#         if (N_save_chain0 > 0):
#             save_chain = True
#             self.decision_chain = np.zeros((N_save_chain0+1, 1), dtype=np.int)
#             self.phi_q = [] # List since the exact dimension is not known before the run.
#         else:
#             save_chain = False
            
#         # Variables for computing acceptance rate
#         accept_counter_warm_up = 0
#         accept_counter = 0
        
#         #---- Executing HMC ---- #
#         # Report time for computing each chain.
#         for m in xrange(self.Nchain): # For each chain            
#             # ---- Initializing the chain
#             # Take the initial value: We treat the first point to be accepted without movement.
#             self.q_chain[m, 0, :] = q_start[m]
#             q_tmp = q_start[m]
#             p_tmp = self.p_sample()[0] # Sample momentun
#             E_initial = self.E(q_tmp, p_tmp)
#             self.N_total_steps += 1 # Energy calculation has likelihood evaluation.
#             self.E_chain[m, 0, 0] = E_initial
#             self.dE_chain[m, 0, 0] = 0 # There is no previous momentum so this is zero.
#             E_previous = E_initial # Initial energy            

#             #---- Start measuring time
#             if verbose:
#                 print "Running chain %d" % m                
#                 start = time.time()                   

#             #---- Execution starts here
#             for i in xrange(1, self.Niter+1): # According to the convention we are using.
#                 # Initial position/momenutm
#                 q_initial = q_tmp # Saving the initial point
#                 p_tmp = self.p_sample()[0] # Sample momentun

#                 # Compute initial energy and save
#                 E_initial = self.E(q_tmp, p_tmp)
#                 self.N_total_steps += 1
#                 if i >= self.warm_up_num: # Save the right cadence of samples.
#                     self.E_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = E_initial
#                     self.dE_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = E_initial - E_previous                    
                    
#                 # Draw the length of the trajectory
#                 L_random = np.random.randint(low=self.L_low, high=self.L_high, size=1)[0]
#                 if save_chain and (m==0) and (i<(N_save_chain0+1)):
#                     # Construct an array of length L_random+1 and save the initial point at 0.
#                     phi_q_tmp = np.zeros((L_random+1, self.D))
#                     phi_q_tmp[0, :] = q_tmp

#                 # Take leap frog steps
#                 for l in xrange(1, L_random+1):
#                     p_tmp, q_tmp = self.leap_frog(p_tmp, q_tmp)
#                     self.N_total_steps += L_random * self.D                    
#                     if save_chain and (m==0) and (i<(N_save_chain0+1)):
#                         phi_q_tmp[l, :] = q_tmp

#                 # Compute final energy and save.
#                 E_final = self.E(q_tmp, p_tmp)
#                 self.N_total_steps += 1                               
                    
#                 # With correct probability, accept or reject the last proposal.
#                 dE = E_final - E_initial
#                 E_previous = E_initial # Save the energy so that the energy differential can be computed during the next run.
#                 lnu = np.log(np.random.random(1))        
#                 if (dE < 0) or (lnu < -dE): # If accepted.
#                     if save_chain and (m==0) and (i<(N_save_chain0+1)):
#                         self.decision_chain[i-1, 0] = 1
#                     if i >= self.warm_up_num: # Save the right cadence of samples.
#                         self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_tmp # save the new point
#                         accept_counter +=1                            
#                     else:
#                         accept_counter_warm_up += 1                        
#                 else: # Otherwise, proposal rejected.
#                     self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_initial # save the old point
#                     q_tmp = q_initial

#                 if save_chain and (m==0) and (i<(N_save_chain0+1)):                    
#                     self.phi_q.append(phi_q_tmp)
            
#             #---- Finish measuring time
#             if verbose:
#                 dt = time.time() - start
#                 self.dt_total += dt
#                 print "Time taken: %.2f\n" % dt 

#         print "Compute acceptance rate"
#         if self.warm_up_num > 0:
#             self.accept_R_warm_up = accept_counter_warm_up / float(self.Nchain * self.warm_up_num)
#             print "During warm up: %.3f" % self.accept_R_warm_up            
#         self.accept_R = accept_counter / float(self.Nchain * (self.Niter - self.warm_up_num + 1))
#         print "After warm up: %.3f" % self.accept_R
#         print "Completed."            

#         return 
  



#     def K(self, p):
#         """
#         The user supplies potential energy and its gradient.
#         User also sets the form of kinetic distribution.
#         Kinetic energy -ln P(p)
#         """
#         return np.dot(p, np.dot(self.inv_cov_p, p)) / 2.

#     def E(self, q, p):
#         """
#         Kinetic plus potential energy    
#         """
#         return self.V(q) + self.K(p)

#     def p_sample(self):
#         """
#         Return a random sample from a unit multi-variate normal of dimension D
#         """
#         return np.random.multivariate_normal(np.zeros(self.D), self.cov_p, size=1)
        
#     def leap_frog(self, p_old, q_old):
#         """
#         dVdq is the gradient of poential function supplied by the user.
#         """
#         p_half = p_old - self.dt * np.dot(self.inv_cov_p,  self.dVdq(q_old)) / 2.
#         q_new = q_old + self.dt * p_half 
#         p_new = p_half - self.dt * np.dot(self.inv_cov_p,  self.dVdq(q_new)) / 2. 

#         return p_new, q_new
    

    
#     def make_movie(self, title_prefix, var_idx1=0, var_idx2=1):
#         """
#         Creates a deck of png files that can be turned into a movie.
        
#         Note each slide is indexed by the iteration number, i, and
#         the time step number l = 0, ..., L. 

#         var_idx1, 2 specifies which variables to plot.
#         """
#         assert self.sampler_type == "Random" # Only random trajectory sampler is supported.

#         # Take only the accepted points
#         q_accepted = np.zeros((len(self.phi_q), self.D))
#         for i in range(len(self.phi_q)): # For each iteration.
#             q_accepted[i, :] = self.phi_q[i][0, :] # Take the first point.
#         q1 = q_accepted[:, var_idx1]
#         q2 = q_accepted[:, var_idx2]
#         qmin1, qmax1 = np.min(q1), np.max(q1)
#         qmin2, qmax2 = np.min(q2), np.max(q2)

#         # Total number of samples from which to make the movie.
#         idx = 0
#         for i in range(len(self.phi_q)): # For each iteration.
#             phi_q_tmp = self.phi_q[i] # Grab the trajectory.
#             phi_q_tmp_len = phi_q_tmp.shape[0] # Length of the trajectory.
#             decision = self.decision_chain[i]
#             for j in range(phi_q_tmp_len): # For each point in the trajectory, make a slide.
#                 if (idx % 100)==0:
#                     print "Working on slide %d" % idx
#                 self.make_slide(title_prefix, idx, phi_q_tmp[:j+1], q_accepted[:i,:], decision, \
#                     qmin1, qmax1, qmin2, qmax2, var_idx1, var_idx2)
#                 idx += 1 # Increment the index every time 

#         print "Use the following command to make a movie:\nffmpeg -r 1 -start_number 0 -i %s-slide-%%d.png -vcodec mpeg4 -y %s-movie.mp4"  % (title_prefix, title_prefix)
        
#         return 
    
#     def make_slide(self, title_prefix, idx, phi_q, q_accepted, decision, qmin1, qmax1, qmin2, qmax2, var_idx1, var_idx2):
#         fig, ax = plt.subplots(1, figsize=(5, 5))

#         # Plot the current trajectory
#         phi_q1 = phi_q[:, var_idx1]
#         phi_q2 = phi_q[:, var_idx2]
#         color = "black"
#         if decision:
#             color = "red"
#         ax.scatter(phi_q1, phi_q2, s=5, edgecolor="none", c=color)
#         ax.scatter(phi_q1[-1], phi_q2[-1], c=color, s=30, edgecolor="")
#         ax.plot(phi_q1, phi_q2, c=color, ls="--", lw=0.5)

#         # Plot all the previously accepted points
#         if q_accepted.shape[0] > 0:
#             q1 = q_accepted[:, var_idx1]
#             q2 = q_accepted[:, var_idx2]
#             ax.scatter(q1, q2, c="black", s=10, edgecolor="none")

#         # Adjust the maximum size depending on the current trajectory
#         qmin1 = np.min((qmin1, np.min(phi_q1)))
#         qmin2 = np.min((qmin2, np.min(phi_q2)))
#         qmax1 = np.max((qmax1, np.max(phi_q1)))
#         qmax2 = np.max((qmax2, np.max(phi_q2)))


#         ax.set_xlim([qmin1, qmax1])
#         ax.set_ylim([qmin2, qmax2])        

#         # Save it
#         plt.savefig("%s-slide-%d.png" % (title_prefix, idx), bbox_inches="tight", dpi=200)
#         plt.close()

#         return 