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


    def dVdq_single(self, q_single, model_data, f_only=True, return_all=False):
        """
        Return the gradient of a single object based on model data where an object with f, x, y is to be added. 
        If f_only is True, then return only f gradient. If False, return xy gradients.

        If all True, then over-ride f_only and return all grads.
        """
        f, x, y = q_single
        Lambda = model_data + f * gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)

        # Variable to be recycled
        rho = (self.D/Lambda)-1.# (D_lm/Lambda_lm - 1)

        # Compute f, x, y gradient for each object
        PSF = gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)

        if return_all:
            grad_f = -np.sum(rho * PSF) # flux grad
            lv = np.arange(0, self.num_rows)
            mv = np.arange(0, self.num_cols)
            mv, lv = np.meshgrid(lv, mv)
            var = (self.PSF_FWHM_pix/2.354)**2 
            grad_x = -np.sum(rho * (lv - x + 0.5) * PSF) * f / var
            grad_y = -np.sum(rho * (mv - y + 0.5) * PSF) * f / var
            return grad_f, grad_x, grad_y
        elif f_only:
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
        return self.V_single(q_single, model_data) + np.dot(p_single, p_single) / 2.


    def V_single(self, q_single, model_data):
        """
        Energy of a single particle given q_singe = [f, x, y] and its corresponding momenta.
        """
        f0, x0, y0 = q_single
        Lambda = model_data + f0 * gauss_PSF(self.num_rows, self.num_cols, x0, y0, FWHM=self.PSF_FWHM_pix)
        return -np.sum(self.D * np.log(Lambda) - Lambda)
    
    def find_peaks(self, linear_pix_density=0.2, dr_tol = 1., dmag_tol = 0.5, mag_lim = None, Nstep=1000,\
        dt_f_coeff=1e-1, dt_xy_coeff=1e-1, no_perturb=False):
        """
        Determine likely positions of stars through a deterministic algorithm but with random jitter.

        Propose a particle at a uniform grid with density determined by the linear pixel density.
        Apply a small random jitter to each particle. Let the particles role for Nstep steps through
        gradient descent which terminates either if the potential doesn't improve by 1 percent compared to the previous.
        Particles excete independent motion (i.e., consider motion of one particle at a time).

        If any two particles are very close (dr < dr_tol) and magnitude very similar (dmag < dmag_tol), then
        the two particles are replaced by one particle.

        Repeat this process until there are no other particles as such. 

        If magnitude limit is None, then it is automatically set to be equal to 1.5 magnitude brighter 
        than the background brightness.

        mag_lim can be set by considering how frequent false detection comes about at a particular magnitude.

        If no_perturb, then do not perform gradient descent.

        dr_tol should depend on the flux.
        """
        if self.num_rows is None or self.D is None:
            print "The image must be specified first."
            assert False

        if mag_lim is None:
            mag_lim = self.mB - 1.

        # Compute flux limit
        f_lim = mag2flux(mag_lim) * self.flux_to_count
        # print "f_lim: %.2f" % f_lim

        # Seed flux 
        f_seed = mag2flux(mag_lim-0.5) * self.flux_to_count
        # print "f_seed: %.2f" % f_seed

        # Seed objects. Compute Nobjs and initialize
        Nobjs_row = int(linear_pix_density * self.num_rows) - 1 # Number of objects in row direction
        Nobjs_col = int(linear_pix_density * self.num_cols) - 1 # Number of objects in col direction
        Nobjs_tot = Nobjs_row * Nobjs_col # Total number of objects
        grid_spacing = 1/float(linear_pix_density)
        q_seed = np.zeros((Nobjs_tot, 3), dtype=float)
        A_seed = np.ones(Nobjs_tot, dtype=bool) # Vector that tells whether an object is considered to have been localized or not.
        for i in xrange(Nobjs_row): # For each row
            for j in xrange(Nobjs_col): # For each col
                idx = i * Nobjs_row + j # obj index
                x = grid_spacing * (i+0.5 + 0.1 * np.random.randn()) 
                y = grid_spacing * (j+0.5 + 0.1 * np.random.randn()) 
                q_seed[idx] = np.array([f_seed, x, y])

        # If not perturbation asked, then simply return the seeds. 
        if no_perturb:
            self.q_seed = q_seed
            return

        # Construct model data image. Pure background.
        model_data = np.ones((self.num_rows, self.num_cols), dtype=float) * self.B_count

        # Perform gradient descent of each object, checking whether an object should be retained or not.
        for idx in xrange(Nobjs_tot):
            f, x, y = q_seed[idx]

            # Initial potential
            V_previous = self.V_single([f, x, y], model_data)

            for i in range(Nstep):
                grad_f, grad_x, grad_y = self.dVdq_single([f, x, y], model_data, return_all=True)

                # Time step size
                dt_f = f * dt_f_coeff
                dt_xy = dt_xy_coeff / f

                # Update the params
                f -= grad_f * dt_f
                x -= grad_x * dt_xy
                y -= grad_y * dt_xy

                if (f < f_lim): # If the object is very faint then make it disappear
                    A_seed[idx] = False
                    break

                V_current = self.V_single([f, x, y], model_data)

                if np.abs((V_current - V_previous)/V_previous) < 1e-9: # If the relative change is less than 0.1%
                    break

                V_previous = V_current

            # Set the q_seed with 
            q_seed[idx] = np.array([f, x, y])

        # Return a subset that did not disappear
        q_seed = q_seed[A_seed]

        # If there are no seeds, return
        if q_seed.shape[0] == 0:
            self.q_seed = q_seed
            print "No peaks were found."
            return

        # Perform reduction operation, eliminating redundant objects. This is NOT optimized.
        q_seed_list_final = []
        while True:
            q_ref = q_seed[0]
            q_seed_list_final.append(q_ref)
            q_seed = q_seed[1:, :]

            # Compute distance from ref to all the others
            dist_sq = (q_seed[:, 1] - q_ref[1])**2 + (q_seed[:, 2] - q_ref[2])**2
            mag_seed = flux2mag(q_seed[:, 0] / self.flux_to_count)
            mag_ref = flux2mag(q_ref[0] / self.flux_to_count)

            # Sub select those that are more than some tolerance amount away or close but magnitude far away
            ibool = np.logical_or((dist_sq > dr_tol**2), (np.abs(mag_ref - mag_seed) > dmag_tol))
            q_seed = q_seed[ibool]

            if q_seed.shape[0] == 0:
                break

        # Concatenate and save
        self.q_seed = np.vstack(q_seed_list_final)

        return


    def HMC_find_best_dt(self, q_model_0=None, steps_min=10, steps_max=50, Niter_per_trial = 10, Ntrial = 10, \
        dt_f_coeff=1., dt_xy_coeff=10., default=False, A_target_f = 0.9, A_target_xy=0.5):
        """
        Find the optimal dt through the following heuristic method.

        Set the initial step size as follows. For each particle,
        - dt_xy: dt_xy_coeff/f
        - dt_f: dt_f_coeff * f

        Start from large step sizes and reduce them incrementally.
    
        For each parameter, fixing every other parameter, do the following search
            Coarse finding:
            Perform Niter_per_trial HMC iterations. Compute the acceptance rate.
            Until acceptance rate becomes 1, either increase or decrease the step size by a factor of 2. 

            Fine finding:
            Once the acceptance has become equal to 1, increase the step size by a factor 2.
            Compute the acceptance rate again, and if it falls below 1, then the next step size to try is
            the mid-point of the current and the previous step size. Continue with the "bi-section search".
            This process terminates when Ntrial number has been exhausted.

        Note that only one particle is being "perturbed" at a time. Also, position step sizes dt_xy are tuned together.

        If default False, then use the intial step size.
        """
        if q_model_0 is None: 
            print "Use found seeds for inference."
            q_model_0 = self.q_seed

        #--- Number of objects
        self.Nobjs = q_model_0.shape[0]
        self.d = self.Nobjs * 3

        #---- Placeholder for step size
        self.dt = np.zeros(self.d)

        #---- If default parameters asked for.
        if default:
            # assert False
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
                    # print "flux"
                    A_target = A_target_f
                else:
                    dt = np.array([dt_f, dt_xy, dt_xy]) # Note that we perturb flux as well.
                    # print "xy"
                    A_target = A_target_xy

                #---- Coarse finding
                run = True
                # While the acceptance is not equal to one, keep adjusting the step size.                
                while run:
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
                    # print dt, A_rate

                    if A_rate < A_target:
                        if k == 0:
                            dt /= 10.
                        else:
                            dt[1:] = dt[1:]/10.
                    else: 
                        run = False

                #---- Fine finding
                counter = 0
                dt_left = dt # The smallest dt where A_rate is greater than A_target
                dt_right = 10. * dt # The largest considered dt where A_rate is below A_target
                if k == 1: 
                    dt_right[0] = dt_f

                # Keep trying to find the precise A_rate = 0 turning point.
                while counter < Ntrial:
                    counter += 1

                    # dt to try
                    dt = (dt_left + dt_right)/2.
                    if k == 1:
                        dt[0] = dt_f

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
                    # print dt, A_rate

                    if np.abs(A_rate-A_target) < 1e-2:
                        break
                    elif A_rate > A_target: 
                        dt_left = dt
                    else: # If acceptance rate is smaller then decrease the maximum limit.
                        dt_right = dt

                # dt = (dt_left+dt_right)/2. # Use the smaller one to be conservative.

                if k == 0:
                    dt_f = dt[0]
                else:
                    dt_xy = dt[1] * 10

            # print np.array([dt_f, dt_xy, dt_xy])
            self.dt[3*l:3*l+3] = np.array([dt_f, dt_xy, dt_xy])



    def HMC_random(self, q_model_0=None, Nchain=1, Niter=1000, thin_rate=0, Nwarmup=0, steps_min=10, steps_max = 50,\
        f_lim = 0., f_lim_default = False):
        """
        Perform Bayesian inference with HMC given an initial model q_model_0 (Nobjs, 3). 
        No change in dimension is implemented. 

        Random trajectory length is used with steps ~ [steps_min, steps_max]
        dt_xy_coeff and dt_f_coeff are parameters.

        f_lim_default: If set True, then set f_lim to *default* value instead of the user provided f_lim value.
        """
        #---- Set inference variables as provided by the user. 
        assert Nchain == 1 # Currently we do not support any other.
        self.Nchain = Nchain
        self.Niter = Niter
        self.thin_rate = thin_rate
        self.Nwarmup = Nwarmup

        #---- Number of objects should have been already determined via optimal step search
        assert self.d is not None

        #---- Min flux
        if f_lim_default:
            self.f_lim = mag2flux(self.mB - 1.) * self.flux_to_count
        else:   
            self.f_lim = f_lim 
        
        #---- Reshape the initial point
        if q_model_0 is None: 
            print "Use found seeds for inference."
            q_model_0 = self.q_seed
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
                p_half = p_tmp - self.dt * self.dVdq(q_tmp) / 2. # First half step
                iflip = np.zeros(self.d, dtype=bool) # Flip array.                
                for _ in xrange(steps_sample): 
                    flip = False
                    q_tmp = q_tmp + self.dt * p_half
                    # We only consider constraint in the flux direction.
                    # If any of the flux is below the limiting point, then change the momentum direction
                    for l in xrange(self.Nobjs):
                        if q_tmp[3 * l] < self.f_lim:
                            iflip[3 * l] = True
                            flip = True
                    if flip: # If fix due to constraint.
                        p_half_tmp = -p_half[iflip] # Flip the direction.
                        p_half = p_half - self.dt * self.dVdq(q_tmp) # Update as usual
                        p_half[iflip] = p_half_tmp # Make correction
                    else:
                        p_half = p_half - self.dt * self.dVdq(q_tmp) # If no correction, then regular update.

                # Final half step correction
                if flip:
                    p_half_tmp = p_half[iflip] # Save 
                    p_half = p_half + self.dt * self.dVdq(q_tmp) / 2.# Update as usual 
                    p_half[iflip] = p_half_tmp # Make correction                    
                else:
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

    def RHMC_efficient_computation(self, q_tmp, p_tmp, debug=True, dVdqq_only=False):
        """
        Given current parameters, return quantities of interest.
        """
        for k in xrange(self.Nobjs):
            if dVdqq_only:
                break
            f, x, y = q_tmp[3*k:3*k+3]
            if f < self.f_lim:
                return np.infty, np.infty, np.infty            

        # Place holders
        dVdq = np.zeros(self.d)
        dVdqq = np.zeros(self.d)
        dVdqqq = np.zeros(self.d)                

        # Construct current model
        Lambda = np.ones_like(self.D) * self.B_count # Model set to background                
        for k in xrange(self.Nobjs):
            f, x, y = q_tmp[3*k:3*k+3]
            Lambda += f * gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)

        # Compute reused quantities
        rho0 = self.D/Lambda
        rho1 = 1-rho0
        rho2 = rho0/Lambda
        rho3 = rho2/Lambda
        lv = np.arange(0, self.num_rows) + 0.5
        mv = np.arange(0, self.num_cols) + 0.5
        mv, lv = np.meshgrid(lv, mv)
        var = (self.PSF_FWHM_pix/2.354)**2 

        # Compute grads 
        for k in range(self.Nobjs):
            f, x, y = q_tmp[3*k:3*k+3]

            # f
            PSF = gauss_PSF(self.num_rows, self.num_cols, x, y, FWHM=self.PSF_FWHM_pix)
            PSF_sq = PSF**2                    
            PSF_cube = PSF_sq * PSF                    
            # x
            dPSFdx = PSF * (lv - x) / var
            dPSFdx_sq = dPSFdx**2                    
            dPSFdxx = (dPSFdx * (lv - x) - PSF) / var
            dPSFdxxx = (dPSFdxx * (lv - x) - 2 * dPSFdx) / var
            # y
            dPSFdy = PSF * (mv - y) / var
            dPSFdy_sq = dPSFdy**2
            dPSFdyy = (dPSFdy * (mv - y) - PSF) / var
            dPSFdyyy = (dPSFdyy * (mv - y) - 2 * dPSFdy) / var

            # Derivatives
            # f
            dVdf = np.sum(rho1 * PSF)
            dVdff = np.sum(rho2 * PSF_sq)
            dVdfff = -2 * np.sum(rho3 * PSF_cube)
            # x
            dVdx = f * np.sum(rho1 * dPSFdx)
            dVdxx =  f**2 * np.sum(rho2 * dPSFdx_sq) + f * np.sum(rho1 * dPSFdxx)
            dVdxxx =  - f**3 * np.sum(rho3 * dPSFdx_sq * dPSFdx) + 3 * f**2 * np.sum(rho2 * dPSFdx * dPSFdxx)\
                + f * np.sum(rho1 * dPSFdxxx)  
            # y
            dVdy = f * np.sum(rho1 * dPSFdy)
            dVdyy =  f**2 * np.sum(rho2 * dPSFdy_sq) + f * np.sum(rho1 * dPSFdyy)
            dVdyyy =  - f**3 * np.sum(rho3 * dPSFdy_sq * dPSFdy) + 3 * f**2 * np.sum(rho2 * dPSFdy * dPSFdyy)\
                + f * np.sum(rho1 * dPSFdyyy)  

            # Save the results
            # dVdq
            dVdq[3*k:3*(k+1)] = np.array([dVdf, dVdx, dVdy])

            # dVdqq
            dVdqq[3*k:3*(k+1)] = np.array([dVdff, dVdxx, dVdyy])

            # dVdqqq
            dVdqqq[3*k:3*(k+1)] = np.array([dVdfff, dVdxxx, dVdyyy])                    

        if dVdqq_only:
            return dVdqq

        # Compute quantities of interest.
        dqdt = p_tmp / dVdqq # inv_cov_p = H^-1                 
        dpdt = -dVdqqq * ((1./dVdqq) - dqdt**2) / 2. - dVdq # dqdt = p_tmp **2 / dVdqq**2
        K = np.sum((p_tmp ** 2) / dVdqq) / 2. + np.log(np.product(dVdqq)) / 2. # Frist corresponds to the qudractic term and the other determinant.
        V = -np.sum(self.D * np.log(Lambda) - Lambda)
        E = K+V

        if debug:
            print "q_tmp", q_tmp
            print "dqdt", dqdt
            print "p_tmp", p_tmp            
            print "dpdt", dpdt
            print "dVdq", dVdq
            print "dVdqq", dVdqq
            print "dVdqqq", dVdqqq
            print "K", K
            print "V", V
            print "dpdt from K", -dVdqqq * ((1./dVdqq) - dqdt**2) / 2.
            print "log det", np.log(np.product(dVdqq)) / 2.
        return dqdt, dpdt, E


    def RHMC_random(self, q_model_0=None, Nchain=1, Niter=1000, thin_rate=0, Nwarmup=0, steps_min=10, steps_max = 50,\
        f_lim = 0., f_lim_default = False, dt_RHMC_xy=1., dt_RHMC_f = 0.1, debug=True):
        """
        Perform Bayesian inference with RHMC given an initial model q_model_0 (Nobjs, 3). 
        No change in dimension is implemented. 

        Random trajectory length is used with steps ~ [steps_min, steps_max].

        f_lim_default: If set True, then set f_lim to *default* value instead of the user provided f_lim value.
        """
        #---- Set inference variables as provided by the user. 
        assert Nchain == 1 # Currently we do not support any other.
        self.Nchain = Nchain
        self.Niter = Niter
        self.thin_rate = thin_rate
        self.Nwarmup = Nwarmup

        #---- Determine number of objects here
        self.Nobjs = q_model_0.shape[0]
        self.d = self.Nobjs * 3

        #---- Construct time step vector
        dt_RHMC = np.asarray([dt_RHMC_f, dt_RHMC_xy, dt_RHMC_xy] * self.Nobjs)

        #---- Min flux
        if f_lim_default:
            self.f_lim = mag2flux(self.mB - 1.) * self.flux_to_count
        else:   
            self.f_lim = f_lim 
        
        #---- Reshape the initial point
        if q_model_0 is None: 
            print "Use found seeds for inference."
            q_model_0 = self.q_seed
        q_model_0 =  q_model_0.reshape((self.d,))# Flatten 

        #---- Allocate storage for variables being inferred.
        # The zeroth slot is reserved for the initial. The first iteration takes 0 --> 1.
        self.q_chain = np.zeros((self.Nchain, self.Niter+1, self.d))
        self.E_chain = np.zeros((self.Nchain, self.Niter+1, 1))
        self.dE_chain = np.zeros((self.Nchain, self.Niter+1, 1))
        self.A_chain = np.zeros((self.Nchain, self.Niter, 1)) # Acceptance rate chain. There are only Niter transitions.
        # Samples are taken from [1, Niter + 1]

        # Set first point
        q_tmp = q_model_0
        p_tmp = np.zeros_like(q_tmp)
        #---- Looping over chains
        for m in xrange(self.Nchain):
            #---- Looping over iterations
            for i in xrange(0, self.Niter+1, 1):
                #---- Initial
                q_initial = q_tmp

                # Resample moementum
                dVdqq = self.RHMC_efficient_computation(q_tmp, p_tmp, debug=False, dVdqq_only=True)
                # if len(dVdqq) == self.d:
                #     pass
                # else:
                #     print q_tmp
                #     print dVdqq
                #     assert False
                p_tmp = self.p_sample() * np.sqrt(dVdqq)

                #---- Efficient computation of grads and energies.                 
                if debug:
                    print "/--- %d" % i 
                    print "Initial"
                dqdt, dpdt, E = self.RHMC_efficient_computation(q_tmp, p_tmp, debug)

                if i == 0:
                    self.q_chain[m, 0, :] = q_tmp
                    self.E_chain[m, 0, 0] = E #
                    self.dE_chain[m, 0, 0] = 0 # Arbitrarily set to zero.
                    E_previous = self.E_chain[m, 0, 0]
                    q_tmp = q_initial
                else:
                    # Compute E and dE and save
                    E_initial = E # 
                    self.E_chain[m, i, 0] = E_initial
                    self.dE_chain[m, i, 0] = E_initial - E_previous                    

                    #---- Looping over a random number of steps
                    steps_sample = np.random.randint(low=steps_min, high=steps_max, size=1)[0]

                    #---- First half step for momentum
                    p_half = p_tmp + dt_RHMC * dpdt / 2.# 
                    iflip = np.zeros(self.d, dtype=bool) # Flip array.                
                    iflip_x = np.zeros(self.d, dtype=bool) # Flip array.                
                    iflip_y = np.zeros(self.d, dtype=bool) # Flip array.                                    
                    for z in xrange(steps_sample):
                        if debug:
                            print "/- %d" % z  
                            print "q step"                                       
                        dqdt, dpdt, E = self.RHMC_efficient_computation(q_tmp, p_half, debug)                        
                        if E == np.infty: 
                            print "Divergence encountered at (%d ,%d)" % (i, z)
                            break
                        flip = False
                        flip_x = False
                        flip_y = False
                        q_tmp += dt_RHMC * dqdt
                        # We only consider constraint in the flux direction.
                        # If any of the flux is below the limiting point, then change the momentum direction
                        for l in xrange(self.Nobjs):
                            if q_tmp[3 * l] < self.f_lim:
                                iflip[3 * l] = True
                                flip = True
                            if (q_tmp[3 * l + 1] < 0) or (q_tmp[3 * l + 1] < self.num_rows):
                                iflip_x[3 * l + 1] = True
                                flip_x = True
                            if (q_tmp[3 * l + 2] < 0) or (q_tmp[3 * l + 2] < self.num_rows):
                                iflip_y[3 * l + 2] = True
                                flip_y = True

                        # If this is the last step
                        if z == steps_sample-1:
                            dt_tmp = dt_RHMC/2.
                        else:
                            dt_tmp = dt_RHMC

                        if debug:
                            print "\n"
                            print "p step"
                        dqdt, dpdt, E = self.RHMC_efficient_computation(q_tmp, p_half, debug)

                        #-- Bounce off boundary
                        p_half_tmp = -p_half # Flip the direction.                        
                        p_half += dt_tmp * dpdt # Update as usual                        
                        # f
                        if flip: # If fix due to constraint.
                            p_half[iflip] = p_half_tmp[iflip] # Make correction
                        # x
                        if flip_x: # If fix due to constraint.
                            p_half[iflip_x] = p_half_tmp[iflip_x] # Make correction
                        # f
                        if flip_y: # If fix due to constraint.
                            p_half[iflip_y] = p_half_tmp[iflip_y] # Make correction

                        if debug:
                            print "\n"

                    # Compute and save the final energy
                    if debug: 
                        print "\n"   
                        print "Energy compute"                
                    _, _, E_final = self.RHMC_efficient_computation(q_tmp, p_half, debug)
                    if debug:
                        print "dE", E_final - E_initial
                        
                    # With correct probability, accept or reject the last proposal.
                    dE = E_final - E_initial
                    E_previous = E_initial # Save the energy so that the energy differential can be computed during the next run.
                    lnu = np.log(np.random.random(1))        
                    if (dE < 0) or (lnu < -dE): # If accepted.
                        self.A_chain[m, i-1, 0] = 1
                        self.q_chain[m, i, :] = q_tmp # save the new point
                        if debug:
                            print "Accepted."
                    else: # Otherwise, proposal rejected.
                        self.q_chain[m, i, :] = q_initial # save the old point
                        q_tmp = q_initial
                        if debug:
                            print "Rejected"

                    if debug:
                        print "\n\n"

                    if (i % 100) == 0: 
                        self.R_accept = np.sum(self.A_chain[m, :i] * 100)/float(i)
                        if self.R_accept < 50:
                            break
            self.R_accept = np.sum(self.A_chain[m, :] * 100)/float(self.Niter)
            print "Chain %d Acceptance rate: %.2f%%" % (m, self.R_accept)

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
        Nobjs = q.size // 3 # Assume that q is flat.s
        for l in xrange(Nobjs):
            if q[3 * l] < self.f_lim:
                return np.infty

        return self.V(q) + self.K(p)

    def K(self, p):
        """
        The user supplies potential energy and its gradient.
        User also sets the form of kinetic distribution.
        Kinetic energy -ln P(p)
        """
        # return np.dot(p, np.dot(self.inv_cov_p, p)) / 2. 
        return np.dot(p, p) / 2. # We assume identity covariance.


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

    def display_data(self, figsize=(5, 5), vmax_percentile = 100, vmin_percentile=1, plot_seed=False, size_seed = 10):
        """
        A light display of the mock data.
        """
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(self.D, interpolation="none", vmin=np.percentile(self.D, vmin_percentile), vmax=np.percentile(self.D, vmax_percentile), cmap="gray")
        if plot_seed:
            x = self.q_seed[:, 1]
            y = self.q_seed[:, 2]            
            ax.scatter(x, y, c="red", edgecolor="none", s=size_seed)
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

        #---- Default mag set up
        self.mB = 23
        B_count = mag2flux(self.mB) * flux_to_count

        # Size of the image
        num_rows = num_cols = 48 # Pixel index goes from 0 to num_rows-1

        return num_rows, num_cols, flux_to_count, PSF_FWHM_pix, B_count, arcsec_to_pix



class RHMC_GMM(object):
    """
    Practice implementing RHMC with a GMM model.
    """
    
    def __init__(self, A_ls=None, mu_ls=None, S_ls=None):
        """
        Input quantities define the Gaussian mixture model.
        """
        # If any of the quantities not provided, then go with a default.
        if (A_ls is None) or (S_ls is None) or (mu_ls is None):
#             # --- Signel component example
#             A_ls = np.array([1.])
#             S_ls = np.array([
#                             [[1., 0.], # first
#                              [0., 0.5]]
#                             ])
#             mu_ls = np.array([
#                             [0., 0.] # first
#                             ]) 
            
            # --- Two component example
            A_ls = np.array([0.5, 0.5])
            S_ls = np.array([
                            [[1., 0.], # first
                             [0., 1.]],
                             [[1e-4, 0.], # second
                              [0., 1e-4]]
                            ])
            mu_ls = np.array([
                            [0., 0.], # first
                            [0., 0.] # second
                            ]) 
        
        self.A_ls = A_ls
        self.mu_ls = mu_ls
        self.S_ls = S_ls
        
        self.K = self.S_ls.shape[0] # Number of components        
        self.D = self.mu_ls[0].size # Dimensions of the problem        
        
        # Compute inverse of covariance matrices
        self.inv_S_ls = np.zeros_like(self.S_ls)
        if self.D == 1:
            for l in xrange(self.K):
                self.inv_S_ls[l] = 1./self.S_ls[l]
        else:
            for l in xrange(self.K):
                self.inv_S_ls[l] = np.linalg.inv(self.S_ls[l])
        
        return
    
    #---- pi_l and its q-derivatives
    def pi_l(self, q, l):
        """
        l-th Normal
        """
        return multivariate_normal.pdf(q, mean=self.mu_ls[l], cov=self.S_ls[l])
    
    def pi_ls_and_derivatives(self, q, HMC=False):
        """
        l-th normal
        """
        # Compute pi_ls
        pi_ls = np.zeros_like(self.A_ls)
        for l in xrange(self.K):
            pi_ls[l] = self.pi_l(q, l)
            
        # Cache a quantity used repeatedly. (i, l)
        G = np.zeros((self.D, self.K)) # Just a lousy name.
        for l in xrange(self.K):
            G[:, l] = np.dot(self.inv_S_ls[l], (q - self.mu_ls[l]))
        
        # Compute first derivative. 
        pi_ls_Dq = -G * pi_ls # (i, l)
        
        if HMC:
            return pi_ls, pi_ls_Dq
        
        # Compute second derivative
        pi_ls_Dqq = np.zeros((self.D, self.D, self.K)) # (i, j, l)
        pi_ls_Dqq_diag = np.zeros((self.D, self.K))         
        for l in xrange(self.K):
            pi_ls_Dqq[:, :, l] = (G[:, l].T * G[:, l] - self.inv_S_ls[l]) * pi_ls[l]
            
        # Compute restricted third derivative
        pi_ls_Dqqq = np.zeros((self.D, self.D, self.K)) # (n, i, l)
        for l in xrange(self.K):
            pi_ls_Dqqq[:, :, l] = G[:, l].T * (-G[:, l]**2 + np.diag(self.inv_S_ls[l]) ) * pi_ls[l]
            
        #--- Debug lines
        # print RHMC.pi_l(0, 0)        
        # print pi_ls_Dq
        # print "Second", pi_ls_Dqq
        # print "Second diag", pi_ls_Dqq_diag
        # print "Third", pi_ls_Dqqq
            
        return pi_ls, pi_ls_Dq, pi_ls_Dqq, pi_ls_Dqqq
    
    def pi_and_derivatives(self, q, HMC=False):
        if HMC:
            pi_ls, pi_ls_Dq = self.pi_ls_and_derivatives(q, HMC=HMC)            
        else:
            pi_ls, pi_ls_Dq, pi_ls_Dqq, pi_ls_Dqqq = self.pi_ls_and_derivatives(q)
        
        # The default sum
        pi = np.sum(self.A_ls * pi_ls)
        
        # First derivative
        pi_Dq = np.dot(pi_ls_Dq, self.A_ls) # i
        
        if HMC:
            return pi, pi_Dq
        
        # Second derivative
        pi_Dqq = np.dot(pi_ls_Dqq, self.A_ls) # i, j
        pi_Dqq_diag = np.diag(pi_Dqq) # i
            
        # Third restricted derivative
        pi_Dqqq = np.dot(pi_ls_Dqqq, self.A_ls) # (n, i)
                
        return pi, pi_Dq, pi_Dqq, pi_Dqq_diag, pi_Dqqq

    
    def V_and_derivatives(self, q):
        pi, pi_Dq, pi_Dqq, pi_Dqq_diag, pi_Dqqq = self.pi_and_derivatives(q)
        
        # Potential
        V = -np.log(pi)
        
        # First derivative
        V_Dq = -pi_Dq / pi
        
        # Second derivative
        V_Dqq = pi_Dq.T * pi_Dq - pi_Dqq / pi
        
        # Second diagonal
        H_ii = np.diag(V_Dqq)
        
        # Third restricted derivative
        H_ii_Dq = 2 * V_Dqq * V_Dq + pi_Dq.T * pi_Dqq_diag / pi*2 - pi_Dqqq / pi
        
        return V, V_Dq, V_Dqq, H_ii, H_ii_Dq
    
    def V_dVdq_HMC(self, q):
        """
        Efficient version of V_and_derivatives.
        """
        pi, pi_Dq = self.pi_and_derivatives(q, HMC=True)
        
        # Potential
        V = -np.log(pi)
        
        # First derivative
        V_Dq = -pi_Dq / pi
        
        return V, V_Dq
        
        
    def T_HMC(self, p):
        return np.dot(p, p) / 2.
    
    def HMC_single_simulator(self, q0, Nsteps = 100, dt=1e-2, p=None): 
        """
        Given the intial point q0, randomly sample a momentum from uni-variate normal 
        and simulate dyanmics for Nsteps.
        
        Args:
             - dt: Global step size.
        
        Save:
            - Position
            - Kinetic, Potential and Total energies
        """
        # Place holder for positions and energies
        self.qs = np.zeros((Nsteps+1, self.D)) # +1 is for the initial position.
        self.Vs = np.zeros(Nsteps+1)
        self.Ts = np.zeros(Nsteps+1)        
        
        # Sample initial momentum if None
        if p is None:
            p = np.random.randn(self.D)
        
        # Save the initial point and energies
        V, dVdq = self.V_dVdq_HMC(q0)
        self.Vs[0] = V
        self.Ts[0] = self.T_HMC(p)
        self.qs[0, :] = q0
        
        # Perform iterations here.
        q = q0        
        for i in xrange(1, Nsteps+1):
            p_half = p - dt * dVdq / 2.            
            q = q + dt * p_half
            V, dVdq = self.V_dVdq_HMC(q)
            p = p_half - dt * dVdq / 2.
            
            # Save the results
            self.Vs[i] = V
            self.Ts[i] = self.T_HMC(p)
            self.qs[i, :] = q
        
        # Total energy
        self.Es = self.Vs + self.Ts
        
        return
    
    def T_RHMC(self, p, H_ii):
        """
        Compute T_RHMC for given a diagonal H matrix.
        """
        return np.sum(0.5 * p * p / H_ii) + 0.5 * np.log(np.abs(np.product(H_ii)))    
    
    def dtau_dp(self, p, H_ii):
        return p/H_ii
    
    def dtau_dq(self, p, H_ii, H_ii_Dq, alpha):
        # Calculation of J_ii
        x_ii = alpha * H_ii
        sinh = np.sinh(x_ii)
        cosh = np.cosh(x_ii)
        J_ii = (cosh/sinh) - x_ii / sinh**2
        
        # Calculation of M_ii
        M_ii = J_ii * p**2
        
        # Calculation of grad
        grad = -0.5 * np.dot(H_ii_Dq, M_ii)
        
        return grad
    
    def dphi_dq(self, H_ii, H_ii_Dq, alpha, V_Dq):
        # Calculation of J_ii
        x_ii = alpha * H_ii
        sinh = np.sinh(x_ii)
        cosh = np.cosh(x_ii)
        J_ii = (cosh/sinh) - x_ii / sinh**2
        
        # Calculation of M_ii
        M_ii = J_ii * sinh / (H_ii * cosh)
        
        # Calculation of grad
        grad = 0.5 * np.dot(H_ii_Dq, M_ii) + V_Dq
        
        return grad
    
    def RHMC_single_simulator(self, q0, Nsteps = 100, eps=0.5, alpha=1e6, delta = 1, p=None): 
        """
        Given the intial point q0, randomly sample a momentum from uni-variate normal 
        and simulate dyanmics for Nsteps.
        
        Args:
            - eps: Global time step size.
            - alpha: Softening paramater for SoftAbs.
            - delta: Parameter used for the fixed point iteration.
        Save:
            - Position
            - Kinetic, Potential and Total energies
        """
        # Place holder for positions and energies
        self.qs = np.zeros((Nsteps+1, self.D)) # +1 is for the initial position.
        self.Vs = np.zeros(Nsteps+1)
        self.Ts = np.zeros(Nsteps+1)        
        
        # Sample initial momentum if None
        if p is None:
            p = np.random.randn(self.D)
        
        # Save the initial point and energies
        V, V_Dq, V_Dqq, H_ii, H_ii_Dq = self.V_and_derivatives(q)
        self.Vs[0] = V
        self.Ts[0] = self.T_RHMC(p, H_ii)
        self.qs[0, :] = q0
                
        # Iterations
        q = q0        
        for i in xrange(1, Nsteps+1):
            # First momentum step
            V, V_Dq, V_Dqq, H_ii, H_ii_Dq = self.V_and_derivatives(q)
            p -= (eps/2.) * self.dphi_dq(H_ii, H_ii_Dq, alpha, V_Dq)
            
            # First fixed iteration
            rho = np.copy(p)
            Dp = np.infty
            while Dp > delta:
                V, V_Dq, V_Dqq, H_ii, H_ii_Dq = self.V_and_derivatives(q)                
                p_tmp = rho - (eps/2.) * self.dtau_dq(p, H_ii, H_ii_Dq, alpha)
                Dp = np.max(np.abs(p_tmp - p))
                p = p_tmp
            
            # Second fixed iteration
            sig = np.copy(q)
            Dq = np.infty
            while Dq > delta:
                V, V_Dq, V_Dqq, H_ii, H_ii_Dq = self.V_and_derivatives(sig)
                grad1 = self.dtau_dp(p, H_ii)
                V, V_Dq, V_Dqq, H_ii, H_ii_Dq = self.V_and_derivatives(q)
                grad2 = self.dtau_dp(p, H_ii)
                q_tmp = sig + (eps/2.) * (grad1 + grad2)
                Dq = np.max(np.abs(q_tmp - q))
                q = q_tmp          
            
            # "Second" momentum update
            V, V_Dq, V_Dqq, H_ii, H_ii_Dq = self.V_and_derivatives(sig)            
            p -= (eps/2.) * self.dtau_dq(p, H_ii, H_ii_Dq, alpha)

            # Last momentum update
            V, V_Dq, V_Dqq, H_ii, H_ii_Dq = self.V_and_derivatives(q)
            p -= (eps/2.) * self.dphi_dq(H_ii, H_ii_Dq, alpha, V_Dq)            
            
            # Save the results
            self.Vs[i] = V
            self.Ts[i] = self.T_RHMC(p, H_ii)
            self.qs[i, :] = q
        
        # Total energy
        self.Es = self.Vs + self.Ts
        
        return 
            
    