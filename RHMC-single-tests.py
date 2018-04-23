# ---- Script used to run test cases showing single trajectories.
from utils import *
from sampler_RHMC import *

# Number of steps/dt -- Paired
Nsteps_list = [20, 200, 2000, 20000]
# dt_list = [5e-2, 5e-3, 5e-4, 5e-5]
dt_list = [1e-1, 1e-2, 1e-3, 1e-4]

# Cross checked
mag_true_list = [15., 18., 20., 21.]
mag_model_list = [15., 18., 20., 21.]

# Case counter: mT and mM pair counts as a case.
counter_case = 0

# Save directory
save_dir = "./RHMC-single-traj/"

for mT in mag_true_list:
	# ---- Generate the gym and mock image and save
	gym = single_gym(dt=0., Nsteps=0, g_xx=1., g_ff=1.)
	# Set the size of the image
	gym.num_rows = gym.num_cols = 16
	q_true = np.array([[mT, gym.num_rows /2., gym.num_cols /2.]])
	gym.gen_mock_data(q_true)
	# Save the image
	image_str = save_dir + "mT%d-image.png" % (mT)
	gym.display_image(figsize=(4, 4), num_ticks=7, save=image_str, show=False)

	# ---- Iterate through different models
	for mM in mag_model_list:
		counter_case += 1
		if counter_case < 2:
			print "/----- Case %d: mT = %2d, mM = %d" % (counter_case, mT, mM)
			q_model = np.array([[mM, gym.num_rows /2. + 2., gym.num_cols /2.]])
			for i in range(len(Nsteps_list)):
				# Number of steps and time step size
				Nsteps = Nsteps_list[i]
				dt = dt_list[i]
				print "Nsteps: %d" % Nsteps

				# Run the simulation.
				gym.Nsteps = Nsteps
				gym.dt = dt
				gym.run_single_RHMC(q_model_0=np.copy(q_model), f_pos=True, solver="implicit", delta=1e-6)
				diag_str1 = save_dir + "%d-mT%d-mM%d-H-Nsteps%d.png" % (counter_case, mT, mM, Nsteps)
				diag_str2 = save_dir + "%d-mT%d-mM%d-HVT-Nsteps%d.png" % (counter_case, mT, mM, Nsteps)
				gym.diagnostics_first(q_true, plot_flux=False, save=diag_str1, show=False)
				gym.diagnostics_first(q_true, plot_flux=False, plot_T=True, plot_V=True, save=diag_str2, show=False)
