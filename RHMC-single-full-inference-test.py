# ---- Script used to run test cases showing single trajectories.
from utils import *
from sampler_RHMC import *

Nsteps = 10
Niter = 1000
dt = 2e-1

# Cross checked
mag_true_list = [15, 19., 20., 21.]
mag_model_list = [15., 19., 20., 21.]

# Case counter: mT and mM pair counts as a case.
counter_case = 0

# Separation
sep_list = [0, 1., 2.]

# For plotting
Nskip_list = [0, 50, 200]

# Save directory
save_dir = "./RHMC-single-full/"

for mT in mag_true_list:
	# ---- Generate the gym and mock image and save
	gym = multi_gym(dt=0., Nsteps=0, g_xx=1., g_ff=1.)
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
		# if counter_case < 2:
		print "/----- Case %d: mT = %2d, mM = %d" % (counter_case, mT, mM)
		for sep in sep_list:
			print "/--- sep: %.1f" % sep
			q_model = np.array([[mM, gym.num_rows /2. + sep, gym.num_cols /2.]])

			# Perform the full inference
			gym.run_RHMC(q_model, f_pos=True, delta=1e-6, Niter = Niter, Nsteps=Nsteps, \
			 dt = dt, save_traj=False)

			for Nskip in Nskip_list:
				diag_str = save_dir + "%d-mT%d-mM%d-sep%d-Niter%d-Nskip%d.png" % (counter_case, mT, mM, sep, Niter, Nskip)
				title_str = "%d-mT%d-mM%d-sep%d-Niter%d-Nskip%d" % (counter_case, mT, mM, sep, Niter, Nskip)
				gym.diagnostics_first(q_true, plot_flux=False, save=diag_str, show=False, Nskip=Nskip, pt_size1=10, title_str=title_str)










