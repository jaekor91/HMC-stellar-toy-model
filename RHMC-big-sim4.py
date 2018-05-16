from utils import *
from sampler_RHMC import *

# Number of steps
Nsteps = 10
Niter = 10000
prior = True
use_Vc = False
RJRHMC = True # Use Transdimensional moves
P_move = [0.6, 0.2, 0.2]
dt = 5e-2

# gff2_list = scheduler(1/10., 4., 500)
beta_list = None # scheduler(1e-2, 1e-12, 500)
gym = multi_gym(dt=0., Nsteps=0, g_xx=0.05, g_ff=4., g_ff2=4.)

# --- Multiple stars
np.random.seed(77)
gym.num_rows = gym.num_cols = 32
Nobjs = int(32**2 * 0.05)
Nobjs_model = 5
q_true = np.zeros((Nobjs, 3))
q_model = np.zeros((Nobjs_model, 3))

# --- Determine step size based on the number of objects.
# dt = 1e-2 # 5e-2 * np.sqrt(30) / np.sqrt(Nobjs_model)

# ---- Truth samples
alpha = 2.
mag_max = 20.
mag_min = 15.
fmin = gym.mag2flux_converter(mag_max)
fmax = gym.mag2flux_converter(mag_min)
mag = gym.flux2mag_converter(gen_pow_law_sample(alpha, fmin, fmax, Nobjs))
for i in xrange(Nobjs):
    x = np.random.random() * (gym.num_rows-2.) + 1.
    y = np.random.random() * (gym.num_cols-2.) + 1.
    q_true[i] = np.array([mag[i], x, y])
if RJRHMC:
    gym.fmin = fmin
    gym.fmax = fmax
    gym.K_split = 1.
    gym.beta_a = 4. 
    gym.beta_b = 4.  
if prior:
    gym.use_prior = True
    gym.alpha = alpha
# if use_Vc:
#     gym.use_Vc = True
#     gym.beta = 1e-4
#     gym.f_expnt = np.zeros(Nobjs_model)
#     gym.Vc_r_pow = 4.

# ---- Model samples
# alpha = 1.5
# mag_max = 23.
# mag_min = 22.9
fmin = gym.mag2flux_converter(mag_max)
fmax = gym.mag2flux_converter(mag_min)
mag = gym.flux2mag_converter(gen_pow_law_sample(alpha, fmin, fmax,  Nobjs_model))
x = np.random.random(size=Nobjs_model) * (gym.num_rows-2.) + 1.
y = np.random.random(size=Nobjs_model) * (gym.num_cols-2.) + 1.
q_model[:, 0] = mag
q_model[:, 1] = x 
q_model[:, 2] = y 

# Generate mock data
gym.gen_mock_data(q_true)
# gym.display_image()
	
# Generate noise profile
gym.gen_noise_profile(q_true, N_trial=1000)

# print "--------------- RHMC"
gym.run_RHMC(q_model, f_pos=True, delta=1e-6, Niter = Niter, Nsteps=Nsteps, \
             dt = dt, save_traj=False, verbose=True, q_true = q_true,
              schedule_beta=beta_list, P_move = P_move, N_max = 120)
# schedule_g_ff2=gff2_list, \


# save_dir = "./RHMC-big-sim4/"
# counter = 0
# j = 0
# for i in xrange(0, Niter+1):
#     if (i % 50) == 0:
#         # for j in xrange(Nsteps+1):
#         title_str = "Niter%05d-Step%03d" % (i, j)
#         fname = save_dir + "slide-%07d" % counter
#         gym.diagnostics_all(q_true, show=False, idx_iter = i, idx_step=j, save=fname,\
#                    m=-15, b =10, s0=23, y_min=5., title_str=title_str)
#         counter+=1 
#         