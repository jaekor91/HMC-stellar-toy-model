from utils import *
from sampler_RHMC import *

# Number of steps
Nsteps = 30
Niter = 1000
dt = 1e-2
prior = True
use_Vc = True

gff2_list = scheduler(1/10., 4., 500)
beta_list = scheduler(1e-2, 1e-12, 500)
gym = multi_gym(dt=0., Nsteps=0, g_xx=0.005, g_ff=25., g_ff2=2.)

# --- Multiple stars
np.random.seed(77)
gym.num_rows = gym.num_cols = 32
Nobjs = 30
Nobjs_model = 30
q_true = np.zeros((Nobjs, 3))
q_model = np.zeros((Nobjs_model, 3))

# ---- Truth samples
alpha = 1.5
mag_max = 20.5
mag_min = 15.
fmin = gym.mag2flux_converter(mag_max)
fmax = gym.mag2flux_converter(mag_min)
mag = gym.flux2mag_converter(gen_pow_law_sample(alpha, fmin, fmax, Nobjs))
for i in xrange(Nobjs):
    x = np.random.random() * (gym.num_rows-2.) + 1.
    y = np.random.random() * (gym.num_cols-2.) + 1.
    q_true[i] = np.array([mag[i], x, y])
if prior:
    gym.use_prior = True
    gym.alpha = alpha
if use_Vc:
    gym.use_Vc = True
    gym.beta = 1e-4
    gym.f_expnt = np.zeros(Nobjs_model)
    gym.Vc_r_pow = 4.

# ---- Model samples
alpha = 1.5
mag_max = 22.9
mag_min = 21.
fmin = gym.mag2flux_converter(mag_max)
fmax = gym.mag2flux_converter(mag_min)
mag = gym.flux2mag_converter(gen_pow_law_sample(alpha, fmin, fmax,  Nobjs_model))
# x = np.linspace(1., gym.num_rows, endpoint=False, num=13)
# y = np.linspace(1., gym.num_cols, endpoint=False, num=13)
# x, y = np.meshgrid(x, y)
# x = x.ravel() + np.random.random(size=Nobjs_model)
# y = y.ravel() + np.random.random(size=Nobjs_model)
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
             dt = dt, save_traj=False, verbose=True, q_true = q_true, schedule_g_ff2=gff2_list, schedule_beta=beta_list)



# save_dir = "./RHMC-big-sim2/"
# counter = 0
# j = 0
# for i in xrange(0, Niter+1):
#     # for j in xrange(Nsteps+1):
#     title_str = "Niter%05d-Step%03d" % (i, j)
#     fname = save_dir + "slide-%07d" % counter
#     gym.diagnostics_all(q_true, show=False, idx_iter = i, idx_step=j, save=fname,\
#                m=-15, b =10, s0=23, y_min=5., title_str=title_str)
#     counter+=1 
#         