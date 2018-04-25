from utils import *
from sampler_RHMC import *

# Number of steps
Nsteps = 50
Niter = 3000

gym = multi_gym(dt=0., Nsteps=0, g_xx=1., g_ff=1.)

# --- Multiple stars
np.random.seed(0)
gym.num_rows = gym.num_cols = 48
Nobjs = 100
Nobjs_model = 200
q_true = np.zeros((Nobjs, 3))
q_model = np.zeros((Nobjs_model, 3))

# ---- Truth samples
alpha = -2.
mag_max = 20.5
mag_min = 15.
fmin = gym.mag2flux_converter(mag_max)
fmax = gym.mag2flux_converter(mag_min)
mag = gym.flux2mag_converter(gen_pow_law_sample(fmin, Nobjs, alpha, fmax=fmax, exact=True))
for i in xrange(Nobjs):
    x = np.random.random() * (gym.num_rows-5) + 2.5
    y = np.random.random() * (gym.num_cols-5) + 2.5
    q_true[i] = np.array([mag[i], x, y])

# ---- Model samples
alpha = -1.5
mag_max = 24.
mag_min = 19.
fmin = gym.mag2flux_converter(mag_max)
fmax = gym.mag2flux_converter(mag_min)
mag = gym.flux2mag_converter(gen_pow_law_sample(fmin, Nobjs_model, alpha, fmax=fmax, exact=True))
# x = np.linspace(1., gym.num_rows, endpoint=False, num=13)
# y = np.linspace(1., gym.num_cols, endpoint=False, num=13)
# x, y = np.meshgrid(x, y)
# x = x.ravel() + np.random.random(size=Nobjs_model)
# y = y.ravel() + np.random.random(size=Nobjs_model)
x = np.random.random(size=Nobjs_model) * (gym.num_rows-5) + 2.5
y = np.random.random(size=Nobjs_model) * (gym.num_cols-5) + 2.5
q_model[:, 0] = mag
q_model[:, 1] = x 
q_model[:, 2] = y 

# # Generate mock data
gym.gen_mock_data(q_true)
gym.display_image()

# print "--------------- RHMC"
gym.run_RHMC(q_model, f_pos=True, delta=1e-6, Niter = Niter, Nsteps=Nsteps, \
             dt = 2e-3, save_traj=True, verbose=True, q_true = q_true)



save_dir = "./RHMC-movie/"
counter = 0
for i in xrange(0, 3000):
    for j in xrange(Nsteps+1):
        title_str = "Niter%05d-Step%03d" % (i, j)
        fname = save_dir + "slide-%07d" % counter
        gym.diagnostics_all(q_true, show=False, idx_iter = i, idx_step=j, save=fname,\
                   m=-15, b =10, s0=23, y_min=5., title_str=title_str)
        counter+=1 
        