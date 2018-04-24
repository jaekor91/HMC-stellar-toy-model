# Matplot ticks
import matplotlib as mpl
mpl.rcParams['xtick.major.size'] = 15
mpl.rcParams['xtick.major.width'] = 1.
mpl.rcParams['ytick.major.size'] = 15
mpl.rcParams['ytick.major.width'] = 1.
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15

from matplotlib import ticker

import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm

from scipy.stats import norm, chi2
from matplotlib.patches import Ellipse

def mag2flux(mag):
    return 10**(0.4*(22.5-mag))

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)    

def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation


def plot_cov_ellipse(ax, mus, covs, var_num1, var_num2, MoG_color="Blue", lw=2):
    N_ellip = len(mus)
    for i in range(N_ellip):
        cov = covs[i]
        cov = [[cov[var_num1, var_num1], cov[var_num1, var_num2]], [cov[var_num2, var_num1], cov[var_num2, var_num2]]]
        mu = mus[i]
        mu = [mu[var_num1], mu[var_num2]]
        for j in [1, 2]:
            width, height, theta = cov_ellipse(cov, q=None, nsig=j)
            e = Ellipse(xy=mu, width=width, height=height, angle=theta, lw=lw)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_facecolor("none")
            e.set_edgecolor(MoG_color)

    return




# ----- Convergence statistics
def convergence_stats(q_chain, thin_rate = 5, warm_up_num = 0):
    """
    Given MCMC chain with dimension (Nchain, Niter, D) return 
    - Gelman-Rubin statistics corresponding to each variable.
    - Effective sample number for each varaible.
    """
    Nchain, Niter, D = q_chain.shape

    assert Nchain > 1 # There should be at least two chains.


    chains = [] # chains to be used to calculate the statistics.
    for m in xrange(Nchain):
        # For each chain dicard warm-up samples.
        q_chain_warmed = q_chain[m, warm_up_num:, :]

        # Thin the resulting chain.
        q_chain_warmed_thinned = q_chain_warmed[::thin_rate, :]
        L_chain = q_chain_warmed_thinned.shape[0]
        if (L_chain % 2) == 0:
            pass
        else:
            q_chain_warmed_thinned = q_chain_warmed_thinned[:L_chain-1]

        # Split the remaining chain in two parts and save.
        n = L_chain/2        
        chains.append(q_chain_warmed_thinned[:n])
        chains.append(q_chain_warmed_thinned[n:])

    m = len(chains)
    # Compute within chain variance, W
    # This an under-estimate of true variance.
    var_within = np.empty((m, D))
    for j in range(m):
        var_within[j, :] = np.std(chains[j], ddof=1, axis=0)
    W = np.mean(var_within, axis=0)

    # Compute between chain variance, B
    # This in general is an over-estimate because of the overdispersion of the starting distribution.
    mean_within = np.empty((m, D))
    for j in range(m):
        mean_within[j, :] = np.mean(chains[j], axis=0)
    mean_all = np.mean(mean_within, axis=0)
    B= np.sum(np.square(mean_within-mean_all), axis=0) * n /float(m-1)     

    # Unbiased posterior variance estimate
    var = W * (n-1)/float(n) + B / float(n)

    # Compute Gelman-Rubin statistics
    R = np.sqrt(var/W)

    #---- n_eff computation
    n_eff = np.zeros(D, dtype=float) # There is an effective number for each.
    for i in range(D): # For each variable
        # First two base cases rho_t
        V_t1 = variogram(chains, i, 1)
        V_t2 = variogram(chains, i, 2)
        rho_t1 = 1. - V_t1/(2*var[i])
        rho_t2 = 1. - V_t2/(2*var[i])
        if (rho_t1 < 5e-2) or (rho_t1 < 5e-2):
            sum_rho = 0
        else:
            rho_t = [rho_t1, rho_t2]# List of autocorrelation numbers: Unknown termination number.
            t = 1 # Current t
            while (t < n-2): # While t is less than the length of the chain
                # Compute V_t and rho_t
                V_t = variogram(chains, i, t+2)
                rho_t.append(1 - V_t/(2*var[i]))

                # Check for termination condition
                # if the sum of rho of T+1 and T+2 are negative then teriminate
                if ((t%2)==1) & ((rho_t[t]+rho_t[t+1]) < 0): # If t is odd and t
                    break

                # Otherwise just update t
                t += 1
            # Sum all rho upto maximum T
            sum_rho = np.sum(rho_t[:t])
            if sum_rho < 0:
                sum_rho = 0
        n_eff[i] = m*n/(1+2*sum_rho)# Computed n_eff

    return R, n_eff

def variogram(chains, var_num, t_lag):
    """
    Variogram as defined in BDA above (11.7).
    
    Args:
    - chains: List with dimension(m chains, (n samples, number of variables)).
    - var_num: Variable of interest
    - t_lag: Time lag
    """
    m = len(chains)
    n = chains[0].shape[0]
    V_t = 0.
    for i in range(m): # For each chain
        chain_tmp = chains[i][:, var_num] # Grab the chain
        # Compute the inner sum and add
        V_t += np.sum(np.square(chain_tmp[t_lag:]-chain_tmp[:-t_lag]))
    V_t /= float(m*(n-t_lag))

    return V_t    
 


def acceptance_rate(decision_chain, start=None, end=None):
    """
    Return acceptance given the record of decisions made
    1: Accepted
    0: Rejected
    
    start, end: Used if average should be taken in a range
    rather than full.
    """
    _, Niter, _ = decision_chain.shape            
    if start is None and end is None:
        return np.sum(decision_chain, axis=(1, 2))/Niter
    else:
        if end > 0:
            Niter = end - start 
        else:
            Niter = Niter - start
        return np.sum(decision_chain[:, start:end, :], axis=(1, 2))/Niter  



def start_pts(q0, cov0, size):
    """
    Returns one starting point from a normal distribution
    with mean "q0" and diagonal covariance "cov".
    """
    return np.random.multivariate_normal(q0, cov0, size=size)



def normal_lnL(q, q0, cov0):
    """
    Multivarite-normal lnL.
    """
    
    return multivariate_normal.logpdf(q, mean=q0, cov=cov0)


# /--- Used for NUTS termination criteria conditions
def find_next(table):
    """
    Given the save index table, return the first empty slot.
    """
    for i, e in enumerate(table):
        if e == -1:
            return i

def retrieve_save_index(table, l):
    """
    Given the save index table and the point number,
    return the save indexe corresponding to the point.s
    """
    for i, m in enumerate(table):
        if m == l:
            return i

def power_of_two(r):
    """
    Return True if r is power of two, False otherwise.
    """
    assert type(r) == int
    return np.bitwise_and(r, r-1) == 0
    
def check_points(m):
    """
    Given the current point m, return all points against which to check
    the terminiation criteria. Assumes m is even.
    """
    assert (m % 2) ==0
    r = int(m)
    # As long as r is not a power of two, keep subtracting the last possible power of two.
    d_last = np.floor(np.log2(r))
    
#     #---- Debug lines
#     counter = 0
#     print counter
#     print "r", r
#     print "d_last", d_last
#     print "\n"
    while ~power_of_two(r) and r>2:
        d_last = np.floor(np.log2(r))
        r -= int(2**d_last)
        d_last -=1
#         #---- Debug lines
#         counter +=1
#         print counter
#         print "r", r
#         print "d_last", d_last
#         print "\n"
        
    pow_tmp = np.log2(r)
    start = m-r+1
    pts = [start]
    
    tmp = start
    while pow_tmp > 1:
        pow_tmp-=1
        tmp += int(2**(pow_tmp))
        pts.append(tmp)
    
    return np.asarray(pts)


def release(m, l):
    """
    Given the current point m and that the termination condition was
    checked against l, return True if the point should no longer be saved.
    Return False, otherwise.
    """
    assert (l != 1) and (m %2) ==0
    r_m, r_l = int(m), int(l)
    d_last = np.floor(np.log2(r_m))
    while ~power_of_two(r_m) and r_m>4:
        tmp = int(2**d_last)
        r_m -= tmp
        r_l -= tmp
        d_last = np.floor(np.log2(r_m))        
    
    if (r_m >= 4) and (r_l>1):
        return True
    else:
        return False

def power_of_two_fast(r):
    """
    Return True if r is power of two, False otherwise.
    """
    # assert type(r) == int
    return np.bitwise_and(r, r-1) == 0
    
def check_points_fast(m, check_pts_cache):
    """
    Note: This just illustrates how the fucntion would work (not even correct!) if check_pts_cache
    could be updated within the function. I had to use copy/paste the code directly
    into the main function.

    Given the current point m, return all points against which to check
    the terminiation criteria. Assume r is even and m is type int.

    Use check_pts_cache to speed up the computation.
    - Calculate idx given m using idx = m/2. 
    - Compare idx to idx_max. If idx <= idx_max, then use pre-computed results from cache. 
    - Else, then compute and store the result and return.
    - The (0, 0) element is used to store idx_max.
    - The 0-th element is used to store the length of the check points array.
    - When cache is requested, check_pts_cache[1:check_pts_cache[idx, 0]+1] is returned.
    """
    # Calculate the idx
    idx = m/2

    # Compare to the max idx and proceed.
    idx_max = check_pts_cache[0, 0]
    if idx > idx_max:
        # As long as r is not a power of two, keep subtracting the last possible power of two.
        r = int(m)
        d_last = np.floor(np.log2(r))
        
        while ~power_of_two_fast(r) and r>2:
            d_last = np.floor(np.log2(r))
            r -= int(2**d_last)
            d_last -=1
            
        pow_tmp = np.log2(r)
        start = m-r+1
        pts = [start]
        
        tmp = start
        while pow_tmp > 1:
            pow_tmp-=1
            tmp += int(2**(pow_tmp))
            pts.append(tmp)
        check_pts = np.asarray(pts)
        check_pts_size = check_pts.size

        # Update the cache
        check_pts_cache[idx+1, 0] = check_pts_size
        check_pts_cache[idx+1, 1:check_pts_size+1] = check_pts
        check_points_cache[0, 0] +=1 # Max idx update.
    else:
        check_pts = check_pts_cache[1:check_pts_cache[idx, 0]+1]
    
    return answer


def release_fast(m, l):
    """
    Given the current point m and that the termination condition was
    checked against l, return True if the point should no longer be saved.
    Return False, otherwise.
    """
    # assert (l != 1) and (m %2) ==0 # Slow down
    r_m, r_l = int(m), int(l)
    d_last = np.floor(np.log2(r_m))
    while ~power_of_two_fast(r_m) and r_m>4:
        tmp = int(2**d_last)
        r_m -= tmp
        r_l -= tmp
        d_last = np.floor(np.log2(r_m))        
    
    if (r_m >= 4) and (r_l>1):
        return True
    else:
        return False

def test_NUTS_binary_tree_flatten():
    """
    Code used to test whether the auxilary functions are working well.
    """
    d = 5
    d_max = 10

    # Index table
    save_index_table = np.ones(d_max+1, dtype=int) * -1

    def print_line(m, save_index_table):
        print_line = "%2d: " % m
        for i in range(1, m+1):
            if (i in save_index_table) or (i == 1) or (i==m):
                print_line = print_line + "x "
            else:
                print_line = print_line + "o "
        print print_line
        return None

    for m in range(2, 2**d+1):
        # Decide whether to save the point for future comparison.
        if (m % 2) == 1: # Only odd numbered points are saved.
            save_index = find_next(save_index_table)
            save_index_table[save_index] = m    
            print_line(m, save_index_table)
        else:
            print_line(m, save_index_table)        
            # Check termination conditions against each point.
            check_pts = check_points(m)
            for l in check_pts:
                # Retrieve a previous point 
                save_index = retrieve_save_index(save_index_table, l)

                # If the point is no longer needed, then release the space.     
                if (l > 1) and release(m, l):
                    save_index_table[save_index] = -1
def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H

def integrate_pow_law(alpha, A, fmin, fmax):
    """
    Given power law model [alpah, A] of A x f**alpha, 
    return the integrated number.
    """
    return A * (fmax**(1 + alpha) - fmin**(1 + alpha))/(1 + alpha)


def gen_pow_law_sample(fmin, nsample, alpha, exact=False, fmax=None, importance_sampling=False, alpha_importance=None):
    """
    Note the convention f**alpha, alpha < 0
    
    If exact, then return nsample number of sample exactly between fmin and fmax.

    If importance_sampling, then generate the samples using the alpha_importance,
    and return the corresponding importance weights along with the sample.
    iw = f_sample^(alpha-alpha_importance)
    """
    flux = None
    if importance_sampling:
        if exact:
            assert (fmax is not None)
            flux = fmin * np.exp(np.log(np.random.rand(nsample))/(alpha_importance+1))
            ibool = (flux>fmin) & (flux<fmax)
            flux = flux[ibool]
            nsample_counter = np.sum(ibool)
            while nsample_counter < nsample:
                flux_tmp = fmin * np.exp(np.log(np.random.rand(nsample))/(alpha_importance+1))
                ibool = (flux_tmp>fmin) & (flux_tmp<fmax)
                flux_tmp = flux_tmp[ibool]
                nsample_counter += np.sum(ibool)
                flux = np.concatenate((flux, flux_tmp))
            flux = flux[:nsample]# np.random.choice(flux, nsample, replace=False)
            iw = flux**(alpha-alpha_importance)
        else:
            pass

        return flux, iw
    else:
        if exact:
            assert (fmax is not None)
            flux = fmin * np.exp(np.log(np.random.rand(nsample))/(alpha+1))
            ibool = (flux>fmin) & (flux<fmax)
            flux = flux[ibool]
            nsample_counter = np.sum(ibool)
            while nsample_counter < nsample:
                flux_tmp = fmin * np.exp(np.log(np.random.rand(nsample))/(alpha+1))
                ibool = (flux_tmp>fmin) & (flux_tmp<fmax)
                flux_tmp = flux_tmp[ibool]
                nsample_counter += np.sum(ibool)
                flux = np.concatenate((flux, flux_tmp))
            flux = flux[:nsample]# np.random.choice(flux, nsample, replace=False)
        else:
            assert False # This mode is not supported.
            # flux = fmin * np.exp(np.log(np.random.rand(nsample))/(alpha+1))
        
        return flux



def gauss_PSF(num_rows, num_cols, x, y, FWHM):
    """
    Given num_rows x num_cols of an image, generate PSF
    at location x, y.
    """
    sigma = FWHM / 2.354
    xv = np.arange(0.5, num_rows)
    yv = np.arange(0.5, num_cols)
    yv, xv = np.meshgrid(xv, yv) # In my convention xv corresponds to rows and yv to columns
    PSF = np.exp(-(np.square(xv-x) + np.square(yv-y))/(2*sigma**2))/(np.pi * 2 * sigma**2)

    return PSF

def poisson_realization(D0):
    """
    Given a truth image D0, make a Poisson realization of it.
    """
    D = np.zeros_like(D0)
    for i in xrange(D0.shape[0]):
        for j in xrange(D0.shape[1]):
            D[i, j] = np.random.poisson(lam=D0[i, j], size=1)
    return D


#---- Define Potential and Gradient
def V(objs_flat, D):
    """
    Negative Poisson log-likelihood given data and model.
    
    The model is specified by the list of objs, which are provided
    as a flattened list [Nobjs x 3](e.g., [f1, x1, y1, f2, x2, y2, ...])
    
    Assume a fixed background.
    """
    Nobjs = objs_flat.size // 3 # Number of objects
    Lambda = np.ones_like(D) * B_count # Model set to background
    for i in range(Nobjs): # Add every object.
        f, x, y = objs_flat[3*i:3*i+3]
        Lambda += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
    return -np.sum(D * np.log(Lambda) - Lambda)

def dVdq(objs_flat, D):
    """
    Gradient of Poisson pontentia above.    
    """
    # Place holder for the gradient.
    grad = np.zeros(objs_flat.size)
    
    # Compute the model.
    Nobjs = objs_flat.size // 3 # Number of objects
    Lambda = np.ones_like(D) * B_count # Model set to background
    for i in range(Nobjs): # Add every object.
        f, x, y = objs_flat[3*i:3*i+3]
        Lambda += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
    
    # Variable to be recycled
    rho = (D/Lambda)-1.# (D_lm/Lambda_lm - 1)
    # Compute f, x, y gradient for each object
    lv = np.arange(0, num_rows)
    mv = np.arange(0, num_cols)
    mv, lv = np.meshgrid(lv, mv)
    var = (PSF_FWHM_pix/2.354)**2
    for i in range(Nobjs):
        f, x, y = objs_flat[3*i:3*i+3]
        PSF = gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
        grad[3*i] = -np.sum(rho * PSF) # flux grad
        grad[3*i+1] = -np.sum(rho * (lv - x + 0.5) * PSF) * f / var
        grad[3*i+2] = -np.sum(rho * (mv - y + 0.5) * PSF) * f / var
    return grad





def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Taken from StackExchange
    
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def plot_range(x, factor = 2.):
    """
    Given quantity x, return a convenient plot range choice.
    """
    x_min, x_max = np.percentile(x, 5), np.percentile(x, 95)
    x_range = (x_max - x_min) * factor
    x_center = (x_min + x_max) / 2.
    x_max = x_center + x_range /2.
    x_min = x_center - x_range /2.    
    
    return x_min, x_max



def factors(num_rows, num_cols, x, y, PSF_FWHM_pix):
    """
    Test of constancy:
    dx = np.random.random() * 2 -1
    dy = np.random.random() * 2 -1

    for _ in range(100):
        print factors(64, 64, 32 + dx , 32+dy)
    """

    # Compute f, x, y gradient for each object
    lv = np.arange(0, num_rows)
    mv = np.arange(0, num_cols)
    mv, lv = np.meshgrid(lv, mv)
    var = (PSF_FWHM_pix/2.354)**2 
    PSF = gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
    PSF_sq = np.square(PSF)
    factor0 = np.sum(PSF_sq)
    factor1 = np.sum(PSF * (x - lv - 0.5)**2) / float(var **2) # sum of (dPSF/dx)^2 / PSF  
    factor2 = np.sum(PSF_sq * (x - lv - 0.5)**2) / float(var**2) # sum of (dPSF/dx)^2
    
    return factor0, factor1, factor2

def gaussian_1D(x, mu=0, sig=1):
    return np.exp(-np.square(x-mu)/ (2 * sig**2)) / (np.sqrt(2 * np.pi) * sig)