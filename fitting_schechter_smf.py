import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import scipy.constants as pc
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.table import vstack
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import ttest_rel
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator
import os
from matplotlib.gridspec import GridSpec
import h5py
import emcee
import corner


def schechter_function(x, alpha, log_mstar, log_phistar):
    """
    x is log(stellar mass) observed values.
    Output is y, where y is linear SMF.
    """

    y = np.log(10) * (10**log_phistar) * np.exp( -10**( x - log_mstar)) * ( 10**( x - log_mstar) )**(alpha + 1)

    return y


def _parse_err(err, n):
    """
    Normalize an error specification into (err_lo, err_hi) arrays of length n.

    Accepted forms for `err`:
      - None                            -> zeros (no error)
      - scalar or 1D array of length n  -> symmetric error (same as before)
      - [err_high, err_low]             -> asymmetric error; each of
        err_high/err_low can be a scalar (applied to all points) or a
        length-n array (per-point)

    Returned lo/hi are positive distances from the central value
    (lower bound = value - lo, upper bound = value + hi).
    """
    if err is None:
        return np.zeros(n), np.zeros(n)

    # Asymmetric: [err_high, err_low]
    if isinstance(err, (list, tuple)) and len(err) == 2:
        hi = np.asarray(err[0], dtype=float) 
        lo = np.asarray(err[1], dtype=float)
        if hi.ndim == 0:
            hi = np.full(n, float(hi))
        if lo.ndim == 0:
            lo = np.full(n, float(lo))
        return lo.copy(), hi.copy()

    # Symmetric: scalar or array, same as before
    arr = np.asarray(err, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    return arr.copy(), arr.copy()


def fit_smf_mcmc(x_in, y_in, x_err=None, y_err=None, x_pivot=None,
                     n_walkers=32, n_steps=5000, plot=True, save_path=None):
    """
    Bayesian gaussian fit without intrinsic scatter,
    supporting asymmetric measurement errors in both x and y (but we have none here).

    x_err / y_err each accept:
      - None, a scalar, or a 1D array of length N (symmetric errors, same
        as before), OR
      - [err_high, err_low] for asymmetric errors, where each of
        err_high/err_low is a scalar (applied to every point) or a
        length-N array (per-point).

    Asymmetric-error handling
    --------------------------
    Each data point's error distribution is modeled as a split-normal
    (two-piece Gaussian): width `sigma_lo` below the measured value and
    `sigma_hi` above it. At every MCMC step, for each point we compare the
    model prediction to the data (delta = model - data) and pick the
    error side consistent with that comparison:
        delta >= 0  ->  use the "hi" error (model sits above the data)
        delta <  0  ->  use the "lo" error (model sits below the data)
    This is applied to both the y errors and the x errors (the latter
    projected into y-space via the slope, m^2 * xerr^2, as in the
    symmetric case). This is the standard practical approximation for
    asymmetric errors in a likelihood — it is exact only when the true
    error distribution really is split-normal, but works well in general
    and reduces to the ordinary symmetric-error likelihood when lo == hi.
    The intrinsic scatter term sigma_int is kept symmetric.
    """
    # --- Step 1: Data Cleaning & Pivot Handling ---
    x = np.asarray(x_in, dtype=float)
    y = np.asarray(y_in, dtype=float)
    n = len(x)

    xe_lo, xe_hi = _parse_err(x_err, n)
    ye_lo, ye_hi = _parse_err(y_err, n)
    # xe_lo = np.zeros(n)
    # xe_hi = np.zeros(n)
    # ye_lo = np.zeros(n)
    # ye_hi = np.zeros(n)
    # My basic version of _parse_err function 

    xp = 0.0 if x_pivot is None else x_pivot

    mask = (
        np.isfinite(x) & np.isfinite(y)
        & np.isfinite(xe_lo) & np.isfinite(xe_hi)
        & np.isfinite(ye_lo) & np.isfinite(ye_hi)
    )

    print("mask: ", mask)
    print("shape of mask: ", np.shape(mask))

    x = x[mask]
    y = y[mask]
    xe_lo, xe_hi = xe_lo[mask], xe_hi[mask]
    ye_lo, ye_hi = ye_lo[mask], ye_hi[mask]
    n = len(x)
    
    
    # Guard against zero/negative error entries (keeps likelihood well-defined)
    xe_lo = np.where(xe_lo <= 0, 1e-10, xe_lo)
    xe_hi = np.where(xe_hi <= 0, 1e-10, xe_hi)
    ye_lo = np.where(ye_lo <= 0, 1e-10, ye_lo)
    ye_hi = np.where(ye_hi <= 0, 1e-10, ye_hi)


    if len(x) < 2:
        raise ValueError("Not enough finite data points.")

    # --- Step 2: Robust Initial Guess ---
    try:
        popt, pcov = curve_fit(schechter_function, x - xp, y, p0=[-2.0, 10.6, -6.5])
        guess_alpha, guess_mstar, guess_phistar = popt
        print(f"Guesses: alpha = {guess_alpha:.3f}, M_star = {guess_mstar:.3f}, phi_star = {guess_phistar:.3f}")
    
    except np.linalg.LinAlgError:
        guess_alpha, guess_log_mstar, guess_phistar = -2.0, 10.6, -6.5 # just rough values from Tom's work, these are z dependent

    # guess_std = np.std(y - gaussian_function(x-xp, guess_sig, guess_mu, guess_amp))

    # --- Step 3: MCMC Functions (3 parameters: m, c, sigma_int) ---
    def log_likelihood(theta, x, y, xerr_lo, xerr_hi, yerr_lo, yerr_hi, xp):
        alpha, log_mstar, log_phistar = theta
        # if sigma_int <= 0: this was one of the params
        #    return -np.inf

        model = schechter_function(x, alpha, log_mstar, log_phistar)
        delta = model - y  # positive => model above data

        y_err = np.where(delta >= 0, yerr_hi, yerr_lo)
        x_err = np.where(delta >= 0, xerr_hi, xerr_lo)

        n = len(x) # no. of data points

        ln_L = -0.5 * np.sum(np.log(2 * np.pi * y_err**2)) - 0.5 * np.sum(((y - model) / y_err)**2)
        # ln_L = - (n/2) * np.log(2*np.pi) - n * np.log(sig) - 0.5 * np.sum( ((delta - mu)  / sig)**2)

        return ln_L

    # Set prior limits safely outside empirical distribution boundaries, from Harvey2025
    min_alpha, max_alpha = -3.5, -1.0
    min_mstar, max_mstar = 8.0, 12.5
    min_phistar, max_phistar = -8, -2

    def log_prior(theta): 
        """
        Uses flat uniform prior.
        """
        alpha, log_mstar, log_phistar = theta
        if min_alpha < alpha < max_alpha and min_mstar < log_mstar < max_mstar and min_phistar < log_phistar < max_phistar:
            return 0 # This is return 0 for flat uniform prior
        return -np.inf

    def log_probability(theta, x, y, xerr_lo, xerr_hi, yerr_lo, yerr_hi, xp):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta, x, y, xerr_lo, xerr_hi, yerr_lo, yerr_hi, xp)
        return lp + ll if np.isfinite(ll) else -np.inf

    # --- Step 4: Run MCMC ---
    ndim = 3
    # initial = np.array([guess_sig, guess_mu, guess_amp if guess_std > 0 else 0.1]) old
    initial = np.array([guess_alpha, guess_mstar, guess_phistar])
    pos = initial + 1e-4 * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_probability,
        args=(x, y, xe_lo, xe_hi, ye_lo, ye_hi, xp)
    )
    sampler.run_mcmc(pos, n_steps, progress=True)

    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)

    x_range = np.linspace(x.min(), x.max(), 100)
    alpha_samples = flat_samples[:, 0]
    mstar_samples = flat_samples[:, 1]
    phistar_samples = flat_samples[:, 2]
    # ------------------------------------------------------ intrinsic scatter was a parameter here but we are fitting to a histogram

    models = schechter_function((x_range - xp), alpha_samples[:, np.newaxis], mstar_samples[:, np.newaxis], phistar_samples[:, np.newaxis])

    alpha_med, mstar_med, phistar_med = np.median(flat_samples, axis=0)
    lower_bound = np.percentile(models, 16, axis=0)
    upper_bound = np.percentile(models, 84, axis=0)

    # --- Step 5: Visualization ---
    if plot:
        """
        plt.figure(figsize=(8, 6))
        print(f"Median sigma: {sig_med:.3f}, mean: {mu_med:.3f}, amplitude: {amp_med:.3f}")

        # matplotlib expects asymmetric errors as [lower_distances, upper_distances]
        # ax.errorbar(x, y, xerr=[xe_lo, xe_hi], yerr=[ye_lo, ye_hi], fmt=".k", alpha=0.4, label="Data", zorder=1)
        
        plt.fill_between(x_range, lower_bound, upper_bound, color="royalblue", alpha=0.3, label=r"1$\sigma$ Confidence")
        plt.plot(x_range, gaussian_function((x_range - xp), sig_med, mu_med), color="darkblue", lw=2, label="Median Fit")
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        """

        labels = [r"alpha $\alpha$", r"$M_\star$", r"$\phi_\star$"]
        corner.corner(flat_samples, labels=labels, truths=[alpha_med, mstar_med, phistar_med], show_titles=True)
        plt.show()

    results = {
        "alpha_median": alpha_med,
        "mstar_median": mstar_med,
        "phistar_median": phistar_med,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "alpha_16": np.percentile(flat_samples[:, 0], 16),
        "alpha_84": np.percentile(flat_samples[:, 0], 84),
        "mstar_16": np.percentile(flat_samples[:, 1], 16),
        "mstar_84": np.percentile(flat_samples[:, 1], 84),
        "phistar_16": np.percentile(flat_samples[:, 2], 16),
        "phistar_84": np.percentile(flat_samples[:, 2], 84),
        "x_pivot": xp,
        "x_range": x_range,
    }

    return results

def plotting(results, ax):

    """
    Parameters
    ----------
    results : arrays
        some shit
    ax : axes
        plotting
    """

    alpha_med = results["alpha_median"]
    alpha_16 = results["alpha_16"]
    alpha_84 = results["alpha_84"]
    mstar_med = results["mstar_median"]
    mstar_16 = results["mstar_16"]
    mstar_84 = results["mstar_84"]
    phistar_med = results["alpha_median"]
    phistar_16 = results["alpha_16"]
    phistar_84 = results["alpha_84"]
    lower_bound = results["lower_bound"]
    upper_bound = results["upper_bound"]
    x_pivot = results["x_pivot"]
    x_range = results["x_range"]

    log_lower_bound = np.log10(lower_bound)
    log_upper_bound = np.log10(upper_bound)
    
    smf = schechter_function((x_range - x_pivot), alpha_med, mstar_med, phistar_med)
    log_smf = np.log10(smf)

    ax.fill_between(x_range, log_lower_bound, log_upper_bound, color="royalblue", alpha=0.3, label=r"1$\sigma$ Confidence")
    ax.plot(x_range, log_smf, color="darkblue", lw=2, label="Median Schechter Fit")