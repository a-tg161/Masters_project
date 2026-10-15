
import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy import units as u
from tqdm import tqdm
from astropy.table import Table
from pathlib import Path
from scipy import stats as scy_stats
from scipy import interpolate
import matplotlib.cm as cm

import Lovell2023_evs as evs
import Lovell2023_stats
import Lovell2023_stellar
from Lovell2023_stats import compute_conf_ints, eddington_bias
from Lovell2023_stellar import apply_fs_distribution

evs_dir = "/nvme/scratch/work/austind/evstats"

from config import home_dir, crop_spec_cat, combined_cat_path

def load_evs_data(survey_area):
    # ensure survey_area has appropriate unit dimensions
    assert survey_area.unit.is_equivalent(u.deg**2), "survey_area must have units of area (e.g., deg^2)"
    with h5py.File(f'{evs_dir}/example/data/evs_all.h5','r') as hf:
        log10m = hf['log10m'][:]
        f = hf['f'][:]
        F = hf['F'][:]
        N = hf['N'][:]
        z = hf['z'][:]
    whole_sky = 41252.96 * u.deg ** 2
    fsky = (survey_area / whole_sky).to(u.dimensionless_unscaled).value
    phi_max = evs._apply_fsky(N, f, F, fsky)
    return log10m, phi_max, z


def calc_pvalue(evs_mstar_pdf, obs_mstar_pdf, n_draw = 10_000, two_tailed=False, one_tailed_dir = "lower"):
    
    idx_ref  = np.random.randint(0, len(evs_mstar_pdf),  n_draw)
    idx_test = np.random.randint(0, len(obs_mstar_pdf), n_draw)
    delta    = obs_mstar_pdf[idx_test] - evs_mstar_pdf[idx_ref]
    #delta_obs = np.mean(obs_mstar_pdf) - np.mean(evs_mstar_pdf)
    if two_tailed:
        p = np.mean(np.abs(delta) >= 0.0)
    else:
        if one_tailed_dir == "upper":
            p = np.mean(delta >= 0.0)
        elif one_tailed_dir == "lower":
            p = np.mean(delta <= 0.0)
    assert p <= 1.0, f"p-value {p} is greater than 1.0, which is not valid."
    #p_ = np.clip(p, 1e-15, 1.0 - 1e-15)
    p_ = p
    if two_tailed:
        p_ /= 2.0
    nsigma = scy_stats.norm.ppf(1.0 - p_)
    return nsigma, p


def combine_pvalues_all(p_values, weights=None):
    """
    Combine an array of one-tailed p-values using six methods from
    Heard & Rubin-Delanchy (2018), Biometrika 105(1), 239-246.

    Methods
    -------
    Fisher   : S_F = -2 * sum(log p_i)         ~ chi2(2n) under H0
               Optimal when p-values ~ Beta(a, 1), a < 1
               → sensitive to any single very small p-value

    Pearson  : S_P = -2 * sum(log(1 - p_i))    ~ chi2(2n) under H0
               Optimal when p-values ~ Beta(1, b), b > 1
               → sensitive to p-values near 1 (i.e. anti-conservative signals)

    George   : S_G = S_F + S_P  (logit method)  ~ scaled chi2 under H0
               Optimal when p-values ~ Beta(a, b), a + b = 2
               → compromise, similar to Stouffer but heavier tails

    Stouffer : S_S = sum(Phi^{-1}(1 - p_i))    ~ N(0, n) under H0
               Optimal when underlying test statistics are Gaussian
               → natural for your nsigma values

    Edgington: S_E = sum(p_i)                   ~ approx Gaussian under H0
               Optimal when p-values ~ Beta(1, b) for small b
               → similar to Pearson for small p, more uniform sensitivity

    Tippett  : S_T = min(p_i)                   ~ Beta(1, n) under H0
               Optimal when only one signal is expected among many nulls
               → most sensitive to a single outlier
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    assert np.all((p > 0) & (p <= 1)), "p-values must be in (0, 1]"

    results = {}

    # Fisher: -2 sum log(p) ~ chi2(2n)
    S_F = -2.0 * np.sum(np.log(p))
    p_fisher = scy_stats.chi2.sf(S_F, df=2 * n)
    results['Fisher'] = p_fisher

    # Pearson: -2 sum log(1-p) ~ chi2(2n)
    S_P = -2.0 * np.sum(np.log(1.0 - p))
    p_pearson = scy_stats.chi2.sf(S_P, df=2 * n)
    results['Pearson'] = p_pearson

    # George (logit): S_G = S_F + S_P, scaled to approximate chi2
    S_G = S_F + S_P   # = -2 * sum log(p / (1-p))
    # Approximation: S_G * (5n+4) / (5n+2) / (pi^2/3) ~ chi2(2n) [H&RD 2018]
    scale = (5 * n + 4) / ((5 * n + 2) * (np.pi**2 / 3.0))
    p_george = scy_stats.chi2.sf(S_G * scale, df=2 * n)
    results['George'] = p_george

    # Stouffer (Z-method): sum Phi^{-1}(1-p) / sqrt(n) ~ N(0,1)
    # With optional weights: sum w_i * Phi^{-1}(1-p_i) / sqrt(sum w_i^2)
    z_i = scy_stats.norm.ppf(1.0 - p)   # individual z-scores (your nsigma)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        S_S = np.sum(w * z_i) / np.sqrt(np.sum(w**2))
    else:
        S_S = np.sum(z_i) / np.sqrt(n)
    p_stouffer = scy_stats.norm.sf(S_S)
    results['Stouffer'] = p_stouffer

    # Edgington: sum(p) — use normal approximation (mean n/2, var n/12)
    S_E = np.sum(p)
    p_edgington = scy_stats.norm.sf(S_E, loc=n / 2.0, scale=np.sqrt(n / 12.0))
    results['Edgington'] = p_edgington

    # Tippett: min(p) ~ Beta(1, n)
    S_T = np.min(p)
    p_tippett = scy_stats.beta.cdf(S_T, 1, n)   # P(min <= S_T) under H0
    results['Tippett'] = p_tippett

    # Convert all to one-tailed nsigma
    nsigma = {k: scy_stats.norm.ppf(1.0 - v) for k, v in results.items()}

    return results, nsigma


def get_spec_tab(
    ax: plt.Axes,
    phot_tab: Table,
    specz_column: str = "zgrade",
    sample_colname: str = "EPOCHS_good_EAZY_sfhz_blue_agn_0.32as",
    phot_zlabel = "EAZY_sfhz_blue_agn_zfree",
):
    combined_spec_tab_path = f"{home_dir}/tabs/spectra/EPOCHS-v2.fits"
    # load spec table
    spec_cat = Table.read(combined_spec_tab_path)
    print(f"Opened combined spec tab with {len(spec_cat)} rows")
    spec_cat_mask = ((spec_cat["grade"] == 3) & (spec_cat[specz_column] > 0.0) & (spec_cat["grating"] == "PRISM"))
    spec_cat = spec_cat[spec_cat_mask]
    spec_tab = crop_spec_cat(spec_tab)
    print(f"Cropped combined spec tab to {len(spec_tab)} rows with grade 3 PRISM spectra and zspec > 0.0")
    select_mask = ((spec_tab[sample_colname]) & (spec_tab[f"{phot_zlabel}_50"] > 6.5))
    select_spec_tab = spec_tab[select_mask]
    # plot only spectra with z_spec>6.5
    select_highz_spec_tab = select_spec_tab[select_spec_tab[specz_column] > 6.5]

    # cross-match with photometric table to get stellar masses
    phot_tab["ID"] = np.array([f"{row['#ID']}_{row['SURVEY']}" for row in phot_tab]).astype(str)
    phot_ids = np.array(phot_tab['ID']).astype(str)
    spec_ids = np.array(select_highz_spec_tab['phot_id']).astype(str)
    phot_mask = np.isin(phot_ids, spec_ids)
    phot_tab_matched = phot_tab[phot_mask]
    # TODO: 2 missing galaxies!!!
    return phot_tab_matched


def extract_pvals(z, z_sed, Mstar, Mstar_err, log10m_CIs, size: int = int(1e4), correct_eddington_bias: bool = True):
    p = np.zeros(len(z_sed))
    nsig = np.zeros(len(z_sed))
    for i, (z_, mstar_50, mstar_err) in tqdm(
        enumerate(zip(z_sed, Mstar, np.mean(Mstar_err, axis=0))),
        total=len(z_sed),
        desc="Calculating p-values and sigma values for each galaxy",
    ):
        # interpolate the EVS predictions to get the max stellar mass at the redshift of the galaxy
        evs_mstar_pdf = np.array([np.interp(z_, z, log10m_CIs[j]) for j in range(7)])
        # construct a (non-Gaussian) pdf for the EVS predictions at this redshift
        quantiles = scy_stats.norm.cdf([-3, -2, -1, 0, 1, 2, 3])
        inv_cdf = interpolate.PchipInterpolator(quantiles, evs_mstar_pdf)
        evs_samples = inv_cdf(np.random.uniform(0, 1, size))
        Mstar_pdf = np.random.normal(loc=mstar_50, scale=mstar_err, size = size)
        nsig_, p_ = calc_pvalue(evs_samples, Mstar_pdf, n_draw = size, two_tailed = False, one_tailed_dir = "lower")
        p[i] = p_
        nsig[i] = nsig_
    mask = ((nsig > 0.0) & (p > 0.0))
    p = p[mask]
    nsig = nsig[mask]
    return p, nsig, mask


def get_ext_src_corrs(tab: Table, sex_tab: Table, filt: str = "F444W"):
    # crop to the same galaxies as the input tab
    sex_ids = np.array(sex_tab['UNIQUE_ID']).astype(str)
    tab_ids = np.array(tab['UNIQUE_ID']).astype(str)
    mask = np.isin(sex_ids, tab_ids)
    sex_tab = sex_tab[mask]
    assert len(tab) == len(sex_tab), "Input tab and sex tab must have the same length"
    ext_src_corrs = np.array(sex_tab[f"ext_src_corr_0.32as_{filt}"]).astype(float)
    # crop ext_src_corrs between 1 and 10
    ext_src_corrs = np.clip(ext_src_corrs, 1.0, 10.0)
    return ext_src_corrs


def get_sigma_z_Mstar(z, Mstar, Mstar_err, z_norm = 6.5, Mstar_norm = 9.0):

    A = np.column_stack([np.ones_like(z), z - z_norm])#, Mstar - Mstar_norm])
    coeffs, *_ = np.linalg.lstsq(A, Mstar_err, rcond=None)
    a, b = coeffs #, c = coeffs

    # Residuals and estimate of noise variance
    Mstar_err_pred = A @ coeffs
    resid = Mstar_err - Mstar_err_pred
    dof = len(z) - len(coeffs)
    sigma2 = np.sum(resid**2) / dof   # unbiased estimate of residual variance

    # Covariance matrix of coefficients
    cov_matrix = sigma2 * np.linalg.inv(A.T @ A)

    # Standard errors = sqrt of diagonal
    se_a, se_b = np.sqrt(np.diag(cov_matrix)) # , se_c

    print(f"a = {a:.4f} ± {se_a:.4f}")
    print(f"b = {b:.4f} ± {se_b:.4f}")
    #print(f"c = {c:.4f} ± {se_c:.4f}")

    # import statsmodels.api as sm
    # model = sm.OLS(Mstar_err, A).fit()

    # print(model.summary())

    # # Or just pull out what you need:
    # print(model.params)      # a, b, c
    # print(model.bse)         # standard errors
    # print(model.conf_int())  # 95% confidence intervals

    lowz_mask = ((z > 6.5) & (z < 7.5))
    highz_mask = ((z > 11.5) & (z < 13.5))
    for mask in [lowz_mask, highz_mask]:
        scatter_ = np.percentile(Mstar_err[mask], [16, 50, 84])
        Mstar_ = np.percentile(Mstar[mask], [16, 50, 84])
        print(f"Scatter in Mstar_err: {scatter_}, Mstar: {Mstar_} for z range {z[mask].min()} < z < {z[mask].max()}")
        z_med = np.median(z[mask])
        Mstar_med = np.median(Mstar[mask])
        # get model prediction
        Mstar_err_pred = a + b * (z_med - z_norm) # + c * (Mstar_med - Mstar_norm)
        print(f"Model prediction for Mstar_err at z={z_med:.2f}, Mstar={Mstar_med:.2f}: {Mstar_err_pred:.4f}")

    return (a, b), (se_a, se_b) # , c, se_c


def plot_epochs_Mstar_z(ax: plt.Axes, z: np.ndarray, log10m_CIs: np.ndarray, cmap = cm.viridis, correct_eddington_bias: bool = True):
    # open bagpipes catalogue
    pipes_tab = Table.read(
        combined_cat_path,
        hdu = "BAGPIPES_SFH_CONT_BURSTY_ZEAZYSFHZBLUEAGN_3.0,10.0MYR_CALZETTI_LOG_10_Z_LOG_10_BPASS_ZGAUSS_3.0SIG"
    )
    sex_tab = Table.read(
        combined_cat_path,
        hdu = "OBJECTS",
    )

    pipes_cropped_spec_tab = get_spec_tab(ax, pipes_tab)
    ext_src_corrs = get_ext_src_corrs(pipes_cropped_spec_tab, sex_tab)
    z_pipes_spec = np.array(pipes_cropped_spec_tab['redshift_50']).astype(float)
    Mstar_spec = np.array(pipes_cropped_spec_tab['stellar_mass_50']).astype(float) + np.log10(ext_src_corrs)
    z_err_spec = np.array([z_pipes_spec - pipes_cropped_spec_tab['redshift_16'], pipes_cropped_spec_tab['redshift_84'] - z_pipes_spec])
    Mstar_err_spec = np.array([Mstar_spec - (pipes_cropped_spec_tab['stellar_mass_16'] + np.log10(ext_src_corrs)), (pipes_cropped_spec_tab['stellar_mass_84'] + np.log10(ext_src_corrs)) - Mstar_spec])
    if correct_eddington_bias:
        # correct spectroscopic masses for Eddington bias - use upper error as the mass error
        Mstar_spec, epsilon_spec = eddington_bias(Mstar_spec, Mstar_err_spec[1, :])
    p_spec, nsig_spec, mask_spec = extract_pvals(z, z_pipes_spec, Mstar_spec, Mstar_err_spec, log10m_CIs)
    combined_pvalues_spec, combined_nsig_spec = combine_pvalues_all(p_spec)
    
    ax.scatter(
        z_pipes_spec[mask_spec],
        Mstar_spec[mask_spec],
        color='black',
        s=30,
        alpha=0.5,
        label='EPOCHS-DR2 (spec)',
        zorder=10,
        marker='*',
    )

    ext_src_corrs = get_ext_src_corrs(pipes_tab, sex_tab)
    z_pipes = np.array(pipes_tab['redshift_50']).astype(float)
    Mstar = np.array(pipes_tab['stellar_mass_50']).astype(float) + np.log10(ext_src_corrs)
    z_err = np.array([z_pipes - pipes_tab['redshift_16'], pipes_tab['redshift_84'] - z_pipes])
    Mstar_err = np.array([Mstar - (pipes_tab['stellar_mass_16'] + np.log10(ext_src_corrs)), (pipes_tab['stellar_mass_84'] + np.log10(ext_src_corrs)) - Mstar])

    coeffs, coeff_errs = get_sigma_z_Mstar(z_pipes, Mstar, np.mean(Mstar_err, axis=0))
    breakpoint()
    
    # correct photometric masses for Eddington bias - use upper error as the mass error
    if correct_eddington_bias:
        Mstar, epsilon = eddington_bias(Mstar, Mstar_err[1, :])
    p_phot, nsig_phot, mask_phot = extract_pvals(z, z_pipes, Mstar, Mstar_err, log10m_CIs)
    try:
        combined_pvalues_phot, combined_nsig_phot = combine_pvalues_all(p_phot)
    except Exception as e:
        print(f"Error combining p-values for photometric sample: {e}")
        breakpoint()

    out_data = [combined_nsig_phot['Fisher'], combined_nsig_spec['Fisher']]
    for i, (label, nsig) in enumerate(zip(["phot", "spec"], out_data)):
        ax.text(
            0.05,
            0.15 - 0.10 * i,
            rf"$\sigma_{{\rm {label}}} = {nsig:.2f}$",
            transform=ax.transAxes,
            ha = "left",
            va = "bottom",
            fontsize = 10.0,
            linespacing = 1.5,
            bbox = dict(boxstyle = "round,pad=0.3", facecolor = "white", edgecolor = "black", alpha = 1.0)
        )
    # plot the most exteme masses as errorbars

    norm = plt.Normalize(vmin=nsig_phot.min(), vmax=nsig_phot.max())
    for i, (nsig_, z_, mstar_, zerr_, mstar_err_) in enumerate(
        zip(nsig_phot, z_pipes[mask_phot], Mstar[mask_phot], z_err[:, mask_phot].T, Mstar_err[:, mask_phot].T)
    ):
        colour = cmap(norm(nsig_))
        ax.errorbar(
            z_,
            mstar_,
            xerr = zerr_.reshape(2, 1),
            yerr = mstar_err_.reshape(2, 1),
            color = colour,
            fmt='o',
            markersize=5.0,
            ecolor='black',
            #alpha=0.5,
            label = "EPOCHS-DR2" if i == 0 else None,
            path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3.0, foreground='white')]
        )
    # scat = ax.errorbar(
    #     z_pipes[mask],
    #     Mstar[mask],
    #     xerr = z_err[:, mask],
    #     yerr = Mstar_err[:, mask],
    #     ecolor = colors,
    #     #fmt='o',
    #     #markersize=1,
    #     #color='black',
    #     #alpha=0.5,
    #     label='EPOCHS-DR2'
    # )
    # plot colorbar for the nsigma values
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('$\sigma$-value', size=15)


def lovell23_evs(f_b: float = 0.16, colors = ['brown', 'lightcoral', 'mistyrose'], correct_eddington_bias: bool = True):

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    evs_path = "../data/EVS_results.ecsv"

    if not Path(evs_path).is_file():

        zbin_areas = {
            "6.50<z<6.53": 651.3,
            "6.53<z<7.27": 652.3,
            "7.27<z<9.97": 733.7,
            "9.97<z<13.09": 743.5,
            "13.09<z<15.19": 738.3,
            "15.19<z<16.50": 737.7,
        }
        zbin_areas = {k: v * u.arcmin ** 2 for k, v in zbin_areas.items()}

        z_arr = []
        mhalo_pc = {
            n: [] for n in range(7)
        }
        baryon_pc = {
            n: [] for n in range(7)
        }

        for zbin_label, zbin_area in tqdm(
            zbin_areas.items(),
            desc="Computing EVS confidence intervals",
            total=len(zbin_areas),
        ):
            zbin_label_split = zbin_label.split("<")
            zbin = [float(zbin_label_split[0]), float(zbin_label_split[-1])]
            log10m, phi_max, z = load_evs_data(zbin_area)
            CIs = compute_conf_ints(phi_max, log10m)
            CI_baryon = CIs + np.log10(f_b)
            mstar_pdf = np.vstack([
                apply_fs_distribution(_phi_max, log10m, _N=int(1e4), method='lognormal') \
                for _phi_max in phi_max
            ])
            CIs = compute_conf_ints(mstar_pdf, log10m)

            z_mask = (z >= zbin[0]) & (z < zbin[1])
            z_arr.extend(z[z_mask])
            CI_baryon = CI_baryon[z_mask]
            CIs = CIs[z_mask]
            for n in range(7):
                mhalo_pc[n].extend(CIs[:,n])
                baryon_pc[n].extend(CI_baryon[:,n])
    
        # save the results to an ecsv file
        out_tab = Table(
            {
                "z": np.array(z_arr),
                **{f"mhalo_pc_{n}": np.array(mhalo_pc[n]) for n in range(7)},
                **{f"baryon_pc_{n}": np.array(baryon_pc[n]) for n in range(7)}
            }
        )
        out_tab.write(evs_path, overwrite=True)
    else:
        out_tab = Table.read(evs_path)
    
    z_arr = out_tab["z"]
    mhalo_pc = {
        n: out_tab[f"mhalo_pc_{n}"] for n in range(7)
    }
    baryon_pc = {
        n: out_tab[f"baryon_pc_{n}"] for n in range(7)
    }

    ax.fill_between(z_arr, np.array(mhalo_pc[0]), np.array(mhalo_pc[6]), alpha=1, color=colors[0])
    ax.fill_between(z_arr, np.array(mhalo_pc[1]), np.array(mhalo_pc[5]), alpha=1, color=colors[1])
    ax.fill_between(z_arr, np.array(mhalo_pc[2]), np.array(mhalo_pc[4]), alpha=1, color=colors[2])
    ax.plot(z_arr, np.array(mhalo_pc[3]), linestyle='dotted', c='black')
    # plot baryon mass confidence intervals
    ax.plot(z_arr, np.array(baryon_pc[6]), linestyle='dashed', color='black')

    ax.set_xlabel(f"Redshift, $z$", size=17.0)
    ax.set_ylabel('$\mathrm{log_{10}}(M^{\star}_{\mathrm{max}} \,/\, M_{\odot})$', size=15)
    #ax.text(0.05, 0.04, '$A = 90 \; \mathrm{arcmin}^2$', size=12, color='black', alpha=0.8, transform = ax.transAxes)

    # make legend
    line1 = plt.Line2D((0,1),(0,0), color=colors[0], linewidth=5)
    line2 = plt.Line2D((0,1),(0,0), color=colors[1], linewidth=5)
    line3 = plt.Line2D((0,1),(0,0), color=colors[2], linewidth=5)
    line4 = plt.Line2D((0,1),(0,0), color='black', linestyle='dotted', linewidth=2)
    line5 = plt.Line2D((0,1),(0,0), color='black', linestyle='dashed', linewidth=2)
    line_dummy = plt.Line2D((0,1),(0,0), color='white')
    leg = ax.legend(handles=[line4,line5,line_dummy,line3,line2,line1], 
            labels=['$\mathrm{med}(M^{\star}_{\mathrm{max}})$','$f_{\mathrm{b}}$ $+3\sigma$','',
                '$1\sigma$', '$2\sigma$', '$3\sigma$'],
                    frameon=False, loc='upper right', fontsize=12, ncol=2)
    vp = leg._legend_box._children[-1]._children[0] 
    for c in vp._children: c._children.reverse() 
    vp.align="right"

    ax.set_xlim(6.5, 15.0)
    ax.set_ylim(7.0, 12.0)

    # reconstruct CIs
    log10m_CIs = np.array([np.array(mhalo_pc[n]) for n in range(7)])
    plot_epochs_Mstar_z(ax, z_arr, log10m_CIs, correct_eddington_bias=correct_eddington_bias)

    save_name = "EVS.pdf"
    if correct_eddington_bias:
        save_name = save_name.replace(".pdf", "_Eddcorr.pdf")
    plt.savefig(f"{home_dir}/plots/{save_name}", dpi=300, bbox_inches='tight')
    plt.close(fig)


def jespersen25_evs(z_arr = np.linspace(6.5, 15.0, 10), n_samp = int(1e5), geometry: str = "rectangle"):
    x = np.linspace(1 / (10 * n_samp), 1 - 1 / (10 * n_samp), n_samp)
    evs = uniform.pdf(x)
    mask = evs > 1e-4
    evs = evs[mask]
    x = x[mask]
    breakpoint()
    
    # Convert angular sizes to physical distances
    d1 = dDEC * np.pi / 180 * cosmo.comoving_distance(z)
    d2 = dRA * np.pi / 180 * cosmo.comoving_transverse_distance(z)
    dz = cosmo.comoving_distance(z + z_r / 2) - cosmo.comoving_distance(z - z_r / 2)

    # Calculate overdensity volume based on selected geometry
    if geometry == 'ellipsoid':
        V_OD = 4 / 3 * np.pi * d1 * d2 * dz / (1 + z) ** 3
    elif geometry == 'cylinder':
        base = np.pi * d1 * d2
        V_OD = base * dz / (1 + z) ** 3
    elif geometry == 'rectangle':
        V_OD = dz * d1 * d2 * 8 / (1 + z) ** 3
    else:
        raise ValueError(f"Unknown overdensity geometry: {geometry}. Possible geometries are: 'ellipsoid', 'cylinder', or 'rectangle'.")

    # Initialize lists to store results
    phi_max_convolved_z = []
    phi_maxs_z = []
    smfs_z = []
    mbins_z = []
    N_trapz_z = []
    f_z = []
    # Loop over all redshifts to calculate extreme value distributions
    for z in tqdm(z_arr, total=len(z_arr)):
        # Calculate the extreme value statistics for the current redshift
        pm, smfs, mbins, N_trapz, f = evs_clustering(cv_df, x, V = V_OD.value, z = z)
        # Mask out bins where the output is NaN (numerical stability)
        mask = ~np.any(np.isnan(pm), axis=0)
        # Select the valid extreme value probabilities
        evs_m = evs[mask]
        # Calculate the convolved maximum PDF at this redshift
        pdf_norm = np.sum(pm[:, mask] * evs_m, axis=1) / np.sum(evs_m)
        # Store results for this redshift
        phi_max_convolved_z.append(pdf_norm)
        phi_maxs_z.append(pm)
        smfs_z.append(smfs)
        mbins_z.append(mbins)
        N_trapz_z.append(N_trapz)
        f_z.append(f)

    # Make a 'data' directory if it doesn't exist
    if not os.path.exists('data'):
        os.mkdir('data')

    # Save all results to an HDF5 file for later analysis
    with h5py.File('data/' + out_str + '.h5', 'w') as hf:
        hf.create_dataset('log10m', data=mbins)  # Mass bins
        hf.create_dataset('z', data=z_arr)  # Redshift array
        hf.create_dataset('smf', data=np.array(smfs_z, dtype=np.float32))  # Stellar mass functions
        hf.create_dataset('phi_max_conv', data=phi_max_convolved_z)  # Convolved maximum PDFs
        hf.create_dataset('phi_maxs', data=np.array(phi_maxs_z, dtype=np.float32))  # Maximum PDFs
        hf.create_dataset('evs', data=evs)  # Extreme value sampling grid
        hf.create_dataset('f', data=[f])  # Bin correction factor
        hf.create_dataset('N_trapz', data=N_trapz_z)  # Number counts in each mass bin

    print(f'Results saved to data/{out_str}.h5')

    return {
        'mbins': mbins,
        'z': z_arr,
        'smfs': smfs_z,
        'phi_max_conv': phi_max_convolved_z,
        'phi_maxs': phi_maxs_z,
        'evs': evs,
        'f': f_z,
        'N_trapz': N_trapz_z
    }
    

if __name__ == "__main__":
    for correct_eddington_bias in [True, False]:
        lovell23_evs(correct_eddington_bias=correct_eddington_bias)