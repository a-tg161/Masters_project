import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as pc
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.table import vstack
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import ttest_rel
from scipy.special import erf
from scipy.special import erfc
import os


def fits_table(file_path):
        # Open the FITS file
        with fits.open(file_path) as hdul:
            # Show the HDU structure
            # hdul.info()
            # Usually the table is in extension 1
            data = hdul[1].data

        # Convert to an Astropy Table
        tbl = Table(data)
        #print("Number of objects = ", len(tbl))

        # Print column names
        # print("\nColumns:")
        # print(tbl.colnames)

        return tbl


def open_catalogue_data(path):
  """
  Reads in a catalogue of data from a FITS file and returns the data as an Astropy Table.
  """

  hdul = fits.open(path)
  hdu_names = [hdu.name for hdu in hdul]

  # sex_cat = [hdu for hdu, name in zip(hdul, hdu_names) if name == "OBJECTS"][0]
    
  # EPOCHS series (Conselice+24, Adams+24, Austin+25, Harvey+25)

  sex_tab = Table.read(path, hdu = "OBJECTS")
  sex_tab_colnames = sex_tab.colnames
  # print(sex_tab_colnames)
  # sky position (Ra = ALPHA_J2000, Dec = DELTA_J2000)
  # flux columns: FLUX_APER_{band}_aper_corr_Jy [Jy]
  # flux error columns: FLUXERR_APER_{band}_loc_depth_5pc_Jy [Jy]

  bagpipes_zfix = Table.read(path, hdu = "BAGPIPES_SFH_CONT_BURSTY_ZEAZYSFHZBLUEAGN_3.0,10.0MYR_CALZETTI_LOG_10_Z_LOG_10_BPASS_ZFIX")
  # ext 3

  bagpipes_zgauss = Table.read(path, hdu = "BAGPIPES_SFH_CONT_BURSTY_ZEAZYSFHZBLUEAGN_3.0,10.0MYR_CALZETTI_LOG_10_Z_LOG_10_BPASS_ZGAUSS_3.0SIG")
  # ext 4

  return sex_tab, bagpipes_zfix, bagpipes_zgauss


def masked_table(sex_tab, bagpipes_zfix, bagpipes_zgauss):

    log_stellar_mass_Q16 = bagpipes_zfix["stellar_mass_16"]
    log_stellar_mass_Q50 = bagpipes_zfix["stellar_mass_50"]
    log_stellar_mass_Q84 = bagpipes_zfix["stellar_mass_84"]

    stellar_mass_upper_err = log_stellar_mass_Q84 - log_stellar_mass_Q50
    stellar_mass_lower_err = log_stellar_mass_Q50 - log_stellar_mass_Q16

    
    redshift = bagpipes_zgauss["redshift_50"]

    z_mask = redshift >= 6.5

    survey_id = bagpipes_zgauss["#ID"]
    sampled_survey_id = survey_id[z_mask]
    # The ids of galaxies in zgauss table with redshift >= 6.5, which are the same in zfix

    print(np.where(bagpipes_zfix["#ID"].data == sampled_survey_id))

    zfix_sample = bagpipes_zfix[np.where(bagpipes_zfix["#ID"].data == sampled_survey_id)]

    mass_mask = (log_stellar_mass_Q50 >= 6) & (log_stellar_mass_Q50 <= 12)
    # We only want galaxies in this range of mass

    zfix_sample = zfix_sample[mass_mask]

    print(zfix_sample.colnames)

    print("Number of galaxies in our sample: ", len(zfix_sample))

    return zfix_sample


def test_plot(sample_table):

    log_stellar_mass_Q16 = sample_table["stellar_mass_16"]
    log_stellar_mass_Q50 = sample_table["stellar_mass_50"]
    log_stellar_mass_Q84 = sample_table["stellar_mass_84"]

    stellar_mass_upper_err = log_stellar_mass_Q84 - log_stellar_mass_Q50
    stellar_mass_lower_err = log_stellar_mass_Q50 - log_stellar_mass_Q16

    redshift = sample_table["redshift_50"]

    plt.errorbar(redshift, log_stellar_mass_Q50, yerr=[stellar_mass_lower_err, stellar_mass_upper_err], fmt='o', ecolor='black', alpha=0.6)
    plt.xscale('linear')
    plt.xlabel('Redshift')
    plt.ylabel('Log Stellar Mass')
    plt.title('Stellar Mass vs Redshift for Sample Galaxies')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    epochs_table_path = "/raid/scratch/work/alberttg/Masters_project/Data_inputs/EPOCHS-v2_EPOCHS_good_EAZY_sfhz_blue_agn_zfree_MASTER_Sel-F277W+F356W+F444W_v12_psfmatch_F444W_empirical+v13_psfmatch_F444W_empirical+v14_psfmatch_F444W_empirical.fits"

    sex_tab, bagpipes_zfix, bagpipes_zgauss = open_catalogue_data(epochs_table_path)

    zfix_sample = masked_table(sex_tab, bagpipes_zfix, bagpipes_zgauss)

    test_plot(zfix_sample)
