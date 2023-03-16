"""

Senior Honours Project: The Near Infrared Properties of Galactic Variables

Author: Selin Ezgi Birincioglu
Student number: s1978067
"""


#importing modules

from astropy.io import fits
from astropy.table import Table
from astropy.io.ascii import write
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import multiprocessing as mp


# by finding the crossmatches between GAIA and SQL query, can get the magnitude and period values from VVV
def find_ID(GAIA_sourceID, results_ID, results_KsMeanMag, results_KsMagRMS, results_JMeanMag, 
results_JMagRMS, results_pixelID, GAIA_r_med_photogeo, GAIA_r_lo_photogeo, GAIA_r_hi_photogeo): 
    
    """
    This function crossmatches the Gaia IDs from the catalogue by Bailer-Jones et al. (2018) with the 
    SLAVEOBJIDs to determine the stars that can be found in the Gaia catalogue to get respective 
    measurements like Ks and J magnitudes, upper-lower-median distances. This is only done to 
    stars that are 10kpc away, because distance error get larger after that point.
    
    Parameters
    ----------  
    GAIA_sourceID (column): Source IDs of the stars from the Gaia catalogue
    results_ID (column): Source IDs of the stars from VVV
    results_KsMeanMag (column): Ks band magnitudes of the stars chosen from VVV
    results_KsMagRMS (column): Ks band RMS magnitudes of the stars chosen from VVV
    results_JMeanMag (column): J band magnitudes of the stars chosen from VVV
    results_JMagRMS (column): J band RMS magnitudes of the stars chosen from VVV
    results_pixelID (column): Pixel IDs of stars chosen from VVV
    GAIA_r_med_photogeo (column): Median distance measurement in the Gaia catalogue 
    GAIA_r_hi_photogeo (column): Higher bounds of distance measurement in the Gaia catalogue 
    GAIA_r_lo_photogeo (column): Higher bounds of distance measurement in the Gaia catalogue 
        
    Returns
    ------- 
    GAIA_ID (array): Source IDs of the stars that have been chosen from VVV and available in Gaia
    Ks_MeanMag (array): Ks band magnitudes of the stars chosen from VVV and available in Gaia
    Ks_MagRMS (array): Ks band RMS magnitudes of the stars chosen from VVV and available in Gaia
    J_MeanMag (array): J band magnitudes of the stars chosen from VVV and available in Gaia
    J_MagRMS (array): J band RMS magnitudes of the stars chosen from VVV and available in Gaia
    pixelID (array): Pixel IDs of stars chosen from VVV and available in Gaia
    GAIA_distance (array): Median distance measurement from Gaia of the stars chosen from VVV
    GAIA_lo (array): Lower distance measurement from Gaia of the stars chosen from VVV
    GAIA_hi (array): Higher distance measurement from Gaia of the stars chosen from VVV
    """
    
    # defining empty arrays to fill
    GAIA_ID = np.array([])                      
    Ks_MeanMag = np.array([])                        
    Ks_MagRMS = np.array([])                         
    J_MeanMag = np.array([])
    J_MagRMS = np.array([])   
    pixelID = np.array([])
    GAIA_distance = np.array([])
    GAIA_lo = np.array([])
    GAIA_hi = np.array([])
    i = 0
    
    # the loop goes through every data point in GAIA_sourceID to find matches with resultsAB_ID so
    # respective values can be appended to the lists. this is only done to distances less than 
    # 10kpc since distance errors get bigger after 10 kpc and there are no extinction values
    for ID in GAIA_sourceID:
        
         if i == 1000: #len(results_ID):
             break
         
         else:
             
            for a, b, c, d, e, f, g, h, j in zip(results_ID, results_KsMeanMag, results_KsMagRMS, results_JMeanMag,
                                                             results_JMagRMS, results_pixelID, GAIA_r_med_photogeo, 
                                                                           GAIA_r_lo_photogeo, GAIA_r_hi_photogeo):
                
                if ID == a and 0 < g <= 10000:
                    
                    GAIA_ID = np.append(GAIA_ID, a)
                    Ks_MeanMag = np.append(Ks_MeanMag, b)
                    Ks_MagRMS = np.append(Ks_MagRMS, c)
                    J_MeanMag = np.append(J_MeanMag, d)
                    J_MagRMS = np.append(J_MagRMS, e)
                    pixelID = np.append(pixelID, f)
                    GAIA_distance = np.append(GAIA_distance, g)
                    GAIA_lo = np.append(GAIA_lo, h)
                    GAIA_hi = np.append(GAIA_hi, j)
                    i += 1
                    
    return GAIA_ID, Ks_MeanMag, Ks_MagRMS, J_MeanMag, J_MagRMS, pixelID, GAIA_distance, GAIA_lo, GAIA_hi



# finding extinction of the pixels in Ks band
def find_extinctionKs(pixelID, GAIA_distance, extinction_data):
    
    """
    This function crossmatches pixel IDs and the distances from the 3D extinction map of VVV which was 
    made by Chen et. al (2013) to get the extinction values (colour excess) of the pixels, therefore 
    relative stars. The extinction measurements are listed in 5 kpc incriments until 10 kpc. The distances
    are converted to pc and since the stars have float distances, the extinction is found with the actual 
    distance, the interval of distance and the relative extinction values. Mostly E(J-Ks) values are used, 
    if they are no available E(H-Ks) is used. These values are multiplied by respective Nihiyama et al. 
    (2009) extinction coefficients for Ks band.
    
    Parameters
    ---------- 
    GAIA_distance (array): Median distance measurement from Gaia of the stars chosen from VVV
    pixelID (array): Pixel IDs of stars chosen from VVV and available in Gaia
    extinction_data["PIXELID"] (column): Pixel IDs of the 3D extinction map of VVV
    extinction_data["R"] (column): Distance of the pixel in the 3D extinction map pf VVV
    extinction_data["EJKS"] (column): E(J-Ks) values of the 3D extinction map of VVV
    extinction_data["EHKS"] (column): E(H-Ks) values of the 3D extinction map of VVV
    
    Methods
    -------
    Mask is used to crossmatch pixel IDs of the stars to the 3D extinction maps. It returns boolean values; 
    True if the pixel IDs match, False if they do not.
    
    Returns
    -------
    eks: extinction in the wavelength of the filter.
    """
    
    # mask will be a boolean when crossmatching pixelIDs in catalogues
    mask = extinction_data["PIXELID"] == pixelID

    # using 3 distances: real distance and the interval its in
    for j, dist in enumerate(extinction_data["R"][mask]):   
                                 
            if (GAIA_distance >= 1000 * dist and GAIA_distance < 1000 * extinction_data["R"][mask][j-1]): 
                
                eks = 0.544696 * (extinction_data["EJKS"][mask][j] + (extinction_data["EJKS"][mask][j-1] 
                    - extinction_data["EJKS"][mask][j]) * ((GAIA_distance - 1000 * extinction_data["R"][mask][j]) 
                    / (1000 * extinction_data["R"][mask][j-1] - 1000 * extinction_data["R"][mask][j])))
                                                           
            if extinction_data["EJKS"][mask][j] == 0 and (1000 * extinction_data["R"][mask][j-1] >
                                                                    GAIA_distance >= 1000 * dist):
                
                eks = 1.666067 * (extinction_data["EHKS"][mask][j] + (extinction_data["EHKS"][mask][j-1]
                    - extinction_data["EHKS"][mask][j]) * ((GAIA_distance - 1000 * extinction_data["R"][mask][j])
                    / (1000 * extinction_data["R"][mask][j-1] - 1000 * extinction_data["R"][mask][j])))
                                                           
    return eks



# finding extinction of the pixels in J band
def find_extinctionJ(pixelID, GAIA_distance, extinction_data):
    
    """
    This function crossmatches pixel IDs and the distances from the 3D extinction map of VVV which was 
    made by Chen et. al (2013) to get the extinction values (colour excess) of the pixels, therefore 
    relative stars. The extinction measurements are listed in 5 kpc incriments until 10 kpc. The distances
    are converted to pc and since the stars have float distances, the extinction is found with the actual 
    distance, the interval of distance and the relative extinction values. Mostly E(J-Ks) values are used, 
    if they are no available E(H-Ks) is used. These values are multiplied by respective Nihiyama et al. 
    (2009) extinction coefficients for J band.
    
    Parameters
    ---------- 
    GAIA_distance (array): Median distance measurement from Gaia of the stars chosen from VVV
    pixelID (array): Pixel IDs of stars chosen from VVV and available in Gaia
    extinction_data["PIXELID"] (column): Pixel IDs of the 3D extinction map of VVV
    extinction_data["R"] (column): Distance of the pixel in the 3D extinction map pf VVV
    extinction_data["EJKS"] (column): E(J-Ks) values of the 3D extinction map of VVV
    extinction_data["EHKS"] (column): E(H-Ks) values of the 3D extinction map of VVV
    
    Methods
    -------
    Mask is used to crossmatch pixel IDs of the stars to the 3D extinction maps. It returns boolean values; 
    True if the pixel IDs match, False if they do not.
    
    Returns
    -------
    ej: extinction in the wavelength of the filter.
    """
    
    # mask will be a boolean when crossmatching pixelIDs in catalogues
    mask = extinction_data["PIXELID"] == pixelID

    # using 3 distances: real distance and the interval its in
    for j, dist in enumerate(extinction_data["R"][mask]): 
                                   
            if (GAIA_distance>=1000*dist and GAIA_distance<1000*extinction_data["R"][mask][j-1]): 
                
                ej = 1.575652 * (extinction_data["EJKS"][mask][j] + (extinction_data["EJKS"][mask][j-1]
                   - extinction_data["EJKS"][mask][j]) * ((GAIA_distance - 1000 * extinction_data["R"][mask][j])
                   / (1000 * extinction_data["R"][mask][j-1] - 1000 * extinction_data["R"][mask][j])))
                                                          
            if extinction_data["EJKS"][mask][j] == 0 and (1000 * extinction_data["R"][mask][j-1] >
                                                                         GAIA_distance>=1000*dist):
                
                ej = 4.819467 * (extinction_data["EHKS"][mask][j] + (extinction_data["EHKS"][mask][j-1 ]
                   -extinction_data["EHKS"][mask][j]) * ((GAIA_distance - 1000 * extinction_data["R"][mask][j])
                   / (1000 * extinction_data["R"][mask][j-1] - 1000 * extinction_data["R"][mask][j])))
    
    return ej



# finding x and y values for the plot
def find_xy(eks_list, ej_list, Ks_MeanMag, Ks_MagRMS, J_MeanMag, J_MagRMS, GAIA_distance, GAIA_lo, GAIA_hi):
    
    """
    This function finds literature magnitudes, absolute magnitude, period and errors of the stars that have 
    exticnction for the main graph. Error for absolute magnitudes are asymmetric so the errors are seperated
    into up and down error bars.
    
    Parameters
    ---------- 
    ejks_list (array): list of extinction of the stars chosen from VVV
    Ks_MeanMag (array): Ks band magnitudes of the stars chosen from VVV
    GAIA_distance (array): Median distance measurement from Gaia of the stars chosen from VVV
    Ks_Period (array): Literature period values of the stars chosen from VVV
    Ks_MagRMS (array): Ks band RMS magnitudes of the stars chosen from VVV
    GAIA_lo (array): Lower distance measurement from Gaia of the stars chosen from VVV
    GAIA_hi (array): Higher distance measurement from Gaia of the stars chosen from VVV
    
    Returns
    -------
    x (list): Absolute Ks band magnitudes of the stars chosen from VVV
    y (list): Literature period values of the stars chosen from VVV
    yerrup (list): Upper error of the absolute Ks band magnitudes of the stars chosen from VVV
    yerrdown (list): Lower error of the absolute Ks band magnitudes of the stars chosen from VVV
    litmag (list): Literature absolute Ks band magnitudes of the stars from Catelan et al. (2004) paper
    """
    
    # defining empty arrays to fill
    yerrup = []
    yerrdown = []
    d = []
    RMS = []
    lo = []
    hi = []
    litmag = []
    x = []
    y = []
    
    # find the absolute magnitudes with the formula M = m - 5 * log10(d/10) - A_lambda and Ks-J magnitudes
    for k, l, m, n, o, p, q, r, s in zip(eks_list, ej_list, Ks_MeanMag, Ks_MagRMS, J_MeanMag, J_MagRMS, GAIA_distance,
                                                                                                    GAIA_lo, GAIA_hi):
        
        if k != 0.:
            
            x.append((o-(5*np.log10(q/10))-l)-(m-(5*np.log10(q/10))-k))
            y.append(m-(5*np.log10(q/10))-k)
            d.append(q)

    return x, y, d



# main function
def main():
    
    """
    FITS
    ----
    GaiaDistViva2.fits: GAIA survey data which includes Source ID, Right Ascension, Declination, r_med_photogeo, 
    r_lo_photogeo, r_hi_photogeo
    RRAB.fits: vivaID, cuEventID, literatureID, crossPeriod, mainVarType, literatureVarType, KsMeanMag, KsMagRMS, 
    masterObjIDs, slaveObjIDs
    RRC.fits: vivaID, cuEventID, literatureID, crossPeriod, mainVarType, literatureVarType, KsMeanMag, KsMagRMS, 
    masterObjIDs, slaveObjIDs

    Methods
    -------
    fits = fits.open("fits_name"): Opens the fits files as variables
    fits_data = Table(fits[1].data): Creates a table from the fits files
    
    Parameters
    ---------- 
    GAIA (list): Variable for GaiaDistViva2.fits
    GAIA_data (Table): Table for GaiaDistViva2.fits
    GAIA_sourceID (column): Source IDs of the stars from the GAIA catalogue
    GAIA_r_med_photogeo (column): Median distance of the stars from the GAIA catalogue
    GAIA_r_lo_photogeo (column): Lo distance of the stars from the GAIA catalogue
    GAIA_r_hi_photogeo (column): Hi distance of the stars from the GAIA catalogue
        
    resultsAB_data (Table): Table for RRAB.fits
    resultsAB_ID (column): Source IDs of the stars chosen from VVV
    resultsAB_KsMeanMag (column): Ks band magnitudes of the stars chosen from VVV
    resultsAB_KsMagRMS (column): Ks band RMS magnitudes of the stars chosen from VVV
    resultsAB_JMeanMag (column): J band magnitudes of the stars chosen from VVV
    resultsAB_JMagRMS (column): J band RMS magnitudes of the stars chosen from VVV
    resultsAB_pixelID (column): Pixel ID of the stars chosen from VVV
    
    resultsC_data (Table): Table for RRC.fits
    resultsC_ID (column): Source IDs of the stars chosen from VVV
    resultsC_KsMeanMag (column): Ks band magnitudes of the stars chosen from VVV
    resultsC_KsMagRMS (column): Ks band RMS magnitudes of the stars chosen from VVV
    resultsC_JMeanMag (column): J band magnitudes of the stars chosen from VVV
    resultsC_JMagRMS (column): J band RMS magnitudes of the stars chosen from VVV
    resultsC_Period (column): Literature period values of the stars chosen from VVV
    resultsC_pixelID (column): Pixel ID of the stars chosen from VVV
    
    Returns
    -------
    the plot: the plot shows the absolute Ks magnitudes (y-axis) with respect to the difference
    of the absolute Ks and J magnitudes (x-axis), which is a colour - magnitude diagram
    
    """

    resultsAB = fits.open("RRABcolour.fits")
    resultsAB_data = Table(resultsAB[1].data)
    
    resultsC = fits.open("RRCcolour.fits")
    resultsC_data = Table(resultsC[1].data)
    
    GAIA = fits.open("GaiaDistViva2.fits")
    GAIA_data = Table(GAIA[1].data)
    
    extinction = fits.open("extinction.fits")
    extinction_data = Table(extinction[1].data)
    
    
    resultsAB_ID = resultsAB_data["SLAVEOBJID"]
    resultsAB_KsMeanMag = resultsAB_data["KSMEANMAG"]
    resultsAB_KsMagRMS = resultsAB_data["KSMAGRMS"]
    resultsAB_JMeanMag = resultsAB_data["JMEANMAG"]
    resultsAB_JMagRMS = resultsAB_data["JMAGRMS"]
    resultsAB_pixelID = resultsAB_data["EXTPIXELID"]
    
    resultsC_ID = resultsC_data["SLAVEOBJID"]
    resultsC_KsMeanMag = resultsC_data["KSMEANMAG"]
    resultsC_KsMagRMS = resultsC_data["KSMAGRMS"]
    resultsC_JMeanMag = resultsC_data["JMEANMAG"]
    resultsC_JMagRMS = resultsC_data["JMAGRMS"]
    resultsC_pixelID = resultsC_data["EXTPIXELID"]
    
    GAIA_sourceID = GAIA_data["source_id"]
    GAIA_r_med_photogeo = GAIA_data["r_med_photogeo"]
    GAIA_r_lo_photogeo = GAIA_data["r_lo_photogeo"]
    GAIA_r_hi_photogeo = GAIA_data["r_hi_photogeo"]
    
    
    # for RRAB
    ABGAIA_ID, ABKs_MeanMag, ABKs_MagRMS, ABJ_MeanMag, ABJ_MagRMS, ABpixelID, ABGAIA_distance, ABGAIA_lo, ABGAIA_hi  = find_ID(
                   GAIA_sourceID, resultsAB_ID, resultsAB_KsMeanMag, resultsAB_KsMagRMS, resultsAB_JMeanMag, resultsAB_JMagRMS, 
                                                resultsAB_pixelID, GAIA_r_med_photogeo, GAIA_r_lo_photogeo, GAIA_r_hi_photogeo)
    
    ABeks_list = np.array([find_extinctionKs(pixID, gaiaDist, extinction_data) for pixID,gaiaDist in zip(ABpixelID, 
                                                                                                 ABGAIA_distance)])
    
    ABej_list = np.array([find_extinctionJ(pixID, gaiaDist, extinction_data) for pixID,gaiaDist in zip(ABpixelID, 
                                                                                                ABGAIA_distance)])
    
    ABx, ABy, ABd = find_xy(ABeks_list, ABej_list, ABKs_MeanMag, ABKs_MagRMS, ABJ_MeanMag, ABJ_MagRMS, ABGAIA_distance, 
                                                                                                  ABGAIA_lo, ABGAIA_hi)
    
    
    # for RRc
    CGAIA_ID, CKs_MeanMag, CKs_MagRMS, CJ_MeanMag, CJ_MagRMS, CpixelID, CGAIA_distance, CGAIA_lo, CGAIA_hi  = find_ID(
               GAIA_sourceID, resultsC_ID, resultsC_KsMeanMag, resultsC_KsMagRMS, resultsC_JMeanMag, resultsC_JMagRMS, 
                                        resultsC_pixelID, GAIA_r_med_photogeo, GAIA_r_lo_photogeo, GAIA_r_hi_photogeo)
    
    Ceks_list = np.array([find_extinctionKs(pixID, gaiaDist, extinction_data) for pixID,gaiaDist in zip(CpixelID, 
                                                                                                CGAIA_distance)])
    
    Cej_list = np.array([find_extinctionJ(pixID, gaiaDist, extinction_data) for pixID,gaiaDist in zip(CpixelID,
                                                                                              CGAIA_distance)])
    
    Cx, Cy, Cd = find_xy(Ceks_list, Cej_list, CKs_MeanMag, CKs_MagRMS, CJ_MeanMag, CJ_MagRMS, CGAIA_distance, CGAIA_lo, 
                                                                                                              CGAIA_hi)
            
    
    plt.figure(dpi=500)
    
    # for RRAB
    plt.scatter(ABx, ABy, s=10, alpha=0.5, c="red")
    
    # for RRC
    plt.scatter(Cx, Cy, s=10, alpha=0.5, c="blue")
    
    # plot properties
    plt.title("Colour - Magnitude Diagram of RR Lyraes")
    plt.xlabel("J - Ks")
    plt.ylabel("Ks")
    plt.gca().invert_yaxis()
    plt.ylim(10, -10)
    plt.xlim(-0.2, 2)
    plt.legend()
      
main()