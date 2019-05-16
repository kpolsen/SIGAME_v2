# coding=utf-8
"""
Module: aux
"""

import numpy as np
import numexpr as ne
import pandas as pd
import pdb as pdb
import scipy as scipy
from scipy import optimize
import scipy.stats as stats
import scipy.integrate as integrate
from scipy.interpolate import RegularGridInterpolator,griddata
from scipy.interpolate import interp1d
import multiprocessing as mp
import os
import time
import matplotlib.pyplot as plt
import linecache as lc
import re as re
import sys as sys
# import cPickle
import sympy as sy
import astropy as astropy


#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

def load_parameters():
    params                      =   np.load('temp_params.npy').item() # insert external parent here
    return(params)

params = load_parameters()
g = globals()
for key,val in params.items():
    exec(key + '=val',g)

#===========================================================================
""" Paths to data etc. """
#---------------------------------------------------------------------------

def get_file_location(**kwargs):
    """
    Finds correct location and file name for a certain file type and galaxy
    """
    import os

    globals()['sim_type'],globals()['ISM_phase'],globals()['ISM_dc_phase'],globals()['map_type'] = '','','tot',''

    for key,val in kwargs.items():
        exec('globals()["' + key + '"]' + '=val')

    if gal_ob_present:
        try:
            globals()['zred'] = kwargs['gal_ob'].zred
            globals()['galname'] = kwargs['gal_ob'].name
            globals()['gal_index'] = kwargs['gal_ob'].gal_index
        except:
            # Assume gal_ob is actually a dictionary
            globals()['zred'] = kwargs['gal_ob']['zred']
            globals()['galname'] = kwargs['gal_ob']['galname']
            globals()['gal_index'] = kwargs['gal_ob']['gal_index']

    # sim particle data name and location
    if sim_type != '':
        path = d_data+'particle_data/sim_data/'
        filename = os.path.join(path, 'z'+'{:.2f}'.format(zred)+'_'+galname+'_sim.'+sim_type)
        try:
            path = d_data+'particle_data/sim_data/'
            if not os.path.exists(path):
                 os.mkdir(path)
            filename = os.path.join(path, 'z'+'{:.2f}'.format(zred)+'_'+galname+'_sim.'+sim_type)
        except:
            print("Need the following to create sim data name: gal_ob, sim_type")
            raise NameError

    # ISM particle data name and location
    if ISM_phase != '':
        try:
            path = d_data+'particle_data/ISM_data/'
            if not os.path.exists(path):
                 os.mkdir(path)
            filename = os.path.join(path, 'z'+'{:.2f}'.format(zred)+'_'+galname+'_'+ISM_phase+'.h5')
        except:
            print("Need the following to create ISM data name: gal_ob, ISM_type")
            raise NameError

    # Datacube name and location
    if ISM_dc_phase != 'tot':
        try:
            target_ext = target
            if target in lines:
                target_ext = 'L_' + target

            path = d_data+'datacubes/'
            if not os.path.exists(path):
                 os.mkdir(path)

            filename = os.path.join(path, '%s_%s_i%s_%s_%s.h5' % (z1, target_ext, inc_dc, galname, ISM_dc_phase))
        except:
            print("Need the following to create datacube name: z1, target, ISM_dc_phase, inc_dc, gal_ob")
            raise NameError

    if map_type != '':
        try:
            path = parent+'sigame/temp/maps/'
            if not os.path.exists(path):
                 os.mkdir(path)
            filename = os.path.join(path, '%s_%s_G%s.h5' % (z1, map_type, gal_index+1))
        except:
            print("Need the following to create map file name: z1, map_type, gal_ob")
            raise NameError

    if debug:
        print("Debugging mode...\n Filename: {:s}...".format(filename))
        import pdb; pdb.set_trace()

    return(filename)

#===========================================================================
""" For classes in general """
#---------------------------------------------------------------------------

def get_UV_str(z1,SFRsd):
    """
    Reads in SFR surface density and compares with SFR surface density of the MW.
    Then finds nearest FUV grid point in cloudy models and returns it as a string.
    """

    if z1 == 'z6':
        UV                  =   [5]
        UV_str              =   str(int(UV[np.argmin(np.abs(np.array(UV)-SFRsd/SFRsd_MW))]))
    if z1 == 'z2':
        UV                  =   [0.001,0.02]
        UV_str              =   str(UV[np.argmin(np.abs(np.array(UV)-SFRsd/SFRsd_MW))])
    if z1 == 'z0':
        UV                  =   [0.1,0.6]
        UV_str              =   str(UV[np.argmin(np.abs(np.array(UV)-SFRsd/SFRsd_MW))])

    return(UV_str)

def update_dictionary(values,new_values):
    """ updates the entries to values with entries from new_values that have
    matching keys. """
    for key in values:
        if key in new_values:
            values[key]     =   new_values[key]
    return values

def save_temp_file(data, subgrid=None, **kwargs):
    """
    Stores temporary files according to their sim or ISM type and stage of processing.
    """
    sim_type,ISM_phase,ISM_dc_phase = '','',''

    for key,val in kwargs.items():
        exec('globals()["' + key + '"]' + '=val')
        # for some reason, the variable ISM_phase is not recognized in the interpolate step if we don't explicitly define using the following (even though we are defining it as a global variable)...
        if key == 'ISM_phase' and 'ISM_phase' in globals():
            ISM_phase = val

        if key == 'ISM_dc_phase' and 'ISM_dc_phase' in globals():
            ISM_dc_phase = val

    filename    =   get_file_location(**kwargs)

    if subgrid or sim_type != '':
        print("saving to pickle: ", filename)
        data.to_pickle(filename)

        # work around so that galaxy.add_FUV() works... while doesn't break add_GMCs() and add_dif() in the subgrid step.
        if subgrid:
            ISM_phase, ISM_dc_phase, map_type = '', '', ''

    try:
        ISM_phase
        if ISM_phase != '':
            print("saving data h5store", filename)
            h5store(data, 'data', filename, **kwargs)
    except NameError:
        ISM_phase = None

    try:
        if ISM_dc_phase != '':
            print("saving data h5store", filename)
            h5store(data, 'data', filename, **kwargs)
    except NameError:
        ISM_dc_phase = None
        print(ISM_dc_phase)

    try:
        map_type
        if map_type != '':
            print("saving data h5store", filename)
            h5store(data, 'data', filename, **kwargs)
    except NameError:
        map_type = None

def load_temp_file(**kwargs):
    """Way to load metadata with dataframe
    """

    for key,val in kwargs.items():
        exec('globals()["' + key + '"]' + '=val')

    filename    =   get_file_location(**kwargs)
    print(filename)
    try:
        data = pd.read_hdf(filename)
        try:
            data            =   data['data'][0]
        except:
            data            =   data
    except:
        try:
            data            =   pd.read_pickle(filename)
        except:
            if verbose: print('Did not find file at '+filename)
            data            =   0

    if verbose:
        if type(data) != int: print('Loaded file at %s' % filename)
    return data

def h5store(df, dc_name, filename, **kwargs):
    """Way to store metadata with dataframe
    """

    try:
        metadata            =   df.metadata
    except:
        metadata            =   {}

    store = pd.HDFStore(filename)
    store.put(dc_name, df)
    store.get_storer(dc_name).attrs.metadata = metadata
    store.close()

#===============================================================================
""" Cosmology """
#-------------------------------------------------------------------------------

def get_lum_dist(zred):
    '''
    Purpose
    ---------
    Calculate luminosity distance for a certain redshift

    returns D_L in Mpc

    '''

    from astropy.cosmology import FlatLambdaCDM
    cosmo               =   FlatLambdaCDM(H0=hubble*100., Om0=omega_m, Ob0=1-omega_m-omega_lambda)

    if len(zred) > 1:
        D_L                 =   cosmo.luminosity_distance(zred).value
        zred_0              =   zred[zred == 0]
        if len(zred_0) > 0:
            D_L[zred == 0]      =   3+27.*np.random.rand(len(zred_0)) # Mpc (see Herrera-Camus+16)

    if len(zred) == 1:
        D_L                 =   cosmo.luminosity_distance(zred).value

    # ( Andromeda is rougly 0.78 Mpc from us )

    return(D_L)

#===========================================================================
""" Some arithmetics """
#---------------------------------------------------------------------------

def rad(foo,labels):
    # Calculate distance from [0,0,0] in 3D for DataFrames!
    if len(labels)==3: return np.sqrt(foo[labels[0]]**2+foo[labels[1]]**2+foo[labels[2]]**2)
    if len(labels)==2: return np.sqrt(foo[labels[0]]**2+foo[labels[1]]**2)

def find_nearest(array,value,find='value'):
    idx = (np.abs(array-value)).argmin()
    if find == 'value': return array[idx]
    if find == 'index': return idx

def lin_reg_boot(x,y,n_boot=5000,plotting=False):
    print('Fit power law and get errors with bootstrapping!')

    n_boot                  =   5000
    nGal                    =   len(x)

    def calc_slope(i, x1, y1, nGal):

        random_i                        =   (np.random.rand(nGal)*nGal).astype(int)

        slope1,inter1,foo1,foo2,foo3    =   stats.linregress(x1[random_i],y1[random_i])
        return slope1,inter1

    slope,inter,x1,x2,x3    =   stats.linregress(x,y)
    boot_results            =   [[calc_slope(i,x1=x,y1=y,nGal=nGal)] for i in range(0,n_boot)]
    slope_boot              =   [boot_results[i][0][0] for i in range(0,n_boot)]
    inter_boot              =   [boot_results[i][0][1] for i in range(0,n_boot)]

    # if plotting:
    #     # Make plot of distribution!
    #     yr = [0,240]
    #     plot.simple_plot(fig=j+1,xlab = ' slope from bootstrapping %s times' % n_boot,ylab='Number',\
    #         xr=plot.axis_range(slope_boot,log=False),yr=yr,legloc=[0.05,0.8],\
    #         histo1='y',histo_real1=True,x1=slope_boot,bins1=n_boot/50.,\
    #         x2=[slope,slope],y2=yr,lw2=1,ls2='--',col2='blue',lab2='Slope from fit to models',\
    #         x3=[np.mean(slope_boot),np.mean(slope_boot)],y3=yr,lw3=1,ls3='--',col3='green',lab3='Mean slope from bootstrapping')
    #         # figname='plots/line_SFR_relations/bootstrap_results/'+line+'slope_boot.png',figtype='png')
    #     plt.show(block=False)

    print('Slope from fit to models: %s' % np.mean(slope))
    print('Bootstrap mean: %s' % np.mean(slope_boot))
    print('Bootstrap std dev: %s' % np.std(slope_boot))

    print('Intercept from fit to models: %s' % np.mean(inter))
    print('Bootstrap mean: %s' % np.mean(inter_boot))
    print('Bootstrap std dev: %s' % np.std(inter_boot))

    return(slope,inter,np.std(slope_boot),np.std(inter_boot))

def rotmatrix(angle,axis='x'):

    cos         =   np.cos(angle)
    sin         =   np.sin(angle)

    if axis == 'x':
        rotmatrix       =   np.array([[1,0,0],[0,cos,-sin],[0,sin,cos]])

    if axis == 'y':
        rotmatrix       =   np.array([[cos,0,sin],[0,1,0],[-sin,0,cos]])

    if axis == 'z':
        rotmatrix       =   np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])

    return rotmatrix

def annulus_area(radius,x0,y0,dr,dx):

    # get upper integration limit
    xf  =   lambda r: min([np.sqrt(r**2 - y0**2), x0+dx])
    rf  =   radius+dr

    # make lambda function for the area under the curve where the annulus intercepts a smal CC region
    f   =   lambda x, r: min([0.5*x*np.sqrt(r**2 - x**2) + np.arctan(x/np.sqrt(r**2-x**2))*r**2, y0+dx])

    # get the area where r = radius+dr
    A1  =   f(xf(rf),rf) - f(x0,rf)

    # get the area where r = radius
    A2  =   f(xf(radius),radius) - f(x0,radius)

    return abs(A1-A2)

def gauss(center,FWHM,x):
    """
    Return value(s) of Gaussian, given the

    .. note:: See https://en.wikipedia.org/wiki/Gaussian_function
    """

    std_dev     =   FWHM_to_stddev(FWHM)

    A           =   1/(std_dev*np.sqrt(2*np.pi))

    return A*np.exp(-4*np.log(2)*(x-center)**2/FWHM**2)

def FWHM_to_stddev(FWHM):

    return(FWHM/(2.*np.sqrt(2*np.log(2))))

#===============================================================================
""" Conversions """
#-------------------------------------------------------------------------------

def LsuntoJydv(Lsun,zred=7,d_L=69727,nu_rest=1900.5369):
    """ Converts luminosity (in Lsun) to velocity-integrated flux (in Jy*km/s)

    args
    ----
    Lsun: numpy array
    solar luminosity (Lsun)

    zred: scalar
    redshift z (num)

    d_L: scalar
    luminosity distance (Mpc)

    nu_rest: scalar
    rest frequency of fine emission line (GHz)

    returns
    Jy*km/s array
    ------
    """

    return Lsun * (1+zred)/(1.04e-3 * nu_rest * d_L**2)

def solLum2Jy(Lsunkms, zred, d_L, nu_rest):
    """ Converts solar luminosity/(km/s) to milli-jansky/(km/s)

    args
    ----
    Lsunkms: numpy array
    solar luminosity / vel bin ( Lsun/(km/s) )

    zred: scalar
    redshift z (num)

    d_L: scalar
    luminosity distance (Mpc)

    nu_rest: scalar
    rest frequency of fine emission line (GHz)

    returns
    Jy/(km/s) array
    ------
    """

    return Lsunkms * (1+zred)/(1.04e-3 * nu_rest * d_L**2)

def Jykm_s_to_ergs_s(Jykms, line, zred, lum_dist):
    """Converts Jy*km/s to ergs/s
    """

    L_ergs              =   Jykm_s_to_L_sun(Jykms, line, zred, lum_dist)*Lsun*1e7 # ergs/s/kpc^2

    return(L_ergs)

def Jykm_s_to_L_sun(Jykms, line, zred, lum_dist):
    """Converts Jy*km/s to solar luminosity

    Parameters
    ----------
    Jykms : scalar
        total in Jy*km/s (10^-26*W/Hz/m^2*km/s)

    line : str
        Line ID

    zred: scalar
        redshift

    lum_dist: scalar
        luminosity distance (Mpc)

    """

    f_line          =   params['f_' + line]

    # Solomon+97 eq. 1:
    return 1.04e-3 * Jykms * (f_line/(1+zred)) * lum_dist**2

def disp2FWHM(sig):
    return 2*np.sqrt(2*np.log(2)) * sig

def W_m2_to_Jykm_s(line,zred,I):
    """Converts flux in W/m^2 to velocity integrated flux [Jy km/s]
    See: https://github.com/aconley/ALMAzsearch/blob/master/ALMAzsearch/radio_units.py

    Parameters
    ----------
    line : str
        line ID
    zred: scalar
        redshift

    """

    f_line          =   params['f_' + line]*1e9
    return I * clight/1e3 / (1e-26*f_line/(1+zred))

def Jykm_s_to_W_m2(line,zred,I):
    """Converts velocity integrated flux [Jy km/s] to W/m^2
    See: https://github.com/aconley/ALMAzsearch/blob/master/ALMAzsearch/radio_units.py

    Parameters
    ----------
    line : str
        line ID
    zred: scalar
        redshift
    """

    f_line          =   params['f_' + line]*1e9
    return I * 1e-26 * f_line/(1+zred) / (clight/1e3)

def arcsec2_to_sr(arcsec2):
    """Converts area on sky in arcsec^2 to steradians
    See: http://cosmos.phy.tufts.edu/cosmicfrontier/quants.html
    """

    return(arcsec2/4.25e10)

#===============================================================================
""" Other functions """
#-------------------------------------------------------------------------------

def diff_percent(x1,x2):
    '''
    Purpose
    -------
    Return difference in percent relative to x1: (x1-x2)/x1


    '''

    diff            =   (x1-x2)/x1*100.

    return(diff)

def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold='nan')
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)

def line_name(line,latex=False):
    '''
    Purpose

    Get name for a line in pretty writing
    '''

    line_dictionary = {\
        'CII':'[CII]',\
        'OI':'[OI]',\
        'OIII':'[OIII]',\
        'NII_122':'[NII]122',\
        'NII_205':'[NII]205',\
        'CI_609':'CI(1-0)609',\
        'CI_369':'CI(2-1)369',\
        'CO32':'CO(3-2)',\
        'CO21':'CO(2-1)',\
        'CO10':'CO(1-0)'}

    return(line_dictionary[line])

def directory_checker(dirname):
    """ if a directory doesn't exist, then creates it """
    dirname =   str(dirname)
    if not os.path.exists(dirname):
        print("creating directory: %s" % dirname)
        try:
            os.mkdir(dirname)
        except:
            os.stat(dirname)

def directory_path_checker(pathway):
    """ checks that all the directories in a pathway exist; if they don't exist,
    then they are created."""

    # create and initialize list of indexes
    indexes =   []
    indexes.append( pathway.find('/') )
    index   =   indexes[0]

    # append index which marks the beginning of a new subdirectory
    while index >= 0:
        index   =   pathway.find('/',indexes[-1]+1)
        if index > 0: indexes.append(index)

    if indexes[0] == 0: indexes = indexes[1:]
    # run directory_checker for each directory in pathway
    for index in indexes:   directory_checker( pathway[:index] )

def check_version(module,version_required):

    version         =   module.__version__

    for i,subversion in enumerate(version.split('.')):
        if int(subversion) < version_required[i]:
            print('\nActive version of module %s might cause problems...' % module.__name__)
            print('version detected: %s' % version)
            print('version required: %s.%s.%s' % (version_required[0],version_required[1],version_required[2]))
            break
        if i == len(version.split('.'))-1:
            print('\nNo version problems for %s module expected!' % module.__name__)



