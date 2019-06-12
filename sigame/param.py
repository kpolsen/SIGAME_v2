# coding=utf-8
"""
Submodule: param
"""

import linecache as lc
import numpy as np
import re
import pandas as pd
import periodictable as per
import matplotlib.colors as colors
import pdb
from argparse import Namespace

def read_params(params_file,params):
    '''Extracts parameters from parameter file set in sigame/__init__.py
    '''

    #===========================================================================
    """ Read from parameters_z*.txt """
    #---------------------------------------------------------------------------

    file                =   open(params_file,'r')
    lc.clearcache()

    parameter_list      =   ['nGal','ow','halos','snaps','haloIDs','z1',\
                            'hubble','omega_m','omega_lambda','omega_r',\
                            'simtype','d_sim',\
                            'v_res','v_max','x_res_pc','x_max_pc','inc_dc',\
                            'simtype','ext_DENSE','ext_DIFFUSE','target','lines',\
                            'frac_h','f_R_gal','d_XL_data','N_cores']

    for i,line in enumerate(file):
        for parameter in parameter_list:
            if line.find('['+parameter+']') >= 0:
                params[parameter]   =   re.sub('\n','',lc.getline(params_file,i+2))

    file.close()

    #===========================================================================
    """ Convert parameters to int/float/lists C """
    #---------------------------------------------------------------------------

    params['zred']          =   float(params['z1'])
    params['nGal']          =   int(params['nGal'])
    if params['ow'] == 'yes': params['ow'] = True
    if params['ow'] == 'no': params['ow'] = False
    params['z1']            =   'z'+str(int(params['z1']))
    params['hubble']        =   float(params['hubble'])
    params['omega_m']       =   float(params['omega_m'])
    params['omega_lambda']  =   float(params['omega_lambda'])
    params['omega_r']       =   float(params['omega_r'])
    params['frac_h']        =   float(params['frac_h'])
    params['f_R_gal']       =   float(params['f_R_gal'])
    params['v_res']         =   float(params['v_res'])
    params['v_max']         =   float(params['v_max'])
    params['x_res_pc']      =   float(params['x_res_pc'])
    params['x_max_pc']      =   float(params['x_max_pc'])
    params['N_cores']       =   int(params['N_cores'])

    # Always extract the z-component of velocity as Line-of-Sight
    params['los_dc'] = 'z'
    # By how much should the galaxy datacube be rotated from face-on around y axis?
    if 'inc_dc' in params:
        inc_dc              =   re.findall(r'\w+',params['inc_dc'])
    if 'haloIDs' in params:
        haloIDs             =   re.findall(r'\w+',params['haloIDs'])
        params['haloIDs']   =   [int(haloIDs[i]) for i in range(0,len(haloIDs))]
    if 'lines' in params:
        lines               =   params['lines']
        lines               =   lines.replace('[','')
        lines               =   lines.replace(']','')
        lines               =   lines.replace(' ','')
        lines               =   lines.replace("'","")
        lines               =   lines.split(',')
        params['lines']     =   lines
    if 'target' in params:
        lines               =   params['target']
        lines               =   lines.replace('[','')
        lines               =   lines.replace(']','')
        lines               =   lines.replace(' ','')
        lines               =   lines.replace("'","")
        # lines               =   lines.split(',')
        params['target']    =   lines

    #===========================================================================
    """ Set directories """
    #---------------------------------------------------------------------------

    # Where large data is stored (alternative SIGAME_dev folder)
    try:
        d_XL_data               =   params['d_XL_data']
    except:
        pass

    # Where temporary files are stored:
    params['d_temp']            =   'sigame/temp/'
    # Where simulations are
    # params['d_sim']         =   d_XL_data+'sim/'
    # Where cloudy and cloudy models are:
    params['d_cloudy_models']   =   params['d_temp']+'%s_data_files/cloud_models/' % params['z1']
    # Where starburst is:
    # params['d_FUV_models']      =   'Tables/starburst99/output/'
    # Where tables are:
    params['d_t']               =   'Tables/'
    # Where data files are stored
    # params['d_data']        =   params['d_temp'] + 'data/'
    # Where global results are stored:
    # params['d_glo_res']     =   params['d_temp'] + 'global_results/'
    # Where diffuse and dense gas emission results are stores (still in use?)
    # params['dif_path']      =   d_XL_data+'sigame/temp/dif/emission/'+params['ext_DIFFUSE']+'/'
    # params['GMC_path']      =   d_XL_data+'sigame/temp/GMC/emission/'+params['ext_DENSE']+'/'
    params['d_sb']              =   '../sb99/'

    # do we need this one?
    params['galaxy_sample_root']    =   params['d_temp'] + 'galaxy_samples/'

    #===========================================================================
    """ Global results save file options """
    #---------------------------------------------------------------------------

    params['global_save_file']      =    params['d_temp']+'global_results/'+params['z1']+'_'+str(params['nGal'])+'gals'+params['ext_DENSE']+params['ext_DIFFUSE']
    params['global_save_files']     =    {'0':params['d_temp']+'global_results/z0_22gals_abun_abun','2':params['d_temp']+'global_results/z2_30gals_abun_abun','6':params['d_temp']+'global_results/z6_30gals_abun_abun'}
    # sample list of different redshifts
    gr_home                         =   'sigame/temp/global_results/'
    gr_names                        =   np.array(['z0_22gals_abun_abun','z2_30gals_abun_abun','z6_30gals_abun_abun'], dtype='O')
    params['sample_list']           =   np.array([ gr_home + name for name in gr_names ])
    params['d_cloud_models']        =    params['d_data']+'cloud_models/'
    params['d_cloud_profiles']      =    params['d_data']+'cloud_profiles/'


    #===========================================================================
    """ Run options for SÍGAME """
    #---------------------------------------------------------------------------

    run_options         =   ['extract','subgrid','interpolate','datacubes','regions','PCR_analysis']

    file                =   open(params_file,'r')
    lc.clearcache()
    for i,line in enumerate(file):
        for run_option in run_options:
            if line.find(run_option) >= 0:
                line1                   =   lc.getline(params_file,i+1)
                params['do_'+run_option]    =   False
                if line1[0:2] == '+1': params['do_'+run_option] = True

    #===========================================================================
    """ Print chosen parameters """
    #---------------------------------------------------------------------------

    print('\n' + (' Parameters chosen').center(20+10+10+10))
    print('+%20s+%10s+%15s+%50s+' % ((20*'-'), (10*'-'), (15*'-'), (50*'-')))
    print('|%20s|%10s|%15s|%50s|' % ('Parameter'.center(20), 'Value'.center(10), 'Name in code'.center(15), 'Explanation'.center(50)))
    print('+%20s+%10s+%15s+%50s+' % ((20*'-'), (10*'-'), (15*'-'), (50*'-')))
    print('|%20s|%10g|%15s|%50s|' % ('Repr. redshift'.center(20), params['zred'], 'zred'.center(15),'Redshift of simulation snapshot'.center(50)))
    print('|%20s|%10g|%15s|%50s|' % ('# galaxies'.center(20), params['nGal'], 'nGal'.center(15),'Number of galaxies in redshift sample'.center(50)))
    print('|%20s|%10s|%15s|%50s|' % ('Dense cloudy grid'.center(20), params['ext_DENSE'], 'ext_DENSE'.center(15),'Extension of desired GMC model grid'.center(50)))
    print('|%20s|%10s|%15s|%50s|' % ('Diffuse cloudy grid'.center(20), params['ext_DIFFUSE'], 'ext_DIFFUSE'.center(15),'Extension of desired HIM model grid'.center(50)))
    print('|%20s|%10g|%15s|%50s|' % ('Fraction of h'.center(20), params['frac_h'], 'frac_h'.center(15),'GMCs are distributed < frac_h * h'.center(50)))
    print('+%20s+%10s+%15s+%50s+' % ((20*'-'), (10*'-'), (15*'-'), (50*'-')))

    print('\nThis is what sigame.run() is set up to do (change in parameter file):')
    if params['do_extract']:            print('- Extract galaxies from simulation snapshots')
    if params['do_subgrid']:            print('- Subgrid the galaxy, creating GMCs and diffuse gas clouds')
    if params['do_interpolate']:        print('- Interpolate in cloud model grid for each gas clouds')
    if params['do_datacubes']:          print('- Create datacubes for ISM phases GMC, DNG and DIG')

    #===========================================================================
    """ Constants and variables used by SIGAME """
    #---------------------------------------------------------------------------

    params['Tkcmb']                 =   2.725*(1+float(params['zred']))      # CMB temperature at this redshift
    params['G_grav']                =   6.67428e-11                          # Gravitational constant [m^3 kg^-1 s^-2]
    params['clight']                =   299792458                            # Speed of light [m/s]
    params['hplanck']               =   4.135667662e-15                      # Planck constant [eV*s]
    params['Ryd']                   =   13.60569253                          # [eV]
    params['eV_J']                  =   1.6021766208e-19                     # [J]
    params['Habing']                =   1.6e-3                               # ergs/cm^2/s
    params['Msun']                  =   1.989e30                             # Mass of Sun [kg]
    params['Lsun']                  =   3.839e26                             # Bol. luminosity of Sun [W]
    params['kB']                    =   1.381e-23                            # Boltzmanns constant [J K^-1]
    params['kB_ergs']               =   1.3806e-16                           # Boltzmanns constant [ergs K^-1]
    params['b_wien']                =   2.8977729e-3                         # Wien's displacement constant [m K]
    params['kpc2m']                 =   3.085677580666e19                    # kpc in m
    params['pc2m']                  =   3.085677580666e16                    # pc in m
    params['kpc2cm']                =   3.085677580666e21                    # kpc in cm
    params['pc2cm']                 =   3.085677580666e18                    # pc in cm
    params['au']                    =   per.constants.atomic_mass_constant   # atomic mass unit [kg]
    params['f_CII']                 =   1900.5369                            # frequency in GHz
    params['f_CI1']                 =   492.160651                           # frequency in GHz
    params['f_CI2']                 =   809.34197                            # frequency in GHz
    params['f_NII_122']             =   2459.370214752                       # frequency in GHz http://www.ipac.caltech.edu/iso/lws/atomic.html
    params['f_NII_205']             =   1461.132118324                       # frequency in GHz http://www.ipac.caltech.edu/iso/lws/atomic.html
    params['f_OI']                  =   4744.774906758                       # frequency in GHz http://www.ipac.caltech.edu/iso/lws/atomic.html
    params['f_OIII']                =   3393.006224818                       # frequency in GHz http://www.ipac.caltech.edu/iso/lws/atomic.html
    els                             =   ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al',\
                                        'Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']
    m_elements  =   {}
    for el in els: m_elements[el] = [getattr(per,el).mass*params['au']]
    m_elements  =   pd.DataFrame(m_elements)
    # solar abundances from cloudy (n_i/n_H)
    n_elements  =   pd.DataFrame(data=np.array([[1],[1e-1],[2.04e-9],[2.63e-11],\
        [6.17e-10],[2.45e-4],[8.51e-5],[4.90e-4],[3.02e-8],[1.0e-4],[2.14e-6],[3.47e-5],\
        [2.95e-6],[3.47e-5],[3.20e-7],[1.84e-5],[1.91e-7],[2.51e-6],[1.32e-7],[2.29e-6],\
        [1.48e-9],[1.05e-7],[1.00e-8],[4.68e-7],[2.88e-7],[2.82e-5],[8.32e-8],[1.78e-6],\
        [1.62e-8],[3.98e-8]]).transpose(),columns=els) # from cloudy
    # solar mass fractions used in simulations
    params['mf_solar']    =   {'tot':0.0134, 'He':0.2485, 'C':2.38e-3,'N': 0.70e-3,'O': 5.79e-3,'Ne': 1.26e-3,
                        'Mg':7.14e-4,'Si': 6.17e-4, 'S':3.12e-4, 'Ca':0.65e-4,'Fe': 1.31e-3}
    elements    =   m_elements.append(n_elements,sort=True)
    elements.index  =   ['mass','[n_i/n_H]']
    params['elements']              =   elements
    params['a_C']                   =   elements.loc['[n_i/n_H]','C'] # solar abundance of carbon
    params['mf_Z1']                 =   0.0134                      # Asplund+09
    params['mH']                    =   elements.loc['mass','H']    # Hydrogen atomic mass [kg]
    params['mC']                    =   elements.loc['mass','C']    # Carbon atomic mass [kg]
    params['me']                    =   9.10938215e-31              # electron mass [kg]
    params['m_p']                   =   1.6726e-24
    params['mCII']                  =   params['mC']-params['me']
    params['mCIII']                 =   params['mC']-2.*params['me']
    params['mCO']                   =   elements.loc['mass','C']+elements.loc['mass','O']
    params['mH2']                   =   2.*elements.loc['mass','H'] # Molecular Hydrogen [kg]
    params['pos']                   =   ['x','y','z']               # set of coordinates (will use often)
    params['posxy']                 =   ['x','y']                   # set of coordinates (will use often)
    params['vpos']                  =   ['vx','vy','vz']            # set of velocities (will use often)
    params['FUV_ISM']               =   0.6*1.6*1e-3                # local FUV field [ergs/s/cm^2]
    params['CR_ISM']                =   3e-17                       # local CR field [s^-1]
    params['SFRsd_MW']              =   0.0033                      # [Msun/yr/kpc^2] https://ned.ipac.caltech.edu/level5/March15/Kennicutt/Kennicutt5.html
    params['Herschel_limits']       =   dict(CII=0.13e-8)           # W/m^2/sr Croxall+17 KINGFISH

    #===========================================================================
    """ For plotting """
    #---------------------------------------------------------------------------

    params['this_work']     =   'Model galaxies at z ~ '+params['z1'].replace('z','')+' (this work)'
    params['sigame_label']  =   r'S$\mathrm{\'I}$GAME at z$\sim$'+params['z1'].replace('z','')+' (this work)'
    # datacubes -> ISM
    params['ISM_dc_phases'] =   ['GMC','DNG','DIG']
    params['ISM_dc_labels'] =   dict( DIG='Diffuse Ionized Gas', DNG='Diffuse Neutral Gas', GMC='Giant Molecular Clouds' )
    params['ISM_dc_colors'] =   dict( DIG='b', DNG='orange', GMC='r' )
    #---------------------------------------------------------------------------
    # particle_data -> sim
    params['sim_types']     =   ['gas', 'dm', 'star']
    params['sim_labels']    =   dict( gas='gas', dm='dark matter', star='star', GMC='Giant Molecular Clouds')
    params['sim_colors']    =   dict( gas='blue', dm='grey', star='orange', GMC='r')
    #---------------------------------------------------------------------------
    # particle_data -> ISM
    params['ISM_phases']    =   ['GMC', 'dif']
    params['ISM_labels']    =   dict(GMC='Giant Molecular Cloud', diff='Diffuse Gas')
    params['ISM_colors']    =   dict(GMC='r', diff='b')
    #---------------------------------------------------------------------------
    params['redshift_colors']       =   {0:'blue',2:'purple',6:'red'}
    # I would take a look at plot.set_mpl_params() for these
    params['galaxy_marker']         =   'o'
    params['galaxy_ms']             =   6
    params['galaxy_alpha']          =   0.7
    params['galaxy_lw']             =   2
    params['galaxy_mew']             =   0
    #---------------------------------------------------------------------------
    params['color_names']           =   colors.cnames
    col                             =   ['']*len(params['color_names'])
    i           =   0
    for key,value in params['color_names'].items():
        col[i]         =    key
        i              +=   1
    params['col']                   =   col
    params['colsel']                =   [u'fuchsia',u'darkcyan',u'indigo',u'hotpink',u'blueviolet',u'tomato',u'seagreen',\
                                    u'magenta',u'cyan',u'darkred',u'purple',u'lightgrey',\
                                    u'brown',u'orange',u'darkgreen',u'black',u'yellow',\
                                    u'darkmagenta',u'olive',u'lightsalmon',u'darkblue',\
                                    u'navajowhite',u'sage']

    #===========================================================================
    """ Save parameters """
    #---------------------------------------------------------------------------

    np.save('temp_params',params)

    params                          =   add_default_args()

    np.save('temp_params',params)


def update_params_file(new_params,verbose=False):

    params                      =   np.load('temp_params.npy', allow_pickle=True).item()

    for key in new_params:
        params[key]               =   new_params[key]

    np.save('temp_params',params)

    if verbose:
        print('Updated params.npy with:')
        for key in new_params:
            print('- '+key)

    return(params)

def add_default_args():
    """ Make custom namespace for specific method
    """

    default_args        =   dict(\

                            add=False,\

                            bins=100,\

                            classification='spherical',\
                            color='k',\
                            convolve=False,\

                            data='sim',\
                            dc_name='data',\
                            debug=False,\

                            flabel='Jy',\
                            fs_labels=15,\
                            FWHM=None,\

                            galname='',\
                            gal_index=0,\
                            gal_ob_present=True,\
                            gal_ob={},\

                            ISM_phase='',\
                            ISM_dc_phase='tot',\
                            Iunits='Jykms',\

                            line='CII',\

                            map_type='',\
                            min_fraction=1./1e6,\

                            N_radial_bins=30,\

                            one_color=True,\

                            R_max=15,\

                            sim_type='',\

                            target='L_CII',\

                            verbose=False,\
                            vlabel='km/s',\

                            xlabel='x [kpc]',\
                            xlim=False,\
                            xyz_units='kpc',\

                            ylabel='y [kpc]',\
                            ylim=False,\

                            zred=0

                            )

    new_params          =   update_params_file(default_args)

    return(new_params)
