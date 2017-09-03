"""
###     Submodule: param_module.py of SIGAME        ###
###                                                 ###
###     Contains functions:                         ###
###                                                 ###
###     read_params: Reads from parameter file      ###
###     (location of parameter file is set in       ###
###     __init__.py)                                ###
###                                                 ###
"""


import linecache as lc
import numpy as np
import re 
import pandas as pd
from periodic import element
import matplotlib.colors as colors
import pdb

global galnames

def read_params(params_file):
    '''
    Purpose
    ---------
    Extracts parameters from parameter file set in sigame/__init__.py
    '''

    params              =   {}

    file                =   open(params_file,'r')
    lc.clearcache()

    parameter_list      =   ['galnames','halos','snaps','haloIDs','z1',\
                            'hubble','omega_m','omega_lamda','omega_r',\
                            'd_sim','D_L',\
                            'simtype','ext_DENSE','ext_DIFFUSE','lines',\
                            'frac_h','f_R_gal']

    for i,line in enumerate(file):
        for parameter in parameter_list:
            if line.find('['+parameter+']') >= 0:
                params[parameter]   =   re.sub('\n','',lc.getline(params_file,i+2))

    # convert some parameters to int/float:
    params['zred']          =   float(params['z1'])
    params['z1']            =   'z'+str(int(params['z1']))
    params['hubble']        =   float(params['hubble'])
    params['omega_m']       =   float(params['omega_m'])
    params['omega_lamda']   =   float(params['omega_lamda'])
    params['omega_r']       =   float(params['omega_r'])
    params['D_L']           =   float(params['D_L'])
    params['frac_h']        =   float(params['frac_h'])
    params['f_R_gal']       =   float(params['f_R_gal'])

    # convert some parameters into lists
    if params.has_key('galnames'):
        galnames            =   params['galnames']
        galnames            =   galnames.replace('[','')
        galnames            =   galnames.replace(']','')
        galnames            =   galnames.replace(' ','')
        galnames            =   galnames.replace("'","")
        galnames            =   galnames.split(",")
        params['galnames']  =   galnames
    if params.has_key('halos'):
        halos               =   re.findall(r'\w+',params['halos'])
    if params.has_key('snaps'):
        snaps               =   re.findall(r'\w+',params['snaps'])
        snaps               =   [int(snaps[i]) for i in range(0,len(snaps))]
    if params.has_key('haloIDs'):
        haloIDs             =   re.findall(r'\w+',params['haloIDs'])
        haloIDs             =   [int(haloIDs[i]) for i in range(0,len(haloIDs))]
    if params.has_key('lines'):
        lines               =   params['lines']
        lines               =   lines.replace('[','')
        lines               =   lines.replace(']','')
        lines               =   lines.replace(' ','')
        lines               =   lines.replace("'","")
        lines               =   lines.split(',')
        params['lines']     =   lines
    
    # decide which parts of SIGAME to run:
    run_options         =   ['load_galaxy','subgrid','create_GMCs',\
                            'line_calc_GMC','create_dif_gas','line_calc_dif']

    file                =   open(params_file,'r')
    lc.clearcache()
    for i,line in enumerate(file):
        for run_option in run_options:
            if line.find(run_option) >= 0:
                line1                   =   lc.getline(params_file,i+1)
                params['do_'+run_option]    =   False
                if line1[0:2] == '+1': params['do_'+run_option] = True

    # Where cloudy is:
    params['d_c']           =   '../cloudy/'
    # Where starburst is:
    params['d_sb']          =   '../starburst99/galaxy/output/'
    # Where tables are:
    params['d_t']           =   'Tables/'
    
    # Print parameters chosen
    print('\n' + (' Parameters chosen').center(20+10+10+10))
    print('+%20s+%10s+%15s+%50s+' % ((20*'-'), (10*'-'), (15*'-'), (50*'-')))
    print('|%20s|%10s|%15s|%50s|' % ('Parameter'.center(20), 'Value'.center(10), 'Name in code'.center(15), 'Explanation'.center(50)))
    print('+%20s+%10s+%15s+%50s+' % ((20*'-'), (10*'-'), (15*'-'), (50*'-')))
    print('|%20s|%10g|%15s|%50s|' % ('Repr. redshift'.center(20), params['zred'], 'zred'.center(15),'Redshift of simulation snapshot'.center(50)))
    print('|%20s|%10s|%15s|%50s|' % ('Dense cloudy grid'.center(20), params['ext_DENSE'], 'ext_DENSE'.center(15),'Extension of desired GMC model grid'.center(50)))
    print('|%20s|%10s|%15s|%50s|' % ('Diffuse cloudy grid'.center(20), params['ext_DIFFUSE'], 'ext_DIFFUSE'.center(15),'Extension of desired HIM model grid'.center(50)))
    print('|%20s|%10g|%15s|%50s|' % ('Fraction of h'.center(20), params['frac_h'], 'frac_h'.center(15),'GMCs are distributed < frac_h * h'.center(50)))
    print('+%20s+%10s+%15s+%50s+' % ((20*'-'), (10*'-'), (15*'-'), (50*'-')))
    
    if params.has_key('galnames'):
        print('Number of selected galaxies:')
        print(len(galnames))
        print('Names of selected galaxies:')
        print(galnames)
    
    # Constants and variables used by SIGAME
    params['Tkcmb']       			=   2.725*(1+float(params['zred']))      # CMB temperature at this redshift
    params['G_grav']      			=   6.67428e-11                          # Gravitational constant [m^3 kg^-1 s^-2]
    params['clight']      			=   299792458                            # Speed of light [m/s]
    params['hplanck']     			=   4.135667662e-15                      # Planck constant [eV*s]
    params['Ryd']                   =   13.60569253                          # eV
    params['Habing']         	    =   1.6e-3                               # ergs/cm^2/s
    params['Msun']        			=   1.989e30                             # Mass of Sun [kg]
    params['Lsun']        			=   3.839e26                             # Bol. luminosity of Sun [W]
    params['kB']          			=   1.381e-23                            # Boltzmanns constant [J K^-1]
    params['kB_ergs']          		=   1.3806e-16                           # Boltzmanns constant [ergs K^-1]
    params['b_wien']                =   2.8977729e-3                         # Wien's displacement constant [m K]
    params['kpc2m']       			=   3.085677580666e19                    # kpc in m
    params['pc2m']        			=   3.085677580666e16                    # pc in m
    params['kpc2cm']      			=   3.085677580666e21                    # kpc in cm
    params['pc2cm']       			=   3.085677580666e18                    # pc in cm
    params['au']          			=   1.660538782e-27                      # atomic unit [kg]
    params['f_CII']                 =   1900.5369                            # frequency in GHz   
    params['f_CI1']                 =   492.160651                           # frequency in GHz   
    params['f_CI2']                 =   809.34197                            # frequency in GHz   
    params['f_NII']                 =   2459.370214752                       # frequency in GHz http://www.ipac.caltech.edu/iso/lws/atomic.html    
    params['f_OI']                  =   4744.774906758                       # frequency in GHz http://www.ipac.caltech.edu/iso/lws/atomic.html 
    params['f_OIII']       			=   3393.006224818                       # frequency in GHz http://www.ipac.caltech.edu/iso/lws/atomic.html  
    els         					=   ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al',\
                                        'Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']
    m_elements  =   pd.DataFrame({'H':[1]})
    for el in els:
        m_elements[el]  =   element(el).mass*params['au']
    # solar abundances from cloudy (n_i/n_H)
    n_elements  =   pd.DataFrame(data=np.array([[1],[1e-1],[2.04e-9],[2.63e-11],\
        [6.17e-10],[2.45e-4],[8.51e-5],[4.90e-4],[3.02e-8],[1.0e-4],[2.14e-6],[3.47e-5],\
        [2.95e-6],[3.47e-5],[3.20e-7],[1.84e-5],[1.91e-7],[2.51e-6],[1.32e-7],[2.29e-6],\
        [1.48e-9],[1.05e-7],[1.00e-8],[4.68e-7],[2.88e-7],[2.82e-5],[8.32e-8],[1.78e-6],\
        [1.62e-8],[3.98e-8]]).transpose(),columns=els) # from cloudy
    elements    =   m_elements.append(n_elements)
    elements.index  =   ['mass','[n_i/n_H]']
    params['elements'] 				=	elements
    params['a_C']         			=   elements.loc['[n_i/n_H]','C'] # solar abundance of carbon
    params['mf_Z1']                 =   0.0134                      # Asplund+09
    params['mH']          			=   elements.loc['mass','H']    # Hydrogen atomic mass [kg]
    params['mC']          			=   elements.loc['mass','C']    # Carbon atomic mass [kg]
    params['me']          			=   9.10938215e-31              # electron mass [kg]
    params['m_p'] 					=	1.6726e-24
    params['mCII']        			=   params['mC']-params['me']
    params['mCIII']       			=   params['mC']-2.*params['me']
    params['mCO']       			=   elements.loc['mass','C']+elements.loc['mass','O']
    params['mH2']         			=   2.*elements.loc['mass','H'] # Molecular Hydrogen [kg]
    params['pos']         			=   ['x','y','z']               # set of coordinates (will use often)
    params['posxy']	    			=	['x','y']                   # set of coordinates (will use often)
    params['vpos']					=	['vx','vy','vz']            # set of velocities (will use often)
    params['FUV_ISM']     			=   0.6*1.6*1e-3                # local FUV field [ergs/s/cm^2]
    params['CR_ISM']      			=   3e-17                       # local CR field [s^-1]
    params['SFRsd_MW']              =   0.0033                      # https://ned.ipac.caltech.edu/level5/March15/Kennicutt/Kennicutt5.html
    
    # Pick some nice colors for plotting
    params['color_names']           =   colors.cnames
    col                             =   ['']*len(params['color_names'])
    i           =   0
    for key,value in params['color_names'].items():
        col[i]         =    key
        i              +=   1
    params['col']                   =   col
    params['colsel']      			=   [u'fuchsia',u'darkcyan',u'indigo',u'hotpink',u'blueviolet',u'tomato',u'seagreen',\
                					u'magenta',u'cyan',u'darkred',u'purple',u'lightgrey',\
                					u'brown',u'orange',u'darkgreen',u'black',u'yellow',\
                					u'darkmagenta',u'olive',u'lightsalmon',u'darkblue',\
                					u'navajowhite',u'sage']
     
    # Derive redshifts from filename
    snapshots                       =   pd.read_table(params['d_t']+'snapshots.txt',names=['snaps','zs','times','D_L'],skiprows=1,sep='\t',engine='python')
    params['snaps_table']           =   snapshots.index
    params['snaps_table']           =   snapshots['snaps'].values
    params['snap_times']            =   snapshots['times'].values
    params['snap_zs']               =   snapshots['zs'].values
    params['snap_D_L']              =   snapshots['D_L'].values
    if galnames != ['']:
        zreds                           =   np.zeros(len(galnames))
        for i in range(0,len(zreds)):
            snap1                           =   str.split(str.split(galnames[i],'s')[1],'_')[0]
            zreds[i]                        =   float(params['snap_zs'][params['snaps_table'] == int(snap1)])
        params['zreds']                 =   zreds

        # Paths to results for using classes
        params['sim_paths']           	=   ['sigame/temp/sim_FUV/z'+'{:.2f}'.format(zred)+'_'+galname+'_sim1.gas' for galname,zred in zip(params['galnames'],params['zreds'])]
        params['star_paths']           	=   ['sigame/temp/sim_FUV/z'+'{:.2f}'.format(zred)+'_'+galname+'_sim1.star' for galname in params['galnames']]
        
        if params['ext_DENSE'] == '_ism': params['GMC_path'] = 'sigame/temp/GMC/emission/'
        else: params['GMC_path'] = 'sigame/temp/GMC/emission/tests/'
        params['GMC_paths']             =   [params['GMC_path']+'z'+'{:.2f}'.format(zred)+'_'+galname+'_GMC'+params['ext_DENSE']+'_em.gas' for galname,zred in zip(params['galnames'],params['zreds'])]

        if params['ext_DIFFUSE'] == '_ism': params['dif_path'] = 'sigame/temp/dif/emission/'
        else: params['dif_path'] = 'sigame/temp/dif/emission/tests/'
        params['dif_paths']             =   [params['dif_path']+'z'+'{:.2f}'.format(zred)+'_'+galname+'_dif'+params['ext_DIFFUSE']+'_em.gas' for galname,zred in zip(params['galnames'],params['zreds'])]
    else:
        params['zreds']                 =   [zred]

    print('This is what sigame.run() is set up to do (change in parameter file):')    
    if params['do_subgrid']:           print('- Subgrid SPH particles')
    if params['do_create_GMCs']:       print('- Generate GMCs')
    if params['do_line_calc_dif']:     print('- Calculate line emission from diffuse gas clouds')
    if params['do_create_dif_gas']:    print('- Generate diffuse gas clouds')
    if params['do_line_calc_GMC']:     print('- Calculate line emission from GMCs')

    params['global_save_file']     =    'sigame/temp/global_results/'+params['z1']+'_'+str(len(galnames))+'gals'+params['ext_DENSE']+params['ext_DIFFUSE']

    np.save('sigame/temp/params',params)


