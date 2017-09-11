"""
###     Submodule: GMC_module.py of SIGAME              ###
"""

import numpy as np
import pandas as pd
import pickle
import pdb
import scipy
from scipy.interpolate import RegularGridInterpolator,griddata
from scipy.interpolate import interp1d
import time
import multiprocessing as mp
import subprocess as sub
import collections
import os
import re
import matplotlib.pyplot as plt
import aux as aux
# import plot as plot

params                      =   np.load('sigame/temp/params.npy').item()
for key,val in params.items():
    exec(key + '=val')

def create_GMCs(galname=galnames[0],zred=zreds[0],verbose=False):
    '''
    Purpose
    -------
    Splits the molecular part of each SPH particle into GMCs (called by main.run)

    Arguments
    ---------
    galname: galaxy name - str
    default = first galaxy name in galnames list from parameter file

    zred: redshift of galaxy - float/int
    default = first redshift name in redshift list from parameter file

    verbose: if True, print out details for each model as it is saved
    default = False

    '''

    print('\n ** Create GMCs **')

    plt.close('all')        # close all windows

    # Load simulation data for gas and stars
    simgas          =   pd.read_pickle('sigame/temp/sim_FUV/z'+'{:.2f}'.format(zred)+'_'+galname+'_sim1.gas')
    simstar         =   pd.read_pickle('sigame/temp/sim_FUV/z'+'{:.2f}'.format(zred)+'_'+galname+'_sim1.star')

    if verbose: print('Number of gas elements: %s' % str(len(simgas)))
    if verbose: print('Number of stars: %s' % str(len(simstar)))

    print('Using f_H2 from simulation to estimate neutral gas mass per gas element')
    Mneu            =   simgas['m'].values*simgas['f_H2'].values
    print('Total dense gas mass: '+str(np.sum(Mneu))+' Msun')
    print('Dense gas mass fraction out of total ISM mass: '+str(np.sum(Mneu)/np.sum(simgas['m'])*100.)+' %')
    print('Min and max particle dense gas mass: '+str(np.min(Mneu))+' '+str(np.max(Mneu))+' Msun')
    print('Throwing away this much gas mass: '+str(np.sum(simgas.loc[Mneu < 1e4]['m']))+' Msun')
    print('In percent: '+str(np.sum(simgas.loc[Mneu < 1e4]['m'])/np.sum(simgas['m']))+' %')

    # Create mass spectrum (probability function = dN/dM normalized)
    b               =   1.8                                # MW powerlaw slope [Blitz+07]
    if ext_DENSE == '_b3p0': b = 3.0                       # outer MW and M33 powerlaw slope [Rosolowsky+05, Blitz+07]
    if ext_DENSE == '_b1p5': b = 1.5                       # inner MW powerlaw slope [Rosolowsky+05, Blitz+07]
    print('Powerlaw slope for mass spectrum (beta) used is %s' % b)
    Mmin            =   1.e4                               # min mass of GMC
    Mmax            =   1.e6                               # max mass of GMC
    tol             =   Mmin                               # precision in reaching total mass
    nn              =   100                                # max draw of masses in each run
    GMCgas          =   abs(simgas[:0])*0                  # 0 gmc
    n_elements      =   simgas.size
    simgas          =   simgas[Mneu > 1e4]                  # Cut out those with masses < 1e4 !!
    Mneu            =   Mneu[Mneu > 1e4]
    h               =   simgas['h'].values
    print('Largest neutral mass is: %s' % (Mneu.max()))
    print('Smallest neutral mass is: %s' % (Mneu.min()))
    print('# Gas particles with Mmol > 2*1e4 Msun: %s out of %s' % (np.size(Mneu[Mneu > 2*Mmin]),np.size(Mneu)))
    simgas.index    =   range(0,len(simgas))
    j               =   0
    print('(Multiprocessing starting up!)')
    pool            =   mp.Pool(processes=3)        # 8 processors on my Mac Pro, 16 on Betty
    np.random.seed(len(GMCgas))                         # so we don't end up with the same random numbers for every galaxy
    results         =   [pool.apply_async(GMC_generator, args=(i,Mneu,h,Mmin,Mmax,b,tol,nn,)) for i in range(0,len(simgas))]#
    GMCs            =   [p.get() for p in results]
    print('(Multiprocessing done!)')
    GMCgas          =   simgas.iloc[0:GMCs[0][0]]
    Mgmc            =   GMCs[0][1]
    newx            =   simgas.loc[0]['x']+GMCs[0][2]
    newy            =   simgas.loc[0]['y']+GMCs[0][3]
    newz            =   simgas.loc[0]['z']+GMCs[0][4]
    i1              =   0
    part            =   0.1
    print('Append results to new dataframe')
    for i in range(1,len(simgas)):
        for ii in range (0,GMCs[i][0]):
            GMCgas = pd.DataFrame.append(GMCgas,simgas.loc[i],ignore_index=True)
        Mgmc         =   np.append(Mgmc,GMCs[i][1])
        newx         =   np.append(newx,simgas.loc[i]['x']+GMCs[i][2])
        newy         =   np.append(newy,simgas.loc[i]['y']+GMCs[i][3])
        newz         =   np.append(newz,simgas.loc[i]['z']+GMCs[i][4])
        i1          +=  len(GMCs[i])
        if 1.*i/len(simgas) > part:
            print(str(int(part*100))+' % done!')
            part    =   part+0.1
    GMCgas['m']     =   Mgmc
    GMCgas['x']     =   newx
    GMCgas['y']     =   newy
    GMCgas['z']     =   newz
    GMCgas['Rgmc']  =   (GMCgas['m']/290.0)**(1.0/2.0)
    print('Min P_ext: %s' % np.log10(min(simgas['P_ext'])))

    print('Mass of all GMCs created: %s - should not exceed:' % np.sum(GMCgas['m']))
    print('Total neutral gas: %s ' % np.sum(Mneu))

    print(str(len(GMCgas))+' GMCs created!')
    ext = ''
    if b == 1.5:    ext = '_b1p5'
    if b == 3.0:    ext = '_b3p0'
    dir_gas             =   'sigame/temp/GMC/'
    GMCgas.to_pickle(dir_gas+'z'+'{:.2f}'.format(zred)+'_'+galname+'_GMC'+ext+'.gas')

def GMC_generator(i,Mneu,h,Mmin,Mmax,b,tol,nn):
    '''
    Purpose
    ---------
    Draws mass randomly from GMC mass spectrum (called by create_GMCs)

    Arguments
    ---------
    '''
    ra      =   np.random.rand(nn)  # draw nn random numbers between 0 and 1
    ra1     =   np.random.rand(nn)  # draw nn random numbers between 0 and 1
    ra2     =   np.random.rand(nn)  # draw nn random numbers between 0 and 1
    Mg      =   np.zeros(nn)
    newm_gmc     =   np.zeros(1)
    if Mneu[i] < Mmin+tol:
        newm_gmc         =   Mneu[i]
    if Mneu[i] > Mmin+tol:
        for ii in range(0,nn):
            k           =   (1/(1-b)*(Mmax**(1-b)-Mmin**(1-b)))**(-1)       # Normalization constant (changing with Mmax)
            Mg[ii]      =   (ra[ii]*(1-b)/k+Mmin**(1-b))**(1./(1-b))        # Draw mass (from cumulative distribution function)
            if ii==0 and np.sum(newm_gmc)+Mg[ii] < Mneu[i]+tol:             # Is it the 1st draw and is the GMC mass below total neutral gas mass (M_neutral) available?
                newm_gmc         =   np.array(Mg[ii])                            # - then add that GMC mass
            if ii>0 and np.sum(newm_gmc)+Mg[ii] < Mneu[i]+tol:              # Is the sum of GMC masses still below M_neutral+tolerance?
                newm_gmc         =   np.append(newm_gmc,Mg[ii])                       # - then add that GMC mass
            if np.sum(newm_gmc) > Mneu[i]-tol:                              # Is the sum of GMC masses above M_neutral-tolerance?
                break                                                       # - fine! then stop here
    # Add SPH info to new DataFrame (same for all GMCs to this SPH parent)
    # Save indices of original SPH particles
    f1      =   np.size(newm_gmc)
    if f1 ==1:
        if newm_gmc == 0:
            print('No GMCs created for this one?')
            pdb.set_trace()
    SPHindex    =   np.zeros(f1)+i
    # but change coordinates!!
    ra_R        =   ra[0:f1]*frac_h*h[i]
    ra_phi      =   ra1[0:f1]*2*np.pi
    ra_theta    =   ra2[0:f1]*np.pi
    ra          =   [ra_R*np.sin(ra_theta)*np.cos(ra_phi),+\
        ra_R*np.sin(ra_theta)*np.sin(ra_phi),+\
        ra_R*np.cos(ra_theta)]
    newx        =   np.array(ra)[0,:]
    newy        =   np.array(ra)[1,:]
    newz        =   np.array(ra)[2,:]
    # Neutral mass that remains is distributed in equal fractions around the GMCs:
    return f1,newm_gmc,newx,newy,newz

def calc_line_emission(galname=galnames[0],zred=zreds[0],SFRsd=SFRsd_MW):
    '''
    Purpose
    ---------
    Interpolates in grid for one galaxy (called by main.run)

    Arguments
    ---------
    galname: galaxy name - str
    default = first galaxy name in galnames list from parameter file

    zred: redshift of galaxy - float/int
    default = first redshift name in redshift list from parameter file

    SFRsd: SFR surface density of this galaxy - float
    default = SFR surface density of the MW

    '''

    print('\n ** Calculate total line emission from GMCs **')

    plt.close('all')        # close all windows

    ext_DENSE0          =   ext_DENSE

    # Load dataframe with GMCs
    GMCgas = pd.read_pickle('sigame/temp/GMC/z'+'{:.2f}'.format(zred)+'_'+galname+'_GMC.gas')
    if ext_DENSE0 == '_b1p5': GMCgas = pd.read_pickle('sigame/temp/GMC/z'+'{:.2f}'.format(zred)+'_'+galname+'_GMC_b1p5.gas')
    if ext_DENSE0 == '_b3p0': GMCgas = pd.read_pickle('sigame/temp/GMC/z'+'{:.2f}'.format(zred)+'_'+galname+'_GMC_b3p0.gas')

    nGMC                =   len(GMCgas)
    print('Number of GMCs: %s' % nGMC)
    print('Mass of GMCs: %s Msun' % np.sum(GMCgas['m']))

    # add diffuse FUV field to GMCs:
    print('SFR surface density is '+str(SFRsd/SFRsd_MW)+' x that of MW ')
    GMCgas['FUV']       =   GMCgas['FUV'].values+SFRsd/SFRsd_MW

    # TEST!!!
    if ext_DENSE0 == '_Z0p05':   GMCgas['Z'],ext_DENSE0     =   GMCgas['Z'].values*0.+0.05,'_Ztest'
    if ext_DENSE0 == '_Z1':      GMCgas['Z'],ext_DENSE0     =   GMCgas['Z'].values*0.+1.,'_Ztest'
    if ext_DENSE0 == '_Zx3':     GMCgas['Z'],ext_DENSE0     =   GMCgas['Z'].values*3.,'_highZ'
    if ext_DENSE0 == '_Zx10':    GMCgas['Z'],ext_DENSE0     =   GMCgas['Z'].values*10.,'_highZ'
    if ext_DENSE0 == '_Zx20':    GMCgas['Z'],ext_DENSE0     =   GMCgas['Z'].values*20.,'_highZ'
    if ext_DENSE0 == '_b1p5':    ext_DENSE0                 =   '_b1p5'
    if ext_DENSE0 == '_b3p0':    ext_DENSE0                 =   '_b3p0'
    cloudy_grid_param   =   pd.read_pickle('cloudy_models/GMC/grids/GMCgrid'+ext_DENSE0+'_'+z1+'.param')
    cloudy_grid         =   pd.read_pickle('cloudy_models/GMC/grids/GMCgrid'+ext_DENSE0+'_'+z1+'_CII.models')

    # make sure we don't go outside of grid:
    GMCgas1             =   np.log10(GMCgas)  # Going to edit a bit in the DataFrame
    GMCgas1.ix[GMCgas1['m']       < min(cloudy_grid_param['Mgmcs']),'m']      =   min(cloudy_grid_param['Mgmcs'])
    GMCgas1.ix[GMCgas1['FUV']     < min(cloudy_grid_param['FUVs']),'FUV']     =   min(cloudy_grid_param['FUVs'])
    GMCgas1.ix[GMCgas1['Z']       < min(cloudy_grid_param['Zs']),'Z']         =   min(cloudy_grid_param['Zs'])
    GMCgas1.ix[GMCgas1['P_ext']   < min(cloudy_grid_param['P_exts']),'P_ext'] =   min(cloudy_grid_param['P_exts'])
    GMCgas1.ix[GMCgas1['m']       > max(cloudy_grid_param['Mgmcs']),'m']      =   max(cloudy_grid_param['Mgmcs'])
    GMCgas1.ix[GMCgas1['FUV']     > max(cloudy_grid_param['FUVs']),'FUV']     =   max(cloudy_grid_param['FUVs'])
    GMCgas1.ix[GMCgas1['Z']       > max(cloudy_grid_param['Zs']),'Z']         =   max(cloudy_grid_param['Zs'])
    GMCgas1.ix[GMCgas1['P_ext']   > max(cloudy_grid_param['P_exts']),'P_ext'] =   max(cloudy_grid_param['P_exts'])
    # Values used for interpolation in cloudy models:
    GMCs                        =   np.column_stack((GMCgas1['m'].values,GMCgas1['FUV'].values,GMCgas1['Z'].values,GMCgas1['P_ext'].values))

    # List of target items that we are interpolating for:
    target_list                 =   ['Mgmc_fit','FUV_fit','Z_fit','P_ext_fit',\
                                    'f_H2','f_HI','m_dust','m_H','m_H2','m_HI','m_HII',\
                                    'm_C','m_CII','m_CIII','m_CIV','m_CO',\
                                    'Tkmw','nHmw','nHmin','nHmax']
    for line in lines: target_list.append('L_'+line)

    cloudy_grid['Mgmc_fit']     =   cloudy_grid['Mgmc']
    cloudy_grid['FUV_fit']      =   cloudy_grid['FUV']
    cloudy_grid['Z_fit']        =   cloudy_grid['Z']
    cloudy_grid['P_ext_fit']    =   cloudy_grid['P_ext']

    for target in target_list:
        # Make grid of corresponding target grid values:
        target_grid                 =   np.zeros([len(cloudy_grid_param['Mgmcs']),len(cloudy_grid_param['FUVs']),len(cloudy_grid_param['Zs']),len(cloudy_grid_param['P_exts'])])
        i                           =   0
        for i1 in range(0,len(cloudy_grid_param['Mgmcs'])):
            for i2 in range(0,len(cloudy_grid_param['FUVs'])):
                for i3 in range(0,len(cloudy_grid_param['Zs'])):
                    for i4 in range(0,len(cloudy_grid_param['P_exts'])):
                        target_grid[i1,i2,i3,i4]    =   cloudy_grid[target][i]
                        i                           +=  1

        # Make function that will do the interpolation:
        interp                      =   RegularGridInterpolator((cloudy_grid_param['Mgmcs'],cloudy_grid_param['FUVs'],cloudy_grid_param['Zs'],cloudy_grid_param['P_exts']), target_grid)
        # And interpolate:
        GMCgas[target]              =   interp(GMCs)

    GMCgas['f_HII']             =   1.-(GMCgas['f_H2']+GMCgas['f_HI'])

    # Find number of closest models!
    model_numbers               =   np.zeros([len(cloudy_grid_param['Mgmcs']),len(cloudy_grid_param['FUVs']),len(cloudy_grid_param['Zs']),len(cloudy_grid_param['P_exts'])])
    j                           =   0
    for i1 in range(0,len(cloudy_grid_param['Mgmcs'])):
        for i2 in range(0,len(cloudy_grid_param['FUVs'])):
            for i3 in range(0,len(cloudy_grid_param['Zs'])):
                for i4 in range(0,len(cloudy_grid_param['P_exts'])):
                    model_numbers[i1,i2,i3,i4]      =   j
                    j                               +=  1
    GMCgas['closest_model_i']   =   np.zeros(nGMC)
    for i in range(0,nGMC):
        lMgmc                       =   GMCgas1['m'][i]
        lFUV                        =   GMCgas1['FUV'][i]
        lZ                          =   GMCgas1['Z'][i]
        lP_ext                      =   GMCgas1['P_ext'][i]
        # Find closest index in each direction of parameter grid:
        i_Mgmc                      =   np.argmin(abs(cloudy_grid_param['Mgmcs']-lMgmc))
        i_FUV                       =   np.argmin(abs(cloudy_grid_param['FUVs']-lFUV))
        i_Z                         =   np.argmin(abs(cloudy_grid_param['Zs']-lZ))
        i_P_ext                     =   np.argmin(abs(cloudy_grid_param['P_exts']-lP_ext))
        GMCgas['closest_model_i'][i] = int(model_numbers[i_Mgmc,i_FUV,i_Z,i_P_ext])

    print('Total GMC mass available: '+str(sum(GMCgas['m'])/1e8)+' x 10^8 Msun')
    print('Mass of selected models: '+str(sum(GMCgas['m_H'])/1e8)+' x 10^8 Msun')
    print('Total mass of CII: '+str(sum(GMCgas['m_CII'])/1e8)+' x 10^8 Msun')
    print('Total mass of CIII: '+str(sum(GMCgas['m_CIII'])/1e8)+' x 10^8 Msun')
    print('Total dust mass: '+str(sum(GMCgas['m_dust'])/1e8)+' x 10^8 Msun')

    print('Integrating entire radial profiles:')
    print(str.format("{0:.2f}",sum(GMCgas['f_H2']*GMCgas['m'])/sum(GMCgas['m'])*100.)+' % of gas mass is molecular')
    print(str.format("{0:.2f}",sum(GMCgas['f_HI']*GMCgas['m'])/sum(GMCgas['m'])*100.)+' % of gas mass is atomic')
    print(str.format("{0:.2f}",sum(GMCgas['f_HII']*GMCgas['m'])/sum(GMCgas['m'])*100.)+' % of gas mass is ionized')
    print('Using boundaries:')
    print(str.format("{0:.2f}",sum(GMCgas['m_H2'])/sum(GMCgas['m_H'])*100.)+' % of gas mass is molecular')
    print(str.format("{0:.2f}",sum(GMCgas['m_HI'])/sum(GMCgas['m_H'])*100.)+' % of gas mass is atomic')
    print(str.format("{0:.2f}",sum(GMCgas['m_HII'])/sum(GMCgas['m_H'])*100.)+' % of gas mass is ionized')

    for line in lines:
        print('Total L_'+line+': %s x 10^8 L_sun' % (sum(GMCgas['L_'+line])/1e8))

    print('Add mass-weighted density and G0 to GMCgas dataframe for diagnostic line ratio plot')

    sigma_vs                =   1.2*(GMCgas['P_ext'].values/1e4)**(1.0/4.0)*(GMCgas['Rgmc'].values)**(1.0/2.0)
    GMCgas['vel_disp_gas']  =    sigma_vs
    GMCgas.to_pickle(GMC_path+'z'+'{:.2f}'.format(zred)+'_'+galname+'_GMC'+ext_DENSE+'_em.gas')

    GMC_results             =   {'M_dust_GMC':sum(GMCgas['m_dust']),'M_GMC':sum(GMCgas['m']),'M_H2_GMC':sum(GMCgas['m_H2']),'M_HI_GMC':sum(GMCgas['m_HI']),'M_HII_GMC':sum(GMCgas['m_HII']),\
                                'M_C_GMC':sum(GMCgas['m_C']),'M_CII_GMC':sum(GMCgas['m_CII']),'M_CIII_GMC':sum(GMCgas['m_CIII']),'M_CO_GMC':sum(GMCgas['m_CO'])}
    for line in lines: GMC_results['L_'+line+'_GMC'] = sum(GMCgas['L_'+line])

    return(GMC_results)


