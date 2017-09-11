"""
###     Submodule: dif_module.py of SIGAME              ###
"""

import numpy as np
import pandas as pd
import pickle
import pdb
import scipy
from scipy.interpolate import RegularGridInterpolator,griddata
from scipy.interpolate import interp1d
import subprocess as sub
import collections
import os
import re
import matplotlib.pyplot as plt
import aux as aux

params                      =   np.load('sigame/temp/params.npy').item()
for key,val in params.items():
    exec(key + '=val')
    
def create_dif(galname=galnames[0],zred=zreds[0]):
    '''
    Purpose
    ---------
    Extracts the non-molecular part of each fluid element (called by main.run)   

    Arguments
    ---------
    galname: galaxy name - str
    default = first galaxy name in galnames list from parameter file

    zred: redshift of galaxy - float/int
    default = first redshift name in redshift list from parameter file

    '''

    plt.close('all')        # close all windows

    # Load simulation data for gas
    simgas                          =   pd.read_pickle('sigame/temp/sim_FUV/z'+'{:.2f}'.format(zred)+'_'+galname+'_sim1.gas')

    # Start new dataframe with only the diffuse gas
    difgas                          =   simgas.copy()
    difgas['m']                     =   difgas['m'].values*(1.-difgas['f_H2'].values)
    print('Total gas mass in galaxy: '+str(sum(simgas['m'])))
    print('Diffuse gas mass in galaxy: '+str(sum(difgas['m'])))
    print('in percent: '+str(sum(difgas['m'])/sum(simgas['m'])*100.)+'%')

    # Calculate radius of diffuse gas clouds
    difgas['R']                     =   difgas['h']
    difgas['R'][difgas['m'] == 0]   =   0

    # Calculate density of diffuse gas clouds
    difgas['nH']                    =   0.75*np.array(difgas['m'],dtype=np.float64)/(4/3.*np.pi*np.array(difgas['R'],dtype=np.float64)**3.)*Msun/mH/kpc2cm**3       # Hydrogen atoms per cm^3
    difgas['nH'][difgas['m'] == 0]  =   0

    # Save results
    dir_gas                         =   'sigame/temp/dif/'
    difgas.to_pickle(dir_gas+'z'+'{:.2f}'.format(zred)+'_'+galname+'_dif.gas')

def calc_line_emission(galname=galnames[0],zred=zreds[0],SFRsd=SFRsd_MW):
    '''
    Purpose
    ---------
    Calculates total line emission etc. from diffuse gas clouds

    Arguments
    ---------
    galname: galaxy name - str
    default = first galaxy name in galnames list from parameter file

    zred: redshift of galaxy - float/int
    default = first redshift name in redshift list from parameter file

    SFRsd: SFR surface density of this galaxy - float
    default = SFR surface density of the MW

    '''

    printe('\n ** Calculate total line emission from diffuse gas **')

    plt.close('all')        # close all windows

    ext_DIFFUSE0        =   ext_DIFFUSE

    # Load dataframe with diffuse gas
    difgas              =   pd.read_pickle('sigame/temp/dif/z'+'{:.2f}'.format(zred)+'_'+galname+'_dif.gas')
    if ext_DIFFUSE0 == '_Z0p05':    difgas['Z'],ext_DIFFUSE0     =   difgas['Z'].values*0.+0.05,'_Ztest'
    if ext_DIFFUSE0 == '_Z1':       difgas['Z'],ext_DIFFUSE0     =   difgas['Z'].values*0.+1.,'_Ztest'
    if ext_DIFFUSE0 == '_Zx3':      difgas['Z'],ext_DIFFUSE0     =   difgas['Z'].values*3.,'_highZ'
    if ext_DIFFUSE0 == '_Zx10':     difgas['Z'],ext_DIFFUSE0     =   difgas['Z'].values*10.,'_highZ'
    if ext_DIFFUSE0 == '_Zx20':     difgas['Z'],ext_DIFFUSE0     =   difgas['Z'].values*20.,'_highZ'
    # Take out any particles with no diffuse gas mass
    difgas              =   difgas[difgas['m'] > 0]
    difgas              =   pd.DataFrame.reset_index(difgas)
    ndif                =   len(difgas)

    # Load the grid that best represents the SFR surface density of this galaxy
    UV                  =   [5,35]
    if ext_DIFFUSE == '_FUV': UV,ext_DIFFUSE0                  =   [5,15,25,35,45,120],'_ism'
    UV1                 =   str(int(UV[np.argmin(np.abs(np.array(UV)-SFRsd/SFRsd_MW))]))
    cloudy_grid_param   =   pd.read_pickle('cloudy_models/dif/grids/difgrid_'+UV1+'UV'+ext_DIFFUSE0+'_'+z1+'.param')
    cloudy_grid         =   pd.read_pickle('cloudy_models/dif/grids/difgrid_'+UV1+'UV'+ext_DIFFUSE0+'_'+z1+'_CII.models')

    print('SFR surface density is '+str(SFRsd/SFRsd_MW)+' x that of MW ')
    print('Using grid at '+UV1+' x ISM FUV field')
    difgas['FUV']       =   difgas['FUV']*0.+UV[np.argmin(np.abs(np.array(UV)-SFRsd/SFRsd_MW))]

    # make sure we don't go outside of grid:
    difgas1             =   difgas.copy()
    difgas1             =   np.log10(difgas1)  # Going to edit a bit in the DataFrame
    difgas1.ix[difgas1['nH']      <= min(cloudy_grid_param['nHs']),'nH']       =   min(cloudy_grid_param['nHs']+1e-4)
    difgas1.ix[difgas1['R']       <= min(cloudy_grid_param['Rs']),'R']         =   min(cloudy_grid_param['Rs']+1e-4)
    difgas1.ix[difgas1['Z']       <= min(cloudy_grid_param['Zs']),'Z']         =   min(cloudy_grid_param['Zs']+1e-4)
    difgas1.ix[difgas1['Tk']      <= min(cloudy_grid_param['Tks']),'Tk']       =   min(cloudy_grid_param['Tks']+1e-4)
    difgas1.ix[difgas1['nH']      >= max(cloudy_grid_param['nHs']),'nH']       =   max(cloudy_grid_param['nHs']-1e-4)
    difgas1.ix[difgas1['R']       >= max(cloudy_grid_param['Rs']),'R']         =   max(cloudy_grid_param['Rs'])
    difgas1.ix[difgas1['Z']       >= max(cloudy_grid_param['Zs']),'Z']         =   max(cloudy_grid_param['Zs'])
    difgas1.ix[difgas1['Tk']      >= max(cloudy_grid_param['Tks']),'Tk']       =   max(cloudy_grid_param['Tks'])

    # Values used for interpolation in cloudy models:
    dif             =   np.column_stack((difgas1['nH'].values,difgas1['R'].values,difgas1['Z'].values,difgas1['Tk'].values))

    # List of target items that we are interpolating for:
    target_list     =   ['index','m_dust','Tk_DNG','Tk_DIG',\
                        'fm_DNG','fm_H_DNG','fm_HI_DNG','fm_HII_DNG','fm_H2_DNG',\
                        'fm_C_DNG','fm_CII_DNG','fm_CIII_DNG','fm_CIV_DNG','fm_CO_DNG',\
                        'fm_H_DIG','fm_HI_DIG','fm_HII_DIG','fm_H2_DIG',\
                        'fm_C_DIG','fm_CII_DIG','fm_CIII_DIG','fm_CIV_DIG','fm_CO_DIG']
    for line in lines: target_list.append('L_'+line)
    for line in lines: target_list.append('f_'+line+'_DNG')

    for target in target_list:
        # Make grid of corresponding target grid values:
        target_grid         =   np.zeros([len(cloudy_grid_param['nHs']),len(cloudy_grid_param['Rs']),len(cloudy_grid_param['Zs']),len(cloudy_grid_param['Tks'])])
        i                   =   0
        for i1 in range(0,len(cloudy_grid_param['nHs'])):
            for i2 in range(0,len(cloudy_grid_param['Rs'])):
                for i3 in range(0,len(cloudy_grid_param['Zs'])):
                    for i4 in range(0,len(cloudy_grid_param['Tks'])):
                        target_grid[i1,i2,i3,i4]    =   cloudy_grid[target][i]
                        i                           +=  1
        # Make function that will do the interpolation:
        interp          =   RegularGridInterpolator((cloudy_grid_param['nHs'],cloudy_grid_param['Rs'],cloudy_grid_param['Zs'],cloudy_grid_param['Tks']), target_grid, method = 'linear')
        # And interpolate:
        difgas[target]  =   interp(dif)

    # Find number of closest models!
    model_numbers               =   np.zeros([len(cloudy_grid_param['nHs']),len(cloudy_grid_param['Rs']),len(cloudy_grid_param['Zs']),len(cloudy_grid_param['Tks'])])
    j                           =   0
    for i1 in range(0,len(cloudy_grid_param['nHs'])):
        for i2 in range(0,len(cloudy_grid_param['Rs'])):
            for i3 in range(0,len(cloudy_grid_param['Zs'])):
                for i4 in range(0,len(cloudy_grid_param['Tks'])):
                    model_numbers[i1,i2,i3,i4]      =   j
                    j                               +=  1
    difgas['closest_model_i']   =   np.zeros(ndif)
    for i in range(0,ndif):
        lnH                         =   difgas1['nH'][i]
        lR                          =   difgas1['R'][i]
        lZ                          =   difgas1['Z'][i]
        lTk                         =   difgas1['Tk'][i]
        # Find closest index in each direction of parameter grid:
        i_nH                        =   np.argmin(abs(cloudy_grid_param['nHs']-lnH))
        i_R                         =   np.argmin(abs(cloudy_grid_param['Rs']-lR))
        i_Z                         =   np.argmin(abs(cloudy_grid_param['Zs']-lZ))
        i_Tk                        =   np.argmin(abs(cloudy_grid_param['Tks']-lTk))
        difgas['closest_model_i'][i] = int(model_numbers[i_nH,i_R,i_Z,i_Tk])

    # Calculate mass fractions in ionized vs neutral diffuse gas:

    difgas['m_DNG']         =   difgas['m']*difgas['fm_DNG']
    difgas['m_H_DNG']       =   difgas['m_DNG']*difgas['fm_H_DNG']
    difgas['m_HI_DNG']      =   difgas['m_DNG']*difgas['fm_HI_DNG']
    difgas['m_H2_DNG']      =   difgas['m_DNG']*difgas['fm_H2_DNG']
    difgas['m_HII_DNG']     =   difgas['m_DNG']*difgas['fm_HII_DNG']
    difgas['m_C_DNG']       =   difgas['m_DNG']*difgas['fm_C_DNG']
    difgas['m_CII_DNG']     =   difgas['m_DNG']*difgas['fm_CII_DNG']
    difgas['m_CIII_DNG']    =   difgas['m_DNG']*difgas['fm_CIII_DNG']
    difgas['m_CIV_DNG']     =   difgas['m_DNG']*difgas['fm_CIV_DNG']
    difgas['m_CO_DNG']      =   difgas['m_DNG']*difgas['fm_CO_DNG']

    difgas['m_DIG']         =   difgas['m']*(1-difgas['fm_DNG'])
    difgas['m_H_DIG']       =   difgas['m_DIG']*difgas['fm_H_DIG']
    difgas['m_HI_DIG']      =   difgas['m_DIG']*difgas['fm_HI_DIG']
    difgas['m_H2_DIG']      =   difgas['m_DIG']*difgas['fm_H2_DIG']
    difgas['m_HII_DIG']     =   difgas['m_DIG']*difgas['fm_HII_DIG']
    difgas['m_C_DIG']       =   difgas['m_DIG']*difgas['fm_C_DIG']
    difgas['m_CII_DIG']     =   difgas['m_DIG']*difgas['fm_CII_DIG']
    difgas['m_CIII_DIG']    =   difgas['m_DIG']*difgas['fm_CIII_DIG']
    difgas['m_CIV_DIG']     =   difgas['m_DNG']*difgas['fm_CIV_DIG']
    difgas['m_CO_DIG']      =   difgas['m_DIG']*difgas['fm_CO_DIG']

    print('Total diffuse gas mass: '+str(sum(difgas['m'])))
    print('Total diffuse neutral gas (DNG) mass: '+str(sum(difgas['m_DNG']))+' or '+str(sum(difgas['m_DNG'])/sum(difgas['m'])*100.)+' %')
    print('Total diffuse ionized gas (DIG) mass: '+str(sum(difgas['m_DIG']))+' or '+str(sum(difgas['m_DIG'])/sum(difgas['m'])*100.)+' %')
    print('Check sum: '+str(sum(difgas['m'])/(sum(difgas['m_DNG'])+sum(difgas['m_DIG']))*100.))

    # Prune the dataframe a bit:
    difgas1                 =   difgas[['x','y','z','vx','vy','vz','vel_disp_gas',\
                                'm','nH','R','Z','Tk','FUV',\
                                'm_DNG','m_DIG','m_H_DNG','m_H_DIG','m_HI_DNG','m_HI_DIG',\
                                'm_H2_DNG','m_H2_DIG','m_HII_DNG','m_HII_DIG',\
                                'm_C_DNG','m_C_DIG','m_CII_DNG','m_CII_DIG','m_CIV_DNG','m_CIV_DIG','m_CIII_DNG','m_CIII_DIG','m_CO_DNG','m_CO_DIG',\
                                'm_dust','Tk_DNG','Tk_DIG','closest_model_i']].copy()

    # Store line emission from ionized vs neutral diffuse gas
    for line in lines:
        difgas1['L_'+line]          =   difgas['L_'+line]
        difgas1['L_'+line+'_DNG']   =   difgas['L_'+line]*difgas['f_'+line+'_DNG']
        difgas1['L_'+line+'_DIG']   =   difgas['L_'+line]*(1.-difgas['f_'+line+'_DNG'])

    print('Total '+line+' luminosity: %s Lsun' % difgas1['L_'+line][i])
    print('with %s %%from DNG' % (difgas['f_'+line+'_DNG'][i]*100.))

    # Dust masses:
    difgas1['m_dust_DNG']    =   difgas1['m_dust']/difgas1['m']*difgas1['m_DNG']
    difgas1['m_dust_DIG']    =   difgas1['m_dust']/difgas1['m']*difgas1['m_DIG']

    # Gas phases within diffuse gas:
    print('In DNG:')
    print(str.format("{0:.2f}",sum(difgas1['m_H2_DNG'])/sum(difgas1['m_H_DNG'])*100.)+' %\t\tof hydrogen is molecular')
    print(str.format("{0:.2f}",sum(difgas1['m_HI_DNG'])/sum(difgas1['m_H_DNG'])*100.)+' %\t\tof hydrogen is atomic')
    print(str.format("{0:.2f}",sum(difgas1['m_HII_DNG'])/sum(difgas1['m_H_DNG'])*100.)+' %\t\tof hydrogen is ionized')
    print('In DIG:')
    print(str.format("{0:.2f}",sum(difgas1['m_H2_DIG'])/sum(difgas1['m_H_DIG'])*100.)+' %\t\tof hydrogen is molecular')
    print(str.format("{0:.2f}",sum(difgas1['m_HI_DIG'])/sum(difgas1['m_H_DIG'])*100.)+' %\t\tof hydrogen is atomic')
    print(str.format("{0:.2f}",sum(difgas1['m_HII_DIG'])/sum(difgas1['m_H_DIG'])*100.)+' %\t\tof hydrogen is ionized')

    print('Mean values of diffuse gas:')
    print('nH: %s' % (np.mean(np.log10(difgas1['nH']))))
    print('R: %s' % (np.mean(np.log10(difgas1['R']))))
    print('Z: %s > 0' % (np.mean(np.log10(difgas1['Z'][difgas1['Z']>0]))))
    print('Tk: %s' % (np.mean(np.log10(difgas1['Tk']))))

    print('Total [CII] luminosity from diffuse gas (DIG/DNG): '+str(sum(difgas1['L_CII_DNG']+difgas1['L_CII_DIG'])/1e8)+' x 10^8 L_sun')

    difgas1.to_pickle(dif_path+'z'+'{:.2f}'.format(zred)+'_'+galname+'_dif'+ext_DIFFUSE0+'_em.gas')

    dif_results             =   {'M_DNG':sum(difgas1['m_DNG']),'M_DIG':sum(difgas1['m_DIG']),\
                                'M_H2_DNG':sum(difgas1['m_H2_DNG']),'M_C_DNG':sum(difgas1['m_C_DNG']),\
                                'M_CII_DNG':sum(difgas1['m_CII_DNG']),'M_CIII_DNG':sum(difgas1['m_CIII_DNG']),\
                                'M_CO_DNG':sum(difgas1['m_CO_DNG']),'M_dust_DNG':sum(difgas1['m_dust_DNG']),\
                                'M_H2_DIG':sum(difgas1['m_H2_DIG']),'M_C_DIG':sum(difgas1['m_C_DIG']),\
                                'M_CII_DIG':sum(difgas1['m_CII_DIG']),'M_CIII_DIG':sum(difgas1['m_CIII_DIG']),\
                                'M_CO_DIG':sum(difgas1['m_CO_DIG']),'M_dust_DIG':sum(difgas1['m_dust_DIG'])}
    for line in lines:
        dif_results['L_'+line+'_DNG']   =   np.sum(difgas1['L_'+line+'_DNG'])
        dif_results['L_'+line+'_DIG']   =   np.sum(difgas1['L_'+line+'_DIG'])
                     

    # make a dataframe to create the line profile    
    data                    =   pd.DataFrame({'v_pos':difgas1['vx'].values,'vel_disp':difgas1['vel_disp_gas'],'L_CII':difgas1['L_CII'].values,'L_OI':difgas1['L_OI'].values})
    data                    =   data[difgas['m'] > 0]
    data.to_pickle('sigame/temp/line_profiles/'+galname+'.diffuse')

    return(dif_results)
